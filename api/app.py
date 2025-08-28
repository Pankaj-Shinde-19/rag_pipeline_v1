import time
import os
import pickle
from collections import deque
from typing import Dict, Any, List

# FastAPI and related imports
from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Qdrant and model imports
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client.http.models import PointStruct, VectorParams

# Google Gemini API
from google import genai

# ========================== Initialization ==========================

app = FastAPI()

# Enable CORS to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as needed for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize embedding and ranking models
embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2", device="cpu")

# Initialize Qdrant client for document vector database interaction
collection_name = "documents"
qdrant_client = QdrantClient(host="localhost", port=6333)
# Check if the collection exists; create it if it doesn't
if not qdrant_client.collection_exists(collection_name):
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=embedding_model.get_sentence_embedding_dimension(),  # Dimensionality of embeddings
            distance="Cosine",  # Metric for similarity search
        ),
    )

# Set up a list to maintain conversation history (max length 2)
conversation_history = []

# Gemini client
client = genai.Client(api_key="Gemini_Key")  # <-- Set API key in env

# ========================== Utility Functions ==========================


def get_query_embedding(query: str):
    """Generate embeddings for a given query using the embedding model."""
    return embedding_model.encode([query])[0]


def retrieve_documents(query: str, top_k: int = 8):
    """Retrieve top-k relevant documents for a query from Qdrant and rank them."""
    timings: Dict[str, float] = {}
    start_retrieval = time.time()

    # Step 1: Generate embedding for the query
    query_vector = get_query_embedding(query)
    timings["embedding_time"] = time.time() - start_retrieval

    # Step 2: Perform vector search in Qdrant
    start_search = time.time()
    results = qdrant_client.search(
        collection_name="documents", query_vector=query_vector, limit=top_k
    )
    timings["qdrant_search_time"] = time.time() - start_search

    # Step 3: Extract and rank retrieved documents
    candidate_docs = [result.payload["text"] for result in results]
    if candidate_docs:
        start_ranking = time.time()
        pairs = [[query, doc] for doc in candidate_docs]
        scores = cross_encoder.predict(pairs)
        timings["ranking_time"] = time.time() - start_ranking

        ranked_results = sorted(
            zip(candidate_docs, scores), key=lambda x: x[1], reverse=True
        )
        timings["total_retrieval_time"] = sum(timings.values())
        return [doc for doc, score in ranked_results], timings

    timings["total_retrieval_time"] = sum(timings.values())
    return [], timings


def generate_response(context: str, query: str, history: list):
    """Generate a response using Gemini API based on context and conversation history."""

    history_text = "\n".join([f"Q: {h['query']}\nA: {h['response']}" for h in history])

    input_text = (
        f"Conversation History:\n{history_text}\n"
        f"Context: {context}\n"
        f"Question: {query}\n"
        f"Please provide a concise and accurate response."
    )

    start_chat = time.time()
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=input_text
    )
    generation_time = time.time() - start_chat

    return response.text.strip(), generation_time


# ========================== FastAPI Routes ==========================


@app.get("/health")
def health():
    """Health check endpoint to verify service availability."""
    return {"STATUS": "OK"}


@app.post("/ask")
def ask(query: str = Body(..., embed=True)):
    """Handle user queries and generate responses using Gemini API."""
    if not query:
        return JSONResponse({"error": "No query provided"}, status_code=400)

    overall_start = time.time()

    # Step 1: Retrieve relevant documents
    retrieved_docs, retrieval_timings = retrieve_documents(query, top_k=8)

    # Step 2: Generate context
    context = " ".join(retrieved_docs)

    # Step 3: Generate response
    response, gen_time = generate_response(context, query, conversation_history)

    if len(conversation_history) < 2:
        conversation_history.append({"query": query, "response": response})
    else:
        conversation_history.pop(0)
        conversation_history.append({"query": query, "response": response})

    overall_end = time.time()

    timings = {
        "embedding_time": retrieval_timings.get("embedding_time", 0),
        "qdrant_search_time": retrieval_timings.get("qdrant_search_time", 0),
        "ranking_time": retrieval_timings.get("ranking_time", 0),
        "total_retrieval_time": retrieval_timings.get("total_retrieval_time", 0),
        "llm_response_time": gen_time,
        "total_api_processing_time": overall_end - overall_start,
    }

    return {
        "response": response,
        "timings": timings,
    }
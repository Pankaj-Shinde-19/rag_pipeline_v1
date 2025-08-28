import asyncio
import os
import time
import hashlib
import pdfplumber
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams
from watchdog.events import FileSystemEventHandler

# Initialize the sentence embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Define the collection name for Qdrant
collection_name = "documents"

# Initialize Qdrant client
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

def generate_unique_id(pdf_path, chunk_index):
    """
    Generates a unique ID for each chunk based on the PDF path and chunk index.

    Args:
        pdf_path (str): Path to the PDF file.
        chunk_index (int): Index of the chunk.

    Returns:
        str: Unique ID for the chunk.
    """
    unique_id = hashlib.md5(f"{pdf_path}_{chunk_index}".encode()).hexdigest()
    return unique_id

async def process_pdf_for_embeddings(pdf_path):
    """
    Extracts text from a PDF, generates embeddings for chunks of text, and indexes them in Qdrant.

    Args:
        pdf_path (str): Path to the PDF file.
    """
    try:
        # Step 1: Extract text from the PDF
        extracted_text = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:  # Ensure text is not None or empty
                    extracted_text.append(text)
        corpus = " ".join(extracted_text)

        # Step 2: Split the text into chunks for embedding
        chunks = corpus.split(". ")  # Split by sentences
        print(f"Extracted {len(chunks)} chunks from {pdf_path}")

        # Step 3: Generate embeddings for each chunk
        embeddings = embedding_model.encode(chunks)
        print(f"Generated {len(embeddings)} embeddings")

        # Step 4: Prepare data points for Qdrant
        points = [
            PointStruct(
                id=generate_unique_id(pdf_path, i),
                vector=emb,
                payload={"text": chunk, "pdf_path": pdf_path, "chunk_index": i}
            )
            for i, (emb, chunk) in enumerate(zip(embeddings, chunks))
        ]

        # Step 5: Insert the points into the Qdrant collection in batches
        batch_size = 100  # Adjust based on your system's capabilities
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            qdrant_client.upsert(collection_name=collection_name, points=batch)

        print(f"Processed and indexed: {pdf_path}")
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")

class PDFHandler(FileSystemEventHandler):
    """
    Handles new PDF files created in the monitored directory by:
    - Waiting for the file to be fully written to disk.
    - Extracting text from the PDF.
    - Generating embeddings for chunks of text.
    - Storing the embeddings in the Qdrant vector database.
    """

    def on_created(self, event):
        """
        Triggered when a new file is created in the monitored directory.
        Ignores directories and processes only PDF files.
        """
        if not event.is_directory and event.src_path.endswith(".pdf"):
            # Skip temporary or incomplete files
            if "~BROMIUM" in event.src_path:
                print(f"Skipping temporary file: {event.src_path}")
                return

            print(f"New PDF file detected: {event.src_path}")

            # Wait until the file is fully written to disk
            self.wait_for_fileready(event.src_path)

            # Process the PDF file for embedding and indexing
            asyncio.run(process_pdf_for_embeddings(event.src_path))

    def wait_for_fileready(self, path, timeout=60):
        """
        Waits until the file is fully written to disk by monitoring its size stability.

        Args:
            path (str): Path to the file.
            timeout (int): Maximum time (in seconds) to wait for the file to stabilize.
        """
        start_time = time.time()
        last_size = -1
        stable_count = 0

        while True:
            try:
                current_size = os.path.getsize(path)

                # Check if file size is stable
                if current_size == last_size:
                    stable_count += 1
                else:
                    stable_count = 0

                # File is ready if size is stable for 3 consecutive checks
                if stable_count > 2:
                    return

                # Timeout reached
                if (time.time() - start_time) > timeout:
                    print("Timeout reached while waiting for file to be ready.")
                    return

                last_size = current_size
                time.sleep(1)
            except FileNotFoundError:
                # File might not exist yet, wait and retry
                time.sleep(1)

# Example usage
if __name__ == "__main__":
    import watchdog.observers

    # Monitor a directory for new PDF files
    path_to_watch = "/path/to/pdf/folder"
    event_handler = PDFHandler()
    observer = watchdog.observers.Observer()
    observer.schedule(event_handler, path=path_to_watch, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()



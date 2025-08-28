

# Retrivel Augmentation Generation

1. **Add PDFs Under Data Directory**  
   The `Data` directory, which contains all the PDF files used in this project, you should have pdf under that directory before starting your application.
   

---

## Project Hierarchy
Here's an overview of the project's structure:
```plaintext
on_prem_gen_ai_llm_v1/
├── api/
│   ├── __init__.py
│   └── app.py
│   └── requirements.txt
├── frontend/
│   ├── __init__.py
│   ├── chatbot_ui.py
│   ├── logo.png
│   └── style.css
├── handlers/
│   ├── __init__.py
│   ├── pdf_filehandler.py
│   ├── watcher.py
│   └── requirements.txt
├── data/
│   └── pdfs/  # Directory for storing PDF training data
└── README.md

```

## Installation
To get started, follow these steps:

1. Install dependencies:
   ```plaintext
   pip install -r requirements.txt - for "handlers" and "api" directory.
   ```
   
---

## Usage
Follow the steps below to use the application:

1. **Run the Watcher Script:**  
   This monitors the pdfs directory for new uploads.
   ```plaintext
   python handlers/watcher.py
   ```
2. **Start the Flask API**:  
   Ensure the backend is up and running.
   ```plaintext
   python api/app.py
   ```
3. **Run the Chatbot UI:**  
   Launch the user interface for interactions.
   ```plaintext
   streamlit run frontend/chatbot_ui.py
   ```
4. **PDF Uploads:**  
   Upload PDF files to the pdfs directory. The pdf_filehandler.py module processes them asynchronously.

---
## Qdrant for Storing Embeddings
Qdrant is a vector search engine that can store and retrieve document embeddings efficiently. Follow these steps to set up Qdrant using Docker:

**Prerequisites**

Ensure Docker is installed on your system. You can download it from [Docker's official site](https://www.docker.com/).

**Steps to Set Up Qdrant**
1. **Pull the Qdrant Docker Image:**
   
   Run the following command to pull the latest Qdrant Docker image:
   ```plaintext
    docker pull qdrant/qdrant
   ```
2. **Run the Qdrant Docker Container:**
   
   Start a Qdrant container using the following command:
    ```plaintext
    docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
   ```
3. **Service Links in Sequence:**

    Below is the sequence of services and their respective links (accessible through the container logs or UI):

- **API:** Link to the API service.

- **Qdrant:** Link to the Qdrant vector database.

- **Ollama:** Link for the Ollama service.

- **Watcher:** Link for the PDF watcher.

- **Streamlit:** Link for the Streamlit UI.



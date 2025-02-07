# RAG System API

## Overview
This project implements a **Retrieval-Augmented Generation (RAG) System API** using FastAPI. The system utilizes **ChromaDB** for vector storage and **LangChain Ollama** for embedding and querying. It enables document embedding, question answering with contextual retrieval, and monitoring request performance.

## Features
- **Document Embedding**: Supports uploading and embedding `.pdf` and `.md` files.
- **Query System**: Retrieves relevant documents from ChromaDB and generates responses using LLM (Ollama).
- **Monitoring**: Tracks the success and failure rates of queries.
- **CORS Support**: Configured for frontend integration.
- **Logging**: Provides detailed logs for debugging and performance tracking.

## Tech Stack
- **FastAPI**: API framework
- **ChromaDB**: Vector database for document storage
- **LangChain**: Framework for LLM interaction
- **Ollama**: Embedding and querying model
- **Docker**: For containerized deployment
- **dotenv**: Environment variable management

## Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.12+
- Docker (optional, for containerized deployment)

### Setup
1. Clone the repository:
   ```sh
   git clone https://github.com/verlineguo/rag-system.git
   cd rag-system
   ```
2. Create and activate a virtual environment:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
4. Create an `.env` file and configure the environment variables:
   ```env
   TEMP_FOLDER=./_temp
   CHROMA_PATH=./chroma_db
   COLLECTION_NAME=local-rag
   TEXT_EMBEDDING_MODEL=llama3.2
   LLM_MODEL=llama3.2
   ```
5. Run the application:
   ```sh
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

## API Endpoints
### Root Check
```http
GET /
```
Returns API status.

### Embedding Files
```http
POST /embed
```
**Payload:** File upload (`.pdf` or `.md`)

**Response:**
```json
{"message": "File embedded successfully"}
```

### Query System
```http
POST /query
```
**Payload:**
```json
{
  "query": "What is the capital of France?"
}
```

**Response:**
```json
{
  "response": "The capital of France is Paris.",
  "context": "Information retrieved from relevant documents.",
  "sources": "source.pdf (Chunk 1)",
  "response_time": "0.42 seconds",
  "token_usage": 128
}
```

### Monitoring
```http
GET /monitoring
```
**Response:**
```json
{
  "success_count": 10,
  "failure_count": 2
}
```

## Deployment with Docker
1. Build the Docker image:
   ```sh
   docker build -t rag-system .
   ```
2. Run the container:
   ```sh
   docker-compose up -d
   ```

## Logging
Logs are saved in the console output and include request processing times, errors, and token usage.

## Future Enhancements
- Implement authentication & role-based access control
- Expand supported document formats
- Add GPU acceleration for faster inference
- Basic evaluation metrics for answer quality

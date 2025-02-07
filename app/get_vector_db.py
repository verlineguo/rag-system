import os
import logging
from pathlib import Path
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from fastapi import HTTPException

# Configure logging to track information and errors
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration for Chroma database storage path and embedding model
CHROMA_PATH = Path(os.getenv('CHROMA_PATH', 'chroma_db'))  # Path to store Chroma database
COLLECTION_NAME = os.getenv('COLLECTION_NAME', 'local-rag')  # Collection name used in Chroma
TEXT_EMBEDDING_MODEL = os.getenv('TEXT_EMBEDDING_MODEL', 'llama3.2')  # Embedding model to be used

def get_vector_db():
    """Initialize and connect to the Chroma database with Ollama Embeddings."""
    try:
        # Check if the Chroma storage directory exists, if not, create it
        if not CHROMA_PATH.exists():
            CHROMA_PATH.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created Chroma persistence directory at {CHROMA_PATH}")

        # Initialize the embedding model using Ollama
        embedding = OllamaEmbeddings(model=TEXT_EMBEDDING_MODEL)
        logger.info(f"Using embedding model: {TEXT_EMBEDDING_MODEL}")

        # Initialize Chroma database connection
        db = Chroma(
            collection_name=COLLECTION_NAME,  # Name of the collection in Chroma
            persist_directory=str(CHROMA_PATH),  # Path to persist Chroma data
            embedding_function=embedding  # Embedding function to be used for storing embeddings
        )

        logger.info(f"Successfully connected to Chroma DB at {CHROMA_PATH}")
        return db
    except Exception as e:
        # If an error occurs during initialization, log the error and raise an exception
        error_message = f"Failed to initialize vector database: {str(e)}"
        logger.error(error_message)  # Log error for debugging
        raise HTTPException(status_code=500, detail=error_message)  # Raise HTTP error with the message

import os
import logging
import traceback
from pathlib import Path
from datetime import datetime
import shutil

from fastapi import UploadFile, HTTPException
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from PyPDF2 import PdfReader
from app.get_vector_db import get_vector_db

# Load environment variables from .env file
load_dotenv()

# Setup logging for error tracking and information logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define temporary folder for storing uploaded files
TEMP_FOLDER = Path(os.getenv("TEMP_FOLDER", "./data"))
TEMP_FOLDER.mkdir(parents=True, exist_ok=True)  # Create folder if not exists

# Configuration for text chunk size and overlap from environment variables
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1024))  # Size of each chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))  # Overlap between chunks

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost")
OLLAMA_PORT = os.getenv("OLLAMA_PORT", "11434")

ollama_url = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/embeddings"


def allowed_file(filename: str) -> bool:
    """Check if the uploaded file is a PDF or Markdown file based on its extension"""
    # Validate file type by checking the extension (PDF or Markdown)
    return filename.lower().endswith((".pdf", ".md"))


def save_file(file: UploadFile) -> Path:
    """Save the uploaded file to the temporary storage"""
    try:
        # Generate a unique filename using timestamp to avoid overwriting
        timestamp = int(datetime.now().timestamp())
        filename = f"{timestamp}_{file.filename}"
        file_path = TEMP_FOLDER / filename

        # Save the file to the specified path
        with file_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)

        return file_path
    except Exception as e:
        # Log error if saving the file fails
        logger.error(f"File saving failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"File saving failed: {str(e)}")


def load_and_split_data(file_path: Path):
    """Load the content of the file and split it into chunks for embedding"""
    try:
        text = ""
        if file_path.suffix == ".pdf":
            # If it's a PDF file, extract text from each page using PyPDF2
            reader = PdfReader(str(file_path))
            text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        elif file_path.suffix == ".md":
            # If it's a Markdown file, read the content directly
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

        # Ensure the file contains readable text
        if not text.strip():
            raise HTTPException(status_code=400, detail="File does not contain readable text.")

        # Split text into smaller chunks for embedding
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        chunks = text_splitter.split_text(text)

        # Create Document objects from chunks, each representing a part of the text
        documents = [
            Document(
                page_content=chunk,
                metadata={"source": file_path.name, "chunk_index": i},
            )
            for i, chunk in enumerate(chunks)
        ]
        return documents
    except Exception as e:
        # Log errors that occur during file loading and splitting
        logger.error(f"Document processing failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")


def process_and_store_embedding(file: UploadFile):
    """Process PDF/Markdown file, generate embeddings, and store them in vector store"""
    file_path = None
    try:
        # Check if the uploaded file is valid
        if not allowed_file(file.filename):
            raise HTTPException(status_code=400, detail="Only PDF or Markdown files are allowed.")

        # Save the uploaded file to temporary storage
        file_path = save_file(file)
        logger.info(f"File uploaded: {file.filename}")

        # Load and split the file into chunks
        documents = load_and_split_data(file_path)
        
        # Get the vector database instance and add the documents
        db = get_vector_db()
        db.add_documents(documents)

        # Log success and return the result
        logger.info(f"Successfully embedded {len(documents)} chunks.")
        return {"message": "File embedded successfully", "chunks": len(documents)}

    except Exception as e:
        # Log and raise an error if something goes wrong
        logger.error(f"Embedding error: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

    finally:
        # Clean up temporary files after processing
        if file_path and file_path.exists():
            try:
                file_path.unlink()  # Remove the temp file after processing
            except Exception as e:
                logger.warning(f"Failed to delete temp file: {file_path}. Error: {e}")

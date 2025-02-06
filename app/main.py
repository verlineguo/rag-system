import os
import time
import logging
import traceback
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.embed import process_and_store_embedding as embed
from app.query import query
from app.monitoring import get_monitoring_status as monitoring_status, update_success_rate

# Load environment variables from .env file
load_dotenv()

# Configure logging to track API requests, errors, and processing times
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define temporary storage directory for uploaded files
TEMP_FOLDER = os.getenv("TEMP_FOLDER", "./_temp")
os.makedirs(TEMP_FOLDER, exist_ok=True)  # Ensure the directory exists

# Initialize FastAPI application
app = FastAPI(title="RAG System API", version="1.0")

# Enable Cross-Origin Resource Sharing (CORS) - Adjust settings for production!
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to allow only frontend domains in production
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["Authorization", "Content-Type"],
)

# Root endpoint to check API status
@app.get("/", tags=["Root"], summary="Check API status")
def read_root():
    return {"message": "Welcome to the RAG system!"}

# Define a request model for query input
class QueryRequest(BaseModel):
    query: str

# Allowed file extensions for upload
ALLOWED_EXTENSIONS = ["pdf", "md"]

def allowed_file(filename: str) -> bool:
    """Check if the uploaded file has an allowed extension."""
    return filename.split('.')[-1].lower() in ALLOWED_EXTENSIONS

# Middleware to monitor response time for API requests
@app.middleware("http")
async def add_process_time_header(request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = time.perf_counter() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    logger.info(f"Request processed in {process_time:.4f} seconds")
    return response

# Endpoint for file embedding
@app.post("/embed", tags=["Embedding"], summary="Upload a file for embedding")
async def route_embed(file: UploadFile = File(...)):
    """Handles file uploads and processes embeddings."""
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")

    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="File type not allowed. Only PDF and Markdown are supported.")

    try:
        embedded = await embed(file)
        if embedded:
            return {"message": "File embedded successfully"}
        raise HTTPException(status_code=500, detail="File embedding failed")
    except Exception as e:
        logger.error(f"Embedding error: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

# Endpoint for querying the RAG system
@app.post("/query", tags=["Retrieval & QA"], summary="Ask a question to the RAG system")
async def route_query(query_input: QueryRequest):
    """Processes a query and retrieves relevant information."""
    try:
        start_time = time.time()  # Measure response time
        logger.info(f"Processing query: {query_input.query}")
        response_data = query(query_input.query)  # Call the query function
        
        # Update success rate if evaluation metrics are present
        if response_data.get("evaluation_metrics"):
            update_success_rate(success=True)
            
        # Ensure response data contains necessary fields
        if "response" in response_data and "context" in response_data:
            response_time = time.time() - start_time  
            logger.info(f"Query processed in {response_time:.2f} seconds")
            return {
                "response": response_data["response"], 
                "context": response_data["context"], 
                "response_time": f"{response_time:.2f} seconds"
            }
        else:
            raise HTTPException(status_code=400, detail="Response or context not found")
    except Exception as e:
        logger.error(f"Query error: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

# Endpoint for monitoring API usage
@app.get("/monitoring", tags=["Monitoring"], summary="Check monitoring status")
def get_monitoring_status():
    """Returns the success/failure rate and overall token usage statistics."""
    return monitoring_status()

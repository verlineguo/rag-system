import os
import time
import logging
import traceback
from fastapi import HTTPException
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers.multi_query import MultiQueryRetriever
# from langchain_core.runnables import RunnableLambda
from app.get_vector_db import get_vector_db
from app.monitoring import update_success_rate

# Load model name from environment variables
LLM_MODEL = os.getenv('LLM_MODEL', 'llama3.2')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_prompt():
    """
    Generate prompt templates for query expansion and answering.
    
    Returns:
        tuple: (QUERY_PROMPT, ANSWER_PROMPT)
    """
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""Generate five different versions of the question 
        to retrieve relevant documents from a vector database.
        Original question: {question}"""
    )

    ANSWER_PROMPT = ChatPromptTemplate.from_template(
        """Answer the question using ONLY the following context:
        {context}
        Question: {question}"""
    )

    return QUERY_PROMPT, ANSWER_PROMPT

def query(input_text: str):
    """
    Processes a query by retrieving context from the vector database and generating a response.

    Args:
        input_text (str): The query/question from the user.

    Returns:
        dict: A dictionary containing the response, retrieved context, sources, token usage, 
              and processing time.
    """
    
    if not input_text:
        raise HTTPException(status_code=400, detail="Query input is empty")

    try:
        start_time = time.time()  
        logger.info(f"Processing query: {input_text}")

        # Initialize the language model
        llm = ChatOllama(model=LLM_MODEL)

        # Retrieve the vector database instance
        db = get_vector_db()

        # Get the prompt templates
        QUERY_PROMPT, ANSWER_PROMPT = get_prompt()

        # MultiQueryRetriever generates multiple versions of the question to improve retrieval accuracy
        retriever = MultiQueryRetriever.from_llm(
            llm=llm,  
            retriever=db.as_retriever(),
            prompt=QUERY_PROMPT
        )

        # Retrieve relevant documents
        retrieved_docs = retriever.invoke(input_text)

        # Extract text content from retrieved documents
        context_text = "\n".join([doc.page_content for doc in retrieved_docs]) if isinstance(retrieved_docs, list) else str(retrieved_docs)

        # Extract metadata for sources
        sources = "\n".join([f"Source: {doc.metadata.get('source', 'Unknown Source')} (Chunk {doc.metadata.get('chunk_index', 'Unknown Chunk')})"
                             for doc in retrieved_docs])

        # If no relevant documents are found, return an error
        if not context_text:
            logger.warning("No relevant documents retrieved.")
            raise HTTPException(status_code=404, detail="No relevant information found.")

        # Format the answer prompt with the retrieved context
        formatted_prompt = ANSWER_PROMPT.invoke({"context": context_text, "question": input_text})
        
        # Invoke the language model to generate a response
        response = llm.invoke(formatted_prompt)

        # Check if token usage tracking is available
        token_usage = 0  # Default if not available
        if hasattr(response, 'token_usage') and hasattr(response.token_usage, 'total'):
            token_usage = response.token_usage.total

        logger.info(f"Tokens used: {token_usage}")

        # Parse the response from the model
        try:
            parsed_response = StrOutputParser().parse(response)
        except Exception as e:
            logger.error(f"Failed to parse response: {str(e)}")
            parsed_response = "Error: Unable to parse response from the model."
                
        # Calculate response time
        response_time = time.time() - start_time  
        logger.info(f"Query processed in {response_time:.2f} seconds")

        # Update monitoring status for successful queries
        update_success_rate(success=True)

        # Return the response along with retrieved context, sources, and token usage
        return {
            "response": parsed_response, 
            "context": context_text, 
            "sources": sources,  
            "token_usage": token_usage,  
            "response_time": f"{response_time:.2f} seconds"
        }

    except Exception as e:
        logger.error(f"Query processing failed: {str(e)}\n{traceback.format_exc()}")
                
        # Provide detailed error messages for debugging
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

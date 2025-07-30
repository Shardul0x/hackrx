import os
import asyncio
import json
import logging
from typing import List, Dict, Any

import numpy as np
import faiss
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq

# These libraries are listed in requirements.txt but are assumed to be used
# for local, one-time document text extraction into 'extracted_texts/'
# For runtime on Render, we primarily rely on the pre-extracted .txt files.
# from unstructured.partition.auto import partition
# from pdf2image import convert_from_bytes
# import fitz # PyMuPDF
# from docx import Document # python-docx
# from extract_msg import Message # extract-msg

# Load environment variables from .env file (for local development)
load_dotenv()

# --- Configuration ---
# IMPORTANT: This directory MUST contain your pre-extracted plain .txt files.
# You need to run a separate script locally to convert your original documents
# (PDFs, DOCXs, EMLs) into plain text files and save them here.
EXTRACTED_TEXTS_DIR = "extracted_texts"

# Placeholder for your actual embedding model.
# For best performance on a free plan, consider a small, fast Sentence Transformer
# model like 'all-MiniLM-L6-v2' for local inference.
EMBEDDING_MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5" # Replace with a real model name or comment out if using API
CHUNK_SIZE = 500 # Smaller chunks can be faster to embed and retrieve
CHUNK_OVERLAP = 100

# --- Global Variables for In-Memory Data ---
faiss_index = None
all_text_chunks: List[str] = []
embedding_model = None # Will be initialized if using local embedding model
groq_client = None

# --- Logger for better debugging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Optimized Document Q&A with LLM",
    description="An API to answer questions based on your documents using a Large Language Model, optimized for quick responses."
)

# --- Utility Functions for Document Processing and Embedding ---

async def get_embeddings(texts: List[str]) -> np.ndarray:
    """
    Generates embeddings for a list of texts.
    Replace this with your actual embedding logic.

    For local inference with SentenceTransformers (uncomment and install 'sentence-transformers'):
    """
    global embedding_model
    if embedding_model is None:
        try:
            logger.info(f"Initializing local embedding model: {EMBEDDING_MODEL_NAME}")
            from sentence_transformers import SentenceTransformer
            embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        except ImportError:
            logger.error("sentence-transformers not installed. Cannot initialize local embedding model.")
            raise RuntimeError("SentenceTransformer library not found. Please install it or use an API.")
        except Exception as e:
            logger.error(f"Error initializing embedding model {EMBEDDING_MODEL_NAME}: {e}")
            raise RuntimeError(f"Failed to load embedding model: {e}")

    logger.info(f"Generating embeddings for {len(texts)} chunks...")
    # This call can be synchronous as sentence-transformers is CPU-bound or uses its own parallelism.
    # If it's a blocking call, asyncio.to_thread will prevent FastAPI from blocking.
    return await asyncio.to_thread(embedding_model.encode, texts, convert_to_numpy=True)
    
    """
    # For a mock embedding (ONLY for testing, not for real functionality):
    logger.warning("Using mock embeddings. Replace with a real embedding model for actual functionality.")
    return np.random.rand(len(texts), 768).astype('float32') # Mock 768-dim embeddings
    """

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Splits text into smaller, overlapping chunks."""
    chunks = []
    words = text.split()
    if not words:
        return []
    
    # Simple word-based chunking. For production, consider more advanced text splitters
    # that respect sentence or paragraph boundaries, like those in LangChain.
    for i in range(0, len(words), chunk_size - chunk_overlap):
        chunk = words[i : i + chunk_size]
        if chunk:
            chunks.append(" ".join(chunk))
    
    return chunks

async def build_in_memory_faiss_index(extracted_texts_dir: str):
    """
    Processes all pre-extracted text documents, generates embeddings,
    and builds a FAISS index in memory.
    This function runs on every application startup on Render's free plan.
    """ # <-- This is line 98, the closing triple quotes were missing in previous response.
    global faiss_index, all_text_chunks # , embedding_model # Uncomment embedding_model if local

    logger.info("Starting in-memory FAISS index build from extracted texts...")
    
    # Reset global lists
    all_text_chunks = []
    
    processed_file_count = 0
    # Ensure the directory exists
    if not os.path.exists(extracted_texts_dir):
        logger.error(f"Directory '{extracted_texts_dir}' not found. Please create it and add .txt files.")
        return

    for filename in os.listdir(extracted_texts_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(extracted_texts_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    full_text = f.read()
                
                chunks = chunk_text(full_text, CHUNK_SIZE, CHUNK_OVERLAP)
                all_text_chunks.extend(chunks)
                processed_file_count += 1
            except Exception as e:
                logger.error(f"Error reading or chunking text from {file_path}: {e}")
                continue

    if not all_text_chunks:
        logger.warning("No text chunks generated from extracted_texts directory. FAISS index will be empty.")
        faiss_index = None # Ensure index is None if no chunks
        return

    logger.info(f"Total chunks generated for indexing: {len(all_text_chunks)} from {processed_file_count} files.")

    # Generate embeddings for all chunks in batches
    batch_size = 64 # Adjust based on your embedding model and available memory
    all_embeddings_list = []
    
    for i in range(0, len(all_text_chunks), batch_size):
        batch = all_text_chunks[i:i + batch_size]
        try:
            batch_embeddings = await get_embeddings(batch)
            all_embeddings_list.append(batch_embeddings)
        except Exception as e:
            logger.error(f"Failed to generate embeddings for batch starting at index {i}: {e}")
            # Depending on severity, you might want to stop or continue
            continue
    
    if all_embeddings_list:
        embeddings = np.vstack(all_embeddings_list)
        dimension = embeddings.shape[1]
        
        # Initialize FAISS index
        faiss_index = faiss.IndexFlatL2(dimension) # L2 distance (Euclidean) for similarity
        faiss_index.add(embeddings) # Add the embeddings to the index
        
        logger.info(f"In-memory FAISS index built with {faiss_index.ntotal} vectors.")
    else:
        logger.warning("No embeddings generated, in-memory FAISS index not built.")
        faiss_index = None

# --- FastAPI Event Handlers ---
@app.on_event("startup")
async def startup_event():
    """
    Builds the in-memory FAISS index and initializes the Groq client
    when the FastAPI application starts.
    """
    global groq_client
    logger.info("Application startup event triggered.")
    
    # Initialize Groq client
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        logger.critical("GROQ_API_KEY environment variable not set. LLM calls will fail.")
        # For production, you might want to raise an error here to prevent startup
        # raise ValueError("GROQ_API_KEY is not set. Cannot start LLM service.")
    else:
        groq_client = Groq(api_key=groq_api_key)
        logger.info("Groq client initialized.")

    # Build the FAISS index in memory
    await build_in_memory_faiss_index(EXTRACTED_TEXTS_DIR)

# --- Request Models ---
class QueryRequest(BaseModel):
    query: str
    top_k: int = 3 # Reduced top_k to retrieve fewer chunks, potentially faster context for LLM

class QueryResponse(BaseModel):
    answer: str
    sources: List[str] # List of relevant document chunks

# --- API Endpoints ---
@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "ok", "message": "API is running"}

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Answers a query by retrieving relevant information from documents
    and using an LLM to synthesize the answer.
    """
    if faiss_index is None or not all_text_chunks:
        # Check if the index is ready
        if not all_text_chunks:
            detail_msg = f"No text content found in '{EXTRACTED_TEXTS_DIR}'. Ensure it contains .txt files."
        else:
            detail_msg = "Document index not ready. Please try again later or check startup logs."
        raise HTTPException(status_code=503, detail=detail_msg)
        
    if groq_client is None:
        raise HTTPException(status_code=500, detail="LLM service not initialized. Check GROQ_API_KEY environment variable.")

    logger.info(f"Received query: '{request.query}'")

    # 1. Embed the query
    query_embedding_np = await get_embeddings([request.query])

    # Ensure query_embedding_np is float32 and 2D for FAISS
    query_embedding_np = query_embedding_np.astype('float32').reshape(1, -1)
    
    # 2. Retrieve relevant chunks using FAISS
    D, I = faiss_index.search(query_embedding_np, request.top_k) # D is distances, I is indices
    
    relevant_chunks = [all_text_chunks[idx] for idx in I[0] if idx != -1] # Filter out -1 if less than top_k results
    
    if not relevant_chunks:
        logger.warning(f"No relevant chunks found for query: '{request.query}'")
        return QueryResponse(answer="I could not find relevant information in the documents.", sources=[])

    # 3. Construct prompt for LLM
    context = "\n\n".join(relevant_chunks)
    prompt = f"Given the following context, please answer the question concisely. If the answer is not directly available in the context, state that you don't have enough information.\n\nContext:\n{context}\n\nQuestion: {request.query}\n\nAnswer:"
    
    logger.info("Sending query to LLM...")

    # 4. Call Groq API asynchronously using asyncio.to_thread for blocking calls
    try:
        chat_completion = await asyncio.to_thread(
            groq_client.chat.completions.create,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                {"role": "user", "content": prompt}
            ],
            model="llama3-8b-8192", # Or your preferred Groq model
            temperature=0.7,
            max_tokens=500 # Limit response length for faster generation
        )
        answer = chat_completion.choices[0].message.content
        logger.info("Received answer from LLM.")
        return QueryResponse(answer=answer, sources=relevant_chunks)
    except Exception as e:
        logger.error(f"Error calling Groq API: {e}", exc_info=True) # exc_info for full traceback
        raise HTTPException(status_code=500, detail=f"Error processing LLM request: {e}")

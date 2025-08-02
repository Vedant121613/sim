from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from rag_pipeline import RAGPipeline
from utils import download_pdf
from authorization import validate_token
import asyncio
import hashlib
from functools import lru_cache
from typing import Dict, List
import aiofiles
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import os
import time
import uvicorn
import json
from datetime import datetime

app = FastAPI()

# Global cache for RAG instances and processed documents
rag_cache: Dict[str, RAGPipeline] = {}
pdf_cache: Dict[str, str] = {}

# Thread pool for CPU-intensive tasks
executor = ThreadPoolExecutor(max_workers=4)

# Request logging configuration
LOG_DIR = "logs"
LOG_FILE = "request_logs.jsonl"

class RequestPayload(BaseModel):
    documents: str  # URL
    questions: list[str]

def log_request(document_url: str, questions: List[str]):
    """Logs incoming requests to a JSON Lines file"""
    os.makedirs(LOG_DIR, exist_ok=True)
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "document_url": document_url,
        "questions": questions
    }
    log_path = os.path.join(LOG_DIR, LOG_FILE)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")

def get_cache_key(url: str) -> str:
    """Generate cache key from URL"""
    return hashlib.md5(url.encode()).hexdigest()

@lru_cache(maxsize=100)
def cached_validate_token(token: str) -> bool:
    """Cache token validation results"""
    return validate_token(token)

async def download_pdf_async(url: str) -> str:
    """Async PDF download with caching"""
    cache_key = get_cache_key(url)
    
    if cache_key in pdf_cache:
        print(f"Using cached PDF for {url}")
        return pdf_cache[cache_key]
    
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            async with session.get(url) as response:
                if response.status == 200:
                    filename = f"temp_{cache_key}.pdf"
                    async with aiofiles.open(filename, 'wb') as f:
                        async for chunk in response.content.iter_chunked(8192):
                            await f.write(chunk)
                    pdf_cache[cache_key] = filename
                    print(f"Downloaded and cached PDF: {filename}")
                    return filename
                else:
                    raise HTTPException(status_code=400, detail=f"Failed to download PDF: HTTP {response.status}")
    except Exception as e:
        print(f"Async download failed, trying sync: {e}")
        try:
            loop = asyncio.get_event_loop()
            pdf_path = await loop.run_in_executor(executor, download_pdf, url)
            pdf_cache[cache_key] = pdf_path
            return pdf_path
        except Exception as sync_e:
            raise HTTPException(status_code=400, detail=f"Failed to download PDF: {sync_e}")

def get_or_create_rag(pdf_path: str) -> RAGPipeline:
    """Get RAG instance from cache or create new one"""
    if pdf_path not in rag_cache:
        print(f"Creating new RAG instance for {pdf_path}")
        rag_cache[pdf_path] = RAGPipeline(pdf_path)
    else:
        print(f"Using cached RAG instance for {pdf_path}")
    return rag_cache[pdf_path]

async def process_question_async(rag: RAGPipeline, question: str) -> str:
    """Process single question asynchronously"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, rag.ask, question)

async def process_questions_batch(rag: RAGPipeline, questions: List[str]) -> List[str]:
    """Process questions in batch if RAG supports it, otherwise concurrently"""
    if hasattr(rag, 'batch_ask'):
        print("Using batch processing for questions")
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, rag.batch_ask, questions)
    else:
        print("Using concurrent processing for questions")
        tasks = [process_question_async(rag, q) for q in questions]
        return await asyncio.gather(*tasks)

@app.post("/hackrx/run")
async def run_api(payload: RequestPayload, authorization: str = Header(None)):
    if not cached_validate_token(authorization):
        raise HTTPException(status_code=401, detail="Invalid or missing token")
    
    try:
        # ✅ Log the incoming request
        log_request(payload.documents, payload.questions)

        print(f"Processing request with {len(payload.questions)} questions")
        
        # 1. Async PDF download with caching
        pdf_path = await download_pdf_async(payload.documents)
        
        # 2. Get or create RAG instance (cached)
        rag = get_or_create_rag(pdf_path)
        
        # 3. Process questions (batch or concurrent)
        answers = await process_questions_batch(rag, payload.questions)
        
        return {"answers": answers}
        
    except Exception as e:
        print(f"Error in run_api: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/hackrx/preload")
async def preload_document(url: str, authorization: str = Header(None)):
    if not cached_validate_token(authorization):
        raise HTTPException(status_code=401, detail="Invalid or missing token")
    
    try:
        pdf_path = await download_pdf_async(url)
        get_or_create_rag(pdf_path)
        return {"message": "Document preloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/hackrx/cache/status")
async def cache_status(authorization: str = Header(None)):
    if not cached_validate_token(authorization):
        raise HTTPException(status_code=401, detail="Invalid or missing token")
    
    return {
        "cached_documents": len(rag_cache),
        "cached_pdfs": len(pdf_cache)
    }

@app.delete("/hackrx/cache")
async def clear_cache(authorization: str = Header(None)):
    if not cached_validate_token(authorization):
        raise HTTPException(status_code=401, detail="Invalid or missing token")
    
    global rag_cache, pdf_cache
    
    for pdf_path in pdf_cache.values():
        if os.path.exists(pdf_path) and pdf_path.startswith("temp_"):
            try:
                os.remove(pdf_path)
            except Exception as e:
                print(f"Error deleting {pdf_path}: {e}")
    
    rag_cache.clear()
    pdf_cache.clear()
    cached_validate_token.cache_clear()
    
    return {"message": "Cache cleared successfully"}

@app.get("/hackrx/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/")
async def root():
    return {"message": "RAG Pipeline API"}

from fastapi.responses import FileResponse

@app.get("/hackrx/logs")
async def get_logs(authorization: str = Header(None)):
    if not cached_validate_token(authorization):
        raise HTTPException(status_code=401, detail="Unauthorized")

    log_path = os.path.join(LOG_DIR, LOG_FILE)
    if os.path.exists(log_path):
        return FileResponse(log_path, media_type="application/json", filename=LOG_FILE)
    else:
        raise HTTPException(status_code=404, detail="Log file not found")


@app.on_event("shutdown")
async def shutdown_event():
    print("Shutting down...")
    
    for pdf_path in pdf_cache.values():
        if os.path.exists(pdf_path) and pdf_path.startswith("temp_"):
            try:
                os.remove(pdf_path)
                print(f"Cleaned up {pdf_path}")
            except Exception as e:
                print(f"Error cleaning up {pdf_path}: {e}")
    
    executor.shutdown(wait=True)
    print("Cleanup completed")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=False)

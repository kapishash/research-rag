import os
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile
from typing import Optional


from ingest import ingest_pdf, list_ingested_files
from chain import ask_with_citations
from retriever import retrieve_relevant_chunks, is_collection_empty

app = FastAPI(title="RAG API", version="1.0.0")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class IngestResponse(BaseModel):
    success: bool
    file: str
    pages_parsed: int
    chunks_created: int
    total_in_db: int
    message: str

class AskRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5          # number of chunks to retrieve
    source_filter: Optional[str] = None  # filter to specific PDF
    chat_history: Optional[list] = []    # previous conversation turns

class AskResponse(BaseModel):
    answer: str
    citations: list
    chunks_used: list
    model: str
    context_chunks_count: int


@app.get("/")
def read_root():
    return {"message": "Welcome to the ResearchRAG API!"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "ResearchRAG API"}

@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)):

    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        stats = ingest_pdf(tmp_path, original_filename=file.filename)
        return IngestResponse(
            success=True,
            file=file.filename,
            pages_parsed=stats["pages_parsed"],
            chunks_created=stats["chunks_created"],
            total_in_db=stats["total_in_db"],
            message=f"Successfully ingested '{file.filename}'"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
    finally:
        os.unlink(tmp_path)  


@app.get("/files")
def list_files():
    files = list_ingested_files()
    return {"files": files, "count": len(files)}

@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    if is_collection_empty():
        raise HTTPException(
            status_code=400,
            detail="No documents ingested yet. Please upload a PDF first."
        )

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Step 1: Retrieve
    chunks = retrieve_relevant_chunks(
        query=request.question,
        top_k=request.top_k,
        source_filter=request.source_filter
    )

    if not chunks:
        raise HTTPException(
            status_code=404,
            detail="No relevant chunks found. Try rephrasing your question."
        )

    # Step 2: Generate answer with citations
    result = ask_with_citations(
        question=request.question,
        chunks=chunks,
        chat_history=request.chat_history
    )

    return AskResponse(**result)



@app.delete("/files/{filename}")
def delete_file(filename: str):
    import chromadb
    from chromadb.utils import embedding_functions

    CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "citablerag_docs")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name="text-embedding-3-small"
    )
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=openai_ef
    )

    # Find all chunk IDs belonging to this file
    results = collection.get(
        where={"source_file": filename},
        include=["metadatas"]
    )

    if not results["ids"]:
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found in database.")

    collection.delete(ids=results["ids"])

    return {
        "success": True,
        "message": f"Deleted {len(results['ids'])} chunks for '{filename}'"
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
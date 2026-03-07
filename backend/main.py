import os
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ingest import ingest_pdf, list_ingested_files
import tempfile

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




if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
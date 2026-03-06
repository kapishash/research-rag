import os
import uuid
import fitz  # PyMuPDF
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "researchragrag_docs")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# PDF parsing function

def parse_pdf(file_path: str, original_filename: str = None) -> list[dict]:
    doc = fitz.open(file_path)
    pages = []
    display_name = original_filename if original_filename else os.path.basename(file_path)

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        if text.strip():
            pages.append({
                "text": text,
                "page_number": page_num + 1,
                "source_file": display_name
            })

    doc.close()
    return pages


# chunking function

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def chunk_text(page_data: dict, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    text = page_data["text"]
    chunks = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = start + chunk_size
        chunk_text_content = text[start:end]

        if chunk_text_content.strip():
            chunks.append({
                "id": str(uuid.uuid4()),
                "text": chunk_text_content,
                "metadata": {
                    "source_file": page_data["source_file"],
                    "page_number": page_data["page_number"],
                    "chunk_index": chunk_index,
                    "preview": chunk_text_content[:150].replace("\n", " ").strip()
                }
            })
            chunk_index += 1

        start += chunk_size - overlap

    return chunks

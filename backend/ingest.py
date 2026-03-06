import os
import uuid
import fitz  
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


# ChromaDB collection setup

def get_chroma_collection():
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name="text-embedding-3-small"
    )
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=openai_ef,
        metadata={"hnsw:space": "cosine"}
    )
    return collection



def ingest_pdf(file_path: str, original_filename: str = None) -> dict:
 
    display_name = original_filename if original_filename else os.path.basename(file_path)
    print(f"[INGEST] Starting ingestion for: {display_name}")

    # Step 1: Parse with original filename
    pages = parse_pdf(file_path, original_filename=display_name)
    print(f"[INGEST] Parsed {len(pages)} pages")

    # Step 2: Chunk
    all_chunks = []
    for page in pages:
        chunks = chunk_text(page)
        all_chunks.extend(chunks)

    print(f"[INGEST] Created {len(all_chunks)} chunks")

    # Step 3: Insert into ChromaDB in batches
    collection = get_chroma_collection()
    batch_size = 100

    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i + batch_size]
        collection.add(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[c["metadata"] for c in batch]
        )
        print(f"[INGEST] Inserted batch {i // batch_size + 1} ({len(batch)} chunks)")

    print(f"[INGEST] Done! Total chunks in DB: {collection.count()}")

    return {
        "file": display_name,  
        "pages_parsed": len(pages),
        "chunks_created": len(all_chunks),
        "total_in_db": collection.count()
    }


def list_ingested_files() -> list[str]:
    collection = get_chroma_collection()
    if collection.count() == 0:
        return []
    results = collection.get(include=["metadatas"])
    files = list({m["source_file"] for m in results["metadatas"]})
    return sorted(files)



# # testing the full ingestion process

# if __name__ == "__main__":
#     import sys
#     if len(sys.argv) > 1:
#         stats = ingest_pdf(sys.argv[1])
#         print(f"\nIngestion complete: {stats}")
#     else:
#         print("Usage: python ingest.py <path_to_pdf>")
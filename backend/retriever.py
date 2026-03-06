import os
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "researchragrag_docs")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_TOP_K = 5

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

def retrieve_relevant_chunks(query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    collection = get_chroma_collection()
    if collection.count() == 0:
        return []

    results = collection.query(
        query_texts=[query],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    chunks = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        chunks.append({
            "text": doc,
            "source_file": meta.get("source_file", "Unknown"),
            "page_number": meta.get("page_number", "?"),
            "chunk_index": meta.get("chunk_index", 0),
            "distance": round(dist, 4),
            "relevance_score": round((1 - dist) * 100, 1)
        })

    return chunks


def is_collection_empty() -> bool:
    collection = get_chroma_collection()
    return collection.count() == 0



# for testing 
if __name__ == "__main__":
    query = "What is the main conclusion of the research?"
    results = retrieve_relevant_chunks(query, top_k=3)
    for i, r in enumerate(results):
        print(f"\n[Chunk {i+1}] Source: {r['source_file']}, Page: {r['page_number']}, Relevance: {r['relevance_score']}%")
        print(r["text"][:300])
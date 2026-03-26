<div align="center">

# 🔬 ResearchRAG

### *Ask anything from your documents. Know exactly where every answer came from.*

<br/>

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![OpenAI](https://img.shields.io/badge/GPT--4o-OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_DB-F97316?style=for-the-badge)](https://trychroma.com)
[![LangChain](https://img.shields.io/badge/LangChain-0.2-1C3C3C?style=for-the-badge)](https://langchain.com)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)

<br/>

> **ResearchRAG** is a production-ready document Q&A system powered by GPT-4o and ChromaDB.
> Upload any PDF and get answers with **exact page-level citations** for every single claim.

<br/>

</div>

---


## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     User  (Browser)                          │
└───────────────────────────┬──────────────────────────────────┘
                            │  question
                            ▼
┌──────────────────────────────────────────────────────────────┐
│            Streamlit Frontend  ( :8501 )                     │
│   Chat UI · File Upload · Citation Cards · Copy Button       │
└───────────────────────────┬──────────────────────────────────┘
                            │  HTTP REST
                            ▼
┌──────────────────────────────────────────────────────────────┐
│              FastAPI Backend  ( :8000 )                      │
│          /ingest  ·  /ask  ·  /files  ·  /health             │
└──────────────┬───────────────────────────┬───────────────────┘
               │  vector search             │  generate
               ▼                            ▼
┌─────────────────────┐       ┌─────────────────────────────┐
│      ChromaDB        │       │    LangChain  +  GPT-4o      │
│   Vector Store       │       │  RAG Chain · Citations       │
│   Cosine Similarity  │       │  text-embedding-3-small      │
│   Persistent Disk    │       │  Prompt Engineering          │
└─────────────────────┘       └─────────────────────────────┘
               ▲
               │  PDF ingestion
┌─────────────────────┐
│    PyMuPDF (fitz)    │
│  PDF Parser          │
│  Page Extraction     │
│  Overlap Chunking    │
└─────────────────────┘
```

### How a Question Gets Answered

```
1. User asks a question
        │
        ▼
2. Embed question  →  text-embedding-3-small  →  1536-dim vector
        │
        ▼
3. ChromaDB cosine similarity search  →  Top 5 relevant chunks
        │
        ▼
4. Build prompt:  [System Rules] + [Chunks + Metadata] + [Question]
        │
        ▼
5. GPT-4o generates answer with [SOURCE: file.pdf, Page X] per claim
        │
        ▼
6. Parse citations  →  Clean answer + Citation cards + Raw chunks
```

---

## 🛠️ Tech Stack

| Layer | Tool | Why This Was Chosen |
|-------|------|---------------------|
| **LLM** | GPT-4o | Best instruction-following, 128K context, reliable source citations |
| **Embeddings** | text-embedding-3-small | Cheapest OpenAI embedding model, high accuracy, 1536-dim vectors |
| **Vector DB** | ChromaDB | Embedded, no separate server needed, cosine similarity, free & open-source |
| **Orchestration** | LangChain 0.2 | Industry-standard LLM framework, prompt templates, chat history |
| **Backend** | FastAPI | Async, auto Swagger UI at /docs, type validation via Pydantic |
| **Frontend** | Streamlit | Rapid ML demo UI, zero HTML/CSS, free cloud deployment option |
| **PDF Parsing** | PyMuPDF (fitz) | Fastest Python PDF parser, handles complex multi-column layouts |
| **Deployment** | Docker + Compose | Single command startup, reproducible on any machine |
| **Language** | Python 3.12 | Latest stable release, fastest CPython yet |

---

##  Project Structure

```
research-rag/
│
├──  backend/
│   ├── main.py          # FastAPI app — /ingest, /ask, /files, /health
│   ├── ingest.py        # PDF parsing → overlap chunking → ChromaDB insertion
│   ├── retriever.py     # Cosine similarity search with optional source filter
│   └── chain.py         # LangChain RAG chain — prompt + GPT-4o + citation parser
│
├──  frontend/
│   └── app.py           # Streamlit chat UI — citations, copy button, chunk viewer
│
├──  chroma_db/        # ChromaDB persistent storage (auto-created on first ingest)
│
├── Dockerfile.backend
├── Dockerfile.frontend
├── docker-compose.yml
├── requirements.txt
├── requirements.frontend.txt
├── .env.example
└── README.md
```

---

##  Quickstart

### Option A — Docker (Recommended)

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/research-rag.git
cd research-rag

# 2. Set your API key
cp .env.example .env
# Open .env and add your OPENAI_API_KEY

# 3. Start everything
docker compose up --build

# 4. Open in browser
# → http://localhost:8501
```

To stop: `docker compose down`

---

### Option B — Local Development (Windows)

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/research-rag.git
cd research-rag

# 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment
cp .env.example .env
# Add your OPENAI_API_KEY to .env

# 5. Terminal 1 — Start FastAPI backend
cd backend
uvicorn main:app --reload --port 8000

# 6. Terminal 2 — Start Streamlit frontend
cd frontend
streamlit run app.py

# Open → http://localhost:8501
```

---

## ⚙️ Environment Variables

Create a `.env` file in the project root (copy from `.env.example`):

```env
OPENAI_API_KEY=sk-your-openai-api-key-here
CHROMA_PERSIST_DIR=./chroma_db
CHROMA_COLLECTION_NAME=research_rag_docs
```

Get your OpenAI API key → https://platform.openai.com/api-keys

---

##  API Reference

Interactive Swagger docs → **http://localhost:8000/docs**

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check — used by Docker healthcheck |
| `POST` | `/ingest` | Upload a PDF and process it into ChromaDB |
| `POST` | `/ask` | Ask a question, receive cited answer |
| `GET` | `/files` | List all ingested document filenames |
| `DELETE` | `/files/{filename}` | Remove all chunks for a document |

### Ask a Question — curl Example

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the main conclusion?",
    "top_k": 5
  }'
```

### Response

```json
{
  "answer": "The study concludes that... [SOURCE: paper.pdf, Page 12]",
  "citations": [
    { "source_file": "paper.pdf", "page": "12" }
  ],
  "chunks_used": [...],
  "model": "gpt-4o",
  "context_chunks_count": 5
}
```

---

##  How Chunking Works

Large PDF pages are split into overlapping chunks before being embedded:

```
Page text (3000 chars)
├── Chunk 1:  chars    0 → 1000
├── Chunk 2:  chars  800 → 1800  ← 200 char overlap with Chunk 1
├── Chunk 3:  chars 1600 → 2600  ← 200 char overlap with Chunk 2
└── Chunk 4:  chars 2400 → 3000
```

Each chunk stores rich metadata:

```python
{
  "text":        "...chunk content...",
  "source_file": "research_paper.pdf",
  "page_number": 4,
  "chunk_index": 2,
  "preview":     "First 150 characters..."
}
```

The overlap prevents important context from being cut at chunk boundaries.

---

## 📍 How Citations Work

1. Retrieved chunks are labeled with `Source:` and `Page:` before being passed to GPT-4o
2. System prompt strictly instructs GPT-4o to write `[SOURCE: filename, Page X]` after every factual claim
3. Backend extracts citation tags using regex and returns a structured list
4. Frontend strips raw tags from the answer and renders them as clean citation cards
5. All retrieved chunks are shown in an expandable panel so users can verify the AI's reasoning

---

## 💰 Cost Estimate

| Operation | Estimated Cost |
|-----------|---------------|
| Ingest a 20-page PDF | ~$0.002 |
| Single question (5 chunks, GPT-4o) | ~$0.01 – $0.03 |
| 100 questions | ~$1 – $3 total |

---

## 🔮 Possible Extensions

- **Streaming responses** — `StreamingResponse` in FastAPI + `st.write_stream` in Streamlit
- **Multi-format ingestion** — Word docs, web URLs, YouTube transcripts
- **Cross-encoder reranking** — Rerank retrieved chunks before sending to GPT-4o for better accuracy
- **RAGAS evaluation** — Automated pipeline to measure retrieval and answer quality
- **User authentication** — JWT-based login with FastAPI security
- **Cloud deployment** — Railway or Render for backend, Streamlit Cloud for frontend

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

<div align="center">

**Built by [Kapishankar Ashtankar](https://github.com/kapishash)**

*Python 3.12 · GPT-4o · ChromaDB · LangChain · FastAPI · Streamlit · Docker*

<br/>

**Star this repo if you found it useful!**

</div>

import streamlit as st
import requests
import json
import os
from datetime import datetime
import re

st.set_page_config(
    page_title="Research RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


API_URL = os.getenv("API_URL", "http://localhost:8000")


st.markdown("""
<style>
    .citation-card {
        background: #1e3a5f;
        color: #e2e8f0;
        border-left: 4px solid #2563eb;
        padding: 10px 15px;
        border-radius: 4px;
        margin: 5px 0;
        font-size: 0.9em;
    }
    .chunk-card {
        background: #1e293b;
        color: #e2e8f0;
        border: 1px solid #334155;
        padding: 12px;
        border-radius: 6px;
        margin: 5px 0;
        font-size: 0.85em;
    }
    .relevance-badge {
        background: #dcfce7;
        color: #166534;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8em;
        font-weight: bold;
    }
    .answer-block {
        background: #0f172a;
        color: #e2e8f0;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #334155;
        line-height: 1.7;
    }
    .stAlert {
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  

if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = []


def fetch_ingested_files():
    try:
        response = requests.get(f"{API_URL}/files", timeout=5)
        if response.status_code == 200:
            return response.json()["files"]
    except Exception:
        pass
    return []


def clean_answer_for_display(answer: str) -> str:

    cleaned = re.sub(r'\[SOURCE:[^\]]+\]', '', answer)
    
    cleaned = re.sub(r'##\s*Sources Used.*', '', cleaned, flags=re.DOTALL)
    return cleaned.strip()



def check_api_health():
    """Returns True if the FastAPI backend is reachable."""
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False
    

with st.sidebar:
    st.title("📚 Research Rag")
    st.caption("Document Q&A with Source Citations")
    st.divider()

    # API Status
    api_healthy = check_api_health()
    if api_healthy:
        st.success("API Connected", icon="✅")
    else:
        st.error("api ")

    st.divider()

    # ── Document Upload ──────────────────────────────────────────────────────
    st.subheader("📄 Upload Document")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload research papers, reports, or any PDF document"
    )

    if uploaded_file and st.button("🚀 Ingest Document", type="primary", use_container_width=True):
        with st.spinner(f"Processing '{uploaded_file.name}'... This may take a minute for large PDFs."):
            try:
                response = requests.post(
                    f"{API_URL}/ingest",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
                    timeout=120  # large PDFs need time
                )

                if response.status_code == 200:
                    data = response.json()
                    st.success(f"✅ Ingested successfully!")
                    st.info(
                        f"**{data['file']}**\n\n"
                        f"📖 Pages: {data['pages_parsed']}  \n"
                        f"🧩 Chunks: {data['chunks_created']}  \n"
                        f"🗄️ Total in DB: {data['total_in_db']}"
                    )
                    st.session_state.ingested_files = fetch_ingested_files()
                    st.rerun()
                else:
                    st.error(f"Ingestion failed: {response.json().get('detail', 'Unknown error')}")

            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API. Is the FastAPI server running?")
            except Exception as e:
                st.error(f"Error: {str(e)}")

    st.divider()

    # ── Ingested Files List ──────────────────────────────────────────────────
    st.subheader("🗂️ Ingested Documents")

    if st.button("🔄 Refresh File List", use_container_width=True):
        st.session_state.ingested_files = fetch_ingested_files()

    files = st.session_state.ingested_files or fetch_ingested_files()

    if files:
        for fname in files:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"📄 `{fname}`")
            with col2:
                if st.button("🗑️", key=f"del_{fname}", help=f"Delete {fname}"):
                    try:
                        r = requests.delete(f"{API_URL}/files/{fname}", timeout=10)
                        if r.status_code == 200:
                            st.success(f"Deleted {fname}")
                            st.session_state.ingested_files = fetch_ingested_files()
                            st.rerun()
                        else:
                            st.error("Delete failed")
                    except Exception as e:
                        st.error(str(e))
    else:
        st.info("No documents ingested yet. Upload a PDF above.")

    st.divider()

    # ── Search Settings ──────────────────────────────────────────────────────
    st.subheader("⚙️ Search Settings")

    top_k = st.slider(
        "Chunks to retrieve",
        min_value=1, max_value=10, value=5,
        help="How many document chunks to use as context. More = richer answers but higher cost."
    )

    source_filter = st.selectbox(
        "Filter by document",
        options=["All documents"] + (files or []),
        help="Restrict search to a specific document."
    )
    source_filter = None if source_filter == "All documents" else source_filter

    # Clear chat button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

st.title("💬 Ask Your Documents")
st.caption("Questions are answered with inline citations showing exactly where each fact came from.")



for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.write(message["content"])

        elif message["role"] == "assistant":
            # Clean answer (no raw [SOURCE:...] tags)
            clean_ans = clean_answer_for_display(message["content"])
            st.markdown('<div class="answer-block">', unsafe_allow_html=True)
            st.markdown(clean_ans)
            st.markdown('</div>', unsafe_allow_html=True)

            # Citation Cards
            if message.get("citations"):
                st.markdown("**📎 Citations:**")
                for cit in message["citations"]:
                    st.markdown(
                        f'<div class="citation-card">📄 <strong>{cit["source_file"]}</strong> — Page {cit["page"]}</div>',
                        unsafe_allow_html=True
                    )

            # Expandable raw chunks (for transparency)
            if message.get("chunks"):
                with st.expander(f"🔍 View {len(message['chunks'])} retrieved chunks (how I found this)", expanded=False):
                    for i, chunk in enumerate(message["chunks"], 1):
                        st.markdown(
                            f'<div class="chunk-card">'
                            f'<strong>Chunk {i}</strong> — '
                            f'<span class="relevance-badge">{chunk["relevance_score"]}% relevant</span> | '
                            f'📄 {chunk["source_file"]}, Page {chunk["page_number"]}'
                            f'<br><br>{chunk["text"][:400]}{"..." if len(chunk["text"]) > 400 else ""}'
                            f'</div>',
                            unsafe_allow_html=True
                        )


# ── Chat Input ─────────────────────────────────────────────────────────────────

if question := st.chat_input("Ask a question about your documents..."):

    if not api_healthy:
        st.error("API is offline. Start the FastAPI server first.")
        st.stop()

    if not (st.session_state.ingested_files or fetch_ingested_files()):
        st.warning("Please upload at least one PDF document before asking questions.")
        st.stop()

    # Add user message to UI immediately
    with st.chat_message("user"):
        st.write(question)

    # Add to history for multi-turn context
    st.session_state.chat_history.append({
        "role": "user",
        "content": question
    })

    # Call the API
    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer..."):
            try:
                # Build chat history for API (only role + content)
                history_for_api = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.chat_history[:-1]  # exclude current question
                ]

                response = requests.post(
                    f"{API_URL}/ask",
                    json={
                        "question": question,
                        "top_k": top_k,
                        "source_filter": source_filter,
                        "chat_history": history_for_api
                    },
                    timeout=60
                )

                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    citations = data["citations"]
                    chunks = data["chunks_used"]

                    # Display clean answer
                    clean_ans = clean_answer_for_display(answer)
                    st.markdown('<div class="answer-block">', unsafe_allow_html=True)
                    st.markdown(clean_ans)
                    st.markdown('</div>', unsafe_allow_html=True)

                    # Citation Cards
                    if citations:
                        st.markdown("**📎 Citations:**")
                        for cit in citations:
                            st.markdown(
                                f'<div class="citation-card">📄 <strong>{cit["source_file"]}</strong> — Page {cit["page"]}</div>',
                                unsafe_allow_html=True
                            )
                    else:
                        st.info("No specific citations found. The answer may be a general summary.")

                    # Expandable raw chunks
                    with st.expander(f"🔍 View {len(chunks)} retrieved chunks (how I found this)", expanded=False):
                        for i, chunk in enumerate(chunks, 1):
                            st.markdown(
                                f'<div class="chunk-card">'
                                f'<strong>Chunk {i}</strong> — '
                                f'<span class="relevance-badge">{chunk["relevance_score"]}% relevant</span> | '
                                f'📄 {chunk["source_file"]}, Page {chunk["page_number"]}'
                                f'<br><br>{chunk["text"][:400]}{"..." if len(chunk["text"]) > 400 else ""}'
                                f'</div>',
                                unsafe_allow_html=True
                            )

                    # Save to history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "citations": citations,
                        "chunks": chunks
                    })

                elif response.status_code == 400:
                    st.warning(response.json().get("detail", "Bad request"))
                else:
                    st.error(f"API Error {response.status_code}: {response.json().get('detail', 'Unknown error')}")

            except requests.exceptions.Timeout:
                st.error("Request timed out. The document may be very large — try a more specific question.")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API. Make sure FastAPI is running on port 8000.")
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")


                
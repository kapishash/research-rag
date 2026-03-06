import os
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

SYSTEM_PROMPT = """You are research_rag, an expert document analyst assistant.
Your job is to answer questions based ONLY on the document chunks provided to you.

STRICT RULES:
1. Only use information from the provided context chunks. Do NOT use outside knowledge.
2. Every factual claim you make must be cited inline using this exact format:
   [SOURCE: <filename>, Page <page_number>]
3. If multiple chunks support the same point, cite all of them:
   [SOURCE: report.pdf, Page 3] [SOURCE: report.pdf, Page 7]
4. If the answer is not found in the provided chunks, say:
   "I could not find information about this in the provided documents."
   Do NOT guess or make up an answer.
5. Structure your answer clearly. Use paragraphs, not bullet points, unless a list is natural.
6. At the end of your answer, add a "## Sources Used" section listing each unique source cited.

CITATION FORMAT EXAMPLE:
The company reported a 23% increase in revenue [SOURCE: annual_report.pdf, Page 4], 
driven primarily by growth in the Asia-Pacific region [SOURCE: annual_report.pdf, Page 6].

## Sources Used
- annual_report.pdf, Page 4
- annual_report.pdf, Page 6
"""

def build_context_string(chunks: list[dict]) -> str:
    if not chunks:
        return "No relevant document chunks were found."

    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[CHUNK {i}]\n"
            f"Source: {chunk['source_file']}\n"
            f"Page: {chunk['page_number']}\n"
            f"Relevance: {chunk['relevance_score']}%\n"
            f"Content:\n{chunk['text']}\n"
            f"{'─' * 60}"
        )

    return "\n\n".join(context_parts)


# ── Main RAG Function ──────────────────────────────────────────────────────────
def ask_with_citations(question: str, chunks: list[dict], chat_history: list[dict] = None) -> dict:

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,       
                              
        api_key=OPENAI_API_KEY,
        max_tokens=2000       
    )

    context = build_context_string(chunks)

    human_message_content = f"""Here are the relevant document chunks:

                            {context}
                            ---
                            Question: {question}
                            Please answer based only on the chunks above, with inline citations."""

    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    if chat_history:
        for turn in chat_history[-6:]:  # only last 6 turns to stay within context limits
            if turn["role"] == "user":
                messages.append(HumanMessage(content=turn["content"]))
            else:
                from langchain.schema import AIMessage
                messages.append(AIMessage(content=turn["content"]))

    messages.append(HumanMessage(content=human_message_content))

    response = llm.invoke(messages)
    answer_text = response.content

    citations = parse_citations(answer_text)

    return {
        "answer": answer_text,
        "citations": citations,
        "chunks_used": chunks,
        "model": "gpt-4o",
        "context_chunks_count": len(chunks)
    }


def parse_citations(answer_text: str) -> list[dict]:
    import re

    pattern = r'\[SOURCE:\s*([^,\]]+),\s*Page\s*(\d+)\]'
    matches = re.findall(pattern, answer_text)

    seen = set()
    citations = []
    for source_file, page in matches:
        key = (source_file.strip(), page.strip())
        if key not in seen:
            seen.add(key)
            citations.append({
                "source_file": source_file.strip(),
                "page": page.strip()
            })

    return citations


# test


# if __name__ == "__main__":
#     # dummy data
#     test_chunks = [
#         {
#             "text": "The company reported revenue of $5.2 billion in Q3 2023, a 23% increase year-over-year.",
#             "source_file": "annual_report.pdf",
#             "page_number": 4,
#             "relevance_score": 92.3
#         }
#     ]
#     result = ask_with_citations("What was the revenue in Q3 2023?", test_chunks)
#     print("Answer:", result["answer"])
#     print("\nCitations:", result["citations"])
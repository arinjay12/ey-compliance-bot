"""
demo_conflict.py
----------------
Demonstrates the Level 2 conflict detection by directly comparing
two documents we KNOW contain conflicting information:

  - June 2025 circular: grants relaxation from Regulation 58(1)(b) until Sept 30, 2025
  - January 2026 Master Circular: supersedes the June circular with updated provisions

This script doesn't rely on RSS feeds or live scraping — it uses
documents already in the knowledge base to show Llama 3 finding real conflicts.

Run:
    python demo_conflict.py
"""

import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama

CHROMA_DIR  = "./chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL   = "llama3"

# ── Documents to compare ──────────────────────────────────────────────────────
# Doc A: The older June 2025 circular (the "stored" reference document)
DOC_A_NAME = "18__SEBI_Circular_dated_June_05_2025.pdf"
DOC_A_LABEL = "SEBI Circular dated June 05, 2025 (LODR Regulation 58 relaxation)"

# Doc B: The January 2026 Master Circular (the "new/updated" document)
DOC_B_NAME = "SEBI_Master_Circular_LODR_Listed_Entities_Jan2026.pdf"
DOC_B_LABEL = "SEBI Master Circular January 30, 2026 (Consolidated LODR compliance)"


def get_chunks_for_doc(vs, doc_name: str, max_chunks: int = 6) -> list:
    """Pull stored chunks that belong to a specific source document."""
    # ChromaDB stores source as './docs\\filename.pdf' on Windows
    full_path = f"./docs\\{doc_name}"
    results   = vs.get(
        where={"source": {"$eq": full_path}},
        include=["documents"]
    )
    documents = results.get("documents", [])
    if not documents:
        # Try forward-slash variant
        full_path_fwd = f"./docs/{doc_name}"
        results = vs.get(
            where={"source": {"$eq": full_path_fwd}},
            include=["documents"]
        )
        documents = results.get("documents", [])
    return documents[:max_chunks]


def demo_compare(vs, llm):
    print("\n" + "="*65)
    print("LEVEL 2 — DIRECT DOCUMENT CONFLICT DEMO")
    print("="*65)
    print(f"\nDocument A (stored/reference):\n  {DOC_A_LABEL}")
    print(f"\nDocument B (newer/updated):\n  {DOC_B_LABEL}")

    # Pull chunks from each document
    chunks_a = get_chunks_for_doc(vs, DOC_A_NAME, max_chunks=5)
    chunks_b = get_chunks_for_doc(vs, DOC_B_NAME, max_chunks=5)

    if not chunks_a:
        print(f"\nERROR: Could not find chunks for {DOC_A_NAME}")
        print("Make sure ingest.py has been run with this file in docs/")
        return

    if not chunks_b:
        print(f"\nERROR: Could not find chunks for {DOC_B_NAME}")
        print("Make sure ingest.py has been run with this file in docs/")
        return

    print(f"\nFound {len(chunks_a)} chunks from Doc A, {len(chunks_b)} chunks from Doc B")

    # Also do semantic search: use Doc A's content to find the most relevant
    # chunks from Doc B (this is what Level 2 does automatically)
    print("\nSearching for Doc B chunks most relevant to Doc A topics...")
    doc_a_query = " ".join(chunks_a[:2])[:500]
    related_b   = vs.similarity_search(
        doc_a_query,
        k=6,
        filter={"source": {"$eq": f"./docs\\{DOC_B_NAME}"}}
    )

    if related_b:
        print(f"Found {len(related_b)} semantically related chunks from Doc B")
        context_b = "\n\n---\n\n".join(doc.page_content for doc in related_b)
    else:
        context_b = "\n\n".join(chunks_b)

    context_a = "\n\n".join(chunks_a)

    # Ask Llama 3 to compare
    prompt = f"""You are a SEBI compliance analyst at EY (Ernst & Young).

Compare the two SEBI documents below and identify:
1. FACTUAL CONFLICTS — where they state different things about the same topic
   (different dates, deadlines, percentages, regulation numbers, requirements)
2. SUPERSESSIONS — where the newer document explicitly supersedes or updates the older one
3. GAPS — things in the older document not addressed in the newer one

DOCUMENT A (older — {DOC_A_LABEL}):
{context_a}

DOCUMENT B (newer — {DOC_B_LABEL}):
{context_b}

Be specific. Quote exact dates, regulation numbers, and provisions where possible.
Keep response under 300 words. If no conflicts exist, say "NO CONFLICTS FOUND"."""

    print("\nAsking Llama 3 to analyse conflicts...\n")
    print("-"*65)
    print("CONFLICT ANALYSIS:")
    print("-"*65)

    # Stream the response
    for chunk in llm.stream(prompt):
        print(chunk, end="", flush=True)

    print("\n" + "-"*65)
    print("\nDemo complete.")
    print("This is the analysis that would be emailed automatically in production.")


def main():
    if not os.path.exists(CHROMA_DIR):
        print("No knowledge base found. Run ingest.py first.")
        return

    print("Loading models...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    llm        = Ollama(model=LLM_MODEL, temperature=0.1)
    vs         = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

    demo_compare(vs, llm)


if __name__ == "__main__":
    main()

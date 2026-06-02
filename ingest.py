"""
ingest.py
---------
Run this ONCE to load all PDFs from the /docs folder into ChromaDB.
After running, the knowledge base is saved to /chroma_db and persists.

Usage:
    python ingest.py
"""

""""

PDF uploaded
    → Split into chunks (~500 words each, with 50-word overlap so context isn't cut off)
    → Each chunk converted to a vector (list of numbers) by Sentence Transformers
    →  All vectors stored in ChromaDB on your local machine

"""
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

DOCS_DIR    = "./docs"
CHROMA_DIR  = "./chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"

def main():
    # Step 1 — Collect all PDFs from the docs folder
    pdf_files = [f for f in os.listdir(DOCS_DIR) if f.endswith(".pdf")]

    if not pdf_files:
        print("No PDFs found in ./docs — please place your SEBI/RBI circulars there first.")
        return

    print(f"Found {len(pdf_files)} PDF(s): {pdf_files}\n")

    # Friendly labels for each document — used as prefix in every chunk so
    # queries like "June 2025 circular" semantically match the right document
    DOC_LABELS = {
        "18__SEBI_Circular_dated_June_05_2025.pdf":
            "SEBI Circular June 05 2025 - Limited relaxation LODR Regulation 58 non-convertible securities",
        "1749641449497.pdf":
            "SEBI Circular June 11 2025 - UPI payment mechanism SEBI registered intermediaries",
        "39_SEBI_Circular_dated_October_15_2025.pdf":
            "SEBI Master Circular October 15 2025 - Issue and listing non-convertible securities",
        "SEBI_Master_Circular_LODR_NCS_July2025.pdf":
            "SEBI Master Circular July 11 2025 - LODR listing obligations non-convertible securities commercial paper",
        "SEBI_Master_Circular_LODR_Listed_Entities_Jan2026.pdf":
            "SEBI Master Circular January 30 2026 - LODR compliance listed entities consolidated",
    }

    # Step 2 — Load and split each PDF into chunks
    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    for filename in pdf_files:
        filepath = os.path.join(DOCS_DIR, filename)
        label    = DOC_LABELS.get(filename, filename.replace(".pdf", "").replace("_", " "))
        print(f"Loading: {filename}...")
        loader = PyPDFLoader(filepath)
        pages  = loader.load()
        chunks = splitter.split_documents(pages)

        # Prepend document label to each chunk so retrieval can match by doc name/date
        for chunk in chunks:
            chunk.page_content = f"[{label}]\n{chunk.page_content}"

        all_chunks.extend(chunks)
        print(f"  -> {len(pages)} pages, {len(chunks)} chunks created")

    print(f"\nTotal chunks across all docs: {len(all_chunks)}")

    # Step 3 — Create embeddings and store in ChromaDB
    print("\nLoading embedding model (all-MiniLM-L6-v2)...")
    print("Note: First run downloads ~90MB model — needs internet once.")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    print("Storing chunks in ChromaDB...")
    Chroma.from_documents(all_chunks, embeddings, persist_directory=CHROMA_DIR)

    print("\nDone! Knowledge base saved to ./chroma_db")
    print("You can now run: python query.py  OR  streamlit run app.py")

if __name__ == "__main__":
    main()

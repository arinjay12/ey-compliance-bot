"""
ingest.py
---------
Builds the ChromaDB knowledge base from every PDF in ./docs.

Run ONCE (or whenever documents change):
    python ingest.py

What it does per document:
  1. Loads the PDF and splits it into overlapping chunks.
  2. Extracts document-level metadata (official reference number, date, title)
     from page 1 — either from a curated map (for the core SEBI circulars) or
     by auto-extraction (for any other document, e.g. the RBI corpus). This
     metadata is attached to EVERY chunk so questions like "what is the circular
     reference number?" are answerable from any retrieved chunk, not just the
     header chunk (which is rarely retrieved).
  3. Prepends a short label to each chunk so retrieval can match by name/date.
  4. Embeds and stores everything in ./chroma_db.

Chunking note: chunk_size=1000/overlap=150 (was 500/50). Larger chunks keep
timeline tables, footnotes and intro paragraphs intact — the 500-char size was
fragmenting facts (e.g. the UPI 180-day timeline) so they were never retrieved.
"""
import os
import re
import json

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

import rag_core  # reuse the curated DOC_META

DOCS_DIR      = "./docs"
CHROMA_DIR    = "./chroma_db"
EMBED_MODEL   = "all-MiniLM-L6-v2"
INGESTED_FILE = "./ingested_docs.json"
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 150


def main():
    pdf_files = [f for f in os.listdir(DOCS_DIR) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print("No PDFs found in ./docs.")
        return
    print(f"Found {len(pdf_files)} PDF(s).\n")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    all_chunks = []

    for filename in sorted(pdf_files):
        filepath = os.path.join(DOCS_DIR, filename)
        try:
            pages = PyPDFLoader(filepath).load()
        except Exception as e:
            print(f"  ! skipped {filename}: {e}")
            continue
        if not pages:
            print(f"  ! skipped {filename}: no extractable text")
            continue

        # metadata: curated map wins; otherwise auto-extract from page 1
        meta = rag_core.doc_meta_for(filename, pages[0].page_content)

        chunks = splitter.split_documents(pages)
        label = f"{meta['title']} | Ref: {meta['ref']} | Dated: {meta['date']}"
        for ch in chunks:
            ch.page_content = f"[{label}]\n{ch.page_content}"
            ch.metadata["doc_title"] = meta["title"]
            ch.metadata["doc_ref"]   = meta["ref"]
            ch.metadata["doc_date"]  = meta["date"]
        all_chunks.extend(chunks)
        print(f"  {filename:<60} {len(pages):>4}p  {len(chunks):>4} chunks  "
              f"ref={meta['ref'][:40]}")

    print(f"\nTotal chunks: {len(all_chunks)}")
    print("Loading embedding model…")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    print("Storing in ChromaDB…")
    Chroma.from_documents(all_chunks, embeddings, persist_directory=CHROMA_DIR)

    # keep the app sidebar's document list in sync
    with open(INGESTED_FILE, "w", encoding="utf-8") as f:
        json.dump(sorted(pdf_files), f)

    print(f"\nDone. Knowledge base saved to {CHROMA_DIR}")
    print(f"Ingested-docs list written to {INGESTED_FILE}")


if __name__ == "__main__":
    main()

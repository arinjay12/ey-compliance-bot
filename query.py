"""
query.py
--------
Terminal-based Q&A for quick testing without the Streamlit UI.
Run this after ingest.py has processed your documents.

Usage:
    python query.py
"""

import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama

CHROMA_DIR  = "./chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL   = "llama3"

def ask_question(question, retriever, llm):
    docs    = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""You are a SEBI compliance expert assistant.
Use the context below from SEBI circulars to answer accurately and concisely.
If the answer is not found in the context, say:
"This information is not available in the uploaded documents."

Context:
{context}

Question: {question}

Answer:"""

    answer = llm.invoke(prompt)
    return answer, docs

def main():
    if not os.path.exists(CHROMA_DIR):
        print("No knowledge base found. Please run ingest.py first.")
        return

    print("Loading knowledge base and models...")
    embeddings  = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    llm         = Ollama(model=LLM_MODEL, temperature=0.1)
    vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    retriever   = vectorstore.as_retriever(search_kwargs={"k": 4})

    print("\nSEBI Compliance Bot ready. Type 'exit' to quit.\n")
    print("-" * 60)

    while True:
        question = input("\nYour question: ").strip()
        if question.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break
        if not question:
            continue

        print("\nSearching circulars...")
        answer, sources = ask_question(question, retriever, llm)

        print(f"\nAnswer:\n{answer}")

        if sources:
            print("\nSources:")
            for i, doc in enumerate(sources[:3], 1):
                src  = os.path.basename(doc.metadata.get("source", "Unknown"))
                page = doc.metadata.get("page", "N/A")
                print(f"  [{i}] {src} — Page {page}")

        print("-" * 60)

if __name__ == "__main__":
    main()

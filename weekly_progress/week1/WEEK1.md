# Week 1 — Project Setup, Level 1 (Q&A), Level 2 (Monitoring)

**Period:** May 26 – June 1, 2026  
**Intern:** Arinjay (EY Technology Risk)  
**Supervisor:** Ashwini S Cheriyerimmel

---

## What Was Built

### Architecture Decision: Local RAG Pipeline

The core architectural choice was to use a **Retrieval-Augmented Generation (RAG)** pattern with fully local components — no external API calls, no data leaving the machine. This was important for a compliance use case where regulatory documents may be confidential.

```
PDF circulars → ChromaDB (local vector store) → Llama 3 (local LLM via Ollama)
```

Components chosen:
| Component | Choice | Reason |
|---|---|---|
| LLM | Llama 3 via Ollama | Free, local, no API key |
| Vector DB | ChromaDB | Lightweight, local, Python-native |
| Embeddings | `all-MiniLM-L6-v2` (HuggingFace) | Fast, accurate, ~90MB, runs on CPU |
| UI | Streamlit | Python-native, minimal boilerplate |
| Retrieval | LangChain 0.3+ (direct calls, no chains) | RetrievalQA was removed in newer LangChain |

---

## Level 1 — Document Q&A (`app.py`, `ingest.py`)

### What it does
- Upload SEBI/RBI circulars (PDF) via the sidebar
- Ask natural language questions about the documents
- Get answers with source citations (document name + page number)
- Answers stream word-by-word (like ChatGPT)
- 3 follow-up questions are automatically suggested after each answer
- Chat history persists to disk and reloads on app restart

### How ingestion works (`ingest.py`)
```
PDF → PyPDFLoader (extracts text page by page)
    → RecursiveCharacterTextSplitter (500-word chunks, 50-word overlap)
    → HuggingFaceEmbeddings (text → 384-dimensional vector)
    → Chroma.from_documents (vectors + text stored to ./chroma_db)
```

Each chunk is labelled with its document's name and date before embedding — this means queries like "June 2025 circular" semantically match the right document even if the chunk text doesn't explicitly repeat the date.

### How retrieval works (`app.py`)
```
Query → embedding → ChromaDB MMR search (k=4, fetch_k=30)
     → top chunks passed as context to Llama 3 prompt
     → answer streamed back via llm.stream()
```

MMR (Maximal Marginal Relevance) is used instead of plain similarity search — it fetches 30 candidates and picks 4 diverse results, preventing the same paragraph from being returned multiple times.

### Key technical challenges and fixes

**LangChain version conflicts:** `langchain.chains.RetrievalQA` was removed in LangChain 0.3+. `langchain.text_splitter` moved to `langchain_text_splitters`. Fixed by removing all chain usage and calling retriever + LLM directly — fewer abstractions, more predictable behaviour.

**Windows PATH:** `streamlit run` fails if Python's Scripts folder isn't on PATH (common with Python 3.14 on Windows). Fix: `python -m streamlit run app.py`.

**OneDrive Desktop:** `C:\Users\arinj\Desktop\` doesn't exist — Desktop is at `C:\Users\arinj\OneDrive\Desktop\`. All paths must use the OneDrive route.

**ChromaDB duplicate chunks:** Running `ingest.py` multiple times without clearing `chroma_db/` stacks chunks on top of each other. Fix: always delete `chroma_db/` before rebuilding. Added `Get-Process python | Stop-Process -Force` before delete because ChromaDB holds a file lock while the app is running.

---

## Level 2 — Automated Conflict Monitoring (`level2.py`)

### What it does
- Fetches live regulatory updates from SEBI RSS and RBI website
- Compares each new circular against stored documents using semantic similarity
- If a related stored document is found (L2 distance < 1.5), sends both to Llama 3
- Llama 3 checks if there's a conflict — superseded deadline, contradictory requirement, etc.
- If conflict found: sends email alert with circular details and conflict summary
- All results written to `level2_status.json`, read by `app.py` for the sidebar panel

### Data sources investigated
| Source | Result |
|---|---|
| `sebi.gov.in/sebirss.xml` (SEBI RSS) | ✅ Accessible — 20–30 recent items |
| `rbi.org.in/Scripts/BS_CircularIndexDisplay.aspx` | ✅ Accessible via BeautifulSoup table parse |
| `sebi.gov.in` (main website) | ❌ 403 Forbidden — blocks scrapers |

RBI table structure discovered: col0 = reference number + hyperlink, col1 = date, col2 = department, col3 = title.

### Email setup
Uses Gmail SMTP with App Password (not regular password). Port 587 STARTTLS with 465 SSL fallback for compatibility. College and corporate networks frequently block both ports — confirmed working on mobile hotspot.

### Scheduler
`python level2.py --schedule` runs a daily check at 09:00 using `schedule` library. `python level2.py` runs once immediately.

### Sidebar panel (`app.py`)
The "Level 2 Monitor" section in the sidebar shows:
- Last run timestamp
- Number of circulars checked
- Alert count (red if conflicts found, green if clear)
- "Run Check Now" button that calls `level2.run_check()` inline

---

## Knowledge Base Built

5 SEBI PDFs sourced from public regulatory repositories:

| File | Pages | Chunks | Source |
|---|---|---|---|
| SEBI Circular Jun 05 2025 | 3 | ~9 | SEBI website |
| SEBI Circular Jun 11 2025 (UPI) | 15 | ~50 | SEBI website |
| SEBI Master Circular Oct 15 2025 (NCS) | 215 | ~933 | SEBI website |
| SEBI Master Circular Jul 11 2025 (NCS LODR) | 114 | ~517 | Bajaj Housing Finance public server |
| SEBI Master Circular Jan 30 2026 (Listed Entities LODR) | 292 | ~1456 | NSDL public portal |

Total: **~2965 chunks** across 5 documents.

---

## Conflict Demonstrated (`demo_conflict.py`)

Tested and confirmed: the bot correctly identifies that the **SEBI Master Circular January 30 2026** supersedes the **June 2025 LODR relaxation circular**.

- June 2025 circular: grants relaxation from Regulation 58(1)(b) until **September 30, 2025**
- January 2026 Master Circular: consolidates all LODR obligations — the relaxation period has expired
- Llama 3 output: correctly stated the Sept 30 2025 deadline had passed and the January 2026 circular now governs

---

## Repo
GitHub: [arinjay12/ey-compliance-bot](https://github.com/arinjay12/ey-compliance-bot)  
Commits: Initial commit + README

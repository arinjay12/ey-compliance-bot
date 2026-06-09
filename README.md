# EY SEBI Compliance Bot

A local, AI-powered compliance assistant built for EY. Upload SEBI/RBI regulatory circulars and ask natural language questions. The bot retrieves relevant sections, generates accurate answers, and automatically monitors live regulatory sources for conflicts with stored documents.

All processing happens **locally on your machine** — no data is sent to external APIs, no cloud cost.

---

## Weekly Progress

| Week | Dates | What Was Built | Status |
|---|---|---|---|
| [Week 1](weekly_progress/week1/WEEK1.md) | May 26 – Jun 1, 2026 | Project setup · Level 1 (Q&A) · Level 2 (monitoring + email) | Complete |
| [Week 2](weekly_progress/week2/WEEK2.md) | Jun 2 – Jun 8, 2026 | Level 3 (PDF/Excel export) · **evaluation harness · corpus 5→25 docs · pipeline hardening** | Complete |
| [Week 3](weekly_progress/week3/WEEK3.md) | Jun 9 – Jun 15, 2026 | **Conversational follow-ups · generalised retrieval · deployability hygiene** | Complete |
| Week 4 | Jun 16 – Jun 22, 2026 | Level 4 (charts) | Upcoming |
| Week 5 | Jun 23 – Jun 29, 2026 | Level 5 (audio) | Upcoming |

> Detailed writeups of decisions, challenges, and solutions for each week are in the [`weekly_progress/`](weekly_progress/) folder.

---

## Feature Levels

| Level | Feature | Status |
|---|---|---|
| 1 | Document Q&A + source citations + follow-up suggestions | ✅ Complete |
| 2 | Automated web monitoring + conflict detection + email alerts | ✅ Complete |
| 3 | Download chat history as PDF / Excel report | ✅ Complete |
| 4 | Chart responses for numerical / tabular data | ✅ Complete |
| 5 | Audio input and audio responses | Upcoming |

## Evaluation

The pipeline is evaluated against a gold Q&A set grounded in the source circulars
(see [`evaluation/`](evaluation/)). On a **25-document** corpus:

| Metric | RAG bot | Raw Llama 3 (no retrieval) |
|---|---|---|
| Factual accuracy (21 answerable questions) | **86–90%** | ~15–19% |
| Out-of-scope questions correctly refused (3) | **3 / 3** | 0 / 3 |
| Retrieval hit-rate (hybrid) | **100%** | — |

The pipeline is several times more accurate than the same model without retrieval, and
refuses questions outside the knowledge base rather than answering from unrelated text.
The accuracy range reflects run-to-run nondeterminism in the local model (temperature
0.1) on a few borderline questions; retrieval matching is deterministic.
Reproduce with `python evaluation/eval_answers.py`.

---

## What It Does

### Level 1 — Document Q&A
- Upload any SEBI/RBI circular (PDF) through the sidebar
- Ask questions in plain English — the bot finds the most relevant sections and answers using Llama 3
- Every answer shows the source circular and page number it came from
- After each answer, 3 follow-up questions are suggested automatically
- Chat history persists across sessions

### Level 2 — Automated Conflict Monitoring
- Fetches the SEBI RSS feed and RBI circulars page daily
- Compares new circulars against stored documents using semantic search
- Llama 3 identifies factual conflicts or gaps (e.g. a deadline in an older circular superseded by a newer one)
- Sends an automated email alert when a conflict is detected
- Results visible in the app sidebar under "Level 2 Monitor"

### Level 3 — Export Reports
- Download the full chat session as a formatted **PDF** (EY-branded, A4, timestamped)
- Download as an **Excel** spreadsheet (Q&A rows, yellow header, zebra-striped)
- Export buttons appear in the sidebar after the first answer

### Level 4 — Chart Responses
- When an answer contains numeric data (amounts, percentages, day-based timelines,
  counts), the bot renders a chart beneath the text answer
- The chart is extracted from the answer itself, so it always matches what was said,
  and it only runs when the answer actually has chartable numbers (ordinary answers
  stay fast)
- Charts persist in the chat history
- Example query: *"What are the UPI transaction limits for capital market transactions?"*

---

## Tech Stack

| Component | Purpose |
|---|---|
| Python + Streamlit | Backend logic and chat UI |
| LangChain | Orchestrates the retrieval pipeline |
| ChromaDB | Stores document chunks as vectors (local) |
| Sentence Transformers (`all-MiniLM-L6-v2`) | Converts text to embeddings for semantic search |
| Llama 3 via Ollama | LLM running locally — no API key needed |
| ReportLab | Generates PDF export reports |
| openpyxl | Generates Excel export reports |
| BeautifulSoup + requests | Scrapes SEBI RSS feed and RBI circulars page |

---

## How It Works

### Document Ingestion (`ingest.py`)
```
PDF → split into 500-word chunks (50-word overlap)
    → each chunk labelled with document name/date
    → each chunk converted to a vector (Sentence Transformers)
    → vectors stored in ChromaDB on disk
```

### Query Pipeline (`app.py`) — Hybrid Retrieval
```
User question → MMR semantic search (4 chunks from 30 candidates)
             → if query mentions specific circular by date/topic →
               inject up to 3 chunks directly from that document
             → all chunks + question sent to Llama 3
             → answer streamed back with source citations
             → 3 follow-up suggestions generated
```

> Hybrid retrieval prevents large documents (e.g. a 292-page master circular) from drowning out smaller, more specific ones.

### Level 2 Pipeline (`level2.py`)
```
Daily trigger → fetch SEBI RSS + RBI circulars page
             → for each new circular, search ChromaDB for related stored content
             → if related content found → send both to Llama 3 for conflict analysis
             → if conflict detected → send email alert automatically
             → results saved to level2_status.json (shown in app sidebar)
```

---

## Project Structure

```
ey-bot/
├── app.py                     # Main Streamlit chat UI (Levels 1, 2, 3)
├── rag_core.py                # Shared RAG pipeline: retrieval, prompting, LLM, refusal gate
├── ingest.py                  # Processes PDFs into ChromaDB with metadata (run once)
├── download_docs.py           # Expands the corpus with public RBI Master Directions
├── query.py                   # Terminal-based Q&A for testing
├── level2.py                  # Automated daily monitoring + email alerts
├── demo_conflict.py           # Demonstrates conflict detection between stored docs
├── scraper_test.py            # Tests which regulatory sources are scrapable
├── requirements.txt           # All Python dependencies
├── docs/                      # Place your PDF circulars here (gitignored)
├── chroma_db/                 # Auto-generated vector database (gitignored)
├── evaluation/                # Gold Q&A set + eval harness + results
│   ├── gold_qa.json
│   ├── eval_retrieval.py
│   ├── eval_answers.py
│   └── EVAL_REPORT.md
└── weekly_progress/           # Week-by-week progress writeups
    ├── week1/WEEK1.md
    └── week2/WEEK2.md
```

---

## Setup

### 1. Prerequisites
- Python 3.11+
- [Ollama](https://ollama.com) installed and running

### 2. Pull the LLM
```bash
ollama pull llama3
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your documents
Place SEBI/RBI circular PDFs into the `docs/` folder.

### 5. Build the knowledge base
```bash
python ingest.py
```
> First run downloads the ~90MB embedding model. Needs internet once.

### 6. Launch the app
```bash
python -m streamlit run app.py
```
Opens at `http://localhost:8501`

---

## Level 2 — Email Setup

Open `level2.py` and fill in the config section at the top:

```python
SENDER_EMAIL    = "your_gmail@gmail.com"
SENDER_PASSWORD = "your_app_password"   # Gmail App Password (not regular password)
RECEIVER_EMAIL  = "recipient@example.com"
```

**Getting a Gmail App Password:**
1. Go to [myaccount.google.com](https://myaccount.google.com) → Security → 2-Step Verification
2. Scroll down → App Passwords → create one named `EY Bot`
3. Paste the 16-character password into `SENDER_PASSWORD`

**Run once manually:**
```bash
python level2.py
```

**Run on daily schedule (9AM every day):**
```bash
python level2.py --schedule
```

**Demo conflict detection without live scraping:**
```bash
python demo_conflict.py
```

---

## Knowledge Base (current)

**25 documents · ~4,125 chunks.** Five core SEBI circulars plus 20 public RBI Master
Directions (added to evaluate retrieval robustness at scale; download with
`python download_docs.py --download 20`).

| Core SEBI document | Description |
|---|---|
| SEBI Circular Jun 05, 2025 | Limited relaxation from LODR Regulation 58(1)(b) |
| SEBI Circular Jun 11, 2025 | UPI payment mechanism for SEBI intermediaries |
| SEBI Master Circular Oct 15, 2025 | Issue and listing of non-convertible securities |
| SEBI Master Circular Jul 11, 2025 | LODR obligations — non-convertible securities |
| SEBI Master Circular Jan 30, 2026 | Consolidated LODR compliance for listed entities |
| + 20 RBI Master Directions | Public banking/NBFC regulations (corpus distractors) |

---

## Notes

- Llama 3 running locally is slower without a GPU (20–40s per response). This is a hardware constraint, not a code issue.
- The SEBI website blocks direct scraping. Level 2 uses the SEBI RSS feed (`sebi.gov.in/sebirss.xml`) and RBI's circulars page instead, both of which are accessible.
- Email sending requires an open network — corporate/university networks often block SMTP ports. Test on a hotspot or home network if needed.
- Run the app with `python -m streamlit run app.py` (not `streamlit run`) if `streamlit` is not on your PATH.

# SEBI Compliance Bot

A local, AI-powered compliance assistant Upload SEBI/RBI regulatory circulars and ask natural language questions. The bot retrieves relevant sections, generates accurate answers, and automatically monitors live regulatory sources for conflicts with stored documents.

All processing happens **locally on your machine** — no data is sent to external APIs, no cloud cost.

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
- Llama 3 identifies factual conflicts or gaps (e.g. a deadline in an older circular that a newer one has superseded)
- Sends an automated email alert when a conflict is detected
- Results visible in the app sidebar under "Level 2 Monitor"

---

## Tech Stack

| Component | Purpose |
|---|---|
| Python + Streamlit | Backend logic and chat UI |
| LangChain | Orchestrates the retrieval pipeline |
| ChromaDB | Stores document chunks as vectors (local) |
| Sentence Transformers (`all-MiniLM-L6-v2`) | Converts text to embeddings for semantic search |
| Llama 3 via Ollama | LLM running locally — no API key needed |
| BeautifulSoup + requests | Scrapes SEBI RSS feed and RBI circulars page |

---

## How It Works

### Document Ingestion (`ingest.py`)
```
PDF → split into 500-word chunks (50-word overlap)
    → each chunk converted to a vector (Sentence Transformers)
    → vectors stored in ChromaDB on disk
```

### Query Pipeline (`app.py`)
```
User question → converted to vector
             → ChromaDB finds 4-6 most similar chunks (hybrid: MMR + document-targeted)
             → chunks + question sent to Llama 3
             → answer streamed back with source citations
             → 3 follow-up suggestions generated
```

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
├── app.py                  # Main Streamlit chat UI
├── ingest.py               # Processes PDFs into ChromaDB (run once)
├── query.py                # Terminal-based Q&A for testing
├── level2.py               # Automated daily monitoring + email alerts
├── demo_conflict.py        # Demonstrates conflict detection between stored docs
├── scraper_test.py         # Tests which regulatory sources are scrapable
├── requirements.txt        # All Python dependencies
├── docs/                   # Place your PDF circulars here
└── chroma_db/              # Auto-generated vector database (gitignored)
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
streamlit run app.py
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

| Document | Description |
|---|---|
| SEBI Circular Jun 05, 2025 | Limited relaxation from LODR Regulation 58(1)(b) |
| SEBI Circular Jun 11, 2025 | UPI payment mechanism for SEBI intermediaries |
| SEBI Master Circular Oct 15, 2025 | Issue and listing of non-convertible securities |
| SEBI Master Circular Jul 11, 2025 | LODR obligations — non-convertible securities |
| SEBI Master Circular Jan 30, 2026 | Consolidated LODR compliance for listed entities |

---

## Roadmap

| Level | Feature | Status |
|---|---|---|
| 1 | Document Q&A + trailing suggestions | Complete |
| 2 | Automated web monitoring + email alerts | Complete |
| 3 | Download responses as PDF / Excel | Upcoming |
| 4 | Chart responses for numerical data | Upcoming |
| 5 | Audio input and audio responses | Upcoming |

---

## Notes

- Llama 3 running locally is slower without a GPU (20–40s per response). This is a hardware constraint, not a code issue.
- The SEBI website blocks direct scraping. Level 2 uses the SEBI RSS feed (`sebi.gov.in/sebirss.xml`) and RBI's circulars page instead, both of which are accessible.
- Email sending requires an open network — corporate/university networks often block SMTP ports. Test on a hotspot or home network if needed.

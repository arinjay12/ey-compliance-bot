# Week 2 — Retrieval Fix + Level 3 (PDF/Excel Export)

## What Was Built

### Problem: Retrieval Domination Bug

After Week 1, the bot was giving wrong answers to queries about specific smaller circulars.

**Root cause:** The January 2026 Master Circular (292 pages, 1456 chunks) represents **49% of all vectors** in ChromaDB. Pure semantic search almost always returned chunks from this document regardless of what was asked. For example:

> Query: *"What is the deadline mentioned in the June 2025 LODR relaxation circular?"*
>
> Wrong answer: *"According to the context, there is no specific deadline mentioned in the June 2025 LODR relaxation circular. The provided documents are from January 30, 2026..."*

The June 2025 circular is only 3 pages (9 chunks). It was being completely drowned out.

**Secondary cause:** The `chroma_db/` directory had been rebuilt multiple times without clearing first, creating duplicate chunks from repeated `ingest.py` runs. This made the imbalance even worse.

---

### Fix 1 — Clean ChromaDB Rebuild

Killed all Python processes (`Get-Process python | Stop-Process -Force`) to release the ChromaDB file lock, then deleted `chroma_db/` and ran `ingest.py` once from scratch.

Result: Clean 2965 chunks, no duplicates.

---

### Fix 2 — Hybrid Retrieval (`app.py`)

The core fix is a two-stage retrieval pipeline:

**Stage 1 — MMR semantic search (broad)**
```python
retriever = vs.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 30})
docs = list(retriever.invoke(query))
```

**Stage 2 — Keyword-triggered direct injection (targeted)**
```python
DOC_KEYWORD_MAP = {
    r"june\s*2025|jun\s*2025|lodr relaxation|regulation 58\(1\)|reg 58":
        "18__SEBI_Circular_dated_June_05_2025.pdf",
    r"upi.*intermediar|intermediar.*upi|upi payment|payment.*upi":
        "1749641449497.pdf",
    r"october\s*2025|oct\s*2025|october 15":
        "39_SEBI_Circular_dated_October_15_2025.pdf",
    r"july\s*2025|jul\s*2025|master circular.*ncs|ncs.*master circular":
        "SEBI_Master_Circular_LODR_NCS_July2025.pdf",
    r"jan\w*\s*2026|january\s*2026":
        "SEBI_Master_Circular_LODR_Listed_Entities_Jan2026.pdf",
}

for pattern, filename in DOC_KEYWORD_MAP.items():
    if re.search(pattern, query_lower):
        src_path = f"./docs\\{filename}"
        targeted = vs.similarity_search(query, k=4, filter={"source": {"$eq": src_path}})
        # inject up to 3 new chunks not already in results
        break
```

If the user's query contains a recognizable date or document keyword (e.g. "June 2025", "lodr relaxation", "reg 58"), chunks are injected directly from that specific document via ChromaDB's `$eq` metadata filter, bypassing semantic ranking entirely. These are added on top of the MMR results.

**Verification test result:**
```
Pattern matched! Filtering for: './docs\18__SEBI_Circular_dated_June_05_2025.pdf'
Targeted results: 4
→ "For the period June 06, 2025 to September 30, 2025, similar relaxation..."
```
The September 30, 2025 deadline is now correctly retrieved.

**Important implementation note:** ChromaDB on Windows stores source paths with a backslash: `./docs\filename.pdf`. The filter must match this exactly. Using `f"./docs\\{filename}"` in an f-string produces a literal backslash when the filename variable is substituted — no escape interpretation occurs on the substituted value.

---

## Level 3 — Export as PDF / Excel

### New dependencies
```
reportlab >= 4.0.0   # PDF generation
openpyxl  >= 3.1.0   # Excel generation
```

### PDF export (`generate_pdf_report`)
Generates an A4 PDF using ReportLab's Platypus layout engine:
- EY yellow `#FFE600` horizontal rule as header divider
- Dark `#2E2E38` text for headings (EY brand colours)
- Each Q&A pair numbered and separated by a light divider
- Timestamp and footer note
- XML special characters (`&`, `<`, `>`) escaped for ReportLab compatibility

### Excel export (`generate_excel_report`)
Generates an `.xlsx` using openpyxl:
- Columns: `#`, `Question`, `Answer`
- Header row: yellow fill (`FFE600`), bold dark font, centred
- Wide `Answer` column (90 chars) with `wrap_text=True` so long answers are readable
- Alternating row fill for readability (zebra stripe)

### UI integration
Both exports appear as `st.download_button` elements in the sidebar under "Export Chat History". The buttons only render after at least one assistant message exists (checked via `any(m["role"] == "assistant" for m in messages)`). Generating the report bytes happens at render time — Streamlit's download button accepts raw bytes directly.

---

## Files Changed This Week

| File | Change |
|---|---|
| `app.py` | Added `DOC_KEYWORD_MAP`, hybrid retrieval in `retrieve_and_build_prompt()`, `generate_pdf_report()`, `generate_excel_report()`, `_extract_qa_pairs()`, export sidebar panel, Level 3 imports |
| `requirements.txt` | Added `reportlab>=4.0.0`, `openpyxl>=3.1.0` |
| `README.md` | Added weekly progress table, Level 3 to feature list, hybrid retrieval explanation, updated project structure |
| `.gitignore` | Added `HANDOFF.md` |

---

## Commits This Week

- `c277d1b` — Add Level 3: PDF and Excel export of chat history
- `7b6a33d` — (remote) Minor README update from GitHub
- `3d47006` — Add README with full project documentation *(end of Week 1)*

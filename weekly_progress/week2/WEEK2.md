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

---

# Part 2 — Evaluation & Hardening

**Why:** the Week 1 ChromaDB issue showed retrieval quality needed to be measured
systematically rather than spot-checked, and five documents were too few to validate
retrieval at a realistic scale. This part adds a repeatable evaluation harness, expands
the corpus, and fixes the issues the evaluation surfaced.

## 1. Evaluation harness (`evaluation/`)

- `gold_qa.json` — 24 questions grounded in the source circulars (reference numbers,
  deadlines, the supersession/conflict case, and out-of-scope questions that should be
  refused).
- `eval_retrieval.py` — measures whether the correct document is retrieved (no LLM).
- `eval_answers.py` — measures answer correctness, and compares the RAG pipeline against
  the same LLM with no retrieval (the baseline that quantifies what retrieval adds).
- `calibrate_gate.py` — sets the out-of-scope refusal threshold from measured data.

## 2. Corpus expansion (5 → 25 documents)

The client's documents are confidential and unavailable, so the corpus was expanded
with **20 public RBI Master Directions** (`download_docs.py`, from `rbidocs.rbi.org.in`).
These act as **distractors** for the SEBI questions, testing retrieval against a
realistic document volume. Result: **4,125 chunks across 25 documents** (was ~2,965
across 5).

## 3. Pipeline extracted and hardened (`rag_core.py`)

The retrieval and prompting logic was moved out of the Streamlit UI into `rag_core.py`,
shared by the app, Level 2 and the evaluation harness. This also removed a redundant
ChromaDB re-instantiation on every query. The evaluation surfaced five issues, each fixed:

| Issue | Root cause | Fix |
|---|---|---|
| Reg-number questions wrong | The official ref number lives only in the page-1 header chunk, which is rarely retrieved | Extract ref/date as **metadata** on every chunk; surface it in the prompt's source tag |
| Out-of-scope questions answered | LLM answered from loosely-related chunks | Relevance gate + "a passing mention of a regulation is not the answer" instruction |
| Over-refusal regression | Too-strict refusal prompt refused even when the answer was present | Rebalanced to "answer if present, refuse only if absent" |
| Wrong document injected | Keyword map stopped at the first match | Inject from **all** matched documents |
| Table/footnote facts lost | 500-char chunks fragmented them | 1,000-char chunks (150 overlap) |

`ingest.py` now auto-extracts each document's reference number, date and title (curated
values for the core SEBI circulars, regex extraction for the rest), and `app.py` was
rewired to use `rag_core`.

## 4. Results (before vs after)

| Metric | Baseline (5 docs) | **After (25 docs)** | Raw Llama 3 (no retrieval) |
|---|---|---|---|
| Factual accuracy | 71.4% (15/21) | **90.5% (19/21)** | 14.3% (3/21) |
| Out-of-scope correctly refused | 1 / 3 | **3 / 3** | 0 / 3 |
| Retrieval hit-rate (hybrid) | — | **95.2%** (vs 71.4% semantic-only) | — |

- The RAG pipeline is ~6× more accurate than the same model used without retrieval. The
  raw LLM fabricated official circular numbers (e.g. `SEBI/HO/CFD/DIL/CIR/P/2022/0003`)
  and denied that real circulars existed.
- Accuracy improved despite the corpus growing 5×, indicating the pipeline scales.
- The keyword-injection step adds +4.8 pts on 5 documents but +23.8 pts against 20
  distractors — its contribution shows mainly at scale.

## 5. Known limitations

- **q04** — phrased across two dates ("June 06 to September 30, 2025") with no document
  or regulation named; neither keyword nor semantic retrieval returns the right chunk.
- **q11** — the answer (a 180-day timeline) sits inside a PDF **table** that extracts and
  embeds poorly, so it is not retrieved. A proper fix needs table-aware parsing (future
  work).

## Files Added / Changed in Part 2

| File | Change |
|---|---|
| `rag_core.py` | **New.** Shared RAG pipeline: cached embeddings/vector-store/LLM, hybrid retrieval, metadata source tags, relevance-gated refusal, vanilla-LLM baseline. |
| `download_docs.py` | **New.** Downloads public RBI Master Direction PDFs to expand the corpus. |
| `evaluation/` | **New.** Gold Q&A set, retrieval + answer eval scripts, gate calibration, results, and `EVAL_REPORT.md`. |
| `ingest.py` | Rewritten: auto-extracts per-document metadata, 1,000-char chunks, keeps `ingested_docs.json` in sync. |
| `app.py` | Rewired to use `rag_core` (no more per-query ChromaDB reload); relevance gate added to the chat flow. |
| `requirements.txt` | Added `requests`, `beautifulsoup4`, `schedule`. |

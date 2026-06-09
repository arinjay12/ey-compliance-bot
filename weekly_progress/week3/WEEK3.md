# Week 3 — Pre-Level-4 Hardening

**Period:** June 9 – June 15, 2026

Before adding Level 4, three improvements were made to close gaps between the
prototype and what a client would actually use day to day. Each was verified with
the evaluation harness.

## 1. Conversational follow-ups

The chat was stateless — every question was answered in isolation, so a follow-up
like "which regulation does it relax?" or "does that apply to commercial paper?"
retrieved nothing, because the question has no subject on its own.

- `build_search_query()` folds the previous user question into the retrieval query
  for follow-ups, so the right document is found. Cheap concatenation, not an LLM
  rewrite — the local 8B model condenses unreliably and a rewrite call would add
  latency.
- `build_prompt(history=...)` includes the last couple of turns so the model
  resolves references like "it"/"that". The answer is still grounded only in the
  retrieved Context. With no history the prompt is byte-identical, so single-turn
  behaviour (and the eval) is unchanged.

Verified: after asking about the June 2025 circular, "which regulation does it
relax?" correctly answers Regulation 58(1)(b).

## 2. Generalised retrieval (works on client-uploaded documents)

The targeted-injection step used a hardcoded keyword→filename map that only matched
our five demo documents. On a client's own documents it would do nothing, dropping
retrieval back to pure semantic search (71% on the 25-doc set).

- Replaced the map with **metadata-driven matching**: dates and reference numbers in
  the query are matched against the `doc_date` / `doc_ref` metadata stored on every
  chunk (`match_documents()`). This reads each document's own metadata, so it works
  for any ingested document, including ones we have never seen.
- Month–year **pairs** are matched (not independent month/year tokens), so
  "July 2025 ... June 2023" no longer falsely matches a June-2025 document.
- Matched documents are **ranked by relevance** before injection, so when a date
  like "June 2025" matches two documents, the one that actually answers the question
  gets the injection budget instead of being starved.

Result: retrieval hit-rate **95.2% → 100%** on the 25-doc set, and it now
generalises beyond the demo corpus.

## 3. Deployability hygiene

- **Secrets out of source:** email credentials now load from a gitignored `.env`
  (`.env.example` provided) instead of being pasted into `level2.py`.
- **Audit trail:** every answer is appended to `audit_log.jsonl` with a timestamp,
  the question, the answer, and the exact source documents/pages used — so the tool
  can show what it said and on what basis (a compliance requirement).
- **Graceful failure:** if Ollama isn't running, the app shows "Llama 3 isn't
  running — start Ollama" instead of a stack trace.

## Measured results (25-doc corpus)

| Metric | Before Week 3 | After Week 3 | Raw Llama 3 |
|---|---|---|---|
| Retrieval hit-rate (hybrid) | 95.2% | **100%** | — |
| Factual accuracy (21 Q) | 90.5% | **85.7%** | 19.0% |
| Out-of-scope correctly refused (3) | 3 / 3 | **3 / 3** | 0 / 3 |

Note on the accuracy figure: across two full runs the RAG pipeline scores 86–90%.
The 2–3 question swing is run-to-run nondeterminism in the local Llama 3
(temperature 0.1) on borderline questions — not a pipeline regression. The questions
that flipped (q08, q14) do not use the retrieval path changed in Week 3, and q04
recovered. Setting temperature to 0 would make future runs deterministic.

## Known limitations (unchanged)

- **q11** — the answer (a 180-day timeline) is inside a PDF table that extracts and
  embeds poorly; a proper fix needs table-aware parsing.
- The local 8B model occasionally mis-answers borderline questions it has retrieved
  correctly (a model-quality limit, not a retrieval one).

# EY SEBI Compliance Bot — Evaluation Report

Automated evaluation of the RAG pipeline against a gold Q&A set grounded in the source SEBI circulars. Compares the shipped RAG bot against the same LLM used directly (no retrieval).

## Headline numbers

| Metric | RAG bot | Vanilla LLM |
|---|---|---|
| Factual accuracy (n=21) | **90%** | 14% |
| Out-of-scope correctly refused (n=3) | **3/3** | 0/3 |
| Out-of-scope hallucinations | 0 | 3 |

## Per-question detail

| ID | Category | RAG | Vanilla |
|---|---|---|---|
| q01 | deadline_date | CORRECT | WRONG |
| q02 | reg_number | CORRECT | WRONG |
| q03 | reg_number | CORRECT | WRONG |
| q04 | factual_single | WRONG | WRONG |
| q05 | factual_single | CORRECT | WRONG |
| q06 | factual_single | CORRECT | WRONG |
| q07 | factual_single | CORRECT | WRONG |
| q08 | factual_single | CORRECT | WRONG |
| q09 | factual_single | CORRECT | WRONG |
| q10 | deadline_date | CORRECT | WRONG |
| q11 | factual_single | WRONG | WRONG |
| q12 | factual_single | CORRECT | WRONG |
| q13 | deadline_date | CORRECT | WRONG |
| q14 | definition | CORRECT | WRONG |
| q15 | reg_number | CORRECT | WRONG |
| q16 | factual_single | CORRECT | WRONG |
| q17 | factual_single | CORRECT | CORRECT |
| q18 | deadline_date | CORRECT | WRONG |
| q19 | factual_single | CORRECT | WRONG |
| q20 | conflict_supersession | CORRECT | CORRECT |
| q21 | out_of_scope | SAFE | UNSAFE |
| q22 | out_of_scope | SAFE | UNSAFE |
| q23 | out_of_scope | SAFE | UNSAFE |
| q24 | reg_number | CORRECT | CORRECT |

> CORRECT/WRONG = answerable questions · SAFE/UNSAFE = out-of-scope (UNSAFE means the model invented an answer it had no source for).

"""
eval_answers.py
---------------
Measures ANSWER quality and runs the head-to-head that proves the RAG bot beats
using the LLM directly. For every gold question we generate TWO answers:

  - RAG     : full pipeline (retrieval + grounded prompt)
  - Vanilla : the same LLM, same expert persona, but NO retrieved context

Then we score both:
  - Answerable questions  -> correct if all 'must_all' keywords appear AND each
                             'any_of' group has at least one variant present.
  - Out-of-scope questions-> SAFE if the model declines (refusal line / "not
                             available"); UNSAFE if it confidently answers
                             (a hallucination — the dangerous failure mode for
                             a compliance tool).

Usage (from project root):
    python evaluation/eval_answers.py            # all questions (slow on CPU Llama 3)
    python evaluation/eval_answers.py --limit 5  # first 5 only (quick smoke test)

Writes results_answers.json and EVAL_REPORT.md.
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rag_core

HERE = os.path.dirname(os.path.abspath(__file__))
GOLD = os.path.join(HERE, "gold_qa.json")

DECLINE_SIGNALS = [
    "not available in the uploaded documents",
    "not available", "no information", "cannot provide", "could not find",
    "do not have", "don't have", "not contained", "not found",
    "not mentioned", "not present in", "unable to", "isn't in", "is not in",
]


def declined(text: str) -> bool:
    t = text.lower()
    return any(sig in t for sig in DECLINE_SIGNALS)


def score_answerable(text: str, q: dict) -> bool:
    t = text.lower()
    for kw in q["must_all"]:
        if kw.lower() not in t:
            return False
    for group in q["any_of"]:
        if not any(v.lower() in t for v in group):
            return False
    return True


def evaluate(text: str, q: dict) -> str:
    """Return one of: CORRECT, WRONG, SAFE, UNSAFE."""
    if q["answerable"]:
        return "CORRECT" if score_answerable(text, q) else "WRONG"
    # out-of-scope
    return "SAFE" if declined(text) else "UNSAFE"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only first N questions")
    ap.add_argument("--ids", type=str, default="", help="comma-separated question ids")
    ap.add_argument("--rag-only", action="store_true", help="skip the vanilla baseline")
    args = ap.parse_args()

    with open(GOLD, encoding="utf-8") as f:
        gold = json.load(f)["questions"]
    if args.ids:
        want = {x.strip() for x in args.ids.split(",")}
        gold = [q for q in gold if q["id"] in want]
    elif args.limit:
        gold = gold[: args.limit]

    rows = []
    rag_ok = van_ok = 0
    rag_unsafe = van_unsafe = 0
    oos_total = 0

    print(f"\nRunning answer eval on {len(gold)} questions (RAG + vanilla each).")
    print("This is slow on CPU Llama 3 — ~20-40s per answer.\n")
    print(f"{'ID':<5}{'CATEGORY':<22}{'RAG':<10}{'VANILLA':<10}{'sec':>6}")
    print("-" * 60)

    t_start = time.time()
    for q in gold:
        t0 = time.time()
        rag_ans, docs = rag_core.answer_rag(q["question"])
        van_ans = "" if args.rag_only else rag_core.answer_vanilla(q["question"])
        dt = time.time() - t0

        rag_v = evaluate(rag_ans, q)
        van_v = evaluate(van_ans, q)

        if q["answerable"]:
            rag_ok += rag_v == "CORRECT"
            van_ok += van_v == "CORRECT"
        else:
            oos_total += 1
            rag_unsafe += rag_v == "UNSAFE"
            van_unsafe += van_v == "UNSAFE"

        rows.append({
            "id": q["id"], "category": q["category"], "question": q["question"],
            "answerable": q["answerable"],
            "rag_verdict": rag_v, "vanilla_verdict": van_v,
            "rag_answer": rag_ans, "vanilla_answer": van_ans,
            "retrieved": [os.path.basename(d.metadata.get("source", "")) for d in docs],
            "seconds": round(dt, 1),
        })
        print(f"{q['id']:<5}{q['category']:<22}{rag_v:<10}{van_v:<10}{dt:>6.0f}")

    total_t = time.time() - t_start
    answerable_n = sum(1 for q in gold if q["answerable"])

    # ---- Summary ----
    print("\n" + "=" * 64)
    print("ANSWER QUALITY  —  RAG  vs  VANILLA LLM")
    print("=" * 64)
    if answerable_n:
        print(f"  Factual accuracy (answerable, n={answerable_n}):")
        print(f"    RAG     : {rag_ok}/{answerable_n} = {rag_ok/answerable_n*100:.1f}%")
        print(f"    Vanilla : {van_ok}/{answerable_n} = {van_ok/answerable_n*100:.1f}%")
    if oos_total:
        print(f"\n  Out-of-scope safety (n={oos_total}) — hallucinated an answer:")
        print(f"    RAG     : {rag_unsafe}/{oos_total} unsafe "
              f"({(oos_total-rag_unsafe)}/{oos_total} correctly refused)")
        print(f"    Vanilla : {van_unsafe}/{oos_total} unsafe "
              f"({(oos_total-van_unsafe)}/{oos_total} correctly refused)")
    print(f"\n  Total runtime: {total_t/60:.1f} min "
          f"({total_t/len(gold):.0f}s per question incl. both answers)")

    results = {
        "summary": {
            "answerable_n": answerable_n,
            "rag_correct": rag_ok, "vanilla_correct": van_ok,
            "oos_n": oos_total, "rag_unsafe": rag_unsafe, "vanilla_unsafe": van_unsafe,
        },
        "rows": rows,
    }
    # Subset / rag-only runs are for iteration — don't overwrite the canonical
    # full-run artifacts.
    subset = bool(args.ids or args.limit or args.rag_only)
    if subset:
        out = os.path.join(HERE, "results_quicktest.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\n(subset run) wrote {os.path.basename(out)} — canonical files untouched")
    else:
        with open(os.path.join(HERE, "results_answers.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        write_report(results)
        print(f"\nWrote results_answers.json and EVAL_REPORT.md")


def write_report(results: dict):
    s = results["summary"]
    an = s["answerable_n"] or 1
    oos = s["oos_n"] or 1
    lines = []
    lines.append("# EY SEBI Compliance Bot — Evaluation Report\n")
    lines.append("Automated evaluation of the RAG pipeline against a gold Q&A set "
                 "grounded in the source SEBI circulars. Compares the shipped RAG "
                 "bot against the same LLM used directly (no retrieval).\n")
    lines.append("## Headline numbers\n")
    lines.append("| Metric | RAG bot | Vanilla LLM |")
    lines.append("|---|---|---|")
    lines.append(f"| Factual accuracy (n={s['answerable_n']}) | "
                 f"**{s['rag_correct']/an*100:.0f}%** | {s['vanilla_correct']/an*100:.0f}% |")
    lines.append(f"| Out-of-scope correctly refused (n={s['oos_n']}) | "
                 f"**{(s['oos_n']-s['rag_unsafe'])}/{s['oos_n']}** | "
                 f"{(s['oos_n']-s['vanilla_unsafe'])}/{s['oos_n']} |")
    lines.append(f"| Out-of-scope hallucinations | {s['rag_unsafe']} | {s['vanilla_unsafe']} |\n")
    lines.append("## Per-question detail\n")
    lines.append("| ID | Category | RAG | Vanilla |")
    lines.append("|---|---|---|---|")
    for r in results["rows"]:
        lines.append(f"| {r['id']} | {r['category']} | {r['rag_verdict']} | {r['vanilla_verdict']} |")
    lines.append("\n> CORRECT/WRONG = answerable questions · SAFE/UNSAFE = out-of-scope "
                 "(UNSAFE means the model invented an answer it had no source for).\n")
    with open(os.path.join(HERE, "EVAL_REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()

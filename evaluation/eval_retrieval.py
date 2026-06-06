"""
eval_retrieval.py
-----------------
Measures RETRIEVAL quality against the gold Q&A set — the layer that caused the
ChromaDB domination bug. For each answerable question we check whether a chunk
from the EXPECTED document appears in the retrieved set (hit@k).

We run it two ways:
  - hybrid   = the shipped pipeline (MMR + keyword-map injection)
  - semantic = pure MMR only (no keyword injection)

The gap between them tells us how much the bot leans on the hardcoded
DOC_KEYWORD_MAP crutch — i.e. how fragile retrieval would be on documents the
map doesn't know about.

Run from the project root:
    python evaluation/eval_retrieval.py

No LLM required — fast.
"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rag_core

GOLD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gold_qa.json")


def retrieved_docnames(query, use_hybrid):
    docs = rag_core.retrieve(query, use_hybrid=use_hybrid)
    return [Path(d.metadata.get("source", "")).name for d in docs]


def main():
    with open(GOLD, encoding="utf-8") as f:
        gold = json.load(f)["questions"]

    answerable = [q for q in gold if q["answerable"]]

    rows = []
    hyb_hits = sem_hits = 0
    cat_hyb = defaultdict(lambda: [0, 0])   # category -> [hits, total]

    print(f"\nEvaluating retrieval on {len(answerable)} answerable questions...\n")
    print(f"{'ID':<5}{'CATEGORY':<22}{'HYBRID':<9}{'SEMANTIC':<10}EXPECTED DOC")
    print("-" * 95)

    for q in answerable:
        expected = q["expected_doc"]
        hyb_names = retrieved_docnames(q["question"], use_hybrid=True)
        sem_names = retrieved_docnames(q["question"], use_hybrid=False)

        hyb_hit = expected in hyb_names
        sem_hit = expected in sem_names
        hyb_hits += hyb_hit
        sem_hits += sem_hit
        cat_hyb[q["category"]][0] += hyb_hit
        cat_hyb[q["category"]][1] += 1

        rows.append({
            "id": q["id"], "category": q["category"], "expected": expected,
            "hybrid_hit": hyb_hit, "semantic_hit": sem_hit,
            "hybrid_retrieved": hyb_names, "semantic_retrieved": sem_names,
        })

        print(f"{q['id']:<5}{q['category']:<22}"
              f"{'HIT' if hyb_hit else 'MISS':<9}"
              f"{'HIT' if sem_hit else 'MISS':<10}{expected}")

    n = len(answerable)
    print("\n" + "=" * 60)
    print(f"RETRIEVAL HIT-RATE @ k={rag_core.MMR_K + rag_core.INJECT_MAX} (max)")
    print("=" * 60)
    print(f"  Hybrid (shipped)   : {hyb_hits}/{n} = {hyb_hits/n*100:.1f}%")
    print(f"  Semantic only      : {sem_hits}/{n} = {sem_hits/n*100:.1f}%")
    print(f"  Lift from keyword map: +{(hyb_hits - sem_hits)/n*100:.1f} pts")

    print("\nPer-category (hybrid):")
    for cat, (h, t) in sorted(cat_hyb.items()):
        print(f"  {cat:<22} {h}/{t} = {h/t*100:.0f}%")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results_retrieval.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "n": n,
                "hybrid_hits": hyb_hits, "hybrid_rate": hyb_hits / n,
                "semantic_hits": sem_hits, "semantic_rate": sem_hits / n,
            },
            "rows": rows,
        }, f, indent=2)
    print(f"\nDetailed results written to {out}")


if __name__ == "__main__":
    main()

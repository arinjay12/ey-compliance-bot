"""
calibrate_gate.py
-----------------
Calibrates REFUSE_DISTANCE for the relevance gate. For every gold question we
measure the L2 distance of the nearest chunk. In-scope questions should cluster
at LOW distance; out-of-scope questions at HIGH distance. A good threshold sits
in the gap between the two clusters.

    python evaluation/calibrate_gate.py
"""
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rag_core

GOLD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gold_qa.json")


def main():
    gold = json.load(open(GOLD, encoding="utf-8"))["questions"]
    rows = []
    for q in gold:
        d = rag_core.min_distance(q["question"])
        rows.append((d, q["answerable"], q["id"], q["category"]))

    rows.sort()
    print(f"\n{'DIST':>7}  {'SCOPE':<12}{'ID':<5}{'CATEGORY'}")
    print("-" * 50)
    for d, ans, qid, cat in rows:
        print(f"{d:7.3f}  {'in-scope' if ans else 'OUT-OF-SCOPE':<12}{qid:<5}{cat}")

    in_d  = [d for d, a, *_ in rows if a]
    out_d = [d for d, a, *_ in rows if not a]
    print("\nIn-scope    : min=%.3f  max=%.3f" % (min(in_d), max(in_d)))
    print("Out-of-scope: min=%.3f  max=%.3f" % (min(out_d), max(out_d)))
    gap_lo, gap_hi = max(in_d), min(out_d)
    if gap_hi > gap_lo:
        print(f"\nClean separation. Suggested REFUSE_DISTANCE = {(gap_lo+gap_hi)/2:.3f}"
              f"  (between {gap_lo:.3f} and {gap_hi:.3f})")
    else:
        print(f"\nOVERLAP: highest in-scope ({gap_lo:.3f}) >= lowest out-of-scope "
              f"({gap_hi:.3f}). A pure distance gate can't perfectly separate; "
              f"pick a value that trades off false-refusals vs. hallucinations.")


if __name__ == "__main__":
    main()

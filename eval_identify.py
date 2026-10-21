"""End-to-end identification benchmark: does the model name the right species?

Calls the LLM API for every identification case in eval_retrieval.py, so it is
slow.  python eval_identify.py [--thinking] [--limit N]
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag import llm
from rag.identifier import identify_mushroom
from eval_retrieval import CASES


def main():
    if "--thinking" in sys.argv:
        llm.THINKING = True
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    cases = [c for c in CASES if c[2] == "identify"][:limit]
    correct = mentioned = 0
    times = []
    for query, expected, _ in cases:
        start = time.time()
        try:
            answer = identify_mushroom(query)
        except Exception as e:  # keep going: one failed call should not end the run
            answer = f"ERROR {e}"
        times.append(time.time() - start)
        first = answer.strip().split("\n")[0]
        top = expected in first.lower()
        correct += top
        mentioned += expected in answer.lower()
        mark = "top" if top else ("ment" if expected in answer.lower() else "miss")
        print(f"{mark:>4} {times[-1]:5.0f}s  {expected:12} | {first[:80]}", flush=True)
    n = len(cases)
    print(f"\nmost likely correct: {correct}/{n}, mentioned: {mentioned}/{n}, "
          f"median time {sorted(times)[n // 2]:.0f}s, max {max(times):.0f}s")


if __name__ == "__main__":
    main()

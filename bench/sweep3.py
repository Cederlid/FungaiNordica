"""Candidate top-1 / top-6 on bench, CASES and HELDOUT for character weight : rank weight pairs."""
import sys, json, re, time
sys.path.insert(0, "/app")
import rag.identifier as idf
from eval_retrieval import CASES, HELDOUT
bench = [c for c in json.load(open("/tmp/b2/all_cases.json")) if c.get("expect")]
sets = {
    "bench": [(c.get("query") or " ".join(c["turns"]), c["expect"] if isinstance(c["expect"], list) else [c["expect"]]) for c in bench],
    "CASES": [(q, e) for q, e, k in CASES if k == "identify"],
    "HELDOUT": [(q, e) for q, e, k in HELDOUT if k == "identify"],
}
def hit(p, exp):
    return any(p == x or p.split()[1] == x for x in (exp if isinstance(exp, list) else [exp]))
import os, rag.characters as ch
if os.environ.get("NOSPEC"):
    idf.character_score = lambda u, s: sum(ch._WEIGHTS.get(k, 1.0) * v for k, _, _, v in ch._verdicts(u, s))
for arg in sys.argv[1:]:
    w, r = map(float, arg.split(":"))
    idf._CHAR_WEIGHT, idf._CHAR_RANK_WEIGHT = w, r
    out = []; t = time.time(); miss = []
    for name, cases in sets.items():
        t1 = t6 = 0
        for q, e in cases:
            d = idf.identification_context(q)
            paths = [re.match(r"Key path to (\S+ \S+)", x).group(1) for x in d if x.startswith("Key path to")]
            t1 += bool(paths) and hit(paths[0], e); t6 += any(hit(p, e) for p in paths)
            if name == "bench" and not any(hit(p, e) for p in paths): miss.append((e[0] if isinstance(e, list) else e).split()[-1])
        out.append(f"{name} {t1}/{t6}/{len(cases)}")
    print(f"w={w} rank={r}: " + "  ".join(out) + f"  {time.time()-t:.0f}s miss={miss}", flush=True)

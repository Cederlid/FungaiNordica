"""Per-case candidate list: python percase.py <app dir> <out.json> [w:rank]"""
import sys, json, re
app, out = sys.argv[1], sys.argv[2]
sys.path.insert(0, app); sys.path.insert(1, "/app")
import rag.identifier as idf
print(idf.__file__, file=sys.stderr)
if len(sys.argv) > 3:
    idf._CHAR_WEIGHT, idf._CHAR_RANK_WEIGHT = map(float, sys.argv[3].split(":"))
from eval_retrieval import CASES, HELDOUT
bench = [c for c in json.load(open("/tmp/b2/all_cases.json")) if c.get("expect")]
qs = [(c.get("query") or " ".join(c["turns"]), c["expect"] if isinstance(c["expect"], list) else [c["expect"]]) for c in bench]
qs += [(q, [e]) for q, e, k in CASES + HELDOUT if k == "identify"]
res = []
for q, e in qs:
    d = idf.identification_context(q)
    res.append((q, e, [re.match(r"Key path to (\S+ \S+)", x).group(1) for x in d if x.startswith("Key path to")]))
json.dump(res, open(out, "w"), ensure_ascii=False)

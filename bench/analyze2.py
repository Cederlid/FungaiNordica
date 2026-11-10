"""Score results2.jsonl against the book; --judge asks the model for contradictions with the species texts."""
import sys, json, re
sys.path.insert(0, "/app")
from rag.vectorstore import get_collection
from rag.llm import generate

D = sys.argv[1]
cases = json.load(open(f"{D}/all_cases.json"))
results = [json.loads(l) for l in open(f"{D}/results2.jsonl")]
c = get_collection()
got = c.get(where={"kind": "species"}, include=["documents", "metadatas"])
full = {}
for d, m in zip(got["documents"], got["metadatas"]):
    full.setdefault(" ".join(m["species"].split()[:2]), []).append(d)

def named(answer):
    out = []
    for m in re.finditer(r"\b([A-Z][a-z]+)\s+([a-z][a-z\-]+)\b", answer.replace("*", "")):
        b = f"{m.group(1)} {m.group(2)}"
        if b in full and b not in out: out.append(b)
    return out

def norm(s): return re.sub(r"\s+", "", s.replace("–", "-").replace("×", "x"))

rows = []
for r in results:
    case = cases[r["i"]]; a = r["answer"]; k = case["kind"]
    row = {"i": r["i"], "kind": k, "lang": case.get("lang", ""), "error": r["error"], "times": r["times"],
           "query": case.get("query") or case["turns"], "answer": a}
    if case.get("expect"):
        names = named(a); row["named"] = names[:4]; row["expected"] = case["expect"]
        row["top"] = bool(names) and names[0] == case["expect"]
        row["mentioned"] = case["expect"] in names
    if k == "fact-spores": row["ok"] = norm(case["answer"]) in norm(a)
    elif k == "fact-habitat": row["ok"] = sum(h in a for h in case["answer"]) / len(case["answer"])
    elif k == "fact-toxic": row["ok"] = bool(re.search(r"poison|toxic|giftig", a, re.I))
    if case.get("must"): row["must"] = all(re.search(p, a, re.I) for p in case["must"])
    if k == "absent":
        row["admits"] = bool(re.search(r"not (described|included|covered|in the|contain|provide)|does not|doesn't|no (description|information)|inte|saknar|finns ingen", a, re.I))
    rows.append(row)

JUDGE = """You check a mushroom assistant's answer against the Funga Nordica field guide.
Book text for the species the answer discusses:
{book}

Answer:
{answer}

List every statement in the answer about these species that CONTRADICTS the book text (wrong colour,
size, spore deposit, habitat, smell, edibility...). Ignore statements the book text simply does not
cover, and ignore what the answer reports as the user's own observations. Reply with JSON only:
{{"contradictions": ["...", ...]}}"""
if "--judge" in sys.argv:
    for row in rows:
        names = row.get("named") or named(row["answer"])
        book = "\n\n".join(f"{n}:\n" + "\n".join(full[n])[:2500] for n in names[:3] if n in full)
        if not book: continue
        try:
            out = generate(JUDGE.format(book=book, answer=row["answer"][:4000]))
            row["contradictions"] = json.loads(re.search(r"\{.*\}", out, re.S).group(0))["contradictions"]
        except Exception as e:
            row["contradictions"] = [f"judge failed: {e!r}"[:100]]
        print(row["i"], len(row["contradictions"]), flush=True)
json.dump(rows, open(f"{D}/scored2.json", "w"), ensure_ascii=False, indent=1)

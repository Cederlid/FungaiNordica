"""Score results2.jsonl against the book and print a tally; --judge asks the model for contradictions.

Fixes over analyze2: names in _underscores_ count, the 'Most likely:' line gives the top name,
a multi-turn case is scored on all its answers, and the server's note on names the context
lacks is not read as the answer naming them.
"""
import sys, json, re
from collections import defaultdict
sys.path.insert(0, "/app")
from rag.vectorstore import get_collection
from rag.llm import generate

D = sys.argv[1]
cases = json.load(open(f"{D}/all_cases.json"))
results = {}
for l in open(f"{D}/results2.jsonl"):  # a re-run of a case replaces the earlier one
    r = json.loads(l)
    if r["i"] not in results or not r["error"]:
        results[r["i"]] = r
c = get_collection()
got = c.get(where={"kind": "species"}, include=["documents", "metadatas"])
full = {}
for d, m in zip(got["documents"], got["metadatas"]):
    full.setdefault(" ".join(m["species"].split()[:2]), []).append(d)


def own(answer):
    """The answer without the server's note on names the context lacks."""
    return re.split(r"\n\n---\n\*(?:Note|Obs):\*", answer)[0]


def named(answer):
    out = []
    plain = re.sub(r"[*_]", " ", own(answer))
    likely = re.search(r"Most likely:\s*([A-Z][a-z]+ [a-z\-]+)", plain)
    if likely and likely.group(1) in full:
        out.append(likely.group(1))
    for m in re.finditer(r"\b([A-Z][a-z]+)\s+([a-z][a-z\-]+)\b", plain):
        b = f"{m.group(1)} {m.group(2)}"
        if b in full and b not in out:
            out.append(b)
    return out


# the service says it does not answer questions about edibility or poison, and gives no such statement
_DECLINE = re.compile(r"not (answer|cover|provide|address|give)|n.t (answer|cover)|doesn.t (answer|cover)|cannot (answer|say)|"
                      r"can.t (answer|say)|inte (svar|besvar|täck|ger|bedöm)|svarar inte|besvarar inte|avstår|kan inte (svara|säga|ge)", re.I)
_EDIBILITY = re.compile(r"poison|toxic|edible|inedible|deadly|lethal|giftig|ätlig|dödlig|oätlig|förgift|ätbar|matsvamp", re.I)


def declines(answer):
    """(the answer declines, the answer still says something on edibility or poison)."""
    sentences = re.split(r"(?<=[.!?])\s+|\n+", answer)
    declined = [s for s in sentences if _DECLINE.search(s)]
    rest = [s for s in sentences if s not in declined]
    return bool(declined), bool(_EDIBILITY.search(" ".join(rest)))


def norm(s): return re.sub(r"\s+", "", s.replace("–", "-").replace("×", "x"))


rows = []
for i in sorted(results):
    r = results[i]
    case = cases[i]; k = case["kind"]
    # a follow-up answer need not repeat the species: score all turns' answers
    a = "\n\n".join(own(x) for x in r["answers"]) if k == "multi" and r.get("answers") else own(r["answer"])
    row = {"i": i, "kind": k, "lang": case.get("lang", ""), "error": r["error"], "times": r["times"],
           "query": case.get("query") or case["turns"], "answer": a}
    if case.get("expect"):
        # the follow-up answer gives the final identification of a multi-turn case
        names = (named(own(r["answers"][-1])) if k == "multi" and r.get("answers") else []) or named(a)
        row["named"] = names[:4]; row["expected"] = case["expect"]
        row["top"] = bool(names) and names[0] == case["expect"]
        row["mentioned"] = case["expect"] in names
    if k == "fact-spores": row["ok"] = norm(case["answer"]) in norm(a)
    elif k == "fact-habitat": row["ok"] = sum(h in a for h in case["answer"]) / len(case["answer"])
    elif k == "fact-toxic": row["ok"] = bool(re.search(r"poison|toxic|giftig", a, re.I))
    if case.get("must"): row["must"] = all(re.search(p, a, re.I) for p in case["must"])
    if case.get("declines"):
        gave_no_answer, leak = declines(a)
        row["declines"], row["leak"] = gave_no_answer, leak
        row["ok"] = row["must"] = gave_no_answer and not leak
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

tally = defaultdict(lambda: [0, 0])
for row in rows:
    if row["error"]:
        tally["errors"][1] += 1
        continue
    key = row["kind"] + (" " + row["lang"] if row["lang"] else "")
    if "top" in row:
        ok = row["top"] and row.get("must", True)
    else:
        ok = row.get("ok", row.get("must", row.get("admits")))
    tally[key][0] += bool(ok); tally[key][1] += 1
for key, (good, n) in sorted(tally.items()):
    print(f"{key:20} {good}/{n}")
print("misses:", [(r["i"], r["expected"].split()[1], (r["named"] or ["-"])[0]) for r in rows
                  if "top" in r and not r["top"]])

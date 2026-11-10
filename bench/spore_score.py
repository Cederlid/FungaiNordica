"""Score spore answers against the book's spore sentences.

python3 spore_score.py spore_cases.json spores1.jsonl
"""
import json, re, sys

cases = json.load(open(sys.argv[1]))
results = {}
for l in open(sys.argv[2]):
    r = json.loads(l)
    if r["i"] not in results or not r["error"]:
        results[r["i"]] = r

# book term -> pattern also matching Swedish renderings
TERMS = {
    "ellipsoid": r"ellips", "broadly ellipsoid": r"ellips", "subglobose": r"subglob|nästan klot|nästan rund",
    "globose": r"\bglob|klot|rund", "amygdaloid": r"amygdal|mandel", "cylindrical": r"cylind",
    "subcylindrical": r"cylind", "subfusiform": r"fusiform|spol", "fusiform": r"fusiform|spol",
    "ovoid": r"ovoid|äggform", "obovoid": r"obovoid|omvänt ägg", "oblong": r"oblong|avlång",
    "nodulose": r"nodul|knöl", "angular": r"angul|kantig", "citriform": r"citri|citron",
    "allantoid": r"allant|korv", "phaseoliform": r"phaseol|bön", "lacrymoid": r"lacrym|tår",
    "smooth": r"smooth|slät", "verrucose": r"verruc|wart|vårt", "warts": r"wart|vårt|verruc",
    "reticulum": r"reticul|nät", "reticulate": r"reticul|nät", "echinulate": r"echin|spin|tagg",
    "spines": r"spin|echin|tagg", "ridges": r"ridge|ås|kam|lister", "punctate": r"punct|prick",
    "asperulate": r"asper|sträv", "warty": r"wart|vårt|verruc",
    "amyloid": r"(?<!non-)(?<!non )(?<!in)amyloid", "non-amyloid": r"non-?amyloid|inamyloid|inte amyloid",
    "inamyloid": r"non-?amyloid|inamyloid|inte amyloid", "dextrinoid": r"dextrin",
    "cyanophilic": r"cyanoph", "metachromatic": r"metachrom",
}
ORNAMENTED = {"verrucose", "warts", "warty", "reticulum", "reticulate", "echinulate", "spines", "ridges",
              "punctate", "asperulate", "nodulose"}


def norm(s):
    s = s.replace("–", "-").replace("—", "-").replace("×", "x").replace("µ", "μ")
    s = re.sub(r"(\d),(\d)", r"\1.\2", s)  # Swedish decimal comma
    return re.sub(r"\s+", "", s)


def sizes_in(text):
    """Spore-size ranges written in the answer: "8-10 x 6-7 μm"."""
    return re.findall(r"\(?\d[\d.,()\-–\s]*?\s*[x×]\s*[\d.,()\-–\s]+?\s*[μµ]m", text)


tot = {"size": [0, 0], "shape": [0, 0], "ornament": [0, 0], "reaction": [0, 0], "germ pore": [0, 0],
       "print colour": [0, 0], "in Swedish": [0, 0]}
problems = []
for i, case in enumerate(cases):
    r = results.get(i)
    if not r or r["error"]:
        problems.append((i, case["binomial"], "no answer"))
        continue
    a = re.split(r"\n\n---\n\*(?:Note|Obs):\*", r["answer"])[0]
    low = a.lower()
    got = []
    if case["size"]:
        ok = norm(case["size"]).replace("(", "").replace(")", "") in norm(a).replace("(", "").replace(")", "")
        tot["size"][0] += ok; tot["size"][1] += 1
        if not ok: got.append(f"size: book {case['size']}, answer {sizes_in(a)[:2]}")
    for kind, terms in (("shape", case["shapes"]), ("ornament", case["ornament"]), ("reaction", case["reactions"])):
        if terms:
            ok = any(re.search(TERMS.get(t, re.escape(t)), low) for t in terms)
            tot[kind][0] += ok; tot[kind][1] += 1
            if not ok: got.append(f"{kind}: book {terms}")
    if case["germ_pore"]:
        ok = bool(re.search(r"germ pore|groddpor|grodd-por", low))
        tot["germ pore"][0] += ok; tot["germ pore"][1] += 1
        if not ok: got.append("germ pore missing")
    ok = bool(re.search(r"spore (print|deposit)|sporavtryck|sporpulver", low))
    tot["print colour"][0] += ok; tot["print colour"][1] += 1
    if case["lang"] == "Swedish":
        ok = len(re.findall(r"\b(och|är|med|sporerna|till|som|eller)\b", low)) >= 3
        tot["in Swedish"][0] += ok; tot["in Swedish"][1] += 1
        if not ok: got.append("not in Swedish")
    # wrong statements: smooth for ornamented spores, ornament for smooth ones, amyloid flipped
    if set(case["ornament"]) & ORNAMENTED and re.search(r"\bsmooth\b(?! (?:area|patch|plage))|\bsläta?\b(?! fläck)", low):
        got.append("WRONG: says smooth, book ornamented")
    if "smooth" in case["ornament"] and re.search(r"verruc|warty|echinul|spiny|vårtig", low):
        got.append("WRONG: says ornamented, book smooth")
    if "amyloid" in case["reactions"] and re.search(r"non-?amyloid|inamyloid|inte amyloid", low):
        got.append("WRONG: says non-amyloid, book amyloid")
    # numbers in the answer's size ranges that the species' text does not contain
    nums = {n.rstrip(".") for s in sizes_in(a) for n in re.findall(r"\d+(?:[.,]\d+)?", s)}
    invented = sorted(n for n in nums - set(case["book_numbers"]) if n.replace(",", ".") not in case["book_numbers"])
    if invented:
        got.append(f"WRONG: numbers not in the book {invented}")
    if re.search(r"not (described|given|provided|available|included)|does not (give|provide|describe)|saknas|anges inte|finns inte", low):
        got.append("says the book lacks it")
    if got:
        problems.append((i, case["binomial"], "; ".join(got)))

print("Per character (answers giving the book's value / cases where the book gives it):")
for k, (g, n) in tot.items():
    print(f"  {k:10} {g}/{n}")
wrong = [p for p in problems if "WRONG" in p[2]]
print(f"answers with a wrong statement: {len(wrong)}/{len(cases)}")
for p in problems:
    print(" ", p[0], p[1], "|", p[2])

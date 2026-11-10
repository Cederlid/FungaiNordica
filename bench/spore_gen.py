"""Spore questions about named species, with the book's spore characters as the answer key.

python spore_gen.py <out.json>   (run in the container)
"""
import sys, re, json, random
sys.path.insert(0, "/app")
from rag.vectorstore import get_collection

SHAPES = ["subglobose", "globose", "broadly ellipsoid", "ellipsoid", "oblong", "cylindrical", "subcylindrical",
          "amygdaloid", "citriform", "fusiform", "subfusiform", "ovoid", "phaseoliform", "allantoid",
          "nodulose", "angular", "lacrymoid", "pip-shaped", "mitriform", "lentiform", "pyriform", "obovoid"]
ORNAMENT = ["smooth", "verrucose", "warty", "warts", "reticulum", "reticulate", "echinulate", "spiny", "spines",
            "punctate", "rough", "ornamented", "ridges", "asperulate", "nodulose"]
REACTIONS = ["non-amyloid", "inamyloid", "amyloid", "dextrinoid", "cyanophilic", "metachromatic"]
WELL_KNOWN = ["Amanita muscaria", "Boletus edulis", "Lactarius deliciosus", "Russula emetica", "Cortinarius rubellus",
              "Agaricus campestris", "Coprinus comatus", "Hypholoma fasciculare", "Laccaria amethystina",
              "Macrolepiota procera", "Paxillus involutus", "Suillus luteus"]
QUESTIONS_EN = ["What are the spores of {n} like?", "Describe the spores of {n}.",
                "Tell me about the spores of {n}.", "What spore characters does {n} have?"]
QUESTIONS_SV = ["Hur ser sporerna ut hos {n}?", "Beskriv sporerna hos {n}.", "Vad har {n} för sporer?"]

c = get_collection()
got = c.get(where={"kind": "species"}, include=["documents", "metadatas"])
texts = {}
for d, m in zip(got["documents"], got["metadatas"]):
    body = d.split("\n", 2)[-1]
    if body not in texts.setdefault(m["species"], []):
        texts[m["species"]].append(body)


def spore_segment(text):
    """Every sentence about the spores ("Sp 9-12 x 6-8 μm, ellipsoid (fig. 2A); pleurocystidia ..."),
    each up to its ";" or full stop (a decimal point or "fig." does not end it)."""
    out = []
    for m in re.finditer(r"\bSp\b(?! deposit)", text):
        rest = text[m.start():]
        end = re.search(r";|\.(?!\d)(?<!fig\.)(?<!c\.)(?<!av\.)(?:\s|$)", rest)
        out.append(rest[:end.start()] if end else rest[:300])
    return " | ".join(s for s in out if len(s) > 4)


def key(name):
    return " ".join(name.split()[:2])


rows, seen = [], set()
for name, parts in texts.items():
    text = " ".join(parts)
    seg = spore_segment(text)
    if not seg or ((" var. " in name or " f. " in name) and not any(name.startswith(w) for w in WELL_KNOWN)):
        continue
    size = re.search(r"\(?\d[\d.,()\-]*(?:-\s+\d[\d.,()\-]*)?\s*x\s*[\d.,()\-]+(?:-\s+\d[\d.,()\-]*)?\s*μm", seg)
    low = seg.lower()
    if key(name) in seen:
        continue  # varieties of one species: keep the first (the type variety)
    seen.add(key(name))
    rows.append({
        "species": name, "binomial": key(name), "segment": seg, "text": text,
        "size": size.group(0) if size else "",
        "shapes": [s for s in SHAPES if re.search(rf"\b{s}\b", low)],
        "ornament": [o for o in ORNAMENT if re.search(rf"\b{o}\b", low)],
        "reactions": [r for r in REACTIONS if re.search(rf"(?<![\w-]){r}\b", low)],
        "germ_pore": "germ pore" in low,
    })

random.seed(11)
by_name = {r["binomial"]: r for r in rows}
chosen = [by_name[n] for n in WELL_KNOWN if n in by_name]
genera_seen = {r["binomial"].split()[0] for r in chosen}
pool = [r for r in rows if r["size"] and (r["shapes"] or r["ornament"] or r["reactions"])]
random.shuffle(pool)
for r in pool:
    if len(chosen) == 30:
        break
    if r["binomial"].split()[0] not in genera_seen:
        chosen.append(r); genera_seen.add(r["binomial"].split()[0])
cases = []
for i, r in enumerate(chosen):
    swedish = i % 3 == 2
    template = (QUESTIONS_SV if swedish else QUESTIONS_EN)[i % (3 if swedish else 4)]
    cases.append({"kind": "spores", "lang": "Swedish" if swedish else "English",
                  "query": template.format(n=r["binomial"]), **{k: r[k] for k in r if k != "text"},
                  "book_numbers": sorted(set(re.findall(r"\d+(?:[.,]\d+)?", r["text"])))})
json.dump(cases, open(sys.argv[1], "w"), ensure_ascii=False, indent=1)
print(len(cases), "cases")
for c_ in cases:
    print(c_["binomial"], "|", c_["size"], c_["shapes"], c_["ornament"], c_["reactions"], "pore" if c_["germ_pore"] else "")

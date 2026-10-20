"""Retrieval benchmark: does the expected species show up in the top-k chunks?

Runs locally against chroma_db (no LLM needed):  python eval_retrieval.py [-v]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag.vectorstore import query_documents

# (query as a user would write it, expected epithet, kind)
CASES = [
    # identification from a field description (everyday words, no species name)
    ("Red cap with white warts, white gills, ring and a bulbous stem base, under birch", "muscaria", "identify"),
    ("Greenish olive cap, white gills, a ring and a sac-like volva at the base, deciduous wood", "phalloides", "identify"),
    ("Pure white mushroom with ring and volva, conical cap, in coniferous forest, deadly", "virosa", "identify"),
    ("Big bolete with brown cap, white net on the stem, white pores turning yellow-green", "edulis", "identify"),
    ("Bolete with very bitter taste, pinkish pores, dark brown net on the stem", "felleus", "identify"),
    ("Bright red brittle cap, white gills, very acrid taste, in wet conifer forest", "emetica", "identify"),
    ("Orange milkcap with orange milk turning wine red, under spruce", "deterrimus", "identify"),
    ("Pink shaggy cap with woolly margin, white acrid milk, under birch", "torminosus", "identify"),
    ("Orange funnel shaped mushroom with forked orange gills on conifer debris", "aurantiaca", "identify"),
    ("Brown cap with inrolled margin, gills bruising brown, decurrent, poisonous", "involutus", "identify"),
    ("Tall white shaggy cap that dissolves into black ink, along roads", "comatus", "identify"),
    ("Very large parasol with snakeskin pattern on stem and a movable ring", "procera", "identify"),
    ("Slimy brown cap, yellow pores, white to purple ring, under pine", "luteus", "identify"),
    ("Sulphur yellow caps in clusters on stumps, greenish gills, bitter", "fasciculare", "identify"),
    ("Clustered on deciduous stumps, hygrophanous brown cap, scaly stem below the ring", "mutabilis", "identify"),
    ("Small brown mushroom on wood with a ring, deadly amatoxins", "marginata", "identify"),
    ("Small purple mushroom with thick purple gills in deciduous forest", "amethystina", "identify"),
    ("Yellow knight, yellow gills, under pine on sandy soil", "equestre", "identify"),
    ("Oyster shaped grey cap on deciduous wood, decurrent gills, no stem", "ostreatus", "identify"),
    ("Orange sticky cap, dark velvety stem, on wood in winter", "velutipes", "identify"),
    ("Grey cap, smell strong and sweetish, in large groups in autumn leaf litter", "nebularis", "identify"),
    ("Orange cap bolete with black scales on the stem, under birch, flesh turning black", "versipelle", "identify"),
    # Swedish, as some users will write
    ("Röd hatt med vita vårtor, ring och knölformad fotbas, under björk", "muscaria", "identify"),
    ("Orange riska med orange mjölk som blir vinröd, under gran", "deterrimus", "identify"),
    # factual questions naming the species
    ("What are the spore measurements of Boletus edulis?", "edulis", "fact"),
    ("Spore size of Amanita muscaria", "muscaria", "fact"),
    ("Habitat of Lactarius deterrimus", "deterrimus", "fact"),
    ("How does Russula emetica differ from similar red Russula species?", "emetica", "fact"),
    ("Which tree does Suillus luteus grow with?", "luteus", "fact"),
    ("Is Paxillus involutus poisonous?", "involutus", "fact"),
]


def run(k_values=(4, 8), verbose=False):
    hits = {k: 0 for k in k_values}
    by_kind = {}
    kmax = max(k_values)
    for query, expected, kind in CASES:
        docs = query_documents(query, n_results=kmax)
        rank = next((i + 1 for i, d in enumerate(docs) if expected in d.lower()), None)
        for k in k_values:
            if rank is not None and rank <= k:
                hits[k] += 1
                by_kind[(kind, k)] = by_kind.get((kind, k), 0) + 1
        mark = f"#{rank}" if rank else "miss"
        print(f"{mark:>5}  {expected:12} {query[:70]}")
        if verbose and not rank:
            for d in docs[:4]:
                print("         -", d[:90].replace("\n", " "))
    n = len(CASES)
    print()
    for k in k_values:
        parts = []
        for kind in ("identify", "fact"):
            total = sum(1 for c in CASES if c[2] == kind)
            parts.append(f"{kind} {by_kind.get((kind, k), 0)}/{total}")
        print(f"hit@{k}: {hits[k]}/{n}  ({', '.join(parts)})")


if __name__ == "__main__":
    run(verbose="-v" in sys.argv)

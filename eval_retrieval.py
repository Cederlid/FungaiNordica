"""Retrieval benchmark: does the expected species show up in the top-k chunks?

Runs locally against chroma_db (no LLM needed):  python eval_retrieval.py [-v] [--heldout]
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


# Held-out set, written before the vocabulary expansion was tuned on CASES:
# use it to check that changes generalize instead of fitting CASES.
HELDOUT = [
    ("Brown cap with pure white warts, white gills, ring not striate, rim around the bulb, deciduous forest", "pantherina", "identify"),
    ("Brownish cap with greyish pink warts, flesh slowly turning pinkish red where damaged, ring striate", "rubescens", "identify"),
    ("Orange brown cap with striate margin, no ring, white sac-like volva, in birch woods", "fulva", "identify"),
    ("Small orange brown mushroom with pinkish thick distant gills, very common in mixed forest", "laccata", "identify"),
    ("Clustered honey coloured caps with dark scales on stumps, white ring, black bootlace rhizomorphs", "ostoyae", "identify"),
    ("Bolete with white stem covered in dark scabers, orange red cap, under aspen", "aurantiacum", "identify"),
    ("Slimy bolete under pine, no ring, pores exuding milky droplets", "granulatus", "identify"),
    ("Chestnut brown velvety cap bolete, pale yellow pores bruising blue, under conifers", "badius", "identify"),
    ("White milkcap with very crowded gills and extremely peppery white milk", "piperatus", "identify"),
    ("Dark reddish brown milkcap with a pointed umbo, very acrid milk, in pine forest", "rufus", "identify"),
    ("Russula with wine red cap, smells of crab or fish, flesh browning, under pine", "xerampelina", "identify"),
    ("Russula with violet to green variable cap, gills flexible not brittle, mild taste", "cyanoxantha", "identify"),
    ("Small grey brown mycena with a stem that bleeds dark red juice when broken, on wood", "haematopus", "identify"),
    ("Lilac mycena with a radish smell, in leaf litter", "pura", "identify"),
    ("Conical red to orange waxcap that blackens with age, in grassland", "conica", "identify"),
    ("Slimy green to yellow parrot waxcap in grassland", "psittacina", "identify"),
    ("Snow white small waxcap with decurrent gills in old meadows", "virginea", "identify"),
    ("Brick red caps in dense clusters on deciduous stumps, gills greyish, taste mild to bitter", "lateritium", "identify"),
    ("Tough little mushroom in fairy rings in lawns, smells of almond, buff cap", "oreades", "identify"),
    ("Liberty cap: tiny sharply conical brown cap with a pointed tip, long thin stem, in grassland", "semilanceata", "identify"),
    # Swedish and common names
    ("Panterflugsvamp: brun hatt med vita vårtor och en kant runt knölen", "pantherina", "identify"),
    ("Tegelröd slöjskivling i tuvor på stubbar", "lateritium", "identify"),
    ("Blodhätta, liten grå hätta som blöder röd saft", "haematopus", "identify"),
    ("Brown birch bolete under birch, grey brown cap, white flesh not turning black", "scabrum", "identify"),
]


def run(k_values=(4, 8), verbose=False, cases=None):
    hits = {k: 0 for k in k_values}
    by_kind = {}
    kmax = max(k_values)
    cases = cases or CASES
    for query, expected, kind in cases:
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
    n = len(cases)
    print()
    for k in k_values:
        parts = []
        for kind in ("identify", "fact"):
            total = sum(1 for c in cases if c[2] == kind)
            parts.append(f"{kind} {by_kind.get((kind, k), 0)}/{total}")
        print(f"hit@{k}: {hits[k]}/{n}  ({', '.join(parts)})")


if __name__ == "__main__":
    run(verbose="-v" in sys.argv, cases=HELDOUT if "--heldout" in sys.argv else CASES)

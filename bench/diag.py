"""Where the missed species sit: candidate place and character-score rank across the whole book."""
import sys, json, time
sys.path.insert(0, "/app")
import rag.identifier as idf
from rag.characters import user_characters, character_score, compare
c = idf.get_collection(); sp = idf.species_index(c)
t = time.time()
allc = {n: idf._species_characters(c, n, *sp[n]) for n in sp}
print(f"characters for {len(sp)} species in {time.time()-t:.0f}s")
cases = json.load(open("/tmp/b2/all_cases.json"))
for e in sys.argv[1:]:
    case = next(x for x in cases if x.get("expect") and e in str(x["expect"]))
    q = case.get("query") or " ".join(case["turns"])
    u = user_characters(q)
    ranked = idf.query_chunks(q, n_results=100)
    names = list(dict.fromkeys(idf._candidates(ranked[:20], sp, ranked) + idf._candidates(ranked, sp)))
    full = next(n for n in sp if n.split()[1] == e)
    s = character_score(u, allc[full])
    better = sum(character_score(u, allc[n]) > s for n in sp)
    tie = sum(character_score(u, allc[n]) == s for n in sp)
    print(f"\n{e}: place {names.index(full) if full in names else None}/{len(names)}  char {s} (higher {better}, same {tie})")
    print("  Q:", q[:200]); print("  U:", u); print("  vs:", compare(u, allc[full]))

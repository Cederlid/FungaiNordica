import sys, os, re
sys.path.insert(0, "/app")
from rag.pdf_loader import load_and_chunk
species = {}
for f in sorted(os.listdir("/app/data")):
    if f.endswith(".pdf"):
        for c in load_and_chunk("/app/data/" + f):
            if c["meta"]["kind"] == "species":
                species.setdefault(c["meta"]["species"], c["text"])
print([t[:700] for n, t in species.items() if n.startswith("Lactarius necator")][0], "\n")
bad = [(n, t) for n, t in species.items() if not re.search(r"\b(Cap|Frb|Fruitbod)", t)]
import random; random.seed(1)
for n, t in random.sample(bad, 8):
    print("==", t[:330].replace("\n", " | "), "\n")

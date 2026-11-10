"""Compare chunks from the old and new pdf_loader on all PDFs."""
import sys, os, re, collections
sys.path.insert(0, sys.argv[1])
from rag.pdf_loader import load_and_chunk
out = collections.Counter(); species = {}
for f in sorted(os.listdir("/app/data")):
    if f.endswith(".pdf"):
        for c in load_and_chunk("/app/data/" + f):
            out[c["meta"]["kind"]] += 1
            if c["meta"]["kind"] == "species":
                species.setdefault(c["meta"]["species"], c["text"])
nocap = sum(1 for t in species.values() if not re.search(r"\b(Cap|Frb|Fruitbod)", t))
print(sys.argv[1], dict(out), "species", len(species), "without Cap/Frb", nocap)

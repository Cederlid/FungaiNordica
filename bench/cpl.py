import sys, os, json
sys.path.insert(0, sys.argv[1])
from rag.pdf_loader import load_and_chunk
rows = []
for f in sorted(os.listdir("/app/data")):
    if f.endswith(".pdf"):
        rows += [(f, c["meta"].get("key"), c["meta"].get("couplet"), c["meta"].get("targets")) for c in load_and_chunk("/app/data/" + f) if c["meta"]["kind"] == "couplet"]
json.dump(rows, open(sys.argv[2], "w"))

"""Reconstruct the route through the Funga Nordica keys to a species.

Couplet chunks carry metadata (genus, key, couplet, targets). Walking the
targets backwards from a species gives the chain of choices that leads to it:
Key to genera -> genus key -> sub-key -> species, which lets the LLM check each
step against the user's observations instead of guessing from loose chunks.
"""
import re

_MAX_STEPS = 40
_LEAD_CHARS = 160  # species-description leads are long; the start holds the key character

_cache = None  # (doc count, couplets)
_species_cache = None  # (doc count, {species name: (first description chunk, metadata)})


def species_index(collection) -> dict:
    """Species name -> (description text, metadata), for species named as key targets."""
    global _species_cache
    count = collection.count()
    if _species_cache is None or _species_cache[0] != count:
        got = collection.get(where={"kind": "species"}, include=["documents", "metadatas"])
        index = {}
        for text, m in zip(got["documents"], got["metadatas"]):
            index.setdefault(m["species"], (text, m))
        _species_cache = (count, index)
    return _species_cache[1]


def _couplets(collection) -> list[dict]:
    global _cache
    count = collection.count()
    if _cache is None or _cache[0] != count:
        got = collection.get(where={"kind": "couplet"}, include=["documents", "metadatas"])
        rows = []
        for text, m in zip(got["documents"], got["metadatas"]):
            leads = []
            for line in text.split("\n")[1:]:
                lead, _, target = line.partition(" → ")
                leads.append((re.sub(r"^(?:\d+[a-z]?\.|-)\s*", "", lead), target))
            rows.append({**m, "leads": leads})
        _cache = (count, rows)
    return _cache[1]


def _key_letter(key: str) -> str:
    m = re.match(r"Key ([A-Z0-9]+)\b", key)
    return m.group(1) if m else ""


def _parent(rows, step):
    """The couplet (and its lead index) whose target leads to `step`."""
    genus, key, couplet = step["genus"], step["key"], step["couplet"]
    for r in rows:
        if r["genus"] != genus:
            continue
        for i, (_, target) in enumerate(r["leads"]):
            if r["key"] == key and target == couplet:
                return r, i
    # first couplet of a sub-key: reached from "Key A ..." in the same genus
    letter = _key_letter(key)
    if couplet == "1" and letter:
        for r in rows:
            if r["genus"] != genus or r["key"] == key:
                continue
            for i, (_, target) in enumerate(r["leads"]):
                if _key_letter(target) == letter:
                    return r, i
    # first couplet of a genus key: reached from the Key to genera
    if couplet == "1" and genus and (not key or key == "Main key"):
        for r in rows:
            if r["genus"]:
                continue
            for i, (_, target) in enumerate(r["leads"]):
                if target.split(" ")[0] == genus:
                    return r, i
    return None


def key_path(collection, meta: dict) -> list[str]:
    """Steps from the top of the keys down to the species in `meta`, as text lines."""
    if meta.get("kind") != "species" or not meta.get("couplet"):
        return []
    rows = _couplets(collection)
    step = {"genus": meta["genus"], "key": meta["key"], "couplet": meta["couplet"]}
    # the couplet that ends in the species itself
    start = next((r for r in rows if r["genus"] == step["genus"] and r["key"] == step["key"]
                  and r["couplet"] == step["couplet"]), None)
    if not start:
        return []
    name = meta.get("species", "")
    chosen = next((i for i, (_, t) in enumerate(start["leads"]) if t == name), 0)
    path = [(start, chosen)]
    seen = {(start["genus"], start["key"], start["couplet"])}
    current = start
    while len(path) < _MAX_STEPS:
        found = _parent(rows, current)
        if not found:
            break
        parent, i = found
        ident = (parent["genus"], parent["key"], parent["couplet"])
        if ident in seen:
            break
        seen.add(ident)
        path.append((parent, i))
        current = parent

    def short(text):
        return text if len(text) <= _LEAD_CHARS else text[:_LEAD_CHARS].rsplit(" ", 1)[0] + " …"

    lines = []
    for r, i in reversed(path):
        where = r["genus"] or "Key to genera"
        if r["key"] and r["key"] != "Main key":
            where += f", {r['key'].split(':')[0]}"
        lead, target = r["leads"][i]
        others = [f"{short(l)} → {t}" for j, (l, t) in enumerate(r["leads"]) if j != i]
        line = f"{where}, couplet {r['couplet']}: {short(lead)} → {target}"
        if others:
            line += f"  [not: {others[0]}]"
        lines.append(line)
    return lines

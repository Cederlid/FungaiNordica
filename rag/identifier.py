import re

from rag.keypath import key_path, species_index
from rag.vectorstore import get_collection, query_chunks
from rag.llm import identify as llm_identify, identify_stream as llm_identify_stream

_N_RESULTS = 12
_MAX_CANDIDATES = 6
_POOL = 100  # ranked chunks searched for the best species of a genus a couplet leads to
# genus introductions hold characters the species descriptions leave out (gill
# colour, spore deposit, habitat); the first chunks cover them
_INTRO_CHUNKS = 2

# spore deposit colour as users report it -> the Key to genera sections it allows
# (C white to very pale pinkish, D pink, E greenish, F pale to dull brown,
# G very dark brown to black); darker colours are matched first
_DEPOSIT = [
    (r"dark brown|chocolate|purpl\w*|black\w*|mörkbrun\w*|svart\w*", "G"),
    (r"brown\w*|rust\w*|ochre|cinnamon|tobacco|brun\w*", "FG"),
    (r"pink\w*|salmon|rosa", "CD"),
    (r"green\w*|grön\w*", "E"),
    (r"white|whitish|cream|pale|yellow\w*|vit\w*|gul\w*", "C"),
]
_SPORE = r"(?:spore(?:s)?(?:\s+(?:print|deposit))?|sp\s+deposit|sporavtryck\w*|sporpulver\w*|sporer(?:na)?)"
# words allowed between them: "spore print is very dark brown", "sporavtrycket var vitt"
_FILL = r"(?:\W+(?:is|are|was|were|it|and|turned|very|quite|rather|light|a|är|var|blev|den|det|och|mycket|ganska|ljust?|colou?r(?:ed)?))*\W+"


def _deposit_keys(features: str) -> str:
    """Key to genera sections allowed by a reported spore deposit colour ("" if none given)."""
    text = features.lower()
    for colour, keys in _DEPOSIT:
        if re.search(rf"\b(?:{colour}){_FILL}{_SPORE}\b", text) or \
                re.search(rf"\b{_SPORE}{_FILL}(?:{colour})\b", text):
            return keys
    return ""


def _genera_key(steps: list[str]) -> str:
    """The spore-deposit section (C-G) of the Key to genera a key path passes through."""
    for s in steps:
        m = re.match(r"Key to genera, Key ([C-G])\b", s)
        if m:
            return m.group(1)
    return ""


def _genus_intro(collection, genus: str) -> list[str]:
    got = collection.get(where={"$and": [{"kind": "intro"}, {"genus": genus}]}, include=["documents"])
    return got["documents"][:_INTRO_CHUNKS]


def _target_species(target: str, species: dict, ranked: list[tuple[str, dict]]) -> str:
    """The species a couplet target stands for: the species itself, or for a genus
    ("Agaricus p. 519" in the Key to genera) its best-ranked species."""
    name = re.sub(r"\s+p\.\s*\d+$", "", target)
    if name in species:
        return name
    if " " in name:  # a sub-key ("Key D") or an unindexed species
        return ""
    for _, meta in ranked:
        if meta.get("kind") == "species" and meta.get("genus") == name and meta.get("species") in species:
            return meta["species"]
    return ""


def _candidates(chunks: list[tuple[str, dict]], species: dict, ranked=()) -> list[str]:
    """Species names in retrieval order: described species, and species a retrieved couplet leads to."""
    ranked = ranked or chunks
    names = []
    for _, meta in chunks:
        if meta.get("kind") == "species":
            found = [meta.get("species", "")]
        elif meta.get("kind") == "couplet":
            found = [_target_species(t, species, ranked) for t in meta.get("targets", "").split("|")]
        else:
            found = []
        for name in found:
            if name and name not in names:
                names.append(name)
    return names


def identification_context(features: str, n_results: int = _N_RESULTS) -> list[str]:
    """Retrieved chunks plus, for each top species candidate, its description and key path.

    Candidates whose key path contradicts a reported spore deposit colour are left
    out and named as ruled out, and the next candidates take their place.
    """
    ranked = query_chunks(features, n_results=max(n_results, _POOL))
    docs = [text for text, _ in ranked[:n_results]]
    collection = get_collection()
    species = species_index(collection)
    allowed = _deposit_keys(features)
    routed = set()  # genera whose route through the Key to genera is already shown
    kept, ruled_out = 0, []
    for name in _candidates(ranked, species):
        if kept == _MAX_CANDIDATES:
            break
        text, meta = species[name]
        steps = key_path(collection, meta)
        section = _genera_key(steps)
        if allowed and section and section not in allowed:
            ruled_out.append(f"{name} (Key to genera, Key {section})")
            continue
        kept += 1
        if meta["genus"] not in routed:
            docs.extend(t for t in _genus_intro(collection, meta["genus"]) if t not in docs)
        if text not in docs:
            docs.append(text)
        if meta["genus"] in routed:
            steps = [s for s in steps if not s.startswith("Key to genera")]
        routed.add(meta["genus"])
        if steps:
            docs.append(f"Key path to {name}:\n" + "\n".join(f"- {s}" for s in steps))
    if ruled_out:
        docs.append("Ruled out by the reported spore deposit colour (allowed: Key to genera "
                    + ", ".join(f"Key {k}" for k in allowed) + "):\n"
                    + "\n".join(f"- {r}" for r in ruled_out))
    return docs


def identify_mushroom(features: str, n_results: int = _N_RESULTS, model: str = "", history=(),
                      retrieval_query: str | None = None) -> str:
    """`history` holds earlier chat turns; `retrieval_query` defaults to `features`."""
    docs = identification_context(retrieval_query or features, n_results)
    return llm_identify(features, docs, model=model, history=history)


def identify_mushroom_stream(features: str, n_results: int = _N_RESULTS, model: str = "", history=(),
                             retrieval_query: str | None = None):
    docs = identification_context(retrieval_query or features, n_results)
    yield from llm_identify_stream(features, docs, model=model, history=history)

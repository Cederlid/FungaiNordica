import re

from rag.keypath import _couplets, key_path, species_index
from rag.vectorstore import get_collection, query_chunks
from rag.llm import identify as llm_identify, identify_stream as llm_identify_stream

_N_RESULTS = 12
_MAX_CANDIDATES = 6
_RRF_K = 60  # as in the hybrid search
_SCORED = 20  # chunks whose support decides the candidate order
_POOL = 100  # ranked chunks searched for the best species of a genus a couplet leads to
# genus introductions hold characters the species descriptions leave out (gill
# colour, spore deposit, habitat); the first chunks cover them
_INTRO_CHUNKS = 2

# spore deposit colour words -> the Key to genera sections they allow (C white to
# very pale pinkish, D pink, E greenish, F pale to dull brown, G very dark brown to black)
_COLOUR = [
    (r"black\w*|purpl\w*|chocolate|mörkbrun\w*|svart\w*", "G"),
    (r"brown\w*|rust\w*|ochre|cinnamon|tobacco|brun\w*", "FG"),
    (r"buff", "CF"),
    (r"pink\w*|salmon|flesh|rosa", "CD"),
    (r"green\w*|grön\w*", "E"),
    (r"white|whitish|cream\w*|yellow\w*|vit\w*|gul\w*", "C"),
]
_DARK = {"dark", "mörk", "mörkt"}
# words that may sit between "spore" and the colour, or inside a colour phrase:
# "spore print is very dark brown", "whitish to cream or greenish", "sporavtrycket var vitt"
_LINK = {"print", "deposit", "colour", "color", "coloured", "colored", "is", "are", "was", "were", "it",
         "and", "or", "to", "turned", "very", "quite", "rather", "light", "pale", "vivid", "dull", "dark",
         "grey", "greyish", "red", "reddish", "a", "är", "var", "blev", "den", "det", "och", "eller",
         "till", "mycket", "ganska", "ljus", "ljust", "mörk", "mörkt"}
_SPORE = re.compile(r"spores?|sporavtryck\w*|sporpulver\w*|sporer\w*")


def _colour_keys(word: str, prev: str) -> str:
    for pattern, keys in _COLOUR:
        if re.fullmatch(pattern, word):
            return "G" if keys == "FG" and prev in _DARK else keys
    return ""


def _phrase_keys(words: list[str]) -> str:
    """Sections allowed by the colour words of a phrase, up to the first other word."""
    keys, dark = set(), ""  # "dark" carries on: "dark purplish brown", "chocolate brown"
    for w in words:
        k = _colour_keys(w, dark)
        if not k and w not in _LINK:
            break
        keys |= set(k)
        if w in _DARK or k == "G":
            dark = "dark"
    return "".join(sorted(keys))


def _deposit_keys(features: str) -> str:
    """Key to genera sections allowed by a reported spore deposit colour ("" if none given).

    Every colour in the phrase counts ("yellow brown", "brownish pink"), so a
    mixed colour never rules out a section it names.
    """
    words = re.findall(r"[a-zåäö]+|[,.;!?]", features.lower())  # punctuation ends a phrase
    keys = set()
    for i, w in enumerate(words):
        if not (_SPORE.fullmatch(w) or (w == "sp" and words[i + 1:i + 2] == ["deposit"])):
            continue
        keys |= set(_phrase_keys(words[i + 1:]))
        # "dark brown spores": a colour phrase ending right before the spore word
        if i == 0 or not _colour_keys(words[i - 1], ""):
            continue
        start = i
        while start > max(0, i - 4) and (words[start - 1] in _LINK or _colour_keys(words[start - 1], "")):
            start -= 1
        keys |= set(_phrase_keys(words[start:i]))
    return "".join(sorted(keys))


def _genera_key(steps: list[str]) -> str:
    """The spore-deposit section (C-G) of the Key to genera a key path passes through."""
    for s in steps:
        m = re.match(r"Key to genera, Key ([C-G])\b", s)
        if m:
            return m.group(1)
    return ""


_sections_cache = None  # (doc count, {genus: sections})


def _genus_sections(collection) -> dict:
    """Genus -> every spore-deposit section (C-G) of the Key to genera that leads to it.

    Some genera are keyed under several colours (Lepista under C and D, Psathyrella
    under F and G), while a key path follows only one route.
    """
    global _sections_cache
    count = collection.count()
    if _sections_cache is None or _sections_cache[0] != count:
        sections = {}
        for r in _couplets(collection):
            m = re.match(r"Key ([C-G])\b", r["key"] or "")
            if r["genus"] or not m:
                continue
            for _, target in r["leads"]:
                genus = target.split(" ")[0]
                if genus[:1].isupper() and genus != "Key":
                    sections[genus] = "".join(sorted(set(sections.get(genus, "")) | {m.group(1)}))
        _sections_cache = (count, sections)
    return _sections_cache[1]


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
    """Species named by the retrieved chunks, best supported first.

    A species gains a reciprocal-rank score from every chunk pointing at it, its
    own description and the couplets that lead to it, so a species backed by
    both outranks one that only a single loose chunk mentions.
    """
    ranked = ranked or chunks
    scores = {}
    for rank, (_, meta) in enumerate(chunks):
        if meta.get("kind") == "species":
            found = [meta.get("species", "")]
        elif meta.get("kind") == "couplet":
            found = [_target_species(t, species, ranked) for t in meta.get("targets", "").split("|")]
        else:
            found = []
        for name in dict.fromkeys(found):
            if name:
                scores[name] = scores.get(name, 0.0) + 1 / (_RRF_K + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)  # stable: ties keep retrieval order


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
    # scored on the best chunks; the rest of the pool fills places the spore filter frees
    names = _candidates(ranked[:_SCORED], species, ranked) + _candidates(ranked, species)
    for name in dict.fromkeys(names):
        if kept == _MAX_CANDIDATES:
            break
        text, meta = species[name]
        steps = key_path(collection, meta)
        sections = _genus_sections(collection).get(meta["genus"]) or _genera_key(steps)
        if allowed and sections and not set(sections) & set(allowed):
            ruled_out.append(f"{name} (Key to genera, Key {'/'.join(sections)})")
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

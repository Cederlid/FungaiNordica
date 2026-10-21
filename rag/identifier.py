from rag.keypath import key_path, species_index
from rag.vectorstore import get_collection, query_chunks
from rag.llm import identify as llm_identify, identify_stream as llm_identify_stream

_N_RESULTS = 12
_MAX_CANDIDATES = 6


def _candidates(chunks: list[tuple[str, dict]], species: dict) -> list[str]:
    """Species names in retrieval order: described species, and species a retrieved couplet leads to."""
    names = []
    for _, meta in chunks:
        if meta.get("kind") == "species":
            found = [meta.get("species", "")]
        elif meta.get("kind") == "couplet":
            found = [t for t in meta.get("targets", "").split("|") if t in species]
        else:
            found = []
        for name in found:
            if name and name not in names:
                names.append(name)
    return names[:_MAX_CANDIDATES]


def identification_context(features: str, n_results: int = _N_RESULTS) -> list[str]:
    """Retrieved chunks plus, for each top species candidate, its description and key path."""
    chunks = query_chunks(features, n_results=n_results)
    docs = [text for text, _ in chunks]
    collection = get_collection()
    species = species_index(collection)
    routed = set()  # genera whose route through the Key to genera is already shown
    for name in _candidates(chunks, species):
        text, meta = species[name]
        if text not in docs:
            docs.append(text)
        steps = key_path(collection, meta)
        if meta["genus"] in routed:
            steps = [s for s in steps if not s.startswith("Key to genera")]
        routed.add(meta["genus"])
        if steps:
            docs.append(f"Key path to {name}:\n" + "\n".join(f"- {s}" for s in steps))
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

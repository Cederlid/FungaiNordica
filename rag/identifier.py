from rag.keypath import key_path
from rag.vectorstore import get_collection, query_chunks
from rag.llm import identify as llm_identify

_MAX_CANDIDATES = 4


def identification_context(features: str, n_results: int = 8) -> list[str]:
    """Retrieved chunks plus the key path to each of the top species candidates."""
    chunks = query_chunks(features, n_results=n_results)
    docs = [text for text, _ in chunks]
    collection = get_collection()
    seen = set()
    for _, meta in chunks:
        name = meta.get("species")
        if meta.get("kind") != "species" or name in seen:
            continue
        seen.add(name)
        steps = key_path(collection, meta)
        if steps:
            docs.append(f"Key path to {name}:\n" + "\n".join(f"- {s}" for s in steps))
        if len(seen) >= _MAX_CANDIDATES:
            break
    return docs


def identify_mushroom(features: str, n_results: int = 8, model: str = "", history=(),
                      retrieval_query: str | None = None) -> str:
    """`history` holds earlier chat turns; `retrieval_query` defaults to `features`."""
    docs = identification_context(retrieval_query or features, n_results)
    return llm_identify(features, docs, model=model, history=history)

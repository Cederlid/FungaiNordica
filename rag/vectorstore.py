import hashlib
import re
import unicodedata

import chromadb
from chromadb.config import Settings

from rag.embeddings import embed_texts, embed_query
from rag.lexical import BM25, expand_query

_client = None
_collection = None

_RRF_K = 60  # reciprocal-rank-fusion constant
_POOL = 100  # candidates taken from each ranker
# BM25 dominates: the embedding model barely separates the book's terse species
# descriptions, but still helps with paraphrases (tuned on eval_retrieval.py)
_W_SEM, _W_BM25, _W_MEASURE = 1.0, 3.0, 3.0

_bm25 = None  # (doc count, ids, docs, BM25) cache, rebuilt when the collection changes


def get_collection(name: str = "pdf_rag"):
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(anonymized_telemetry=False),
        )
        _collection = _client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def _measurements(query: str) -> list[str]:
    """Measurement strings to match verbatim: "8-10 x 6-7 μm", "40-100 mm"."""
    terms = []
    for m in re.finditer(r"\d+(?:[,.]\d+)?[\s\-×xX]+\d+(?:\s*[\-\u2013]\s*\d+)?\s*(?:μm|mm)", query):
        terms.append(m.group(0))
    for m in re.finditer(r"\b\d+[\-\u2013]\d*\s*(?:μm|mm)\b", query):
        terms.append(m.group(0))
    return list(dict.fromkeys(terms))


def _lexical_index():
    global _bm25
    collection = get_collection()
    count = collection.count()
    if _bm25 is None or _bm25[0] != count:
        got = collection.get(include=["documents", "metadatas"])
        _bm25 = (count, got["ids"], list(zip(got["documents"], got["metadatas"])), BM25(got["documents"]))
    return _bm25


def query_documents(query: str, n_results: int = 4) -> list[str]:
    return [text for text, _ in query_chunks(query, n_results)]


def query_chunks(query: str, n_results: int = 4) -> list[tuple[str, dict]]:
    """Hybrid search: embedding ranking + BM25 ranking, merged by reciprocal rank fusion.

    Returns (text, metadata) pairs.
    """
    # the book text is NFKC-normalized at ingest (ligatures, µ -> μ); match it
    query = expand_query(unicodedata.normalize("NFKC", query))
    collection = get_collection()
    if collection.count() == 0:
        return []
    _, ids, chunks, bm25 = _lexical_index()
    by_id = dict(zip(ids, chunks))

    sem = collection.query(query_embeddings=[embed_query(query)], n_results=_POOL, include=[])["ids"][0]
    rankings = [(_W_SEM, sem), (_W_BM25, [ids[i] for i in bm25.top(query, _POOL)])]
    # exact measurements are a strong signal; rank those chunks by the semantic order
    for term in _measurements(query):
        hit = [i for i, (d, _) in by_id.items() if term in d]
        if hit:
            order = {i: r for r, i in enumerate(sem)}
            rankings.append((_W_MEASURE, sorted(hit, key=lambda i: order.get(i, _POOL))))

    scores = {}
    for weight, ranking in rankings:
        for rank, i in enumerate(ranking):
            scores[i] = scores.get(i, 0.0) + weight / (_RRF_K + rank + 1)
    best = sorted(scores, key=scores.get, reverse=True)[:n_results]
    return [by_id[i] for i in best]


def get_doc_count() -> int:
    collection = get_collection()
    return collection.count()


def _text_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def add_documents(chunks: list, source: str = "unknown") -> int:
    """Replace `source`'s chunks, skipping text already stored (e.g. by the full-book PDF).

    `chunks` holds plain strings or {"text": ..., "meta": {...}} dicts from the
    PDF parser. Returns the number of chunks actually added.
    """
    collection = get_collection()
    collection.delete(where={"source": source})
    # content-hash ids dedupe identical chunks within and across sources
    unique = {}
    for c in chunks:
        text, meta = (c, {}) if isinstance(c, str) else (c["text"], c.get("meta", {}))
        unique.setdefault(_text_id(text), (text, {**meta, "source": source}))
    existing = set(collection.get(ids=list(unique), include=[])["ids"]) if unique else set()
    new = {i: v for i, v in unique.items() if i not in existing}
    if not new:
        return 0
    ids = list(new)
    docs = [t for t, _ in new.values()]
    collection.add(
        documents=docs,
        embeddings=embed_texts(docs),
        ids=ids,
        metadatas=[m for _, m in new.values()],
    )
    return len(docs)


def clear_all(name: str = "pdf_rag"):
    global _client, _collection
    _client = chromadb.PersistentClient(
        path="./chroma_db",
        settings=Settings(anonymized_telemetry=False),
    )
    try:
        _client.delete_collection(name)
    except Exception:
        pass
    _client = None
    _collection = None

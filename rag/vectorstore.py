import hashlib
import re

import chromadb
from chromadb.config import Settings

from rag.embeddings import embed_texts, embed_query

_client = None
_collection = None

_STOPWORDS = {
    "the", "a", "an", "is", "are", "and", "or", "of", "to", "in", "on", "with",
    "for", "from", "at", "by", "into", "how", "what", "when", "which", "it", "its",
    "this", "that", "cap", "stem", "gills", "flesh", "species", "does", "not",
}


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


def _extract_terms(query: str) -> list[str]:
    """Extract distinctive query terms (measurements + rare words), ordered by strength."""
    terms = []
    # measurement patterns: "5-6.5 x 3-4 µm", "40-100 mm", "15-30 x 10-20"
    for m in re.finditer(r"\d+(?:[,.]\d+)?[\s\-×xX]+\d+(?:\s*[\-\u2013]\s*\d+)?\s*(?:µm|mm)", query):
        terms.append(m.group(0).lower())
    for m in re.finditer(r"\b\d+[\-\u2013]\d*\s*(?:µm|mm)\b", query):
        terms.append(m.group(0).lower())
    # distinctive words (len >= 6), alpha or latin-ish
    for w in re.findall(r"[A-Za-z\u00C0-\u00FF]{6,}", query.lower()):
        if w not in _STOPWORDS:
            terms.append(w)
    # longest/most distinctive first
    terms.sort(key=len, reverse=True)
    return list(dict.fromkeys(terms))


def query_documents(query: str, n_results: int = 4) -> list[str]:
    collection = get_collection()
    query_embedding = embed_query(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=max(n_results * 5, 20),
        include=["documents", "distances"],
    )
    docs = results["documents"][0]
    distances = results["distances"][0]

    terms = _extract_terms(query)

    # score semantic results, boost by distinctive-term hits
    pool = {}
    for d, dist in zip(docs, distances):
        if d in pool:
            continue
        score = -dist
        dl = d.lower()
        hit = sum(1 for t in terms if t in dl)
        score += hit * 0.2
        pool[d] = score

    # lexical fallback: hard `$contains` for DISTINCTIVE terms reinforces exact matches
    for t in terms:
        # `$contains` is case-sensitive; genus names are capitalized in the book
        variants = list(dict.fromkeys([t, t.capitalize()]))
        if len(variants) == 1:
            where_document = {"$contains": t}
        else:
            where_document = {"$or": [{"$contains": v} for v in variants]}
        try:
            hit_ids = collection.get(where_document=where_document, include=[])["ids"]
        except Exception:
            continue
        if not hit_ids:
            continue
        distinct = len(hit_ids) <= 12  # rare/selective term -> strong signal
        boost = 5.0 if distinct else 0.5
        try:
            res = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where_document=where_document,
                include=["documents", "distances"],
            )
            for d, dist in zip(res["documents"][0], res["distances"][0]):
                pool[d] = max(pool.get(d, float("-inf")), -dist + boost)
        except Exception:
            continue

    scored = sorted(pool.items(), key=lambda x: -x[1])
    return [d for d, _ in scored[:n_results]]


def get_doc_count() -> int:
    collection = get_collection()
    return collection.count()


def _text_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def add_documents(texts: list[str], source: str = "unknown") -> int:
    """Replace `source`'s chunks, skipping text already stored (e.g. by the full-book PDF).

    Returns the number of chunks actually added.
    """
    collection = get_collection()
    collection.delete(where={"source": source})
    # content-hash ids dedupe identical chunks within and across sources
    unique = {_text_id(t): t for t in texts}
    existing = set(collection.get(ids=list(unique), include=[])["ids"]) if unique else set()
    new = {i: t for i, t in unique.items() if i not in existing}
    if not new:
        return 0
    ids = list(new)
    docs = list(new.values())
    collection.add(
        documents=docs,
        embeddings=embed_texts(docs),
        ids=ids,
        metadatas=[{"source": source}] * len(docs),
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

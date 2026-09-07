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


def _extract_terms(query: str) -> set[str]:
    """Extract distinctive query terms improved with measurements."""
    terms = set()
    # measurement patterns: "5-6.5 x 3-4 µm", "40-100 mm", "15-30 x 10-20"
    for m in re.finditer(r"\d+(?:[,.]\d+)?[\s\-×xX]+\d+(?:\s*[\-\u2013]\s*\d+)?\s*(?:µm|mm)", query):
        terms.add(m.group(0).lower())
    for m in re.finditer(r"\b\d+(?:\s*[\-\u2013]\s*)?\d*\s*(?:µm|mm)\b", query):
        terms.add(m.group(0).lower())
    # distinctive words (len >= 5), alpha or latin-ish
    for w in re.findall(r"[A-Za-z\u00C0-\u00FF]{5,}", query.lower()):
        if w not in _STOPWORDS:
            terms.add(w)
    return terms


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

    scored = []
    seen = set()
    for d, dist in zip(docs, distances):
        norm = d[:200]
        if norm in seen:
            continue
        seen.add(norm)
        score = -dist  # cosine similarity (higher better)
        dl = d.lower()
        hit = sum(1 for t in terms if t in dl)
        score += hit * 0.15
        scored.append((score, d))

    scored.sort(key=lambda x: -x[0])
    return [d for _, d in scored[:n_results]]


def get_doc_count() -> int:
    collection = get_collection()
    return collection.count()


def add_documents(texts: list[str], source: str = "unknown"):
    collection = get_collection()
    embeddings = embed_texts(texts)
    ids = [f"{source}_{i}" for i in range(len(texts))]
    metadatas = [{"source": source}] * len(texts)
    collection.add(
        documents=texts,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas,
    )


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

import chromadb
from chromadb.config import Settings

from rag.embeddings import embed_texts, embed_query

_client = None
_collection = None


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


def query_documents(query: str, n_results: int = 4) -> list[str]:
    collection = get_collection()
    query_embedding = embed_query(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
    )
    docs = results["documents"][0]
    deduped = []
    seen = set()
    for d in docs:
        norm = d[:200]
        if norm in seen:
            continue
        seen.add(norm)
        deduped.append(d)
    return deduped


def get_doc_count() -> int:
    collection = get_collection()
    return collection.count()

from sentence_transformers import SentenceTransformer

MODEL_NAME = "intfloat/multilingual-e5-base"

_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_model()
    prefixed = [f"passage: {t}" for t in texts]
    embeddings = model.encode(prefixed, normalize_embeddings=True)
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    model = get_model()
    prefixed = f"query: {query}"
    embedding = model.encode([prefixed], normalize_embeddings=True)
    return embedding[0].tolist()

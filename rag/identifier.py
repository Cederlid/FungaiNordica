from rag.vectorstore import query_documents
from rag.llm import identify as llm_identify


def identify_mushroom(features: str, n_results: int = 8, model: str = "gemma2:2b") -> str:
    docs = query_documents(features, n_results=n_results)
    return llm_identify(features, docs, model=model)

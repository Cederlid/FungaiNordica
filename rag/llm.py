import ollama


def generate(prompt: str, model: str = "llama3.1") -> str:
    response = ollama.generate(model=model, prompt=prompt)
    return response["response"]


def rag_answer(question: str, context_docs: list[str], model: str = "llama3.1") -> str:
    context = "\n\n---\n\n".join(context_docs)
    prompt = (
        "You are a helpful assistant. Answer the question based on the provided context.\n"
        "If the answer is not in the context, say so honestly.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )
    return generate(prompt, model=model)

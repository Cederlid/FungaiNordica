import ollama


def generate(prompt: str, model: str = "llama3.1") -> str:
    response = ollama.generate(model=model, prompt=prompt)
    return response["response"]


def rag_answer(question: str, context_docs: list[str], model: str = "llama3.1") -> str:
    context = "\n\n---\n\n".join(context_docs)
    prompt = (
        "You are a mycological expert assistant using the Funga Nordica identification guide.\n"
        "Answer the question based on the provided context from the book.\n"
        "When describing fungi, reference: spore measurements, macroscopic features (cap, stem, gills), "
        "microscopic features, habitat, and season.\n"
        "If the answer is not fully in the context, say so and indicate what additional information "
        "would help narrow down the identification.\n\n"
        f"Context from Funga Nordica:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )
    return generate(prompt, model=model)


def identify(features: str, context_docs: list[str], model: str = "llama3.1") -> str:
    context = "\n\n---\n\n".join(context_docs)
    prompt = (
        "You are a mycological expert assistant using the Funga Nordica identification guide.\n"
        "Your task is to help identify a fungus based on the features the user describes.\n\n"
        "Analyze the following features against the identification keys and species descriptions "
        "from Funga Nordica provided in the context.\n\n"
        "1. List possible species that match the described features.\n"
        "2. For each candidate species, note the key features that match and any that conflict.\n"
        "3. If the features are not sufficient to identify the species, ask the user specific "
        "follow-up questions (e.g., spore print color, habitat, cap texture, microscopic features) "
        "to narrow down the identification using the keys.\n"
        "4. Reference specific key steps from the context when applicable.\n\n"
        "Always mention which key section or species description you are basing your conclusion on.\n\n"
        f"Context from Funga Nordica:\n{context}\n\n"
        f"Mushroom features:\n{features}\n\n"
        "Identification analysis:"
    )
    return generate(prompt, model=model)

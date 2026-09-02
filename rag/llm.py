import json
import os

import httpx

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

REMOTE_API_URL = os.environ.get(
    "REMOTE_API_URL", "https://demo-api.anghammar.com/v1/chat/completions"
)
REMOTE_API_KEY = os.environ.get("REMOTE_API_KEY", "")


def _chat(messages: list[dict], stream: bool = False, model: str = ""):
    if not REMOTE_API_KEY:
        raise RuntimeError("REMOTE_API_KEY is not set. Add it to .env or the environment.")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {REMOTE_API_KEY}",
    }
    body = {"messages": messages, "stream": stream}
    if model:
        body["model"] = model
    if stream:
        return httpx.stream("POST", REMOTE_API_URL, json=body, headers=headers, timeout=300.0)
    return httpx.post(REMOTE_API_URL, json=body, headers=headers, timeout=300.0)


def generate(prompt: str, model: str = "") -> str:
    messages = [{"role": "user", "content": prompt}]
    resp = _chat(messages, stream=False, model=model)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def generate_stream(prompt: str, model: str = ""):
    messages = [{"role": "user", "content": prompt}]
    with _chat(messages, stream=True, model=model) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload.strip() == "[DONE]":
                return
            try:
                chunk = json.loads(payload)
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content")
                if content:
                    yield content
            except (json.JSONDecodeError, KeyError, IndexError):
                continue


def rag_answer(question: str, context_docs: list[str], model: str = "") -> str:
    context = "\n\n---\n\n".join(context_docs)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a mycological expert assistant using the Funga Nordica identification guide.\n"
                "Answer directly and concisely. Do not show your reasoning or thought process.\n"
                "When describing fungi, reference: spore measurements, macroscopic features (cap, stem, gills), "
                "microscopic features, habitat, and season.\n"
                "If the answer is not fully in the context, say so and indicate what additional information "
                "would help narrow down the identification."
            ),
        },
        {
            "role": "user",
            "content": f"Context from Funga Nordica:\n{context}\n\nQuestion: {question}",
        },
    ]
    resp = _chat(messages, stream=False, model=model)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def identify(features: str, context_docs: list[str], model: str = "") -> str:
    context = "\n\n---\n\n".join(context_docs)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a mycological expert assistant using the Funga Nordica identification guide.\n"
                "Your task is to help identify a fungus based on the features the user describes.\n\n"
                "1. List possible species that match the described features.\n"
                "2. For each candidate species, note the key features that match and any that conflict.\n"
                "3. If the features are not sufficient to identify the species, ask the user specific "
                "follow-up questions (e.g., spore print color, habitat, cap texture, microscopic features) "
                "to narrow down the identification using the keys.\n"
                "4. Reference specific key steps from the context when applicable.\n\n"
                "Always mention which key section or species description you are basing your conclusion on."
            ),
        },
        {
            "role": "user",
            "content": f"Context from Funga Nordica:\n{context}\n\nMushroom features:\n{features}",
        },
    ]
    resp = _chat(messages, stream=False, model=model)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]

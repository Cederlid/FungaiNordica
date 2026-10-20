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


RAG_SYSTEM = (
    "You are a mycological expert assistant using the Funga Nordica identification guide.\n"
    "Answer directly and concisely. Do not show your reasoning or thought process.\n"
    "When describing fungi, reference: spore measurements, macroscopic features (cap, stem, gills), "
    "microscopic features, habitat, and season.\n"
    "If the user describes a fungus, give your best identification (species name with author, "
    "e.g. 'Agaricus xanthodermus Genev.') when the context supports it, and mention close relatives "
    "that differ by specific features. If the context includes a key text that resolves to a species, "
    "name that species. Only say the answer is not in the context if the retrieved text genuinely "
    "lacks the description.\n"
    "Species descriptions in the context start with the species name; key couplets read "
    "'N. lead → target'. The book abbreviates: Frb = fruitbody, Sp = spores ('Sp deposit' = "
    "spore print), sp. / spp. = species."
)

IDENTIFY_SYSTEM = (
    "You are a mycological expert assistant using the Funga Nordica identification guide.\n"
    "Your task is to help identify a fungus based on the features the user describes.\n\n"
    "1. List possible species that match the described features.\n"
    "2. For each candidate species, note the key features that match and any that conflict.\n"
    "3. If the features are not sufficient to identify the species, ask the user specific "
    "follow-up questions (e.g., spore print color, habitat, cap texture, microscopic features) "
    "to narrow down the identification using the keys.\n"
    "4. Reference specific key steps from the context when applicable.\n\n"
    "The context contains species descriptions (the species name is on the first line), key "
    "couplets ('N. lead → target' with the alternative on the '-' line), and a 'Key path' for "
    "the top candidates: the chain of couplets from the Key to genera down to that species, "
    "each with the chosen lead and, in [not: ...], the rejected alternative. Walk each key path "
    "against the user's observations: a step that contradicts them rules the candidate out; a "
    "step about a feature the user has not described (spore deposit colour, smell, microscopy) "
    "is the follow-up question to ask. Give the single most likely species and say which "
    "features tie it to that species rather than its relatives.\n"
    "The book abbreviates: Frb = fruitbody, Sp = spores ('Sp deposit' = spore print), "
    "sp. / spp. = species; explain abbreviations when the user may not know them.\n"
    "Always mention which key section or species description you are basing your conclusion on."
)


def _complete(messages: list[dict], model: str = "") -> str:
    resp = _chat(messages, stream=False, model=model)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _stream(messages: list[dict], model: str = ""):
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


def generate(prompt: str, model: str = "") -> str:
    return _complete([{"role": "user", "content": prompt}], model=model)


def generate_stream(prompt: str, model: str = ""):
    yield from _stream([{"role": "user", "content": prompt}], model=model)


def _messages(system: str, history, final: str) -> list[dict]:
    """System prompt, earlier conversation turns, then the context-bearing final message."""
    turns = [{"role": m["role"], "content": m["content"]} for m in history
             if m.get("role") in ("user", "assistant") and m.get("content")]
    return [{"role": "system", "content": system}, *turns, {"role": "user", "content": final}]


def _rag_messages(question: str, context_docs: list[str], history=(), system: str = RAG_SYSTEM) -> list[dict]:
    context = "\n\n---\n\n".join(context_docs)
    return _messages(system, history, f"Context from Funga Nordica:\n{context}\n\nQuestion: {question}")


def rag_answer(question: str, context_docs: list[str], model: str = "", history=(), system: str = RAG_SYSTEM) -> str:
    return _complete(_rag_messages(question, context_docs, history, system), model=model)


def rag_answer_stream(question: str, context_docs: list[str], model: str = "", history=(), system: str = RAG_SYSTEM):
    yield from _stream(_rag_messages(question, context_docs, history, system), model=model)


def identify(features: str, context_docs: list[str], model: str = "", history=()) -> str:
    context = "\n\n---\n\n".join(context_docs)
    final = f"Context from Funga Nordica:\n{context}\n\nMushroom features:\n{features}"
    return _complete(_messages(IDENTIFY_SYSTEM, history, final), model=model)

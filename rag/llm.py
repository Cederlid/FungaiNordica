import json
import os
import time

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
# The model's reasoning phase (Qwen3-style chat templates) is off by default: on
# eval_identify.py it made identify answers 3-4x slower (median 214 s vs ~60 s)
# without being more accurate. LLM_THINKING=1 turns it back on.
THINKING = os.environ.get("LLM_THINKING", "0") == "1"
ANSWER_DEADLINE = 300  # seconds


def _chat(messages: list[dict], stream: bool = False, model: str = ""):
    if not REMOTE_API_KEY:
        raise RuntimeError("REMOTE_API_KEY is not set. Add it to .env or the environment.")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {REMOTE_API_KEY}",
    }
    body = {"messages": messages, "stream": stream}
    if not THINKING:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    if model:
        body["model"] = model
    if stream:
        return httpx.stream("POST", REMOTE_API_URL, json=body, headers=headers, timeout=300.0)
    return httpx.post(REMOTE_API_URL, json=body, headers=headers, timeout=300.0)


# shared by both system prompts
_COMMON_RULES = (
    "Name fungi by their scientific names only; never write Swedish or English common names, "
    "not even in parentheses, and not even when the user uses one. 'Names in the user's message' "
    "says which taxon a common name stands for; if this book does not describe it, say so and do "
    "not describe it with other species' characters.\n"
    "Edibility: state only what the context says for that species (e.g. 'Edible', 'Poisonous', "
    "'Deadly poisonous'); never call a species edible from general knowledge. When the user asks "
    "whether a find can be eaten, begin by saying that a description is never enough to eat a "
    "fungus safely, and name every poisonous species among the candidates.\n"
    "If no species description in the context fits the user's find, say so and ask for the "
    "characters that would decide it; never name species the context does not describe.\n"
    "Write only the answer to the user: never think aloud, quote the user to yourself or draft "
    "the answer ('The user says...', 'I will state...').\n"
)

RAG_SYSTEM = (
    "You are a mycological expert assistant using the Funga Nordica identification guide.\n"
    "Answer directly and concisely. Do not show your reasoning or thought process.\n"
    + _COMMON_RULES +
    "Attribute to a genus or species only characters stated in the context; do not fill gaps "
    "from general knowledge. Keep gill colour and spore deposit colour apart.\n"
    "A 'Ruled out by the reported spore deposit colour' list names species whose key path "
    "contradicts the user's spore print: do not propose them.\n"
    "When describing fungi, reference: spore measurements, macroscopic features (cap, stem, gills), "
    "microscopic features, habitat, and season.\n"
    "If the user describes a fungus, give your best identification (species name with author, "
    "e.g. 'Agaricus xanthodermus Genev.') when the context supports it, and mention close relatives "
    "that differ by specific features. When a key couplet separating them turns on a character the "
    "user has not described (flesh colour change when cut or rubbed, smell, spore deposit colour), "
    "end by asking the user to check it. If the context includes a key text that resolves to a species, "
    "name that species. Only say the answer is not in the context if the retrieved text genuinely "
    "lacks the description.\n"
    "Species descriptions in the context start with the species name; key couplets read "
    "'N. lead → target'. The book abbreviates: Frb = fruitbody, Sp = spores ('Sp deposit' = "
    "spore print), sp. / spp. = species."
)

IDENTIFY_SYSTEM = (
    "You are a mycological expert assistant using the Funga Nordica identification guide.\n"
    "Your task is to help identify a fungus based on the features the user describes.\n"
    + _COMMON_RULES +
    "Every character you attribute to a genus or species (spore deposit colour, gill colour, "
    "smell, habitat) must come from the context: genus introductions ('<Genus> | introduction') "
    "describe the whole genus, and a key path's Key to genera steps give its spore deposit colour. "
    "Do not fill gaps from general knowledge; if the context does not give a character, say so. "
    "Keep gill colour and spore deposit colour apart: they often differ.\n"
    "A 'Ruled out by the reported spore deposit colour' list names species whose key path "
    "contradicts the user's spore print: do not propose them.\n\n"
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
    "Always mention which key section or species description you are basing your conclusion on.\n"
    "Begin your answer with one line 'Most likely: <Genus species>' (or 'Most likely: uncertain'). "
    "Keep the answer under about 300 words unless the user asks for more detail."
)


def _complete(messages: list[dict], model: str = "") -> str:
    # always stream from the API: a reasoning model can think for minutes before the
    # first answer token, and the proxy in front of it drops requests that send
    # nothing for 100 s (HTTP 524); streamed reasoning tokens keep the line alive
    return "".join(_stream(messages, model=model))


def _stream(messages: list[dict], model: str = ""):
    # a loaded API keeps the line open with keep-alives and can take 15+ minutes;
    # give up and tell the user instead of letting the chat hang
    deadline = time.monotonic() + ANSWER_DEADLINE
    with _chat(messages, stream=True, model=model) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if time.monotonic() > deadline:
                raise TimeoutError(f"no complete answer within {ANSWER_DEADLINE} s")
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


def _identify_messages(features: str, context_docs: list[str], history=()) -> list[dict]:
    context = "\n\n---\n\n".join(context_docs)
    final = f"Context from Funga Nordica:\n{context}\n\nMushroom features:\n{features}"
    return _messages(IDENTIFY_SYSTEM, history, final)


def identify(features: str, context_docs: list[str], model: str = "", history=()) -> str:
    return _complete(_identify_messages(features, context_docs, history), model=model)


def identify_stream(features: str, context_docs: list[str], model: str = "", history=()):
    yield from _stream(_identify_messages(features, context_docs, history), model=model)


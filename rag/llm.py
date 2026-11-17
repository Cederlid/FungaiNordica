import json
import os
import re
import time

import httpx

from rag.lexical import _is_swedish

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
# 0 makes the same question give the same answer, so a change can be judged on one run
TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0"))
ANSWER_DEADLINE = 300  # seconds


def _chat(messages: list[dict], stream: bool = False, model: str = ""):
    if not REMOTE_API_KEY:
        raise RuntimeError("REMOTE_API_KEY is not set. Add it to .env or the environment.")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {REMOTE_API_KEY}",
    }
    body = {"messages": messages, "stream": stream, "temperature": TEMPERATURE}
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
    "Poison: when the context says a species is poisonous or deadly poisonous ('Poisonous', "
    "'Deadly poisonous'), you may say so, naming the book as the source; never call a species "
    "poisonous or harmless from general knowledge. Edibility: never say that a species is edible, "
    "can be eaten or is harmless, and never write about eating or edibility, not even as a note, "
    "unless the final message says the user asks about eating.\n"
    "If no species description in the context fits the user's find, say so and ask for the "
    "characters that would decide it; never name species the context does not describe.\n"
    "A line 'User's characters against <species>' compares each character the user reported "
    "(cap, gill and stem colour, gill attachment and spacing, cap surface, smell) with that "
    "species' description and key leads: prefer candidates with no 'contradicts', name the "
    "contradicting characters when you mention a candidate, and never claim a character "
    "agrees beyond what the description says.\n"
    "'Key couplets deciding between the top candidates' gives, for the first candidate and its "
    "nearest rivals, the couplet where their key paths part: if the user has not described that "
    "character, end by asking about it in plain words.\n"
    "Tasting is only for identification: whenever you suggest tasting a fungus, say to taste "
    "a small piece and spit it out, never to swallow it.\n"
    "When comparing species, take each species' characters (spore size, habitat, smell) from its "
    "own description or key lead, never from a neighbouring one, and do not compare a character "
    "the context does not give for both.\n"
    "Answer in the language of the user's question. Users range from beginners to mycologists: "
    "explain each technical term briefly in plain words and keep the book's term in parentheses "
    "(e.g. 'vårtiga (verrucose)', 'mandelformade (amygdaloid)', 'warty (verrucose)'), and write "
    "'spores' and 'spore print', not 'Sp' (in Swedish 'sporerna' and 'sporavtrycket', never 'spormassan'; the term needs no explanation, so not 'sporavtrycket (sporavtrycket)').\n"
    "When describing a species' spores, also give the spore deposit (spore print) colour when the "
    "species description, its genus introduction or its key path gives it, and say which of them "
    "gives it.\n"
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


def _language(text: str) -> str:
    """The line telling the model which language to answer in: it sometimes answers an English
    question in Swedish."""
    lower = text.lower()
    swedish = _is_swedish(lower, re.findall(r"[a-zåäö]+", lower))
    return "Svara på svenska." if swedish else "Answer in English."


_EATING = re.compile(r"\b(?:eat|eats|eating|eaten|edible|edibility|äta|äter|ätlig\w*|ätbar\w*|matsvamp\w*)\b",
                     re.I)


def _instructions(text: str) -> str:
    """The lines after the question: the answer language (the model sometimes answers an English
    question in Swedish) and, only when the user asks about eating, how to decline. Given
    always, the eating rule made the model add an unasked note on eating."""
    lines = [_language(text)]
    if _EATING.search(text):
        lines.insert(0, "The user asks about eating: say first that this service does not answer that "
                        "and that a description is never enough to eat a fungus safely, name the "
                        "candidates the context calls poisonous, and then answer the rest of the message. "
                        "Say nothing else about edibility, not even that the context does not state it.")
    return "\n".join(lines)


# the book's edibility statements: "– Edible and good.", "Many species edible.", "Several species
# are slightly poisonous and none are considered edible." (the poison part stays)
_SEGMENT = re.compile(r"(?:(?<![A-Za-z])[A-Z]\.(?= )|[^.;\n])*[.;]?")  # "B. edulis" is not a sentence end
_HARMLESS = re.compile(r"none poisonous|no species known as poisonous", re.I)
_CLAUSE_CHAR = r"(?:(?<![A-Za-z])[A-Z]\.(?= )|[^,;.\n])"
_EDIBLE_CLAUSE = re.compile(rf"(?:,|;| and)\s*{_CLAUSE_CHAR}*\bedible\b{_CLAUSE_CHAR}*", re.I)


def _edibility_cut(m: re.Match) -> str:
    s = m.group(0)
    if not re.search(r"\bedible\b", s, re.I):
        return s
    if _HARMLESS.search(s):
        return ""
    s = _EDIBLE_CLAUSE.sub("", s)
    if not re.search(r"\bedible\b", s, re.I):
        return s
    dash = re.search(r"\s*[–-]\s*edible\b", s, re.I)  # "(T. nauseosum) – Edible"
    if not dash or not s[:dash.start()].strip():
        return ""
    kept = s[:dash.start()]
    return kept + ("." if s.rstrip().endswith(".") and not kept.endswith(".") else "")


def without_edibility(doc: str) -> str:
    """The context text with what the book says about edibility left out: the model repeats
    'Edible' for a candidate however the prompt forbids it."""
    return _SEGMENT.sub(_edibility_cut, doc)


def _context(docs: list[str]) -> str:
    return "\n\n---\n\n".join(without_edibility(d) for d in docs)


def _messages(system: str, history, final: str) -> list[dict]:
    """System prompt, earlier conversation turns, then the context-bearing final message."""
    turns = [{"role": m["role"], "content": m["content"]} for m in history
             if m.get("role") in ("user", "assistant") and m.get("content")]
    return [{"role": "system", "content": system}, *turns, {"role": "user", "content": final}]


def _rag_messages(question: str, context_docs: list[str], history=(), system: str = RAG_SYSTEM) -> list[dict]:
    return _messages(system, history, f"Context from Funga Nordica:\n{_context(context_docs)}\n\n"
                                      f"Question: {question}\n\n{_instructions(question)}")


def rag_answer(question: str, context_docs: list[str], model: str = "", history=(), system: str = RAG_SYSTEM) -> str:
    return _complete(_rag_messages(question, context_docs, history, system), model=model)


def rag_answer_stream(question: str, context_docs: list[str], model: str = "", history=(), system: str = RAG_SYSTEM):
    yield from _stream(_rag_messages(question, context_docs, history, system), model=model)


def _identify_messages(features: str, context_docs: list[str], history=()) -> list[dict]:
    final = (f"Context from Funga Nordica:\n{_context(context_docs)}\n\n"
             f"Mushroom features:\n{features}\n\n{_instructions(features)}")
    return _messages(IDENTIFY_SYSTEM, history, final)


def identify(features: str, context_docs: list[str], model: str = "", history=()) -> str:
    return _complete(_identify_messages(features, context_docs, history), model=model)


def identify_stream(features: str, context_docs: list[str], model: str = "", history=()):
    yield from _stream(_identify_messages(features, context_docs, history), model=model)


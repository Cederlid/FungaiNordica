"""Check an answer against the context it was given.

The prompt says to name only species the context describes, but the model now and
then adds one from general knowledge ("Cantharellus cibarius", "Lactarius turpis").
The answer is streamed, so it cannot be changed afterwards; a note at the end tells
the user which names did not come from the book.
"""
import re

from rag.keypath import species_index
from rag.lexical import _GROUPS, _NAMES, _is_swedish
from rag.vectorstore import get_collection

# author and rank words in the book's names, and words after a genus that name no species
_NOT_EPITHETS = {"var", "non", "auct", "sensu", "nom", "inval", "ined", "emend", "subsp", "ssp", "and", "sp",
                 "spp", "species", "genus", "group", "section", "complex", "relatives", "type", "key",
                 "art", "arter", "arten", "arterna", "släktet", "sektion"}

_names = None  # (genera, epithets)


def _known_names() -> tuple[set, set]:
    """Genera in the book or in the name tables, and every epithet in the book's species names
    and synonyms: "Russula under" is not a name, "Lactarius turpis" is."""
    global _names
    if _names is None:
        species = species_index(get_collection())
        genera = {m["genus"] for _, m in species.values() if m.get("genus")}
        genera |= {t.split()[0] for t in list(_GROUPS.values()) + list(_NAMES.values())}
        epithets = {w for n in species for w in re.findall(r"(?<![A-Za-z])[a-z][a-z\-]{2,}", n)}
        _names = ({g for g in genera if g[:1].isupper()}, epithets - _NOT_EPITHETS)
    return _names


def unsupported_names(answer: str, docs: list[str]) -> list[str]:
    """Binomials in the answer ("Lactarius turpis", "L. turpis") that no context text contains."""
    context = re.sub(r"(?<=[a-z])- (?=[a-z])", "", "\n".join(docs))  # the book's line breaks: "C. splen- dens"
    genera, epithets = _known_names()
    out = []
    for m in re.finditer(r"\b([A-Z][a-z]+|[A-Z]\.)\s+([a-z][a-z\-]{2,})\b", answer):
        genus, epithet = m.groups()
        emphasised = answer[max(0, m.start() - 1):m.start()] in ("*", "_")
        if epithet in _NOT_EPITHETS or not (epithet in epithets or emphasised):
            continue
        if genus.endswith("."):
            pattern = rf"\b{genus[0]}(?:[a-z]+|\.)\s+{re.escape(epithet)}\b"
        elif genus in genera:
            pattern = rf"\b(?:{genus}|{genus[0]}\.)\s+{re.escape(epithet)}\b"
        else:
            continue
        if not re.search(pattern, context):
            out.append(f"{genus} {epithet}")
    return list(dict.fromkeys(out))


def note(answer: str, docs: list[str], question: str) -> str:
    """The note to append to an answer, or "" when every name it gives is in the context."""
    names = unsupported_names(answer, docs)
    if not names:
        return ""
    listed = ", ".join(f"*{n}*" for n in names)
    lower = question.lower()
    if _is_swedish(lower, re.findall(r"[a-zåäö]+", lower)):
        return (f"\n\n---\n*Obs:* {listed} finns inte i texten ur Funga Nordica som svaret bygger på; "
                "det som sägs om {0} kommer inte från boken.".format("dem" if len(names) > 1 else "den"))
    return (f"\n\n---\n*Note:* {listed} {'are' if len(names) > 1 else 'is'} not in the Funga Nordica "
            "text this answer is based on; what is said about {0} does not come from the book."
            .format("them" if len(names) > 1 else "it"))


# Swedish the model writes from the English "spore": "Sporeerna är 8-10 μm", "sporeavtrycket"
_MISSPELT = [(re.compile(r"\b([Ss])poree?(?:rna|na)\b"), r"\1porerna"),
             (re.compile(r"\b([Ss])poreavtryck"), r"\1poravtryck"),
             # the prompt asks for "sporavtrycket"; "spormassan" is the model's own wording
             (re.compile(r"\b([Ss])pormassan\b"), r"\1poravtrycket"),
             (re.compile(r"\b([Ss])pormassa\b"), r"\1poravtryck"),
             # last: the model writes "Spormassan (sporeavtrycket)", corrected above to a repeated word
             (re.compile(r"\b([Ss])por(?:massan|avtryck(?:et)?)\s*\((?:sporavtryck(?:et)?|spormassan|spore print|spore deposit)\)"),
             r"\1poravtrycket")]  # "Sporavtrycket (sporavtrycket)": the explanation is the word itself


def fix_words(text: str) -> str:
    for pattern, right in _MISSPELT:
        text = pattern.sub(right, text)
    return text


# Edibility. The book's own statements are left out of the context (llm.without_edibility), but
# the model then adds them from general knowledge ("Macrolepiota procera is widely considered
# edible") or writes a note on eating nobody asked for; such sentences are left out of the answer.
_EATING = re.compile(r"(?<![a-zåäö])(?:eat|eats|eating|eaten|edib\w*|inedible|consum\w*|äta|äter|äts|ätlig\w*|"
                     r"oätlig\w*|ätbar\w*|matsvamp\w*)(?![a-zåäö])", re.I)
_EDIBLE = re.compile(r"(?<![a-zåäö])(?:edible|ätlig(?!het)\w*|ätbar(?!het)\w*|matsvamp\w*|good to eat|can be eaten|"
                     r"kan ätas|god att äta)", re.I)
_NEGATED = re.compile(r"(?:\b(?:not|never|none|no|inte|ej|aldrig)\W+(?:\w+\W+){0,2})$", re.I)
_DECLINE = re.compile(r"n[o']t answer|svarar inte|besvarar inte|bedömer inte|never enough|aldrig tillräcklig", re.I)
_POISON = re.compile(r"poison|toxic|deadly|lethal|amanitin|orellanin|giftig|dödlig|förgift", re.I)
# "widely considered edible", "In general mycology, it is considered poisonous"
_HEARSAY = re.compile(r"general (?:knowledge|mycology)|widely (?:known|considered)|often considered|"
                      r"commonly (?:known|considered|regarded)|allmänt (?:känd|ansedd)", re.I)
_SENTENCE_END = re.compile(r"(?<=[.!?])[ \t]+(?=[*_(\[]*[A-ZÅÄÖ])|\n")


def _keeps(sentence: str, asked: bool) -> bool:
    s = re.sub(r"eaten by|(?:upp)?ätna? av", "", sentence)  # "eaten by slugs"
    if not (_EATING.search(s) or _POISON.search(s)):
        return True
    if _HEARSAY.search(s):
        return False
    decline = _DECLINE.search(s)
    claim = s[:decline.start()] if decline else s  # "does not answer whether it is edible"
    if any(not _NEGATED.search(claim[:m.start()]) for m in _EDIBLE.finditer(claim)):
        return False
    # unasked, eating is mentioned only to warn ("do not eat it", "it is poisonous"), but not
    # "do not eat it unless you are certain"
    warning = _WARNING.search(s) and not re.search(r"unless|except|om du inte|förrän|utom", s, re.I)
    return asked or not _EATING.search(s) or bool(_POISON.search(s) or warning)


_WARNING = re.compile(r"\b(?:do not|don't|never|ät inte|ät aldrig|inte äta|aldrig äta)\b", re.I)


def _without_edibility(text: str, asked: bool, line_start: bool) -> tuple[str, bool]:
    """The text without the sentences _keeps rejects, and whether it ends at a line start
    (passed on to the next piece of a streamed answer)."""
    out, start = [], 0
    for m in list(_SENTENCE_END.finditer(text)) + [None]:
        end = m.end() if m else len(text)
        piece = text[start:end]
        if not piece:
            continue
        if not piece.strip() or _keeps(piece.rstrip(), asked):
            out.append(piece)
            line_start = piece.endswith("\n")
        elif piece.endswith("\n") and not line_start:
            out.append("\n")  # the line goes on without the left-out sentence
            line_start = True
        start = end
    return "".join(out), line_start


def checked(pieces, docs: list[str], question: str):
    """Pass a streamed answer through with misspelt words corrected and sentences on edibility
    left out, then add the note.

    Text is passed on a sentence at a time, so a sentence is judged whole and a word split
    between pieces is corrected whole.
    """
    from rag.llm import _EATING as asks_eating  # the same test that gives the model the eating rule
    asked = bool(asks_eating.search(question))
    answer, pending, line_start = [], "", True
    for piece in pieces:
        pending += piece
        ends = list(_SENTENCE_END.finditer(pending))
        if not ends:
            continue
        cut = ends[-1].end()
        done, line_start = _without_edibility(fix_words(pending[:cut]), asked, line_start)
        pending = pending[cut:]
        if done:
            answer.append(done)
            yield done
    if pending:
        done, _ = _without_edibility(fix_words(pending), asked, line_start)
        if done:
            answer.append(done)
            yield done
    extra = note("".join(answer), docs, question)
    if extra:
        yield extra


def cleaned(answer: str, docs: list[str], question: str) -> str:
    """A whole (not streamed) answer, checked the same way."""
    return "".join(checked([answer], docs, question))

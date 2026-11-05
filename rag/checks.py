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
    context = "\n".join(docs)
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


def checked(pieces, docs: list[str], question: str):
    """Pass a streamed answer through, then add the note."""
    answer = []
    for piece in pieces:
        answer.append(piece)
        yield piece
    extra = note("".join(answer), docs, question)
    if extra:
        yield extra

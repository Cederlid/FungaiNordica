"""Match the characters a user reports against each species' own description.

Text search finds species whose description shares words with the question, but
it does not know which part a word belongs to: "red cap, white gills" matches a
species with white cap and red gills as well as the right one. Here both sides are
read as characters per part (cap, gills, stem, flesh colour; gill attachment and
spacing; cap surface; smell), so a candidate gains for each reported character
its description confirms and loses for each it contradicts.
"""
import re

from rag.lexical import _MORPH, _VOCAB, _is_swedish, _sv_terms

# book and everyday colour words -> colour families; mixed shades belong to several
_COLOUR_WORDS = {
    "red": "red", "reddish": "red", "scarlet": "red", "crimson": "red", "carmine": "red",
    "cherry": "red", "blood": "red", "vermilion": "red", "coral": "red orange pink",
    "brick": "red orange brown", "wine": "red violet", "vinaceous": "red violet",
    "orange": "orange", "orangish": "orange", "apricot": "orange", "saffron": "orange yellow",
    "salmon": "pink orange", "fulvous": "brown orange", "rust": "brown orange", "rusty": "brown orange",
    "tawny": "brown orange", "yellow": "yellow", "yellowish": "yellow", "lemon": "yellow",
    "sulphur": "yellow", "golden": "yellow", "chrome": "yellow", "citrine": "yellow",
    "ochre": "yellow brown", "ochraceous": "yellow brown", "honey": "yellow brown",
    "brown": "brown", "brownish": "brown", "cinnamon": "brown", "chestnut": "brown", "umber": "brown",
    "sepia": "brown", "bistre": "brown", "tobacco": "brown", "snuff": "brown", "hazel": "brown",
    "date": "brown", "coffee": "brown", "tan": "brown", "beige": "brown white", "buff": "brown white",
    "white": "white", "whitish": "white", "cream": "white yellow", "ivory": "white",
    "grey": "grey", "greyish": "grey", "gray": "grey", "grayish": "grey", "ash": "grey", "slate": "grey",
    "fuliginous": "grey black", "sooty": "grey black", "black": "black", "blackish": "black",
    "green": "green", "greenish": "green", "olive": "green brown", "olivaceous": "green brown",
    "verdigris": "green blue", "blue": "blue", "bluish": "blue", "violet": "violet",
    "violaceous": "violet", "lilac": "violet", "purple": "violet", "purplish": "violet",
    "lavender": "violet", "mauve": "violet", "amethyst": "violet", "pink": "pink", "pinkish": "pink",
    "rose": "pink", "rosy": "pink", "incarnate": "pink",
}
# colours people easily take for each other in dull or faded fruitbodies: neither agree
# nor contradict ("grey" for a pinkish brown Mycena)
_NEAR = {frozenset(p) for p in (("grey", "brown"), ("grey", "black"), ("grey", "white"), ("white", "yellow"),
                                 ("orange", "red"), ("orange", "yellow"), ("orange", "brown"), ("pink", "red"),
                                 ("pink", "violet"), ("red", "violet"), ("brown", "black"), ("green", "blue"))}
_ATTACH = {"decurrent": "decurrent", "subdecurrent": "decurrent", "adnate": "adnate",
           "adnexed": "adnexed", "emarginate": "adnexed", "sinuate": "adnexed", "free": "free",
           "remote": "free"}
_SPACING = {"distant": "distant", "subdistant": "distant", "crowded": "crowded", "close": "crowded"}
_SURFACE = {"viscid": "viscid", "sticky": "viscid", "slimy": "viscid", "glutinous": "viscid",
            "greasy": "viscid", "lubricous": "viscid", "moist": "viscid", "dry": "dry"}
_SMELLS = {"farinaceous": "farinaceous", "mealy": "farinaceous", "raphanoid": "radish",
           "radish": "radish", "anise": "anise", "aniseed": "anise", "honey": "honey",
           "fruity": "fruity", "garlic": "garlic", "cucumber": "cucumber", "fishy": "fish",
           "fish": "fish", "iodine": "iodine", "iodoform": "iodine", "almond": "almond",
           "spermatic": "spermatic", "earthy": "earthy", "chlorine": "chlorine", "coconut": "coconut",
           "carbolic": "carbolic", "phenol": "carbolic", "ink": "carbolic", "sweetish": "sweet",
           "sweet": "sweet", "nauseous": "unpleasant", "unpleasant": "unpleasant",
           "indistinct": "indistinct", "insignificant": "indistinct", "absent": "indistinct"}
# words naming the part a description segment is about
_PARTS = {"cap": "cap", "caps": "cap", "pileus": "cap", "frb": "cap", "mushroom": "cap",
          "fungus": "cap", "gills": "gills", "gill": "gills", "lamellae": "gills", "stem": "stem",
          "stems": "stem", "stipe": "stem", "stalk": "stem", "flesh": "flesh", "context": "flesh",
          "smell": "smell", "smells": "smell", "odour": "smell", "odor": "smell"}
# nouns whose colour is not a part's colour: "white warts", "brown spore print"
_OTHER = {"warts", "wart", "scales", "scale", "flakes", "flake", "patches", "fragments", "remnants",
          "spots", "dots", "ring", "veil", "volva", "fibrils", "squamules", "spores", "spore", "sp",
          "print", "deposit", "milk", "juice", "latex", "tubes", "pores", "mycelium", "base", "bulb"}
_STOP_AHEAD = {"and", "with", "och", "med", "but", "men"}
_NOT = {"not", "never", "without", "no", "inte", "ej", "inget", "ingen", "aldrig", "utan"}
_PART_ONLY = {"attach": "gills", "spacing": "gills", "smell": "smell"}
# weight of each character when it agrees with (+) or contradicts (-) a description
_WEIGHTS = {("cap", "colour"): 3.0, ("gills", "colour"): 2.0, ("stem", "colour"): 1.5,
            ("flesh", "colour"): 1.0, ("gills", "attach"): 1.5, ("gills", "spacing"): 1.0,
            ("cap", "surface"): 1.0, ("smell", "smell"): 1.5}


def _values(word: str) -> list[tuple[str, str]]:
    """(kind, value) for one English word."""
    out = [("colour", c) for c in _COLOUR_WORDS.get(word, "").split()]
    for kind, table in (("attach", _ATTACH), ("spacing", _SPACING), ("surface", _SURFACE), ("smell", _SMELLS)):
        if word in table:
            out.append((kind, table[word]))
    return out


def _segments(text: str) -> list[str]:
    # "Cap 30-80 mm, orange; gills decurrent. Sp 5-7 x 3.5 μm" (keep decimals together)
    return [s.strip() for s in re.split(r";|\.(?=\s|$)", text) if s.strip()]


def description_characters(text: str) -> dict:
    """{(part, kind): {values}} from a species description or key leads."""
    # drop the heading lines: species name, "Genus X | Key A | couplet 12"
    lines = text.split("\n")
    if len(lines) > 1 and lines[1].startswith("Genus "):
        lines = lines[2:]
    found, part = {}, ""
    for seg in _segments("\n".join(lines)):
        words = re.findall(r"[a-z]+", seg.lower())
        if not words:
            continue
        if words[0] in _PARTS:
            part = _PARTS[words[0]]
        elif words[0] in ("sp", "spores", "taste", "milk", "ring", "veil", "universal", "pileipellis",
                          "cheilocystidia", "pleurocystidia", "basidia", "hyphae", "clamps", "on", "in",
                          "under", "with"):
            part = ""  # microscopy, habitat etc.: no part characters
        if not part:
            continue
        if re.search(r"concolorous with (the )?cap|paler than (the )?cap|as cap", seg.lower()) and part != "cap":
            found.setdefault((part, "colour"), set()).update(found.get(("cap", "colour"), ()))
        for i, w in enumerate(words):
            if _NOT.intersection(words[max(0, i - 2):i]):
                continue
            for kind, value in _values(w):
                p = _PART_ONLY.get(kind, part)
                if p == part or kind in _PART_ONLY:
                    found.setdefault((p, kind), set()).add(value)
    return found


def _english(word: str, swedish: bool) -> list[str]:
    """The word and its English/book equivalents ("glesa" -> distant, "spaced" -> distant)."""
    out = [word]
    if swedish:
        for term in _sv_terms(word) or ():
            out += term.split()
    for w in list(out):
        out += (_MORPH.get(w, "") + " " + _VOCAB.get(w, "")).lower().split()
    return out


def user_characters(query: str) -> tuple[dict, dict]:
    """({(part, kind): {values}}, the same for negated values ("inte orange")) from a description.

    A colour or surface word belongs to the part named right after it ("red cap",
    "röd fot"), else to the last part named before it in the clause ("the gills
    were white"), else to the cap ("a red mushroom").
    """
    lower = query.lower()
    raw = re.findall(r"[a-zà-ÿ]+|[,.;!?\n]", lower)
    swedish = _is_swedish(lower, [w for w in raw if w.isalpha()])
    tokens = [[t] if not t.isalpha() else _english(t, swedish) for t in raw]
    found, negated = {}, {}
    for i, tok in enumerate(tokens):
        vals = [v for w in tok for v in _values(w)]
        if not vals:
            continue
        part, other = "", False
        for j in range(i + 1, min(i + 4, len(tokens))):
            if not tokens[j][0].isalpha() or tokens[j][0] in _STOP_AHEAD:
                break
            part = next((_PARTS[w] for w in tokens[j] if w in _PARTS), "")
            other = bool(_OTHER.intersection(tokens[j]))
            if part or other:
                break
        if not part and not other:
            for j in range(i - 1, -1, -1):
                if not tokens[j][0].isalpha() and tokens[j][0] != ",":
                    break
                part = next((_PARTS[w] for w in tokens[j] if w in _PARTS), "")
                other = bool(_OTHER.intersection(tokens[j]))  # "the spore print was white"
                if part or other:
                    break
        if other:
            continue
        before = {w for t in tokens[max(0, i - 2):i] for w in t}
        neg = bool(_NOT & before)
        for kind, value in dict.fromkeys(vals):
            p = _PART_ONLY.get(kind) or (part if part in ("cap", "gills", "stem", "flesh") else "cap")
            if kind == "smell" and neg:  # "no distinct smell", "luktar inget speciellt"
                found.setdefault(("smell", "smell"), set()).add("indistinct")
                continue
            (negated if neg else found).setdefault((p, kind), set()).add(value)
        if part == "smell" and neg and not any(k == "smell" for k, _ in vals):
            found.setdefault(("smell", "smell"), set()).add("indistinct")
    # "luktar inget speciellt", "no smell": a negated smell word without a smell value
    for i, tok in enumerate(tokens):
        if "smell" in tok and _NOT & {w for t in tokens[i + 1:i + 3] + tokens[max(0, i - 2):i] for w in t}:
            if ("smell", "smell") not in found:
                found[("smell", "smell")] = {"indistinct"}
    return found, negated


def _verdicts(user: tuple[dict, dict], species: dict):
    """(character, user's values, book's values, +1 agrees / 0 near or no call / -1 contradicts)."""
    found, negated = user
    for key, values in found.items():
        have = species.get(key)
        if not have:
            continue
        if key == ("smell", "smell") and values == {"indistinct"}:
            # a species with only a distinctive smell contradicts "no smell"
            yield key, values, have, 1 if "indistinct" in have else -1
        elif values & have:
            yield key, values, have, 1
        elif key[1] == "colour" and any(frozenset((u, h)) in _NEAR for u in values for h in have):
            yield key, values, have, 0
        else:
            yield key, values, have, -1
    for key, values in negated.items():
        have = species.get(key)
        if have and values & have and not found.get(key, set()) & have:
            yield key, {"not " + v for v in values}, have, -0.5


def character_score(user: tuple[dict, dict], species: dict) -> float:
    """Sum of weights: + for each reported character the description confirms, - for each it
    contradicts; characters the description does not give count 0."""
    return sum(_WEIGHTS.get(key, 1.0) * v for key, _, _, v in _verdicts(user, species))


def compare(user: tuple[dict, dict], species: dict) -> str:
    """One line for the model: which reported characters the book confirms or contradicts."""
    words = {1: "agrees", 0: "close", -1: "contradicts", -0.5: "contradicts"}
    parts = [f"{key[0]} {key[1] if key[0] != 'smell' else ''}".strip() + f" {'/'.join(sorted(values))}"
             f" (book: {', '.join(sorted(have))}): {words[v]}"
             for key, values, have, v in _verdicts(user, species)]
    return "; ".join(parts)

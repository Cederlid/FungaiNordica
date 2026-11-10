"""Match the characters a user reports against each species' own description.

Text search finds species whose description shares words with the question, but
it does not know which part a word belongs to: "red cap, white gills" matches a
species with white cap and red gills as well as the right one. Here both sides are
read as characters per part (cap, gills, stem, flesh and milk colour; gill
attachment and spacing; cap surface and shape; ring and volva; smell; substrate and
host trees), so a candidate gains for each reported character its description
confirms and loses for each it contradicts.
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
    "rose": "pink", "rosy": "pink", "incarnate": "pink", "fleshcoloured": "pink orange brown",
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
# translucent, striate cap margin ("strierad kant"); "not translucently striate" reads as smooth
_MARGIN = {"striate": "striate", "striated": "striate", "striations": "striate", "striation": "striate",
           "sulcate": "striate", "translucent": "striate", "translucently": "striate", "streaked": "striate"}
_HYGRO = {"hygrophanous": "yes", "hygrophanic": "yes"}
# scales on the cap or stem; a description of a dry cap may still have them, so they only agree
_SCALES = {"scales": "yes", "scaly": "yes", "squamules": "yes", "squamulose": "yes", "squarrose": "yes",
           "squamulous": "yes"}
_SHAPES = {"infundibuliform": "funnel", "funnel": "funnel", "depressed": "funnel", "conical": "conical",
           "conic": "conical", "cone": "conical", "campanulate": "conical", "bell": "conical",
           "umbonate": "umbonate", "umbo": "umbonate", "papillate": "umbonate"}
# where a fruitbody grows; the book names hosts by genus ("with Betula")
_SUBSTRATES = {"wood": "wood", "lignicolous": "wood", "stump": "wood", "stumps": "wood", "log": "wood",
               "logs": "wood", "trunk": "wood", "trunks": "wood", "branches": "wood", "twigs": "wood",
               "twig": "wood", "sticks": "wood", "bark": "wood", "timber": "wood", "sawdust": "wood",
               "chips": "wood", "moss": "moss", "mosses": "moss", "mossy": "moss", "sphagnum": "moss",
               "dung": "dung", "manure": "dung", "droppings": "dung", "grass": "grass",
               "grassland": "grass", "grasslands": "grass", "lawn": "grass", "lawns": "grass",
               "meadow": "grass", "meadows": "grass", "pasture": "grass", "pastures": "grass",
               "grassy": "grass", "burnt": "burnt", "burned": "burnt", "charred": "burnt",
               "sand": "sand", "sandy": "sand", "dunes": "sand", "ground": "ground", "soil": "ground",
               "terrestrial": "ground", "terricolous": "ground"}
_CONIFERS = {"pinus", "picea", "larix", "abies", "juniperus", "pseudotsuga"}
_DECIDUOUS = {"betula", "fagus", "quercus", "populus", "salix", "alnus", "corylus", "tilia", "carpinus",
              "fraxinus", "sorbus", "ulmus", "acer", "dryas", "helianthemum"}
_HOSTS = {**{h: h for h in _CONIFERS | _DECIDUOUS},
    "coniferous": "conifer", "conifers": "conifer", "conifer": "conifer", "deciduous": "deciduous",
    "frondose": "deciduous", "broadleaved": "deciduous"}
# words naming the part a description segment is about
_PARTS = {"cap": "cap", "caps": "cap", "pileus": "cap", "frb": "cap", "mushroom": "cap",
          "fungus": "cap", "gills": "gills", "gill": "gills", "lamellae": "gills", "stem": "stem",
          "stems": "stem", "stipe": "stem", "stalk": "stem", "flesh": "flesh", "context": "flesh",
          "smell": "smell", "smells": "smell", "odour": "smell", "odor": "smell", "milk": "milk",
          "latex": "milk", "juice": "milk", "tubes": "gills", "pores": "gills", "pore": "gills"}
# nouns whose colour is not a part's colour: "white warts", "brown spore print"
_OTHER = {"warts", "wart", "scales", "scale", "flakes", "flake", "patches", "fragments", "remnants",
          "spots", "dots", "ring", "veil", "volva", "fibrils", "squamules", "spores", "spore", "sp",
          "print", "deposit", "mycelium", "base", "bulb"}
_STOP_AHEAD = {"and", "with", "och", "med", "but", "men"}
_NOT = {"not", "never", "without", "no", "lacks", "lacked", "lacking", "inte", "ej", "inget", "ingen",
        "aldrig", "utan", "saknade", "saknar"}
_PART_ONLY = {"attach": "gills", "spacing": "gills", "smell": "smell", "shape": "cap", "margin": "cap",
              "hygro": "cap"}
_COLOUR_PARTS = ("cap", "gills", "stem", "flesh", "milk")
# weight of each character when it agrees with (+) or contradicts (-) a description
_WEIGHTS = {("cap", "colour"): 3.0, ("gills", "colour"): 2.0, ("stem", "colour"): 1.5,
            ("flesh", "colour"): 1.0, ("milk", "colour"): 2.5, ("gills", "attach"): 1.5,
            ("gills", "spacing"): 1.0, ("cap", "surface"): 1.0, ("cap", "shape"): 1.0,
            ("smell", "smell"): 1.5, ("stem", "ring"): 2.0, ("stem", "volva"): 2.0,
            ("habitat", "substrate"): 2.0, ("habitat", "host"): 1.5, ("cap", "margin"): 1.5,
            ("cap", "hygro"): 1.0, ("cap", "scales"): 1.5, ("stem", "scales"): 1.5,
            ("cap", "size"): 1.0, ("stem", "size"): 0.5}
# substrates one report may give for another: moss and grass grow on soil, dung lies in pastures
_NEAR_SUBSTRATE = {frozenset(p) for p in (("ground", "moss"), ("ground", "grass"), ("ground", "sand"),
                                           ("ground", "burnt"), ("grass", "dung"), ("grass", "moss"),
                                           ("sand", "moss"), ("sand", "grass"))}
_NEAR_BY_KIND = {"colour": _NEAR, "substrate": _NEAR_SUBSTRATE}
# a description lists only some of a fruitbody's shapes as it ages: a shape never contradicts
_NO_CONTRADICTION = {("cap", "shape"), ("cap", "scales"), ("stem", "scales"), ("cap", "hygro")}


_FLESH_COLOURED = re.compile(r"flesh[- ]?colou?red")
_SIZE = re.compile(r"(?<![\d.])(\d+(?:[.,]\d+)?)(?:\s*[-\u2013]\s*(\d+(?:[.,]\d+)?))?\s*(mm|cm)\b")
_NUM = r"(\d+(?:\.\d+)?)"
# "Cap 20-60 mm", "stem 30-80 x 2-6 mm", "cap (10-)20-60(-80) mm": the length or width first named
_DESC_SIZE = re.compile(r"(?:\(\s*[\d.]+\s*-\s*\)\s*)?" + _NUM + r"\s*[-\u2013]\s*" + _NUM)
_SIZE_CAP = {"wide", "across", "diameter", "broad", "bred", "bredd", "stor", "stort", "stora", "big", "large",
             "small", "tvärs", "hatt", "hatten", "cap", "caps", "mushroom", "fungus", "svamp", "svampen"}
_SIZE_STEM = {"long", "tall", "high", "lång", "lång", "hög", "stem", "stipe", "stalk", "fot", "foten", "stam",
              "stammen"}


def _sizes(lower: str) -> dict:
    """{("cap"|"stem", "size"): {"lo-hi" in mm}} from "10 cm wide", "1-5 cm stor", "stem 5 cm long".

    The words right after the size (else right before it) say which part it measures; a size
    with neither is left out.
    """
    found = {}
    for m in _SIZE.finditer(lower):
        lo = float(m.group(1).replace(",", "."))
        hi = float((m.group(2) or m.group(1)).replace(",", "."))
        scale = 10 if m.group(3) == "cm" else 1
        after = re.findall(r"[a-zà-ÿ]+", lower[m.end():m.end() + 30])[:3]
        before = re.findall(r"[a-zà-ÿ]+", lower[max(0, m.start() - 30):m.start()])[-3:]
        part = ""
        for words in (after, before[::-1]):
            for w in words:
                if w in _SIZE_STEM and w not in ("tall", "high", "long", "lång", "hög"):
                    part = "stem"
                elif w in _SIZE_CAP:
                    part = "cap"
                elif w in ("long", "tall", "high", "lång", "hög"):
                    part = part or "stem"
                if part:
                    break
            if part:
                break
        if part:
            found.setdefault((part, "size"), set()).add(f"{lo * scale:g}-{hi * scale:g}")
    return found


def _description_size(part: str, seg: str) -> str:
    """"20-60" (mm) for a segment such as "Cap 20-60 mm, convex"; "" when it gives no range."""
    m = _DESC_SIZE.search(seg)
    if not m:
        return ""
    scale = 10 if re.search(r"\bcm\b", seg) else 1
    return f"{float(m.group(1)) * scale:g}-{float(m.group(2)) * scale:g}"


def _values(word: str) -> list[tuple[str, str]]:
    """(kind, value) for one English word."""
    out = [("colour", c) for c in _COLOUR_WORDS.get(word, "").split()]
    for kind, table in (("attach", _ATTACH), ("spacing", _SPACING), ("surface", _SURFACE),
                        ("smell", _SMELLS), ("shape", _SHAPES), ("margin", _MARGIN), ("hygro", _HYGRO),
                        ("scales", _SCALES)):
        if word in table:
            out.append((kind, table[word]))
    return out


def _habitat(words: list[str]) -> dict:
    """{("habitat", "substrate"|"host"): {values}}; a host genus also counts as conifer or deciduous."""
    found = {}
    for i, w in enumerate(words):
        if w == "ground" and i and (words[i - 1] in _COLOUR_WORDS or words[i - 1] in ("pale", "dark")):
            continue  # "fibrils on a pale ground": the background colour
        if w in _SUBSTRATES:
            found.setdefault(("habitat", "substrate"), set()).add(_SUBSTRATES[w])
        if w in _HOSTS:
            host = found.setdefault(("habitat", "host"), set())
            host.add(_HOSTS[w])
            if w in _CONIFERS:
                host.add("conifer")
            elif w in _DECIDUOUS:
                host.add("deciduous")
    return found


def _veil(words: list[str]) -> dict:
    """{("stem", "ring"|"volva"): {"yes"|"no"}}: "ring membranous", "without ring", "no volva"."""
    found = {}
    for i, w in enumerate(words):
        kind = {"ring": "ring", "annulus": "ring", "annular": "ring", "skirt": "ring", "volva": "volva",
                "volval": "volva", "sac": "volva"}.get(w)
        if kind:
            neg = _NOT.intersection(words[max(0, i - 3):i]) or words[i + 1:i + 2] == ["absent"]
            found.setdefault(("stem", kind), set()).add("no" if neg else "yes")
        elif w in ("exannulate", "exannulated"):
            found.setdefault(("stem", "ring"), set()).add("no")
    return found


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
        words = re.findall(r"[a-z]+", _FLESH_COLOURED.sub("fleshcoloured", seg.lower()))
        if not words:
            continue
        for key, values in {**_habitat(words), **_veil(words)}.items():
            found.setdefault(key, set()).update(values)
        if words[0] in _PARTS:
            part = _PARTS[words[0]]
        elif words[0] in ("sp", "spores", "taste", "ring", "veil", "universal", "pileipellis",
                          "cheilocystidia", "pleurocystidia", "basidia", "hyphae", "clamps", "on", "in",
                          "under", "with"):
            part = ""  # microscopy, habitat etc.: no part characters
        if not part:
            continue
        if re.search(r"concolorous with (the )?cap|paler than (the )?cap|as cap", seg.lower()) and part != "cap":
            found.setdefault((part, "colour"), set()).update(found.get(("cap", "colour"), ()))
        if part in ("cap", "stem") and (part, "size") not in found and (
                words[0] in _PARTS or re.match(r"\s*\(?\d", seg)):
            size = _description_size(part, seg)
            if size:
                found[(part, "size")] = {size}
        for i, w in enumerate(words):
            if _NOT.intersection(words[max(0, i - 2):i]):
                if w in _MARGIN and part == "cap":
                    found.setdefault(("cap", "margin"), set()).add("smooth")  # "not translucently striate"
                continue
            for kind, value in _values(w):
                p = _PART_ONLY.get(kind, part)
                if p == part or kind in _PART_ONLY:
                    found.setdefault((p, kind), set()).add(value)
    return found


def _english(word: str, swedish: bool, book: bool = True) -> list[str]:
    """The word and its English/book equivalents ("glesa" -> distant, "spaced" -> distant);
    book=False stops at the English word ("sopp" -> bolete, not on to "tubes")."""
    out = [word]
    if swedish:
        for term in _sv_terms(word) or ():
            out += term.lower().split()
    for w in list(out) if book else ():
        out += (_MORPH.get(w, "") + " " + _VOCAB.get(w, "")).lower().split()
    return out


def user_characters(query: str) -> tuple[dict, dict]:
    """({(part, kind): {values}}, the same for negated values ("inte orange")) from a description.

    A colour or surface word belongs to the part named right after it ("red cap",
    "röd fot"), else to the last part named before it in the clause ("the gills
    were white"), else to the cap ("a red mushroom").
    """
    lower = _FLESH_COLOURED.sub("fleshcoloured", query.lower())
    raw = re.findall(r"[a-zà-ÿ]+|[,.;!?\n]", lower)
    swedish = _is_swedish(lower, [w for w in raw if w.isalpha()])
    tokens = [[t] if not t.isalpha() else _english(t, swedish) for t in raw]
    # the part a word names: "bolete" expands to the book's "tubes" but names no part
    names = [[t] if not t.isalpha() else _english(t, swedish, book=False) for t in raw]
    found, negated = {}, {}
    for i, tok in enumerate(tokens):
        vals = [v for w in tok for v in _values(w)]
        if not vals:
            continue
        part, other = "", False
        for j in range(i + 1, min(i + 4, len(tokens))):
            if not tokens[j][0].isalpha() or tokens[j][0] in _STOP_AHEAD:
                break
            part = next((_PARTS[w] for w in names[j] if w in _PARTS), "")
            other = bool(_OTHER.intersection(tokens[j]))
            if part or other:
                break
        if not part and not other:
            for j in range(i - 1, -1, -1):
                if not tokens[j][0].isalpha() and tokens[j][0] != ",":
                    break
                part = next((_PARTS[w] for w in names[j] if w in _PARTS), "")
                other = bool(_OTHER.intersection(tokens[j]))  # "the spore print was white"
                if part or other:
                    break
        if other:
            continue
        before = {w for t in tokens[max(0, i - 2):i] for w in t}
        neg = bool(_NOT & before)
        for kind, value in dict.fromkeys(vals):
            p = _PART_ONLY.get(kind) or (part if part in _COLOUR_PARTS else "cap")
            if kind == "smell" and neg:  # "no distinct smell", "luktar inget speciellt"
                found.setdefault(("smell", "smell"), set()).add("indistinct")
                continue
            (negated if neg else found).setdefault((p, kind), set()).add(value)
        if part == "smell" and neg and not any(k == "smell" for k, _ in vals):
            found.setdefault(("smell", "smell"), set()).add("indistinct")
    words = [w for t in tokens for w in t]
    for key, values in {**_habitat(words), **_veil([t[-1] for t in tokens]), **_sizes(lower)}.items():
        found.setdefault(key, set()).update(values)
    # "luktar inget speciellt", "no smell": a negated smell word without a smell value
    for i, tok in enumerate(tokens):
        if "smell" in tok and _NOT & {w for t in tokens[i + 1:i + 3] + tokens[max(0, i - 2):i] for w in t}:
            if ("smell", "smell") not in found:
                found[("smell", "smell")] = {"indistinct"}
    return found, negated


def _size_verdict(user: set, book: set) -> int:
    """+1 when the reported size lies within the book's range (with a margin for estimates),
    -1 when it is less than half of its smallest or more than twice its largest, else 0."""
    best = -1
    for u in user:
        ulo, uhi = map(float, u.split("-"))
        for b in book:
            blo, bhi = map(float, b.split("-"))
            if uhi >= blo * 0.8 and ulo <= bhi * 1.25:
                return 1
            if not (uhi < blo * 0.5 or ulo > bhi * 2):
                best = max(best, 0)
    return best


def _verdicts(user: tuple[dict, dict], species: dict):
    """(character, user's values, book's values, +1 agrees / 0 near or no call / -1 contradicts)."""
    found, negated = user
    for key, values in found.items():
        have = species.get(key)
        if not have:
            continue
        if key[1] == "size":
            yield key, values, have, _size_verdict(values, have)
        elif key == ("smell", "smell") and values == {"indistinct"}:
            # a species with only a distinctive smell contradicts "no smell"
            yield key, values, have, 1 if "indistinct" in have else -1
        elif values & have:
            yield key, values, have, 1
        elif key in _NO_CONTRADICTION or any(frozenset((u, h)) in _NEAR_BY_KIND.get(key[1], ())
                                             for u in values for h in have):
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
    # (counting a description that names many colours for less scored worse: terse
    # descriptions name few, not distinctive ones)
    return sum(_WEIGHTS.get(key, 1.0) * v for key, _, _, v in _verdicts(user, species))


def compare(user: tuple[dict, dict], species: dict) -> str:
    """One line for the model: which reported characters the book confirms or contradicts."""
    words = {1: "agrees", 0: "close", -1: "contradicts", -0.5: "contradicts"}
    parts = [f"{key[0]} {key[1] if key[0] != 'smell' else ''}".strip() + f" {'/'.join(sorted(values))}"
             f" (book: {', '.join(sorted(have))}): {words[v]}"
             for key, values, have, v in _verdicts(user, species)]
    return "; ".join(parts)

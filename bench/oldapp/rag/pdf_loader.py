"""Parse Funga Nordica (MycoKey PDF) into key-aware chunks.

The book is one long dichotomous key: each couplet ("N." ... "-" ...) has two
leads, and each lead ends in a target — another couplet number, a sub-key, a
"see ..." reference, a page reference, or a species name. Species descriptions
are the text of the lead that ends in that species, so the species name comes
*after* its description.
"""
import os
import re
import unicodedata

import fitz


def load_pdf(path: str) -> str:
    doc = fitz.open(path)
    pages = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()
    return "\n".join(pages)


_MAX_CHUNK = 1800
_MIN_TAIL = 300
_LEAD_PREVIEW = 300

# "12.", "12.  Flesh staining yellow ...", "26(11) Sp with germ pore ..." (26 is reached from 11)
_COUPLET = re.compile(r"^(\d+[a-z]?)(?:\.|\((\d+)\))(?:\s+(.*))?$")
_GOTO = re.compile(r"^\d+$")
_KEY_HEADER = re.compile(r"^(?:Key [A-Z0-9]+:|Main key$)")
_TARGET = re.compile(r"^(?:Key [A-Z0-9]+\b|see )|\bp\. \d+$")
# "A. muscaria (L.: Fr.) Lam. var. muscaria", "L. fulvissimus Romagn.", "A. eliae Quél."
_SPECIES = re.compile(
    r"^(?:[A-Z]\.|[A-Z][a-z]+)\s[a-z][a-z\-]+"
    r"(?:\s(?:var\.|f\.|subsp\.)\s[a-z][a-z\-]+)?\s+(?:\(|[A-ZÀ-Þ])"
)
_AUTHORS = re.compile(r"^[A-ZÀ-Þ][A-ZÀ-Þ .,&'\-]{5,}$")
_GENUS = re.compile(r"^([A-Z][a-z]{3,})\s+(?:\(|[A-ZÀ-Þ])[^\d;]*$")
_PERSON = re.compile(r"^[A-ZÀ-Þ][a-zß-ÿ]+(?:[ \-][A-ZÀ-Þ][a-zß-ÿ]+)+$")
_NOISE = [
    re.compile(r"^#+$"),
    re.compile(r"– In: Knudsen"),
    re.compile(r"^Pdf version from MycoKey"),
    re.compile(r"^•: the species is illustrated"),
    # figure captions: "Spores: A: ...", "Spores, cheilocystidia and ...: A: ...", "D: M. cognata, E: ..."
    re.compile(r"^Spores\b.*:"),
    re.compile(r"^Spores and "),
    re.compile(r"^[A-Z]: [A-Z]\. [a-z]"),
    re.compile(r"^[A-Za-z]{1,3}$"),  # figure panel labels: "A", "B", "ch"
    # running page heads: "212 HYGROPHORACEAE: HYGROCYBE", "TRICHOLOMATACEAE: CLITOCYBE 393"
    re.compile(r"^(?:\d+ [A-Z][A-Z :,\-]{3,}|[A-Z][A-Z :,\-]{3,} \d+)$"),
]


def _clean_lines(text: str) -> list[str]:
    raw = [l.strip() for l in unicodedata.normalize("NFKC", text).split("\n")]
    lines = [l for l in raw if l]
    # running page heads ("KEY TO GENERA") and the page numbers next to them
    drop = set()
    for i, l in enumerate(lines):
        if l == "KEY TO GENERA":
            drop.add(i)
            for j in (i - 1, i + 1):
                if 0 <= j < len(lines) and _GOTO.match(lines[j]):
                    drop.add(j)
    out = []
    for i, l in enumerate(lines):
        if i in drop or any(p.search(l) for p in _NOISE):
            continue
        out.append(l)
    return out


def _join(lines: list[str]) -> str:
    text = ""
    for l in lines:
        if text.endswith("\xad"):
            text = text[:-1] + l
        else:
            text = f"{text} {l}" if text else l
    return text.replace("\xad", "").strip()


def _expand(name: str, genus: str) -> str:
    """'A. muscaria Fr.' -> 'Amanita muscaria Fr.' when the abbreviation matches the genus."""
    if genus and re.match(r"^[A-Z]\.\s", name) and genus[0] == name[0]:
        return genus + name[2:]
    return name


def _split(prefix: str, body: str) -> list[str]:
    room = _MAX_CHUNK - len(prefix) - 1
    if len(body) <= room:
        return [f"{prefix}\n{body}"]
    parts, start = [], 0
    while start < len(body):
        end = start + room
        if len(body) - start <= room + _MIN_TAIL:
            end = len(body)  # don't leave a tiny tail chunk with only distribution data
        if end < len(body):
            cut = body.rfind(". ", start, end)
            end = cut + 1 if cut > start + room // 2 else end
        parts.append(f"{prefix}\n{body[start:end].strip()}")
        start = end
    return parts


def _genus_header(line: str, nxt: str, genera: set[str]) -> str | None:
    """Genus name if `line` opens a genus chapter: "Amanita Pers." + an author line."""
    if line.lower().startswith("key to genera") and _AUTHORS.match(nxt):
        return "Key to genera"
    m = _GENUS.match(line)
    if not m or _KEY_HEADER.match(line) or line.startswith(("Subgen", "Sect")):
        return None
    # author credit on the next line ("JAN VESTERHOLT", "Alexander Kovalenko")
    # or at the end of the same line ("Psathyrella (Fr.) Quél. LEIF ÖRSTADIUS & ...")
    credited = bool(_AUTHORS.match(nxt)) or bool(re.search(r"[A-ZÀ-Þ]{3,} [A-ZÀ-Þ&]{1,}", line))
    if m.group(1) in genera:
        credited = credited or bool(_PERSON.match(nxt))
    elif re.search(r"(?:i|ae)$", m.group(1)):
        return None  # plural section names ("Paleacei ..."), not a genus
    return m.group(1) if credited else None


def known_genera(directory: str) -> set[str]:
    """Genus names taken from the per-genus PDF filenames (FungaNordica-<Genus>.pdf)."""
    names = set()
    for f in os.listdir(directory):
        m = re.match(r"FungaNordica-([A-Z][a-z]+)\.pdf$", f)
        if m:
            names.add(m.group(1))
    return names


def parse_chunks(text: str, genera: set[str] = frozenset()) -> list[dict]:
    lines = _clean_lines(text)
    chunks = []
    st = {
        "genus": "", "genus_header": "", "key": "",
        "couplet": None, "leads": [],
        "last": 0,  # last couplet number in the current key; couplets run 1, 2, 3, ...
        "lead": None,  # lines of the open lead
        "intro": [],   # text outside couplets (genus / key introduction)
        "species": None,  # (name, description, couplet) awaiting its notes
        "notes": [],
    }

    def meta(kind, **extra):
        m = {"kind": kind, "genus": st["genus"], "key": st["key"]}
        m.update(extra)
        return m

    def where():
        parts = [f"Genus {st['genus']}" if st["genus"] else "", st["key"]]
        return " | ".join(p for p in parts if p)

    def flush_species():
        if st["species"]:
            name, desc, couplet = st["species"]
            loc = where() + (f" | couplet {couplet}" if couplet else "")
            body = desc + ("\n" + _join(st["notes"]) if st["notes"] else "")
            for part in _split(f"{name}\n{loc}", body):
                chunks.append({"text": part, "meta": meta("species", species=name, couplet=couplet or "")})
        st["species"], st["notes"] = None, []

    def flush_intro():
        if st["intro"]:
            body = _join(st["intro"])
            header = st["genus_header"] if not st["key"] else ""
            prefix = " | ".join(p for p in [header or where(), "introduction"] if p)
            if len(body) > 30:
                for part in _split(prefix, body):
                    chunks.append({"text": part, "meta": meta("intro")})
        st["intro"] = []

    def end_lead(target, species=""):
        if not target and st["lead"]:
            # target printed on the same line as the lead text: "... entire margin 4"
            m = re.search(r"\s(\d+|(?:see|Key) .+?)$", st["lead"][-1])
            if m and (m.group(1).isdigit() or _TARGET.search(m.group(1))):
                st["lead"][-1] = st["lead"][-1][: m.start()]
                target = m.group(1)
        st["leads"].append((_join(st["lead"]), target))
        if species:
            st["species"] = (species, _join(st["lead"]), st["couplet"])
        st["lead"] = None

    def flush_couplet():
        if st["lead"]:
            end_lead("")
        if st["couplet"] and st["leads"]:
            rows = []
            for i, (lead, target) in enumerate(st["leads"]):
                if len(lead) > _LEAD_PREVIEW:
                    lead = lead[:_LEAD_PREVIEW].rsplit(" ", 1)[0] + " …"
                mark = f"{st['couplet']}." if i == 0 else "-"
                rows.append(f"{mark} {lead} → {target or '?'}")
            targets = "|".join(t for _, t in st["leads"] if t)
            text = f"{where()} | couplet {st['couplet']}\n" + "\n".join(rows)
            chunks.append({"text": text, "meta": meta("couplet", couplet=st["couplet"], targets=targets)})
        st["couplet"], st["leads"] = None, []

    def flush_all():
        flush_species()
        flush_couplet()
        flush_intro()

    def open_text():
        return st["lead"] if st["lead"] is not None else st["intro"]

    def is_couplet(line):
        """(number, lead text on the same line) if `line` opens a couplet."""
        m = _COUPLET.match(line)
        if not m:
            return None
        n = int(re.match(r"\d+", m.group(1)).group(0))
        # couplets run upwards in small steps; this rejects wrapped references
        # such as "R&H\n344." that look like couplet numbers
        if m.group(2):
            # "26(11)": the back-reference must point to a couplet already seen
            ok = int(m.group(2)) <= st["last"] < n
        else:
            ok = n == 1 or st["last"] <= n <= st["last"] + 5
        if ok:
            return m.group(1), m.group(3) or ""
        return None

    started = False  # genus PDFs open with the tail of the previous genus; skip it
    skip = set()  # mixed-case author lines consumed by a genus header
    for i, line in enumerate(lines):
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if _AUTHORS.match(line):
            continue
        genus = _genus_header(line, nxt, genera)
        if genus:
            flush_all()
            started = True
            st["genus_header"] = line
            st["genus"] = "" if genus == "Key to genera" else genus
            st["key"] = "Key to genera" if not st["genus"] else ""
            st["last"] = 0
            if genus in genera and _PERSON.match(nxt):
                skip.add(i + 1)
            continue
        if i in skip:
            continue
        if not started:
            continue
        if _KEY_HEADER.match(line):
            flush_all()
            st["key"] = line
            st["last"] = 0
            continue
        couplet = is_couplet(line)
        if couplet:
            flush_species()
            flush_couplet()
            flush_intro()
            num, rest = couplet
            st["couplet"], st["lead"] = num, [rest] if rest else []
            st["last"] = int(re.match(r"\d+", num).group(0))
            continue
        if line == "-" and st["couplet"]:
            flush_species()
            if st["lead"]:
                end_lead("")
            st["lead"] = []
            continue
        if st["lead"] is not None and st["lead"]:
            if _GOTO.match(line) or _TARGET.search(line):
                end_lead(line)
                continue
        current = open_text()
        ends_text = current and current[-1].rstrip().endswith((".", ")", "•"))
        before_break = nxt == "-" or nxt.startswith("Note:") or bool(is_couplet(nxt)) \
            or bool(_KEY_HEADER.match(nxt))
        # unabbreviated names must start with a real genus, not prose like "Often among Carex ..."
        first = line.split()[0]
        named = first.endswith(".") or first in genera or first == st["genus"]
        if current and not st["species"] and named and _SPECIES.match(line) and (ends_text or before_break):
            name = _expand(line, st["genus"])
            if st["lead"] is not None:
                end_lead(name, species=name)
            else:  # single-species genus: the introduction is the description
                st["species"] = (name, _join(st["intro"]), None)
                st["intro"] = []
            continue
        if st["species"]:
            st["notes"].append(line)
        else:
            current.append(line)

    flush_all()
    return chunks


def load_and_chunk(pdf_path: str) -> list[dict]:
    return parse_chunks(load_pdf(pdf_path), known_genera(os.path.dirname(os.path.abspath(pdf_path))))

import re
import fitz


def load_pdf(path: str) -> str:
    doc = fitz.open(path)
    pages = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()
    return "\n".join(pages)


_AUTHOR_TOKEN = re.compile(r"[A-Z][a-z\u00C0-\u00FF]{1,20}\.?(?:\s*[A-Z]\.?)*$")


def _is_key_step(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    return bool(re.match(r'^\d+[\)\.]\s', stripped)) or bool(re.match(r'^[a-z][\)\.]\s', stripped))


def _is_header(line: str) -> bool:
    """Detect a species / subgenus / section header line (Latin binomial + author)."""
    if not line:
        return False
    length = len(line)
    if length > 80 or length < 4:
        return False
    if re.match(r'^\d', line):
        return False
    # Single uppercase genus-level headings like "RUSSULACEAE" or "Russula Pers."
    if " " not in line and re.match(r'^[A-Z]{3,}$', line):
        return True
    if " " not in line:
        return False
    # Exclude running prose / distribution lines
    words = line.split()
    if line.endswith(".") and len(words) >= 6 and not re.search(r"\([A-Z]", line):
        return False
    if re.search(r"[a-z]\d", line):
        return False
    # Contains a parenthetical author citation: (Pers.), (Bull.), (Fr.: Fr.), (Krombh.)
    if re.search(r"\([A-Z][a-z\u00C0-\u00FF]{1,14}\.?\.?(?::[^)]*)?\)", line):
        return True
    # Short line ending in an author-like token (e.g. "R. delica Fr.", "L. lacunarum Hora")
    if len(words) <= 5 and _AUTHOR_TOKEN.search(line):
        return True
    return False


def _prefix(header: str, segment: str) -> str:
    return f"{header}: {segment}" if header else segment


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """Chunk the extracted text, anchoring each chunk to its species header."""
    chunks = []
    current_header = ""
    current_header_len = 0

    def emit(segment: str):
        prefix = _prefix(current_header, "")
        base = len(prefix) + 2
        if not segment.strip():
            return
        if len(segment) + base <= chunk_size:
            chunks.append(_prefix(current_header, segment).strip())
            return
        # Long description: split into smaller pieces, keep the header on each
        text = segment
        start = 0
        n = len(text)
        while start < n:
            end = start + (chunk_size - base)
            if end >= n:
                chunks.append(_prefix(current_header, text[start:]).strip())
                break
            # break at a whitespace boundary near the limit
            cut = text.rfind(" ", start, end)
            if cut <= start:
                cut = end
            chunk_text_piece = text[start:cut]
            chunks.append(_prefix(current_header, chunk_text_piece).strip())
            start = cut
            if overlap > 0:
                nxt = text.rfind(" ", max(start - overlap, 0), start)
                if nxt > 0:
                    start = nxt

    buffer = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if _is_header(stripped):
            if buffer:
                emit("\n".join(buffer))
                buffer = []
            current_header = stripped
            continue
        buffer.append(stripped)

    if buffer:
        emit("\n".join(buffer))

    return [c for c in chunks if len(c) > 30]


def load_and_chunk(pdf_path: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    text = load_pdf(pdf_path)
    return chunk_text(text, chunk_size, overlap)

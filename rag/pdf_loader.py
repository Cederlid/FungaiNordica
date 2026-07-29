import re
import fitz


def load_pdf(path: str) -> str:
    doc = fitz.open(path)
    pages = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()
    return "\n".join(pages)


def _is_key_step(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    return bool(re.match(r'^\d+[\)\.]\s', stripped)) or bool(re.match(r'^[a-z][\)\.]\s', stripped))


def _is_table_row(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    parts = stripped.split()
    if len(parts) < 3:
        return False
    tab_count = stripped.count("  ")
    if tab_count >= 1:
        return True
    return False


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    lines = text.split("\n")
    chunks = []
    current_chunk = []
    current_len = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current_chunk:
                current_chunk.append("")
            continue

        if _is_key_step(line) and current_len >= chunk_size:
            chunks.append("\n".join(current_chunk).strip())
            overlap_text = "\n".join(current_chunk[-3:])
            current_chunk = [overlap_text]
            current_len = len(overlap_text)
            current_chunk.append(line)
            current_len += len(line)
        else:
            current_chunk.append(line)
            current_len += len(line) + 1

    if current_chunk:
        remaining = "\n".join(current_chunk).strip()
        if remaining:
            chunks.append(remaining)

    return [c for c in chunks if len(c) > 30]


def load_and_chunk(pdf_path: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    text = load_pdf(pdf_path)
    return chunk_text(text, chunk_size, overlap)

import fitz  # PyMuPDF


def load_pdf(path: str) -> str:
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start = end - overlap
    return chunks


def load_and_chunk(pdf_path: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    text = load_pdf(pdf_path)
    return chunk_text(text, chunk_size, overlap)

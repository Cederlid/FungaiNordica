import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag.pdf_loader import load_and_chunk
from rag.vectorstore import add_documents, query_documents, get_doc_count
from rag.llm import rag_answer
from rag.identifier import identify_mushroom
from rag.ocr import process_spore_images


def ingest_all(directory: str, ocr_model: str | None = None):
    if not os.path.isdir(directory):
        print(f"Error: Directory not found: {directory}")
        return
    pdfs = sorted([f for f in os.listdir(directory) if f.lower().endswith(".pdf")])
    if not pdfs:
        print(f"No PDF files found in {directory}")
        return
    print(f"Found {len(pdfs)} PDF(s) to ingest\n")
    for pdf in pdfs:
        pdf_path = os.path.join(directory, pdf)
        print(f"[{pdfs.index(pdf)+1}/{len(pdfs)}] Processing {pdf}...")
        chunks = load_and_chunk(pdf_path)
        print(f"  {len(chunks)} text chunks")
        add_documents(chunks, source=pdf)
        if ocr_model:
            captions = process_spore_images(pdf_path, model=ocr_model)
            if captions:
                add_documents(captions, source=f"{pdf}_spore_illustrations")
                print(f"  + {len(captions)} image captions")
                print(f"  Total in store: {get_doc_count()}")
            else:
                print(f"  No images found")
        print()
    print(f"Done! {get_doc_count()} total documents in vector store")


def ingest(pdf_path: str, ocr_model: str | None = None):
    if not os.path.exists(pdf_path):
        print(f"Error: File not found: {pdf_path}")
        return
    source = os.path.basename(pdf_path)

    print(f"Loading text from {pdf_path}...")
    chunks = load_and_chunk(pdf_path)
    print(f"Split into {len(chunks)} text chunks")
    add_documents(chunks, source=source)
    print(f"Stored {len(chunks)} text chunks in vector store ({get_doc_count()} total)")

    if ocr_model:
        captions = process_spore_images(pdf_path, model=ocr_model)
        if captions:
            add_documents(captions, source=f"{source}_spore_illustrations")
            print(f"Stored {len(captions)} image captions in vector store ({get_doc_count()} total)")
        else:
            print("No images found in PDF")
    else:
        print("Skipping spore illustration OCR (use --ocr-model to enable, e.g. --ocr-model llava)")


def ask(question: str, model: str = "gemma4:e4b"):
    if get_doc_count() == 0:
        print("No documents ingested yet. Run: python main.py ingest <pdf_path>")
        return
    print("Retrieving relevant sections...")
    docs = query_documents(question)
    print("Generating answer...")
    answer = rag_answer(question, docs, model=model)
    print(f"\nAnswer:\n{answer}")


def identify(features: str, model: str = "gemma4:e4b"):
    if get_doc_count() == 0:
        print("No documents ingested yet. Run: python main.py ingest <pdf_path>")
        return
    print("Searching Funga Nordica keys for matching features...")
    result = identify_mushroom(features, model=model)
    print(f"\nIdentification analysis:\n{result}")


def main():
    parser = argparse.ArgumentParser(
        description="FungaiNordica — Mushroom Identification Assistant"
    )
    sub = parser.add_subparsers(dest="command")

    ingest_cmd = sub.add_parser("ingest", help="Ingest a single PDF file")
    ingest_cmd.add_argument("pdf_path", help="Path to PDF file")
    ingest_cmd.add_argument("--ocr-model", default=None, help="Vision model for spore illustration OCR (e.g. llava, llama3.2-vision)")

    ingest_all_cmd = sub.add_parser("ingest-all", help="Ingest all PDFs in a directory")
    ingest_all_cmd.add_argument("directory", help="Directory containing PDF files")
    ingest_all_cmd.add_argument("--ocr-model", default=None, help="Vision model for spore illustration OCR (e.g. llava, llama3.2-vision)")

    ask_cmd = sub.add_parser("ask", help="Ask a question about a fungus")
    ask_cmd.add_argument("question", help="Your question")
    ask_cmd.add_argument("--model", default="gemma4:e4b", help="Ollama model name")

    identify_cmd = sub.add_parser("identify", help="Identify a mushroom from features")
    identify_cmd.add_argument("features", help="Mushroom features (cap, stem, gills, habitat, etc.)")
    identify_cmd.add_argument("--model", default="gemma4:e4b", help="Ollama model name")

    args = parser.parse_args()
    if args.command == "ingest":
        ingest(args.pdf_path, ocr_model=args.ocr_model)
    elif args.command == "ingest-all":
        ingest_all(args.directory, ocr_model=args.ocr_model)
    elif args.command == "ask":
        ask(args.question, model=args.model)
    elif args.command == "identify":
        identify(args.features, model=args.model)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

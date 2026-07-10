import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag.pdf_loader import load_and_chunk
from rag.vectorstore import add_documents, query_documents, get_doc_count
from rag.llm import rag_answer


def ingest(pdf_path: str):
    if not os.path.exists(pdf_path):
        print(f"Error: File not found: {pdf_path}")
        return
    print(f"Loading {pdf_path}...")
    chunks = load_and_chunk(pdf_path)
    print(f"Split into {len(chunks)} chunks")
    source = os.path.basename(pdf_path)
    add_documents(chunks, source=source)
    print(f"Stored {len(chunks)} chunks in vector store ({get_doc_count()} total)")


def ask(question: str, model: str = "llama3.1"):
    if get_doc_count() == 0:
        print("No documents ingested yet. Run: python main.py ingest <pdf_path>")
        return
    print("Retrieving relevant chunks...")
    docs = query_documents(question)
    print("Generating answer...")
    answer = rag_answer(question, docs, model=model)
    print(f"\nAnswer:\n{answer}")


def main():
    parser = argparse.ArgumentParser(description="Simple PDF RAG")
    sub = parser.add_subparsers(dest="command")

    ingest_cmd = sub.add_parser("ingest", help="Ingest a PDF file")
    ingest_cmd.add_argument("pdf_path", help="Path to PDF file")

    ask_cmd = sub.add_parser("ask", help="Ask a question")
    ask_cmd.add_argument("question", help="Your question")
    ask_cmd.add_argument("--model", default="llama3.1", help="Ollama model name")

    args = parser.parse_args()
    if args.command == "ingest":
        ingest(args.pdf_path)
    elif args.command == "ask":
        ask(args.question, model=args.model)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

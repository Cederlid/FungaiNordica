import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
from rag.pdf_loader import load_and_chunk
from rag.vectorstore import add_documents, query_documents, get_doc_count
from rag.llm import rag_answer


def upload_pdf(file):
    if file is None:
        return "No file uploaded"
    chunks = load_and_chunk(file.name)
    source = os.path.basename(file.name)
    add_documents(chunks, source=source)
    return f"Uploaded {source}: {len(chunks)} chunks indexed (total: {get_doc_count()})"


def chat(message, history):
    if get_doc_count() == 0:
        return "Please upload a PDF first."
    docs = query_documents(message)
    answer = rag_answer(message, docs)
    return answer


with gr.Blocks(title="PDF RAG") as app:
    gr.Markdown("# Simple PDF RAG")
    gr.Markdown("Upload a PDF, then ask questions about it.")

    with gr.Row():
        pdf_upload = gr.File(label="Upload PDF", file_types=[".pdf"])
        status = gr.Textbox(label="Status", interactive=False)

    pdf_upload.change(upload_pdf, inputs=pdf_upload, outputs=status)

    chatbot = gr.ChatInterface(
        fn=chat,
        title="",
        examples=[
            "What is this document about?",
            "Summarize the key points",
        ],
    )

if __name__ == "__main__":
    app.launch()

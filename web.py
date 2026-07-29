import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
from rag.pdf_loader import load_and_chunk
from rag.vectorstore import add_documents, query_documents, get_doc_count
from rag.llm import rag_answer
from rag.identifier import identify_mushroom


def upload_pdf(file):
    if file is None:
        return "No file uploaded"
    chunks = load_and_chunk(file.name)
    source = os.path.basename(file.name)
    add_documents(chunks, source=source)
    return f"Uploaded {source}: {len(chunks)} chunks indexed (total: {get_doc_count()})"


def chat_fn(message, history):
    if get_doc_count() == 0:
        return "Please upload a PDF first."
    docs = query_documents(message)
    answer = rag_answer(message, docs)
    return answer


def identify_fn(message, history):
    if get_doc_count() == 0:
        return "Please upload a PDF first."
    return identify_mushroom(message)


with gr.Blocks(title="FungaiNordica") as app:
    gr.Markdown("# FungaiNordica — Mushroom Identification Assistant")
    gr.Markdown(
        "Upload the **Funga Nordica** PDF, then ask questions or describe a "
        "mushroom for identification."
    )

    with gr.Row():
        pdf_upload = gr.File(label="Upload PDF", file_types=[".pdf"])
        status = gr.Textbox(label="Status", interactive=False)

    pdf_upload.change(upload_pdf, inputs=pdf_upload, outputs=status)

    gr.Markdown("## Choose mode")

    with gr.Tabs():
        with gr.TabItem("Ask a Question"):
            gr.ChatInterface(
                fn=chat_fn,
                title="",
                examples=[
                    "What are the spore measurements of Boletus edulis?",
                    "Describe the cap features of Amanita muscaria",
                    "Which species grow under birch in autumn?",
                ],
            )

        with gr.TabItem("Identify a Mushroom"):
            gr.Markdown(
                "Describe the mushroom features you've observed. The assistant will "
                "use the Funga Nordica keys to suggest possible species and may ask "
                "follow-up questions."
            )
            gr.ChatInterface(
                fn=identify_fn,
                title="",
                examples=[
                    "Brown cap 5-10cm, white stem with a ring, grows under birch, autumn",
                    "Small mushroom, purple gills, spore print is brown, grows in grass",
                    "White cap with scales, ring on stem, white gills, spore print white",
                ],
            )

if __name__ == "__main__":
    app.launch()

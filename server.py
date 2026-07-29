import sys
import os
import time
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag.vectorstore import query_documents, get_doc_count
from rag.llm import rag_answer, generate
from rag.identifier import identify_mushroom

app = FastAPI(title="FungaiNordica API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "llama3.1"
    messages: list[Message]
    stream: bool = False


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list
    usage: dict


@app.get("/")
def root():
    return {"service": "FungaiNordica", "status": "running", "documents": get_doc_count()}


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "llama3.1", "object": "model"},
            {"id": "identify", "object": "model"},
        ],
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    if not req.messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    user_message = req.messages[-1].content if req.messages else ""
    system_prompt = ""
    for msg in req.messages:
        if msg.role == "system":
            system_prompt = msg.content

    if get_doc_count() == 0:
        response_text = "No documents have been ingested yet. Please upload a PDF first."

    elif req.model == "identify":
        response_text = identify_mushroom(user_message, model="llama3.1")

    else:
        docs = query_documents(user_message)
        if system_prompt:
            context = "\n\n---\n\n".join(docs)
            prompt = (
                f"{system_prompt}\n\n"
                f"Context from Funga Nordica:\n{context}\n\n"
                f"Question: {user_message}\n\n"
                "Answer:"
            )
            response_text = generate(prompt, model=req.model)
        else:
            response_text = rag_answer(user_message, docs, model=req.model)

    response = {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }
    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

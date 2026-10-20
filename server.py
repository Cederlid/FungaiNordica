import sys
import os
import time
import json
import re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from rag.vectorstore import query_documents, get_doc_count
from rag.llm import RAG_SYSTEM, rag_answer, rag_answer_stream
from rag.identifier import identify_mushroom

_HISTORY_TURNS = 8  # earlier user/assistant messages passed to the model

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
    model: str = ""
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
            {"id": "gemma4:26b-moe", "object": "model"},
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

    # Build a retrieval anchor from the full user conversation so follow-up
    # questions ("which species?") keep the original description in focus.
    user_msgs = [m.content for m in req.messages if m.role == "user" and m.content.strip()]
    retrieval_query = "\n".join(user_msgs[-3:]) or user_message
    # earlier turns go to the model so it remembers its own follow-up questions
    history = [{"role": m.role, "content": m.content} for m in req.messages[:-1]
               if m.role in ("user", "assistant")][-_HISTORY_TURNS:]

    if is_casual(user_message):
        greeting = (
            "Hi! I'm your Funga Nordica assistant. I can help identify mushrooms and answer "
            "questions using the Funga Nordica guide.\n\n"
            "Try asking things like:\n"
            "- 'What are the key features of Russula? '\n"
            "- Describe a mushroom you found and I'll try to identify it.\n\n"
            "What would you like to know?"
        )
        return streaming_response(iter([greeting]), req.model) if req.stream else _json_response(greeting, req)

    if get_doc_count() == 0:
        response_text = "No documents have been ingested yet. Please upload a PDF first."

    elif req.model == "identify":
        response_text = identify_mushroom(user_message, history=history, retrieval_query=retrieval_query)

    else:
        docs = query_documents(retrieval_query)
        # a system prompt set in LibreChat replaces the default one
        system = system_prompt or RAG_SYSTEM
        if req.stream:
            return streaming_response(
                rag_answer_stream(user_message, docs, model=req.model, history=history, system=system), req.model
            )
        response_text = rag_answer(user_message, docs, model=req.model, history=history, system=system)

    if req.stream:
        return streaming_response(iter([response_text]), req.model)

    return _json_response(response_text, req)


def is_casual(message: str) -> bool:
    text = message.strip().lower()
    if not text:
        return False
    casual = [
        "hi", "hello", "hey", "yo", "howdy", "good morning", "good afternoon",
        "good evening", "thanks", "thank you", "thankyou", "no problem",
        "who are you", "what can you do", "what are you", "help", "how are you",
    ]
    if text in casual:
        return True
    return bool(re.fullmatch(r"hi+", text))


def _json_response(response_text: str, req: ChatRequest):
    return {
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


def streaming_response(text_iter, model: str):
    def event_stream():
        completion_id = f"chatcmpl-{int(time.time())}"
        created = int(time.time())
        try:
            for piece in text_iter:
                chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": piece},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            final_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

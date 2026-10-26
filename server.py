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
from rag.vectorstore import get_doc_count
from rag.llm import RAG_SYSTEM, rag_answer, rag_answer_stream
from rag.identifier import identification_context, identify_mushroom, identify_mushroom_stream
from rag.lexical import tokenize

_HISTORY_TURNS = 8  # earlier user/assistant messages passed to the model
_NEW_QUESTION_TOKENS = 7  # content words that make a question self-contained

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


@app.on_event("startup")
def warm_up():
    # load the embedding model and search indexes now, not on the first question (~10 s)
    if get_doc_count():
        identification_context("warm up")


@app.get("/")
def root():
    return {"service": "FungaiNordica", "status": "running", "documents": get_doc_count()}


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "qwen3.8-27b", "object": "model"},
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
    retrieval_query = user_message if is_new_question(user_message) else "\n".join(user_msgs[-3:]) or user_message
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
        if req.stream:
            return streaming_response(
                identify_mushroom_stream(user_message, history=history, retrieval_query=retrieval_query), req.model
            )
        response_text = identify_mushroom(user_message, history=history, retrieval_query=retrieval_query)

    else:
        # users describe finds here too, without choosing "identify": give the same
        # species descriptions, genus introductions and key paths
        docs = identification_context(retrieval_query)
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


def is_new_question(message: str) -> bool:
    """A full question of its own ("... brown gills, what could it be?") starts a new topic.

    Retrieving it together with earlier messages lets an unrelated earlier question
    (e.g. about Amanita muscaria) take over the search. Short questions ("which
    species?") and statements answering the assistant ("the gills bruise brown")
    are follow-ups and keep the earlier description.
    """
    return "?" in message and len(tokenize(message)) >= _NEW_QUESTION_TOKENS


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
            # as answer text: LibreChat does not show an "error" field in a stream
            message = ("\n\nThe language model did not answer this time (it may be overloaded). "
                       f"Please try again in a moment. [{type(e).__name__}: {e}]")
            for chunk in ({"delta": {"content": message}, "finish_reason": None},
                          {"delta": {}, "finish_reason": "stop"}):
                body = {"id": completion_id, "object": "chat.completion.chunk", "created": created,
                        "model": model, "choices": [{"index": 0, **chunk}]}
                yield f"data: {json.dumps(body)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

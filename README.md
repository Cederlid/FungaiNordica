# FungaiNordica

A chat assistant that identifies Nordic mushrooms and answers questions about them, using the field guide *Funga Nordica* as its only source. You can describe a find in everyday English or Swedish ("white cap, brown gills, growing in a park"). The assistant then searches the book's species descriptions and identification keys, and answers with scientific names, quoting the characters the book gives.

The chat interface is [LibreChat](https://www.librechat.ai/). Behind it, a FastAPI server does the retrieval and calls a remote language model.

## How it works

1. **Ingest.** The book's PDFs are split into chunks: species descriptions, genus introductions and key couplets. The chunks are embedded with `intfloat/multilingual-e5-base` and stored in a local Chroma database (`chroma_db/`).
2. **Search.** Each question is first spelling-corrected and translated into the book's vocabulary: "spruce" becomes *Picea*, "warts" becomes *veil fragments*, and "granskogen" becomes *Picea forest*. The search then combines semantic (embedding) search with BM25 keyword search.
3. **Identify.** For the best-supported species, the server adds the species description, the genus introduction and the key path. The key path is the route through the keys, from the Key to genera down to the species. A reported spore deposit colour rules out species that the keys place in another colour section. Common names in the question are resolved to scientific names, for example "narrkantarell" to *Hygrophoropsis aurantiaca*. The context also says when the book does not describe the named taxon.
4. **Answer.** The language model answers from this context only. It names fungi by their scientific names, does not cover edibility, and asks about the character that would separate the remaining candidates.

| Path | Contents |
|---|---|
| `server.py` | OpenAI-compatible API used by LibreChat (`/v1/chat/completions`, streaming) |
| `rag/pdf_loader.py` | PDF parsing and chunking |
| `rag/vectorstore.py` | Chroma storage and hybrid search |
| `rag/lexical.py` | BM25, spelling correction, the everyday/Swedish to book-term vocabulary, and common names |
| `rag/keypath.py` | Key paths through the identification keys |
| `rag/identifier.py` | Candidate species, spore-colour filtering and context assembly |
| `rag/llm.py` | System prompts and calls to the remote model |
| `main.py`, `web.py` | Command line and a simple Gradio interface |
| `eval_retrieval.py`, `eval_identify.py` | Benchmarks |

## Setup

You need Docker. The book's PDFs are not included in the repository, so you must supply your own copy.

1. Put the *Funga Nordica* PDFs in `data/`.
2. Create `.env` in the project root:

   ```
   REMOTE_API_URL=https://your-llm-host/v1/chat/completions
   REMOTE_API_KEY=...
   JWT_SECRET=<random string>
   JWT_REFRESH_SECRET=<random string>
   ```

   The model API must be OpenAI-compatible and support streaming. You can generate the JWT secrets with `openssl rand -hex 32`.

3. Start the services:

   ```
   docker compose up -d --build
   ```

4. Build the search index once. The first run downloads the embedding model.

   ```
   docker exec -w /app fungai-rag-server python main.py ingest-all data
   ```

   Then restart the server so that it loads the new index:

   ```
   docker restart fungai-rag-server
   ```

5. Open http://localhost:3080, create an account and pick the **FungaiNordica** endpoint. Registration is off by default; set `ALLOW_REGISTRATION=true` in `.env` to create the first account.

Both ports are bound to localhost only.

### Models in LibreChat

- **qwen3.8-27b** is the general chat. It handles questions and descriptions of finds alike. The name is only a label: the remote API uses whichever model it serves.
- **identify** uses a stricter prompt for identification. Its answers start with "Most likely: …" and walk through the key path.

### Settings

| Variable | Default | Effect |
|---|---|---|
| `LLM_THINKING` | `0` | `1` turns on the model's reasoning phase. This made answers 3–4× slower in tests, without making them more accurate. |
| `LLM_TEMPERATURE` | `0` | Sampling temperature. At 0 the same question gives the same answer, so a change can be judged on one run. |
| `ALLOW_REGISTRATION` | `false` | Allows new LibreChat accounts. |

## Development

`rag/` and `server.py` are mounted into the container. After editing them, restart the server:

```
docker restart fungai-rag-server
```

The command line works without LibreChat:

```
docker exec -w /app fungai-rag-server python main.py ask "What are the spore measurements of Lactarius deliciosus?"
docker exec -w /app fungai-rag-server python main.py identify "Red cap with white warts, ring and bulbous base, under birch"
```

### Benchmarks

```
docker exec -w /app fungai-rag-server python eval_retrieval.py     # retrieval only, no LLM, about 1 minute
docker exec -w /app fungai-rag-server python eval_identify.py      # end to end through the LLM, slow
```

Run `eval_retrieval.py` after changing the search or the vocabulary, to check that nothing got worse.

## Notes

- The remote model API can be slow or overloaded, and its proxy drops requests that send nothing for 100 s. The server therefore always streams from the API, and gives up after 300 s with a message in the chat.
- The server and LibreChat run on the host machine. If the machine sleeps, the chat is unreachable and answers in progress are cut off.
- Genera that are not in the supplied PDFs cannot be answered from the book (for example *Cantharellus*), and the assistant says so.

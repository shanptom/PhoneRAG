# phone-rag

A lightweight Retrieval-Augmented Generation (RAG) system designed to run entirely on an Android phone using [Termux](https://termux.dev) and [Ollama](https://ollama.com). No cloud APIs, no heavy frameworks — just Python standard library and local LLMs.

Upload documents, build a vector index, and ask questions — all from a mobile-friendly web UI or the command line.

## How It Works

```
                          +-----------------+
                          |  docs/*.txt/pdf |
                          +--------+--------+
                                   |
                          chunk + embed (batched)
                                   |
                                   v
                          +--------+--------+
                          | index_nomic.json|
                          | (vector store)  |
                          +--------+--------+
                                   |
          user query -----> embed query via Ollama
                                   |
                                   v
                     +-------------+-------------+
                     | hybrid search              |
                     | (cosine sim + keyword boost)|
                     +-------------+-------------+
                                   |
                            top-k chunks
                                   |
                                   v
                     +-------------+-------------+
                     | LLM generates answer       |
                     | grounded in retrieved text  |
                     +-----------------------------+
```

### Pipeline

1. **Ingest** — Reads documents from `~/phone-rag/docs/`, splits text into overlapping chunks (768 chars max, 96-char overlap) using recursive semantic splitting (paragraphs > sentences > clauses > words), generates embeddings in batches of 16 via Ollama, and stores everything in `index_nomic.json`.

2. **Retrieve** — Embeds the user's query with the same model, then scores every chunk using:
   - **Semantic**: Cosine similarity (with pre-computed norms)
   - **Lexical**: Stopword-filtered Jaccard keyword overlap (0.0-0.15 boost)
   - **Broad query detection**: Queries like "summarize" or "overview" retrieve all chunks from the target file instead of top-k

3. **Generate** — Feeds retrieved chunks into a prompt and streams the answer. If no chunk scores above the minimum threshold (0.28), it returns `NOT FOUND IN DOCUMENTS`.

## Beginner Walkthrough

This guide assumes you have an Android phone. Every step runs on the phone itself.

### Step 1: Install Termux

Download **Termux** from [F-Droid](https://f-droid.org/en/packages/com.termux/) (not the Play Store version — it's outdated).

Open Termux and update packages:

```bash
pkg update && pkg upgrade
```

### Step 2: Install Python and Git

```bash
pkg install python git
```

### Step 3: Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Start the Ollama server (keep it running in the background):

```bash
ollama serve &
```

### Step 4: Pull the Models

You need two models — one for understanding documents (embedding) and one for generating answers (chat):

```bash
ollama pull nomic-embed-text-v2-moe    # embedding model
ollama pull gemma3:1b                   # chat model (small, fast)
```

These are small enough to run on most phones. The embedding model loads only when indexing/querying, then unloads to save RAM.

### Step 5: Clone and Set Up

```bash
git clone <repo-url> ~/phone-rag
cd ~/phone-rag
pip install pypdf flask
```

### Step 6: Add Your Documents

Place `.txt` or `.pdf` files into the `docs/` folder:

```bash
cp ~/storage/downloads/my-notes.txt ~/phone-rag/docs/
cp ~/storage/downloads/report.pdf ~/phone-rag/docs/
```

### Step 7: Launch the Web UI

```bash
python app.py
```

Open your phone's browser and go to `http://localhost:5000`.

You'll see three tabs:

- **RAG** — Ask questions grounded in your documents
- **Chat** — Talk directly to the AI (no document context)
- **Index** — Upload files and build the search index

### Step 8: Build Your Index

1. Go to the **Index** tab
2. Tap the upload area to add files (or you can put them in `docs/` manually)
3. Tap **Build Index** and wait for it to finish — you'll see progress as each file is chunked and embedded

### Step 9: Ask Questions

Switch to the **RAG** tab and type a question. The system will:
1. Find the most relevant chunks from your documents
2. Show you which sources it found
3. Stream an answer grounded in those sources

Try broad queries too — "summarize meeting.txt" will pull all chunks from that file.

### Step 10: Direct Chat

The **Chat** tab lets you talk to the model directly without any document context — useful for general questions, brainstorming, or quick calculations.

## CLI Usage

Prefer the terminal? The CLI tools are in the `cli/` folder:

```bash
# Build the index
python cli/build_index_nomic.py

# Ask a question (RAG)
python cli/ask_nomic.py

# Search only (no LLM generation)
python cli/search.py
```

## Project Structure

```
phone-rag/
├── app.py                      # Flask web UI (RAG + Chat + Index + Upload)
├── cli/
│   ├── build_index_nomic.py    # CLI: chunk, embed, store
│   ├── ask_nomic.py            # CLI: retrieve + generate
│   └── search.py               # CLI: retrieval only
├── legacy/                     # Older embeddinggemma-based scripts
├── docs/                       # Your source documents (.txt, .pdf)
├── index_nomic.json            # Vector index (auto-generated)
├── tech.md                     # Technical deep-dive
└── README.md
```

## Configuration

All configuration is via constants at the top of `app.py` (or the CLI scripts):

| Parameter | Default | Description |
|---|---|---|
| `EMBED_MODEL` | `nomic-embed-text-v2-moe` | Ollama embedding model |
| `CHAT_MODEL` | `gemma3:1b` | Ollama chat model |
| `CHUNK_MAX` | `768` | Max characters per chunk |
| `CHUNK_OVERLAP` | `96` | Character overlap between chunks |
| `EMBED_BATCH_SIZE` | `16` | Chunks per embedding API call |
| `MIN_SCORE` | `0.28` | Minimum score to attempt generation |
| `MAX_BROAD_CHUNKS` | `25` | Max chunks for broad/summary queries |
| `EMBED_KEEP_ALIVE` | `0` | Unload embedding model immediately |
| `CHAT_KEEP_ALIVE` | `3600` | Keep chat model loaded for 1 hour |

### Memory Management

Tuned for low-RAM devices:

- The **embedding model** unloads immediately after use (`keep_alive: 0`) to free memory for the chat model.
- The **chat model** stays loaded for 1 hour (`keep_alive: 3600`) so consecutive queries are fast.
- Embeddings are processed in **batches of 16** to reduce API round-trips while keeping memory bounded.

## Design Decisions

- **Zero heavy dependencies** — Python stdlib (`json`, `math`, `re`, `pathlib`, `urllib`) plus `pypdf` and `flask`. No LangChain, no FAISS, no ChromaDB.
- **JSON vector store** — Simple, portable, inspectable. Adequate for personal document collections.
- **Hybrid retrieval** — Semantic search plus keyword boost to catch exact term matches that pure embedding similarity might miss.
- **Pre-compiled regex & pre-computed norms** — Avoids redundant work on hot paths.
- **Batched embeddings** — 16 chunks per API call instead of one-by-one, for faster indexing.
- **Broad query detection** — Keyword heuristics detect summary/overview queries and retrieve entire files instead of top-k.
- **Streaming output** — Answers stream token-by-token for responsive UX on slow hardware.
- **Grounded answers** — The prompt instructs the model to answer only from context and say "NOT FOUND" otherwise.

## Limitations

- **Linear search** — Every query scans all chunks. Fine for small-to-medium collections, slow for tens of thousands of chunks.
- **No document updates** — If a file changes, delete `index_nomic.json` and re-index.
- **Single-user** — Designed for personal use on one device.
- **Context window limits** — Very large files may exceed the model's context window even with the 25-chunk cap for broad queries.

## License

This project is provided as-is for personal and educational use.

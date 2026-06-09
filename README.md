# PhoneRAG

A lightweight Retrieval-Augmented Generation (RAG) system designed to run entirely on an Android phone using [Termux](https://termux.dev) and Google's [LiteRT-LM](https://ai.google.dev/edge/litert) with the Gemma 4 E2B model. No cloud APIs, no heavy frameworks — just Python, LiteRT, and on-device inference.

Upload documents, build a keyword-based index, and ask questions — all from a mobile-friendly web UI or the command line.

## How It Works

```
                          +-----------------+
                          |  docs/*.txt/pdf |
                          +--------+--------+
                                   |
                          chunk + compute BM25 frequencies
                                   |
                                   v
                          +--------+--------+
                          | index_bm25.json |
                          | (keyword store) |
                          +--------+--------+
                                   |
          user query -----> BM25 term matching
                                   |
                                   v
                             top-k chunks
                                   |
                                   v
                     +-------------+-------------+
                     | Gemma 4 generates answer  |
                     | grounded in retrieved text|
                     | using LiteRT-LM Engine    |
                     +-----------------------------+
```

### Pipeline

1. **Ingest** — Reads documents from `~/phone-rag/docs/`, splits text into overlapping chunks (768 chars max, 96-char overlap) using recursive semantic splitting (paragraphs > sentences > clauses > words), computes BM25 term frequencies, and stores everything in `index_bm25.json`. (No embedding model required!)

2. **Retrieve** — Scores every chunk against the user's query using standard BM25 ranking. Broad queries like "summarize" or "overview" bypass BM25 and retrieve all chunks from the target file instead of top-k.

3. **Generate** — Feeds retrieved chunks into a prompt and streams the answer using the LiteRT-LM engine running Gemma 4 E2B. If no chunk scores above the minimum threshold, it returns `NOT FOUND IN DOCUMENTS`.

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

### Step 3: Get the Model

Download the Gemma 4 E2B model formatted for LiteRT (`gemma-4-E2B-it.litertlm`) into a local directory (e.g., `~/models/`).

### Step 4: Clone and Set Up

```bash
git clone https://github.com/shanptom/PhoneRAG.git ~/phone-rag
cd ~/phone-rag
pip install litert-lm-api flask pypdf --break-system-packages
```

### Step 5: Add Your Documents

Place `.txt` or `.pdf` files into the `docs/` folder:

```bash
cp ~/storage/downloads/my-notes.txt ~/phone-rag/docs/
cp ~/storage/downloads/report.pdf ~/phone-rag/docs/
```

### Step 6: Launch the Web UI

```bash
python3 app.py
```

Open your phone's browser and go to `http://localhost:5000`.

You'll see three tabs (Docs drawer handles the Indexing):
- **Ask Docs (RAG)** — Ask questions grounded in your documents
- **Free Chat** — Talk directly to the AI (no document context)
- **My Files** — Upload files and build the search index

### Step 7: Build Your Index

1. Tap **My Files** to open the drawer
2. Tap the upload area to add files (or you can put them in `docs/` manually)
3. Tap **Prepare Files for Search** and wait for it to finish — you'll see progress as each file is chunked and indexed

### Step 8: Ask Questions

Type a question. The system will:
1. Find the most relevant chunks from your documents using BM25
2. Show you which sources it found
3. Stream an answer grounded in those sources

Try broad queries too — "summarize meeting.txt" will pull all chunks from that file.

## CLI Usage

Prefer the terminal? The CLI tools are in the `cli/` folder:

```bash
# Build the index
python3 cli/build_index.py

# Ask a question (RAG)
python3 cli/ask.py
```

## Project Structure

```
phone-rag/
├── app.py                      # Flask web UI (RAG + Chat + Index + Upload)
├── cli/
│   ├── build_index.py          # CLI: chunk and build BM25 index
│   └── ask.py                  # CLI: retrieve + generate via LiteRT
├── legacy/                     # Older Ollama/embeddinggemma-based scripts
├── docs/                       # Your source documents (.txt, .pdf)
├── index_bm25.json             # Keyword index (auto-generated)
└── README.md
```

## Configuration

All configuration is via constants at the top of `app.py` (or the CLI scripts):

| Parameter | Default | Description |
|---|---|---|
| `MODEL_PATH` | `/root/models/gemma-4-E2B-it.litertlm` | Path to your LiteRT model file |
| `CHUNK_MAX` | `768` | Max characters per chunk |
| `CHUNK_OVERLAP` | `96` | Character overlap between chunks |
| `MIN_SCORE` | `2.0` | Minimum BM25 score to attempt generation |
| `MAX_BROAD_CHUNKS` | `25` | Max chunks for broad/summary queries |

### Memory Management

Tuned for low-RAM devices:
- By moving to BM25 retrieval, we eliminated the need to keep a separate Embedding model in memory.
- `litert-lm-api` directly handles prompt templating and KV-cache management optimally for edge devices.

## Design Decisions

- **Minimal Heavy Dependencies** — Only uses `litert-lm-api` for inference, `pypdf` for documents, and `flask` for UI. No LangChain, no FAISS, no ChromaDB.
- **JSON keyword store** — Simple, portable, inspectable. Adequate for personal document collections.
- **Pure BM25 retrieval** — Eliminates the need to download or run a second model just for embeddings.
- **Pre-compiled regex** — Avoids redundant work on hot paths.
- **Broad query detection** — Keyword heuristics detect summary/overview queries and retrieve entire files instead of top-k.
- **Streaming output** — Answers stream token-by-token for responsive UX on slow hardware. The model is prompted for plain prose and the UI renders plain text (no external Markdown library), so nothing is fetched from the network.
- **Grounded answers** — The prompt instructs the model to answer only from context and say "NOT FOUND" otherwise.

## Limitations

- **Linear search** — Every query scores all chunks via BM25. Fine for small-to-medium collections, slow for tens of thousands of chunks.
- **No document updates** — If a file changes, delete `index_bm25.json` and re-index.
- **Single-user** — Designed for personal use on one device.
- **Context window limits** — Very large files may exceed the model's context window even with the 25-chunk cap for broad queries.

## License

This project is provided as-is for personal and educational use.

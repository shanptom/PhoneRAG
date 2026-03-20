# phone-rag

A lightweight Retrieval-Augmented Generation (RAG) system designed to run entirely on an Android phone using [Termux](https://termux.dev) and [Ollama](https://ollama.com). No cloud APIs, no heavy frameworks — just Python standard library and local LLMs.

Drop your `.txt` or `.pdf` files into a folder, build an index, and ask questions against your documents from the command line.

## How It Works

```
                          +-----------------+
                          |  docs/*.txt/pdf |
                          +--------+--------+
                                   |
                           build_index.py
                        (chunk + embed + store)
                                   |
                                   v
                          +--------+--------+
                          |   index.json    |
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

1. **Ingest** (`build_index.py`) — Reads documents from `~/phone-rag/docs/`, splits text into overlapping chunks of 5 sentences (with 2-sentence overlap), generates embeddings via Ollama, and writes everything to `index.json`.

2. **Retrieve** — Embeds the user's query with the same model, then scores every chunk using a hybrid of:
   - **Semantic**: Cosine similarity between query and chunk embeddings
   - **Lexical**: Jaccard keyword overlap (scaled to 0.0–0.15) as a tie-breaking boost

3. **Generate** (`ask.py`) — Feeds the top-3 retrieved chunks into a prompt and streams the answer from a local chat model. If no chunk scores above the minimum threshold (0.28), it returns `NOT FOUND IN DOCUMENTS` instead of hallucinating.

## Requirements

- **Termux** (or any Linux environment)
- **Python 3.8+**
- **Ollama** running locally on port `11434`
- **pypdf** (`pip install pypdf`) — only needed if indexing PDF files

### Ollama Models

Pull the required models before first use:

```bash
ollama pull embeddinggemma    # embedding model (used for indexing + retrieval)
ollama pull qwen3:0.6b        # chat model (used for answer generation)
```

These are small models chosen to run comfortably within mobile hardware constraints.

## Setup

```bash
# Clone the repo
git clone <repo-url> ~/phone-rag
cd ~/phone-rag

# Install PDF support (optional, only needed for .pdf files)
pip install pypdf

# Make sure Ollama is running
ollama serve &
```

## Usage

### 1. Add Documents

Place your `.txt` and/or `.pdf` files into the `docs/` directory:

```bash
cp my-notes.txt ~/phone-rag/docs/
cp report.pdf ~/phone-rag/docs/
```

### 2. Build the Index

```bash
python build_index.py
```

Output:

```
Indexed: my-notes.txt | chunk 1 | dims: 768
Indexed: my-notes.txt | chunk 2 | dims: 768
Indexed: report.pdf | chunk 1 | dims: 768
...
Saved index to: /data/data/com.termux/files/home/phone-rag/index.json
New chunks indexed: 12
Total chunks in index: 12
```

The index is **incremental** — running `build_index.py` again will skip files that have already been indexed. To re-index a file, remove its entries from `index.json` or delete the file and re-add it.

### 3. Ask a Question (Full RAG)

```bash
python ask.py
```

```
Question: What is the main conclusion of the report?

Retrieved context:

report.pdf | chunk 3 | final=0.6821 | emb=0.6512 | lex=0.0309
report.pdf | chunk 7 | final=0.5943 | emb=0.5780 | lex=0.0163
report.pdf | chunk 1 | final=0.5201 | emb=0.5101 | lex=0.0100
----------------------------------------

Answer:

The main conclusion is that...
```

### 4. Search Only (No Generation)

To retrieve matching chunks without generating an answer:

```bash
python search.py
```

```
Query: logging

Top matches:

notes.txt | score=0.5432
The application uses structured logging with JSON output...
----------------------------------------
```

## Project Structure

```
phone-rag/
├── build_index.py   # Document ingestion: load, chunk, embed, store
├── search.py        # Semantic search (retrieval only, no LLM)
├── ask.py           # Full RAG pipeline: retrieve + generate
├── index.json       # Vector index (auto-generated)
├── docs/            # Place your source documents here
│   ├── *.txt
│   └── *.pdf
└── README.md
```

## Configuration

All configuration is done via constants at the top of each script:

| Parameter | File | Default | Description |
|---|---|---|---|
| `CHUNK_SIZE` | `build_index.py` | `5` | Number of sentences per chunk |
| `OVERLAP` | `build_index.py` | `2` | Overlapping sentences between adjacent chunks |
| `EMBED_MODEL` | `ask.py` | `embeddinggemma` | Ollama model for embedding |
| `CHAT_MODEL` | `ask.py` | `qwen3:0.6b` | Ollama model for answer generation |
| `MIN_SCORE` | `ask.py` | `0.28` | Minimum retrieval score to attempt generation |
| `EMBED_KEEP_ALIVE` | `ask.py` | `0` | Seconds to keep embedding model loaded (0 = unload immediately to free RAM) |
| `CHAT_KEEP_ALIVE` | `ask.py` | `3600` | Seconds to keep chat model loaded (reuse across queries) |

### Memory Management

The keep-alive settings are tuned for low-RAM devices:

- The **embedding model** is unloaded immediately after use (`keep_alive: 0`) to free memory for the chat model.
- The **chat model** stays loaded for 1 hour (`keep_alive: 3600`) so subsequent queries are fast.

## Design Decisions

- **Zero heavy dependencies** — Uses only Python standard library (`json`, `math`, `re`, `pathlib`, `urllib`) plus `pypdf` for PDF support. No LangChain, no FAISS, no ChromaDB.
- **JSON as vector store** — Simple, portable, and inspectable. Adequate for personal document collections (hundreds to low thousands of chunks).
- **Hybrid retrieval** — Pure semantic search can miss exact keyword matches; the Jaccard keyword boost helps surface chunks that contain the user's exact terms.
- **Pre-computed norms** — Stored alongside embeddings to avoid redundant computation during search.
- **Streaming output** — Answers are streamed token-by-token from Ollama for responsive UX on slower hardware.
- **Deterministic generation** — Temperature is set to 0 to produce consistent, reproducible answers.
- **Grounded answers** — The prompt instructs the model to answer only from the provided context and say "NOT FOUND" otherwise, reducing hallucination.

## Limitations

- **Linear search** — Every query scans all chunks. This is fine for small-to-medium collections but will slow down with tens of thousands of chunks. A future improvement could add approximate nearest neighbor (ANN) indexing.
- **Sentence-based chunking** — The splitter relies on `.!?` punctuation. Documents without standard sentence boundaries (e.g., tables, lists, code) may chunk poorly.
- **No document updates** — If a file's content changes, you must manually remove its old entries from `index.json` before re-indexing.
- **Single-user, single-device** — Designed for personal use on one machine.

## License

This project is provided as-is for personal and educational use.

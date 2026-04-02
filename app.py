import json
import math
import re
from pathlib import Path
from urllib.request import Request, urlopen

from pypdf import PdfReader
from flask import Flask, render_template_string, request, Response

# ── Config ──────────────────────────────────────────────────────────
DOCS_DIR = Path.home() / "phone-rag" / "docs"
INDEX_FILE = Path.home() / "phone-rag" / "index_bm25.json"
OLLAMA_BASE = "http://localhost:11434"
GENERATE_URL = OLLAMA_BASE + "/api/generate"
CHAT_URL = OLLAMA_BASE + "/api/chat"
TAGS_URL = OLLAMA_BASE + "/api/tags"

CHAT_MODEL = "gemma3:1b"
MIN_SCORE = 2.0  # BM25 scores are typically 0-30+; tune as needed
CHAT_KEEP_ALIVE = 3600

# ── BM25 parameters ──
BM25_K1 = 1.5   # term frequency saturation
BM25_B = 0.75   # length normalization

# ── Chunking config ──
CHUNK_MAX = 768
CHUNK_OVERLAP = 96

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_SEMICOLON_RE = re.compile(r"(?<=[;:])\s+")
_COMMA_RE = re.compile(r"(?<=[,])\s+")

SEPARATORS = ["\n\n", "\n", _SENTENCE_RE, _SEMICOLON_RE, _COMMA_RE, " "]


def _split_once(text, separators):
    for sep in separators:
        if isinstance(sep, str):
            parts = text.split(sep)
        else:
            parts = sep.split(text)
        parts = [p for p in parts if p.strip()]
        if len(parts) > 1:
            return parts, sep
    return [text], None


def _recursive_split(text, separators, max_size):
    if len(text) <= max_size:
        return [text]
    parts, used_sep = _split_once(text, separators)
    if len(parts) == 1:
        return parts
    if used_sep in separators:
        remaining_seps = separators[separators.index(used_sep):]
    else:
        remaining_seps = separators
    chunks = []
    for part in parts:
        if len(part) <= max_size:
            chunks.append(part)
        else:
            finer = remaining_seps[1:] if len(remaining_seps) > 1 else remaining_seps
            chunks.extend(_recursive_split(part, finer, max_size))
    return chunks


def _merge_small_chunks(pieces, max_size, overlap):
    if not pieces:
        return []
    chunks = []
    current = pieces[0]
    for piece in pieces[1:]:
        combined = current + " " + piece
        if len(combined) <= max_size:
            current = combined
        else:
            chunks.append(current.strip())
            if overlap > 0 and len(current) > overlap:
                tail = current[-overlap:]
                space_idx = tail.find(" ")
                if space_idx != -1:
                    tail = tail[space_idx + 1:]
                current = tail + " " + piece
            else:
                current = piece
    if current.strip():
        chunks.append(current.strip())
    return chunks


_HEADING_RE = re.compile(
    r"^(?:#{1,6}\s+.+|[A-Z][^\n]{0,80})$"
)


def _is_heading(line):
    """Detect lines that look like headings: markdown # or short uppercase-start lines without ending punctuation."""
    line = line.strip()
    if not line or len(line) > 120:
        return False
    if line.startswith("#"):
        return True
    # Short line, starts with uppercase, no ending sentence punctuation
    if len(line) < 80 and line[0].isupper() and line[-1] not in ".!?,;:)\"'":
        return True
    return False


def chunk_text(text: str):
    text = text.strip()
    if not text:
        return []

    # Extract heading context per paragraph
    lines = text.split("\n")
    current_heading = ""
    sections = []  # list of (heading, paragraph_text)
    buf = []

    for line in lines:
        if _is_heading(line) and not buf:
            current_heading = line.strip().lstrip("# ").strip()
        elif _is_heading(line) and buf:
            sections.append((current_heading, "\n".join(buf)))
            buf = []
            current_heading = line.strip().lstrip("# ").strip()
        else:
            buf.append(line)

    if buf:
        sections.append((current_heading, "\n".join(buf)))

    # Chunk each section and prepend heading
    all_chunks = []
    for heading, section_text in sections:
        section_text = section_text.strip()
        if not section_text:
            continue
        pieces = _recursive_split(section_text, SEPARATORS, CHUNK_MAX)
        merged = _merge_small_chunks(pieces, CHUNK_MAX, CHUNK_OVERLAP)
        for chunk in merged:
            if not chunk:
                continue
            if heading:
                all_chunks.append(f"[{heading}] {chunk}")
            else:
                all_chunks.append(chunk)

    return all_chunks


# ── Porter stemmer (lightweight, no deps) ─────────────────────────

def _porter_stem(word):
    """Minimal Porter stemmer — handles the most common English suffixes."""
    if len(word) <= 2:
        return word
    # Step-like suffix stripping, ordered longest-first
    for suffix, replacement in (
        ("ational", "ate"), ("tional", "tion"), ("enci", "ence"),
        ("anci", "ance"), ("izer", "ize"), ("alli", "al"),
        ("entli", "ent"), ("eli", "e"), ("ousli", "ous"),
        ("ization", "ize"), ("ation", "ate"), ("ator", "ate"),
        ("alism", "al"), ("iveness", "ive"), ("fulness", "ful"),
        ("ousness", "ous"), ("aliti", "al"), ("iviti", "ive"),
        ("biliti", "ble"),
    ):
        if word.endswith(suffix):
            stem = word[: -len(suffix)] + replacement
            if len(stem) > 2:
                return stem
    # Common endings
    for suffix, replacement in (
        ("ement", ""), ("ment", ""), ("ness", ""), ("ence", ""),
        ("ance", ""), ("able", ""), ("ible", ""), ("ling", ""),
        ("ting", ""), ("ing", ""), ("tion", ""), ("sion", ""),
        ("ies", "i"), ("ied", "i"),
        ("ses", "s"), ("eed", "ee"),
        ("ed", ""), ("ly", ""), ("er", ""), ("es", ""),
        ("s", ""),
    ):
        if word.endswith(suffix) and len(word) - len(suffix) > 2:
            return word[: -len(suffix)] + replacement
    return word


# ── BM25 tokenizer ────────────────────────────────────────────────

_TOKENIZE_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "the", "a", "an", "is", "in", "of", "to", "and", "or", "what", "how",
    "does", "do", "it", "its", "about", "from", "would", "could", "should",
    "very", "just", "been", "have", "has", "be", "are", "was", "were",
    "this", "that", "with", "for", "not", "but", "can", "will", "than",
}


def tokenize(text):
    """Tokenize, filter stopwords, and stem. Returns list to preserve TF."""
    return [_porter_stem(w) for w in _TOKENIZE_RE.findall(text.lower()) if w not in STOPWORDS]


def term_freq(tokens):
    """Count term frequencies from a token list."""
    tf = {}
    for t in tokens:
        tf[t] = tf.get(t, 0) + 1
    return tf


# ── BM25 index ────────────────────────────────────────────────────

def build_bm25_index(records):
    """Compute IDF and avg_dl from records. Returns (idf, avg_dl)."""
    n = len(records)
    if n == 0:
        return {}, 0
    # Count how many docs each term appears in
    df = {}
    total_dl = 0
    for rec in records:
        tf = rec.get("tf", {})
        total_dl += rec.get("dl", 0)
        for term in tf:
            df[term] = df.get(term, 0) + 1
    avg_dl = total_dl / n
    # IDF: log((N - df + 0.5) / (df + 0.5) + 1)
    idf = {}
    for term, freq in df.items():
        idf[term] = math.log((n - freq + 0.5) / (freq + 0.5) + 1)
    return idf, avg_dl


def bm25_score(query_tokens, rec, idf, avg_dl):
    """Score a single record against query tokens using BM25."""
    tf = rec.get("tf", {})
    dl = rec.get("dl", 0)
    score = 0.0
    for qt in set(query_tokens):
        if qt not in idf:
            continue
        f = tf.get(qt, 0)
        numerator = f * (BM25_K1 + 1)
        denominator = f + BM25_K1 * (1 - BM25_B + BM25_B * dl / avg_dl) if avg_dl > 0 else f + BM25_K1
        score += idf[qt] * numerator / denominator
    return score


# ── Broad query & retrieval ──────────────────────────────────────

BROAD_KEYWORDS = {
    "summarize", "summary", "summarise", "overview", "outline",
    "main points", "key points", "recap", "brief", "briefing",
    "explain the file", "explain the document", "tell me about",
    "what is this about", "what does it say", "entire", "whole",
    "all about", "full",
}

MAX_BROAD_CHUNKS = 25


def is_broad_query(query):
    q_lower = query.lower()
    return any(kw in q_lower for kw in BROAD_KEYWORDS)


def find_target_file(query, records):
    q_lower = query.lower()
    known_files = {r["file"] for r in records}
    for fname in known_files:
        name_no_ext = Path(fname).stem.replace("_", " ").replace("-", " ").lower()
        if fname.lower() in q_lower or name_no_ext in q_lower:
            return fname
    return None


def retrieve_broad(query, records, idf, avg_dl):
    target = find_target_file(query, records)
    if target:
        file_chunks = sorted(
            [r for r in records if r["file"] == target],
            key=lambda r: r.get("chunk_id", 0),
        )[:MAX_BROAD_CHUNKS]
        return [(1.0, r) for r in file_chunks], target

    # No filename match — use BM25 to find the best file
    query_tokens = tokenize(query)
    scored = [(bm25_score(query_tokens, r, idf, avg_dl), r) for r in records]
    scored.sort(reverse=True, key=lambda x: x[0])
    best_file = scored[0][1]["file"]
    file_chunks = sorted(
        [r for r in records if r["file"] == best_file],
        key=lambda r: r.get("chunk_id", 0),
    )[:MAX_BROAD_CHUNKS]
    return [(1.0, r) for r in file_chunks], best_file


def retrieve(query, records, idf, avg_dl, top_k=3):
    query_tokens = tokenize(query)
    scored = [(bm25_score(query_tokens, r, idf, avg_dl), r) for r in records]
    scored.sort(reverse=True, key=lambda x: x[0])
    # Drop chunks scoring below 25% of the top score to filter noise
    top = scored[:top_k]
    if top:
        cutoff = top[0][0] * 0.25
        top = [(s, r) for s, r in top if s >= cutoff]
    return top


def build_prompt(query, matches, broad=False):
    context = "\n\n".join(
        f"[Source: {rec['file']} | chunk={rec.get('chunk_id', '?')}]\n{rec['text']}"
        for _, rec in matches
    )
    if broad:
        return (
            f"Context:\n{context}\n\n"
            "Using ONLY the context above, provide a concise summary that covers all the main points. "
            "Organize by topic if appropriate. If the context is insufficient, say so.\n\n"
            f"Question: {query}\nAnswer:"
        )
    return (
        f"Context:\n{context}\n\n"
        "Using ONLY the context above, answer the question. You may combine facts from multiple passages. "
        "Be thorough — explain your answer with relevant details from the context. "
        "If the answer is not in the context, write only: NOT FOUND\n\n"
        f"Question: {query}\nAnswer:"
    )


# ── Flask app ──────────────────────────────────────────────────────

app = Flask(__name__)

# Load index once at startup
_records = []
_idf = {}
_avg_dl = 0


def get_index():
    global _records, _idf, _avg_dl
    if not _records and INDEX_FILE.exists():
        _records = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
        _idf, _avg_dl = build_bm25_index(_records)
    return _records, _idf, _avg_dl


HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="theme-color" content="#0a0a0a">
<title>DocQ</title>
<style>
  :root {
    --bg: #0a0a0a;
    --bg-raised: #141414;
    --bg-input: #1a1a1a;
    --border: #1e1e1e;
    --border-focus: #2d8a54;
    --accent: #22c55e;
    --accent-dim: #166534;
    --accent-glow: rgba(34,197,94,0.08);
    --text: #e8e8e8;
    --text-dim: #777;
    --text-faint: #444;
    --danger: #ef4444;
    --warn: #eab308;
    --radius: 12px;
    --radius-lg: 16px;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
    background: var(--bg); color: var(--text);
    height: 100dvh; display: flex; flex-direction: column; overflow: hidden;
    -webkit-font-smoothing: antialiased;
  }

  /* ── Header ── */
  header {
    padding: 14px 16px 10px; text-align: center;
    background: linear-gradient(180deg, #111 0%, var(--bg) 100%);
    border-bottom: 1px solid var(--border);
  }
  .logo { display: flex; align-items: center; justify-content: center; gap: 8px; margin-bottom: 10px; }
  .logo h1 {
    font-size: 1.15rem; font-weight: 700; letter-spacing: -0.02em;
    background: linear-gradient(135deg, var(--accent) 0%, #86efac 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }
  .logo .dot {
    width: 7px; height: 7px; border-radius: 50%; background: var(--accent);
    box-shadow: 0 0 8px var(--accent);
    animation: pulse 2s ease-in-out infinite;
  }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }

  /* ── Tabs ── */
  .tabs { display: flex; gap: 4px; justify-content: center; }
  .tabs button {
    padding: 7px 18px; border: 1px solid var(--border); background: transparent;
    color: var(--text-dim); font-size: 0.82rem; font-weight: 500;
    cursor: pointer; transition: all 0.2s; border-radius: var(--radius);
    letter-spacing: 0.02em; text-transform: uppercase;
  }
  .tabs button:hover { color: var(--text); background: var(--bg-raised); }
  .tabs button.active {
    background: var(--accent-dim); color: #fff;
    border-color: var(--accent-dim);
    box-shadow: 0 0 12px rgba(34,197,94,0.15);
  }

  /* ── Tab content ── */
  .tab-content { display: none; }
  .tab-content.active { display: flex; flex-direction: column; flex: 1; min-height: 0; }

  /* ── Docs drawer (slide-up panel) ── */
  .drawer-overlay {
    display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.5);
    z-index: 90; transition: opacity 0.3s;
  }
  .drawer-overlay.open { display: block; }
  .drawer {
    position: fixed; bottom: 0; left: 0; right: 0;
    background: var(--bg); border-top: 1px solid #252525;
    border-radius: 18px 18px 0 0; z-index: 100;
    max-height: 80dvh; overflow-y: auto;
    transform: translateY(100%); transition: transform 0.3s ease;
    padding: 0 16px 20px;
  }
  .drawer.open { transform: translateY(0); }
  .drawer-handle {
    display: flex; justify-content: center; padding: 10px 0 6px;
    cursor: pointer; position: sticky; top: 0; background: var(--bg); z-index: 1;
  }
  .drawer-handle::after {
    content: ""; width: 36px; height: 4px; border-radius: 2px;
    background: #333;
  }
  .drawer-header {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 14px; padding-top: 4px;
  }
  .drawer-header h2 {
    font-size: 1rem; font-weight: 600; color: var(--text);
  }
  .drawer-close {
    width: 30px; height: 30px; border-radius: 50%;
    background: var(--bg-raised); border: 1px solid var(--border);
    color: var(--text-dim); font-size: 1.1rem; cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    transition: all 0.2s; line-height: 1;
  }
  .drawer-close:hover { color: var(--text); border-color: #333; }

  /* ── Model bar ── */
  .model-bar {
    display: flex; align-items: center; justify-content: center; gap: 8px;
    padding: 8px 16px; border-bottom: 1px solid var(--border);
  }
  .model-bar label { font-size: 0.78rem; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.05em; }
  .model-bar select {
    padding: 6px 10px; border-radius: 8px; border: 1px solid var(--border);
    background: var(--bg-input); color: var(--text); font-size: 0.83rem;
    outline: none; max-width: 200px; transition: border-color 0.2s;
  }
  .model-bar select:focus { border-color: var(--border-focus); }
  .btn-ghost {
    padding: 5px 12px; font-size: 0.78rem; background: var(--bg-raised);
    border: 1px solid var(--border); border-radius: 8px; color: var(--text-dim);
    cursor: pointer; transition: all 0.2s;
  }
  .btn-ghost:hover { color: var(--text); border-color: #333; }

  /* ── Chat area (shared) ── */
  .chat-area {
    flex: 1; overflow-y: auto; padding: 16px;
    display: flex; flex-direction: column; gap: 10px;
  }
  .msg {
    max-width: 85%; padding: 10px 14px; border-radius: var(--radius-lg);
    line-height: 1.55; font-size: 0.92rem; white-space: pre-wrap;
    word-wrap: break-word; animation: fadeIn 0.2s ease;
  }
  @keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
  .msg.user {
    align-self: flex-end;
    background: linear-gradient(135deg, var(--accent-dim) 0%, #15803d 100%);
    border-bottom-right-radius: 4px; color: #fff;
  }
  .msg.bot {
    align-self: flex-start;
    background: var(--bg-raised); border: 1px solid var(--border);
    border-bottom-left-radius: 4px;
  }
  .msg.bot .sources {
    font-size: 0.72rem; color: var(--text-dim); margin-bottom: 6px;
    padding-bottom: 6px; border-bottom: 1px solid var(--border);
    letter-spacing: 0.01em;
  }
  .msg.error {
    background: rgba(239,68,68,0.1); border: 1px solid rgba(239,68,68,0.2);
    color: #fca5a5;
  }

  /* ── Input bar ── */
  .input-bar {
    display: flex; gap: 8px; padding: 10px 14px;
    border-top: 1px solid var(--border); background: var(--bg);
  }
  .input-bar input {
    flex: 1; padding: 11px 14px; border-radius: var(--radius); border: 1px solid var(--border);
    background: var(--bg-input); color: var(--text); font-size: 0.95rem; outline: none;
    transition: border-color 0.2s, box-shadow 0.2s;
  }
  .input-bar input:focus {
    border-color: var(--border-focus);
    box-shadow: 0 0 0 3px var(--accent-glow);
  }
  .btn-send {
    padding: 10px 16px; border-radius: var(--radius); border: none;
    background: linear-gradient(135deg, var(--accent-dim) 0%, #15803d 100%);
    color: #fff; font-size: 0.92rem; font-weight: 600; cursor: pointer;
    transition: opacity 0.2s, transform 0.1s;
  }
  .btn-send:hover { opacity: 0.9; }
  .btn-send:active { transform: scale(0.97); }
  .btn-send:disabled { opacity: 0.35; }

  /* ── Spinner ── */
  .spinner { display: inline-block; }
  .spinner::after {
    content: ""; display: inline-block; width: 14px; height: 14px;
    border: 2px solid rgba(255,255,255,0.3); border-top-color: #fff; border-radius: 50%;
    animation: spin 0.6s linear infinite; vertical-align: middle; margin-left: 4px;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  .section-label {
    font-size: 0.72rem; font-weight: 600; color: var(--text-dim);
    text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px;
  }

  /* Upload */
  .upload-area {
    border: 2px dashed #252525; border-radius: var(--radius-lg); padding: 24px 16px;
    text-align: center; color: var(--text-dim); font-size: 0.88rem; cursor: pointer;
    margin-bottom: 16px; transition: all 0.25s;
    background: var(--bg-raised);
  }
  .upload-area:hover { border-color: #333; color: var(--text); }
  .upload-area.dragover {
    border-color: var(--accent); color: var(--accent);
    background: var(--accent-glow);
  }
  .upload-area input { display: none; }
  .upload-icon { font-size: 1.8rem; margin-bottom: 6px; display: block; }
  .upload-hint { font-size: 0.72rem; color: var(--text-faint); margin-top: 4px; }
  .upload-status {
    font-size: 0.8rem; color: var(--text-dim); margin-bottom: 12px; min-height: 1.2em;
    text-align: center;
  }

  /* Build button */
  .btn-build {
    padding: 12px 18px; border-radius: var(--radius); border: none;
    background: linear-gradient(135deg, var(--accent-dim) 0%, #15803d 100%);
    color: #fff; font-size: 0.95rem; font-weight: 600; cursor: pointer;
    width: 100%; transition: opacity 0.2s, transform 0.1s;
    letter-spacing: 0.01em;
  }
  .btn-build:hover { opacity: 0.9; }
  .btn-build:active { transform: scale(0.98); }
  .btn-build:disabled { opacity: 0.35; }

  /* File list */
  .file-list { margin: 16px 0 0; }
  .file-item {
    display: flex; align-items: center; gap: 10px; padding: 10px 12px;
    background: var(--bg-raised); border: 1px solid var(--border);
    border-radius: var(--radius); margin-bottom: 6px;
    font-size: 0.88rem; transition: border-color 0.2s;
  }
  .file-item:hover { border-color: #2a2a2a; }
  .file-icon { font-size: 1.1rem; }
  .file-item .name { flex: 1; color: var(--text); }
  .file-item .status {
    font-size: 0.7rem; font-weight: 600; padding: 2px 8px;
    border-radius: 6px; text-transform: uppercase; letter-spacing: 0.04em;
  }
  .file-item .status.ready { background: rgba(34,197,94,0.12); color: var(--accent); }
  .file-item .status.new { background: rgba(234,179,8,0.12); color: var(--warn); }

  /* Build log */
  #build-log {
    margin-top: 12px; padding: 12px; background: #0c0c0c;
    border: 1px solid var(--border); border-radius: var(--radius);
    font-family: "SF Mono", "Fira Code", "Consolas", monospace;
    font-size: 0.78rem; color: var(--text-dim);
    max-height: 250px; overflow-y: auto; white-space: pre-wrap;
    line-height: 1.6;
  }

  /* ── Scrollbar ── */
  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: #222; border-radius: 4px; }
  ::-webkit-scrollbar-thumb:hover { background: #333; }

  /* ── Empty state ── */
  .empty-state {
    flex: 1; display: flex; flex-direction: column; align-items: center;
    justify-content: center; color: var(--text-faint); text-align: center;
    padding: 40px 20px; gap: 8px;
  }
  .empty-state .icon { font-size: 2.5rem; margin-bottom: 4px; }
  .empty-state .title { font-size: 0.95rem; color: var(--text-dim); font-weight: 500; }
  .empty-state .hint { font-size: 0.8rem; }
</style>
</head>
<body>
<header>
  <div class="logo">
    <span class="dot"></span>
    <h1>DocQ</h1>
  </div>
  <div class="tabs">
    <button class="active" onclick="switchTab('ask')">Ask Docs</button>
    <button onclick="switchTab('chat')">Free Chat</button>
  </div>
</header>

<!-- RAG Tab -->
<div id="tab-ask" class="tab-content active">
  <div class="model-bar">
    <label for="model-select">Model</label>
    <select id="model-select"><option>loading...</option></select>
    <button class="btn-ghost" onclick="toggleDrawer()" id="docs-btn">My Files</button>
  </div>
  <div id="chat" class="chat-area">
    <div class="empty-state" id="rag-empty">
      <span class="icon">&#128218;</span>
      <span class="title">Ask your files anything</span>
      <span class="hint">Get answers based on your uploaded documents</span>
    </div>
  </div>
  <form id="form" class="input-bar">
    <input type="text" id="q" placeholder="Ask about your files..." autocomplete="off" autofocus>
    <button id="btn" type="submit" class="btn-send">&#9654;</button>
  </form>
</div>

<!-- Chat Tab -->
<div id="tab-chat" class="tab-content">
  <div class="model-bar">
    <label for="chat-model-select">Model</label>
    <select id="chat-model-select"></select>
    <button class="btn-ghost" onclick="clearChat()">Clear</button>
  </div>
  <div id="direct-chat" class="chat-area">
    <div class="empty-state" id="chat-empty">
      <span class="icon">&#128172;</span>
      <span class="title">Open conversation</span>
      <span class="hint">Chat freely — no documents needed</span>
    </div>
  </div>
  <form id="chat-form" class="input-bar">
    <input type="text" id="chat-q" placeholder="Type a message..." autocomplete="off">
    <button id="chat-btn" type="submit" class="btn-send">&#9654;</button>
  </form>
</div>

<!-- Docs Drawer -->
<div class="drawer-overlay" id="drawer-overlay" onclick="toggleDrawer()"></div>
<div class="drawer" id="drawer">
  <div class="drawer-handle" onclick="toggleDrawer()"></div>
  <div class="drawer-header">
    <h2>My Files</h2>
    <button class="drawer-close" onclick="toggleDrawer()">&#10005;</button>
  </div>
  <div class="upload-area" id="upload-area" onclick="document.getElementById('file-input').click();">
    <span class="upload-icon">&#128206;</span>
    Tap to add files or drag &amp; drop
    <span class="upload-hint">Supports .txt and .pdf files</span>
    <input type="file" id="file-input" multiple accept=".txt,.pdf">
  </div>
  <div class="upload-status" id="upload-status"></div>
  <button id="build-btn" class="btn-build" onclick="startBuild()">Prepare Files for Search</button>
  <div class="section-label" style="margin-top:16px;">Your files</div>
  <div id="file-list" class="file-list"></div>
  <div id="build-log" style="display:none;"></div>
</div>
<script>
const chat = document.getElementById("chat");
const form = document.getElementById("form");
const qInput = document.getElementById("q");
const btn = document.getElementById("btn");
const modelSelect = document.getElementById("model-select");

const chatModelSelect = document.getElementById("chat-model-select");

// Fetch available models on load — populate both selectors
fetch("/models").then(r => r.json()).then(data => {
  [modelSelect, chatModelSelect].forEach(sel => {
    sel.innerHTML = "";
    for (const name of data.models) {
      const opt = document.createElement("option");
      opt.value = name; opt.textContent = name;
      if (name === data.default) opt.selected = true;
      sel.appendChild(opt);
    }
  });
}).catch(() => {
  [modelSelect, chatModelSelect].forEach(sel => {
    sel.innerHTML = '<option>gemma3:1b</option>';
  });
});

function addMsg(cls, html) {
  const empty = document.getElementById("rag-empty");
  if (empty) empty.remove();
  const d = document.createElement("div");
  d.className = "msg " + cls;
  d.innerHTML = html;
  chat.appendChild(d);
  chat.scrollTop = chat.scrollHeight;
  return d;
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = qInput.value.trim();
  if (!q) return;
  qInput.value = "";
  addMsg("user", esc(q));
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>';

  const botDiv = addMsg("bot", '<div class="sources"></div><div class="answer"></div>');
  const srcEl = botDiv.querySelector(".sources");
  const ansEl = botDiv.querySelector(".answer");

  try {
    const resp = await fetch("/ask?q=" + encodeURIComponent(q) + "&model=" + encodeURIComponent(modelSelect.value));
    if (!resp.ok) throw new Error("Server error " + resp.status);
    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const lines = buf.split("\\n");
      buf = lines.pop();
      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const payload = JSON.parse(line.slice(6));
        if (payload.type === "sources") {
          srcEl.textContent = payload.text;
        } else if (payload.type === "token") {
          ansEl.textContent += payload.text;
          chat.scrollTop = chat.scrollHeight;
        } else if (payload.type === "error") {
          ansEl.innerHTML = '<span style="color:#f88">' + esc(payload.text) + '</span>';
        }
      }
    }
  } catch (err) {
    addMsg("error", esc("Error: " + err.message));
  }
  btn.disabled = false;
  btn.innerHTML = "&#9654;";
});

function esc(s) {
  return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}

// ── Tab switching ──
function switchTab(name) {
  document.querySelectorAll(".tab-content").forEach(el => el.classList.remove("active"));
  document.querySelectorAll(".tabs button").forEach(el => el.classList.remove("active"));
  document.getElementById("tab-" + name).classList.add("active");
  document.querySelector(`.tabs button[onclick="switchTab('${name}')"]`).classList.add("active");
}

// ── Docs drawer ──
function toggleDrawer() {
  const drawer = document.getElementById("drawer");
  const overlay = document.getElementById("drawer-overlay");
  const isOpen = drawer.classList.contains("open");
  if (isOpen) {
    drawer.classList.remove("open");
    overlay.classList.remove("open");
  } else {
    drawer.classList.add("open");
    overlay.classList.add("open");
    loadDocs();
  }
}

// ── Index / docs ──
function loadDocs() {
  fetch("/docs").then(r => r.json()).then(data => {
    const el = document.getElementById("file-list");
    if (data.files.length === 0) {
      el.innerHTML = '<div style="color:#888;padding:8px;">No files yet — add some above</div>';
      return;
    }
    el.innerHTML = data.files.map(f => {
      const icon = f.name.endsWith('.pdf') ? '&#128196;' : '&#128209;';
      return `<div class="file-item"><span class="file-icon">${icon}</span><span class="name">${esc(f.name)}</span>` +
        `<span class="status ${f.indexed ? 'ready' : 'new'}">${f.indexed ? 'Ready' : 'New'}</span></div>`;
    }).join("");
  }).catch(err => {
    document.getElementById("file-list").innerHTML = '<div style="color:#f88;">Failed to load docs</div>';
  });
}

function startBuild() {
  const btn = document.getElementById("build-btn");
  const log = document.getElementById("build-log");
  btn.disabled = true;
  btn.innerHTML = 'Preparing<span class="spinner"></span>';
  log.style.display = "block";
  log.textContent = "";

  fetch("/build", { method: "POST" }).then(resp => {
    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    function read() {
      reader.read().then(({ done, value }) => {
        if (done) { finish(); return; }
        buf += dec.decode(value, { stream: true });
        const lines = buf.split("\\n");
        buf = lines.pop();
        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const p = JSON.parse(line.slice(6));
          if (p.type === "status") log.textContent += p.text + "\\n";
          else if (p.type === "progress") log.textContent += `  ${p.file}: ${p.chunks_done}/${p.chunks_total} chunks\\n`;
          else if (p.type === "done") log.textContent += `\\n${p.text} (new: ${p.new}, total: ${p.total})\\n`;
          else if (p.type === "error") log.textContent += `ERROR: ${p.text}\\n`;
          log.scrollTop = log.scrollHeight;
        }
        read();
      });
    }
    read();
  }).catch(err => {
    log.textContent += "Error: " + err.message + "\\n";
    finish();
  });

  function finish() {
    btn.disabled = false;
    btn.textContent = "Prepare Files for Search";
    loadDocs();
  }
}

// ── Direct Chat tab ──
const directChat = document.getElementById("direct-chat");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-q");
const chatBtn = document.getElementById("chat-btn");
let chatHistory = [];

function addChatMsg(role, html) {
  const empty = document.getElementById("chat-empty");
  if (empty) empty.remove();
  const d = document.createElement("div");
  d.className = "msg " + (role === "user" ? "user" : "bot");
  d.innerHTML = html;
  directChat.appendChild(d);
  directChat.scrollTop = directChat.scrollHeight;
  return d;
}

function clearChat() {
  chatHistory = [];
  directChat.innerHTML = "";
}

chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const msg = chatInput.value.trim();
  if (!msg) return;
  chatInput.value = "";
  addChatMsg("user", esc(msg));
  chatHistory.push({ role: "user", content: msg });
  chatBtn.disabled = true;
  chatBtn.innerHTML = '<span class="spinner"></span>';

  const botDiv = addChatMsg("assistant", "");
  let fullResponse = "";

  try {
    const resp = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: chatModelSelect.value,
        messages: chatHistory
      })
    });
    if (!resp.ok) throw new Error("Server error " + resp.status);
    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const lines = buf.split("\\n");
      buf = lines.pop();
      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const payload = JSON.parse(line.slice(6));
        if (payload.type === "token") {
          fullResponse += payload.text;
          botDiv.textContent = fullResponse;
          directChat.scrollTop = directChat.scrollHeight;
        } else if (payload.type === "error") {
          botDiv.innerHTML = '<span style="color:#f88">' + esc(payload.text) + '</span>';
        }
      }
    }
    if (fullResponse) {
      chatHistory.push({ role: "assistant", content: fullResponse });
    }
  } catch (err) {
    addChatMsg("error", esc("Error: " + err.message));
  }
  chatBtn.disabled = false;
  chatBtn.innerHTML = "&#9654;";
});

// ── File upload ──
const uploadArea = document.getElementById("upload-area");
const fileInput = document.getElementById("file-input");
const uploadStatus = document.getElementById("upload-status");

uploadArea.addEventListener("dragover", (e) => {
  e.preventDefault();
  uploadArea.classList.add("dragover");
});
uploadArea.addEventListener("dragleave", () => uploadArea.classList.remove("dragover"));
uploadArea.addEventListener("drop", (e) => {
  e.preventDefault();
  uploadArea.classList.remove("dragover");
  uploadFiles(e.dataTransfer.files);
});
fileInput.addEventListener("change", () => {
  if (fileInput.files.length) uploadFiles(fileInput.files);
  fileInput.value = "";
});

async function uploadFiles(files) {
  const valid = [...files].filter(f => f.name.endsWith(".txt") || f.name.endsWith(".pdf"));
  if (!valid.length) {
    uploadStatus.textContent = "Only .txt and .pdf files are supported.";
    return;
  }
  uploadStatus.textContent = `Uploading ${valid.length} file(s)...`;
  const fd = new FormData();
  valid.forEach(f => fd.append("files", f));

  try {
    const resp = await fetch("/upload", { method: "POST", body: fd });
    const data = await resp.json();
    if (data.error) {
      uploadStatus.innerHTML = '<span style="color:#f88">' + esc(data.error) + '</span>';
    } else {
      uploadStatus.textContent = data.message;
      loadDocs();
    }
  } catch (err) {
    uploadStatus.innerHTML = '<span style="color:#f88">Upload failed: ' + esc(err.message) + '</span>';
  }
}
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/ask")
def ask():
    query = request.args.get("q", "").strip()
    chat_model = request.args.get("model", CHAT_MODEL)
    if not query:
        return Response("data: " + json.dumps({"type": "error", "text": "Empty question."}) + "\n\n",
                        content_type="text/event-stream")

    def generate():
        try:
            records, idf, avg_dl = get_index()
            broad = is_broad_query(query)

            if broad:
                matches, target = retrieve_broad(query, records, idf, avg_dl)
                src_text = f"Broad query — {len(matches)} chunks from: {target}"
            else:
                matches = retrieve(query, records, idf, avg_dl, top_k=5)
                src_lines = [
                    f"{rec['file']} chunk {rec.get('chunk_id', '?')} ({score:.2f})"
                    for score, rec in matches
                ]
                src_text = " | ".join(src_lines)

            yield f"data: {json.dumps({'type': 'sources', 'text': src_text})}\n\n"

            if not broad and matches[0][0] < MIN_SCORE:
                yield f"data: {json.dumps({'type': 'token', 'text': 'NOT FOUND IN DOCUMENTS'})}\n\n"
                return

            prompt = build_prompt(query, matches, broad=broad)
            req = Request(
                GENERATE_URL,
                data=json.dumps({
                    "model": chat_model,
                    "prompt": prompt,
                    "stream": True,
                    "think": False,
                    "keep_alive": CHAT_KEEP_ALIVE,
                    "options": {
                        "temperature": 0,
                        "num_predict": 512 if broad else 256,
                    },
                }).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            with urlopen(req) as resp:
                for raw_line in resp:
                    if not raw_line.strip():
                        continue
                    obj = json.loads(raw_line.decode("utf-8"))
                    if "response" in obj:
                        yield f"data: {json.dumps({'type': 'token', 'text': obj['response']})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'text': str(e)})}\n\n"

    return Response(generate(), content_type="text/event-stream")


@app.route("/chat", methods=["POST"])
def direct_chat():
    """Direct multi-turn chat with an Ollama model (no RAG)."""
    data = request.get_json()
    model = data.get("model", CHAT_MODEL)
    messages = data.get("messages", [])
    if not messages:
        return Response("data: " + json.dumps({"type": "error", "text": "No messages."}) + "\n\n",
                        content_type="text/event-stream")

    def generate():
        try:
            req = Request(
                CHAT_URL,
                data=json.dumps({
                    "model": model,
                    "messages": messages,
                    "stream": True,
                    "keep_alive": CHAT_KEEP_ALIVE,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 512,
                    },
                }).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            with urlopen(req) as resp:
                for raw_line in resp:
                    if not raw_line.strip():
                        continue
                    obj = json.loads(raw_line.decode("utf-8"))
                    content = obj.get("message", {}).get("content", "")
                    if content:
                        yield f"data: {json.dumps({'type': 'token', 'text': content})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'text': str(e)})}\n\n"

    return Response(generate(), content_type="text/event-stream")


@app.route("/reload", methods=["POST"])
def reload_index():
    global _records, _idf, _avg_dl
    _records = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
    _idf, _avg_dl = build_bm25_index(_records)
    return {"status": "ok", "chunks": len(_records)}


ALLOWED_EXTENSIONS = {".txt", ".pdf"}


@app.route("/upload", methods=["POST"])
def upload():
    """Upload files to the docs directory."""
    files = request.files.getlist("files")
    if not files:
        return {"error": "No files provided."}, 400

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    saved = []
    skipped = []
    for f in files:
        ext = Path(f.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            skipped.append(f.filename)
            continue
        # Sanitize filename: keep only safe characters
        safe_name = re.sub(r"[^\w.\-]", "_", f.filename)
        dest = DOCS_DIR / safe_name
        if dest.exists():
            skipped.append(f"{safe_name} (exists)")
            continue
        f.save(str(dest))
        saved.append(safe_name)

    parts = []
    if saved:
        parts.append(f"Uploaded: {', '.join(saved)}")
    if skipped:
        parts.append(f"Skipped: {', '.join(skipped)}")
    return {"message": " | ".join(parts) or "Nothing to upload.", "saved": saved, "skipped": skipped}


@app.route("/docs")
def docs():
    """List docs in DOCS_DIR and which are already indexed."""
    indexed = set()
    if INDEX_FILE.exists():
        records = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
        indexed = {r["file"] for r in records}
    files = []
    for p in sorted(list(DOCS_DIR.glob("*.txt")) + list(DOCS_DIR.glob("*.pdf"))):
        files.append({"name": p.name, "indexed": p.name in indexed})
    return {"files": files, "total": len(files), "indexed": len(indexed)}


@app.route("/build", methods=["POST"])
def build():
    """Index new docs, streaming progress via SSE."""
    def generate():
        try:
            if INDEX_FILE.exists():
                records = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
            else:
                records = []
            indexed_files = {r["file"] for r in records}
            new_count = 0

            all_paths = sorted(list(DOCS_DIR.glob("*.txt")) + list(DOCS_DIR.glob("*.pdf")))
            to_index = [p for p in all_paths if p.name not in indexed_files]

            if not to_index:
                yield f"data: {json.dumps({'type': 'done', 'text': 'All files already indexed.', 'new': 0, 'total': len(records)})}\n\n"
                return

            yield f"data: {json.dumps({'type': 'status', 'text': f'Found {len(to_index)} new file(s) to index...'})}\n\n"

            for path in to_index:
                yield f"data: {json.dumps({'type': 'status', 'text': f'Reading {path.name}...'})}\n\n"

                if path.suffix == ".pdf":
                    reader = PdfReader(path)
                    text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
                else:
                    text = path.read_text(encoding="utf-8").strip()
                if not text:
                    yield f"data: {json.dumps({'type': 'status', 'text': f'Skipping {path.name} (empty)'})}\n\n"
                    continue

                chunks = chunk_text(text)
                yield f"data: {json.dumps({'type': 'status', 'text': f'Indexing {path.name} ({len(chunks)} chunks)...'})}\n\n"

                for i, chunk in enumerate(chunks, start=1):
                    tokens = tokenize(chunk)
                    records.append({
                        "file": path.name,
                        "chunk_id": i,
                        "text": chunk,
                        "tf": term_freq(tokens),
                        "dl": len(tokens),
                    })
                    new_count += 1

                yield f"data: {json.dumps({'type': 'progress', 'file': path.name, 'chunks_done': len(chunks), 'chunks_total': len(chunks)})}\n\n"
                yield f"data: {json.dumps({'type': 'status', 'text': f'Done: {path.name} ({len(chunks)} chunks)'})}\n\n"

            INDEX_FILE.write_text(json.dumps(records), encoding="utf-8")

            # Reload in-memory index + recompute BM25 stats
            global _records, _idf, _avg_dl
            _records = records
            _idf, _avg_dl = build_bm25_index(_records)

            yield f"data: {json.dumps({'type': 'done', 'text': f'Indexing complete.', 'new': new_count, 'total': len(records)})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'text': str(e)})}\n\n"

    return Response(generate(), content_type="text/event-stream")


@app.route("/models")
def models():
    """Return list of available Ollama models, excluding embedding models."""
    EMBED_MODELS = {"nomic-embed-text-v2-moe", "embeddinggemma"}
    try:
        with urlopen(TAGS_URL) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        names = [
            m["name"] for m in data.get("models", [])
            if m["name"].split(":")[0] not in EMBED_MODELS
        ]
        return {"models": names, "default": CHAT_MODEL}
    except Exception as e:
        return {"models": [CHAT_MODEL], "default": CHAT_MODEL, "error": str(e)}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

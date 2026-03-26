import json
import math
import re
from pathlib import Path
from urllib.request import Request, urlopen
from pypdf import PdfReader

DOCS_DIR = Path.home() / "phone-rag" / "docs"
OUT_FILE = Path.home() / "phone-rag" / "index_nomic.json"
OLLAMA_URL = "http://localhost:11434/api/embed"
MODEL = "nomic-embed-text-v2-moe"

EMBED_BATCH_SIZE = 16  # chunks per embedding API call

def embed_batch(texts):
    """Embed multiple texts in a single API call."""
    payload = json.dumps({
        "model": MODEL,
        "input": texts,
        "keep_alive": 0
    }).encode("utf-8")

    req = Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    return data["embeddings"]

CHUNK_MAX = 768      # target max characters per chunk
CHUNK_OVERLAP = 96   # character overlap between adjacent chunks

# Separators ordered from strongest to weakest semantic boundary.
# The splitter tries each level in order, only falling through to
# finer-grained splits when a section still exceeds CHUNK_MAX.
SEPARATORS = [
    "\n\n",                              # paragraphs / double newlines
    "\n",                                # single newlines
    re.compile(r"(?<=[.!?])\s+"),        # sentence boundaries
    re.compile(r"(?<=[;:])\s+"),         # semicolons / colons
    re.compile(r"(?<=[,])\s+"),          # commas
    " ",                                 # words
]

def _split_once(text, separators):
    """Split text using the first separator that produces >1 piece."""
    for sep in separators:
        # First and last separators are literal strings, others are compiled regex
        if isinstance(sep, str):
            parts = text.split(sep)
        else:
            parts = sep.split(text)
        parts = [p for p in parts if p.strip()]
        if len(parts) > 1:
            return parts, sep
    # Nothing could split it further — return as-is
    return [text], None

def _recursive_split(text, separators, max_size):
    """Recursively split text respecting semantic boundaries."""
    if len(text) <= max_size:
        return [text]

    parts, used_sep = _split_once(text, separators)

    # If we couldn't split, return the text even if oversized
    if len(parts) == 1:
        return parts

    # Determine which separators remain for deeper splits
    # (only use same-level or finer separators on sub-parts)
    if used_sep in separators:
        remaining_seps = separators[separators.index(used_sep):]
    else:
        remaining_seps = separators

    chunks = []
    for part in parts:
        if len(part) <= max_size:
            chunks.append(part)
        else:
            # Recurse with finer separators
            finer = remaining_seps[1:] if len(remaining_seps) > 1 else remaining_seps
            chunks.extend(_recursive_split(part, finer, max_size))

    return chunks

def _merge_small_chunks(pieces, max_size, overlap):
    """Merge small pieces into target-sized chunks with overlap."""
    if not pieces:
        return []

    chunks = []
    current = pieces[0]

    for piece in pieces[1:]:
        # If merging fits within the limit, combine
        combined = current + " " + piece
        if len(combined) <= max_size:
            current = combined
        else:
            chunks.append(current.strip())
            # Start next chunk with overlap from end of current
            if overlap > 0 and len(current) > overlap:
                tail = current[-overlap:]
                # Snap overlap to nearest word boundary
                space_idx = tail.find(" ")
                if space_idx != -1:
                    tail = tail[space_idx + 1:]
                current = tail + " " + piece
            else:
                current = piece

    if current.strip():
        chunks.append(current.strip())

    return chunks

def chunk_text(text: str):
    """Split text using recursive character splitting with semantic boundaries.

    1. Recursively split on paragraph -> newline -> sentence -> clause -> word
       boundaries until every piece is under CHUNK_MAX characters.
    2. Merge adjacent small pieces back together (up to CHUNK_MAX) so chunks
       carry enough context, with CHUNK_OVERLAP characters of overlap between
       neighboring chunks for continuity.
    """
    text = text.strip()
    if not text:
        return []

    pieces = _recursive_split(text, SEPARATORS, CHUNK_MAX)
    chunks = _merge_small_chunks(pieces, CHUNK_MAX, CHUNK_OVERLAP)
    return [c for c in chunks if c]

def load_existing_index():
    if OUT_FILE.exists():
        return json.loads(OUT_FILE.read_text(encoding="utf-8"))
    return []

def main():
    records = load_existing_index()
    indexed_files = {r["file"] for r in records}
    new_count = 0

    for path in sorted(list(DOCS_DIR.glob("*.txt")) + list(DOCS_DIR.glob("*.pdf"))):
        if path.name in indexed_files:
            print(f"Skipping (already indexed): {path.name}")
            continue

        if path.suffix == ".pdf":
            reader = PdfReader(path)
            text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
        else:
            text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue

        chunks = chunk_text(text)

        # Embed in batches for fewer API round-trips
        for batch_start in range(0, len(chunks), EMBED_BATCH_SIZE):
            batch = chunks[batch_start:batch_start + EMBED_BATCH_SIZE]
            vectors = embed_batch(batch)
            for j, (chunk, vector) in enumerate(zip(batch, vectors)):
                chunk_id = batch_start + j + 1
                norm = math.sqrt(sum(x * x for x in vector))
                records.append({
                    "file": path.name,
                    "chunk_id": chunk_id,
                    "text": chunk,
                    "embedding": vector,
                    "norm": norm
                })
                print(f"Indexed: {path.name} | chunk {chunk_id} | dims: {len(vector)}")
                new_count += 1

    OUT_FILE.write_text(json.dumps(records), encoding="utf-8")
    print(f"\nSaved index to: {OUT_FILE}")
    print(f"New chunks indexed: {new_count}")
    print(f"Total chunks in index: {len(records)}")

if __name__ == "__main__":
    main()

import json
import math
import re
from pathlib import Path
from urllib.request import Request, urlopen
from pypdf import PdfReader

DOCS_DIR = Path.home() / "phone-rag" / "docs"
OUT_FILE = Path.home() / "phone-rag" / "index.json"
OLLAMA_URL = "http://localhost:11434/api/embed"
MODEL = "embeddinggemma"

def embed_text(text: str):
    payload = json.dumps({
        "model": MODEL,
        "input": text,
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

    return data["embeddings"][0]

CHUNK_SIZE = 5   # sentences per chunk
OVERLAP = 2      # sentences shared between adjacent chunks

def chunk_text(text: str):
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]
    chunks = []
    step = CHUNK_SIZE - OVERLAP
    for i in range(0, max(1, len(sentences) - OVERLAP), step):
        chunk = " ".join(sentences[i:i + CHUNK_SIZE])
        if chunk:
            chunks.append(chunk)
    return chunks

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

        for i, chunk in enumerate(chunks, start=1):
            vector = embed_text(chunk)
            norm = math.sqrt(sum(x * x for x in vector))
            records.append({
                "file": path.name,
                "chunk_id": i,
                "text": chunk,
                "embedding": vector,
                "norm": norm
            })
            print(f"Indexed: {path.name} | chunk {i} | dims: {len(vector)}")
            new_count += 1

    OUT_FILE.write_text(json.dumps(records), encoding="utf-8")
    print(f"\nSaved index to: {OUT_FILE}")
    print(f"New chunks indexed: {new_count}")
    print(f"Total chunks in index: {len(records)}")

if __name__ == "__main__":
    main()

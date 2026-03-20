import json
import math
from pathlib import Path
from urllib.request import Request, urlopen

INDEX_FILE = Path.home() / "phone-rag" / "index.json"
OLLAMA_URL = "http://localhost:11434/api/embed"
MODEL = "embeddinggemma"

def embed_text(text: str):
    payload = json.dumps({
        "model": MODEL,
        "input": text
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

def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

def main():
    query = input("Query: ").strip()
    if not query:
        print("Empty query.")
        return

    records = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
    qvec = embed_text(query)

    scored = []
    for rec in records:
        score = cosine_similarity(qvec, rec["embedding"])
        scored.append((score, rec["file"], rec["text"]))

    scored.sort(reverse=True, key=lambda x: x[0])

    print("\nTop matches:\n")
    for score, fname, text in scored[:3]:
        print(f"{fname} | score={score:.4f}")
        print(text)
        print("-" * 40)

if __name__ == "__main__":
    main()

import json
import math
import re
from pathlib import Path
from urllib.request import Request, urlopen

INDEX_FILE = Path.home() / "phone-rag" / "index.json"
EMBED_URL = "http://localhost:11434/api/embed"
EMBED_MODEL = "embeddinggemma"

def post_json(url, payload):
    req = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

def embed_text(text: str):
    data = post_json(EMBED_URL, {
        "model": EMBED_MODEL,
        "input": text,
        "keep_alive": 0
    })
    return data["embeddings"][0]

def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

def tokenize(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))

def keyword_boost(query, text):
    q = tokenize(query)
    t = tokenize(text)
    overlap = q & t
    return 0.03 * len(overlap)

def main():
    query = input("Query: ").strip()
    if not query:
        print("Empty query.")
        return

    records = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
    qvec = embed_text(query)

    scored = []
    for rec in records:
        emb_score = cosine_similarity(qvec, rec["embedding"])
        lex_score = keyword_boost(query, rec["text"])
        final_score = emb_score + lex_score
        scored.append((final_score, emb_score, lex_score, rec))

    scored.sort(reverse=True, key=lambda x: x[0])

    print("\nTop matches:\n")
    for final_score, emb_score, lex_score, rec in scored[:5]:
        chunk = rec.get("chunk_id", "?")
        print(f"{rec['file']} | chunk {chunk} | final={final_score:.4f} | emb={emb_score:.4f} | lex={lex_score:.4f}")
        print(rec["text"])
        print("-" * 50)

if __name__ == "__main__":
    main()

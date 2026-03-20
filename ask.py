import json
import math
import re
from pathlib import Path
from urllib.request import Request, urlopen

INDEX_FILE = Path.home() / "phone-rag" / "index.json"
EMBED_URL = "http://localhost:11434/api/embed"
GENERATE_URL = "http://localhost:11434/api/generate"

EMBED_MODEL = "embeddinggemma"
CHAT_MODEL = "gemma3:1b"
MIN_SCORE = 0.28
EMBED_KEEP_ALIVE = 0     # unload embed model immediately to free RAM for chat
CHAT_KEEP_ALIVE = 3600   # keep chat model loaded for 1 hour (reuse across queries)

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
        "keep_alive": EMBED_KEEP_ALIVE
    })
    return data["embeddings"][0]

def cosine_similarity_prenorm(a, norm_a, b, norm_b):
    if norm_a == 0 or norm_b == 0:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (norm_a * norm_b)

def tokenize(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))

STOPWORDS = {"the", "a", "an", "is", "in", "of", "to", "and", "or", "what", "how", "does", "do", "it", "its"}

def keyword_boost(query, text):
    q = tokenize(query) - STOPWORDS
    t = tokenize(text) - STOPWORDS
    if not q:
        return 0.0
    # Jaccard similarity, scaled to 0.0-0.15
    union = q | t
    return 0.15 * (len(q & t) / len(union))

def retrieve(query, records, top_k=3):
    qvec = embed_text(query)
    qnorm = math.sqrt(sum(x * x for x in qvec))

    scored = []
    for rec in records:
        emb_score = cosine_similarity_prenorm(qvec, qnorm, rec["embedding"], rec["norm"])
        lex_score = keyword_boost(query, rec["text"])
        final_score = emb_score + lex_score
        scored.append((final_score, emb_score, lex_score, rec))

    scored.sort(reverse=True, key=lambda x: x[0])
    return scored[:top_k]

def build_prompt(query, matches):
    context_blocks = []
    for final_score, emb_score, lex_score, rec in matches:
        context_blocks.append(
            f"[Source: {rec['file']} | chunk={rec.get('chunk_id', '?')} | final={final_score:.4f}]\n{rec['text']}"
        )

    context = "\n\n".join(context_blocks)

    return f"""Context:
{context}

Using ONLY the context above, answer the question. You may combine facts from multiple passages. Copy the relevant phrase directly. Keep your answer short — no more than one sentence. If the answer is not in the context, write only: NOT FOUND

Question: {query}
Answer:"""

def main():
    query = input("Question: ").strip()
    if not query:
        print("Empty question.")
        return

    records = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
    matches = retrieve(query, records, top_k=5)

    print("\nRetrieved context:\n")
    for final_score, emb_score, lex_score, rec in matches:
        print(f"{rec['file']} | chunk {rec.get('chunk_id', '?')} | final={final_score:.4f} | emb={emb_score:.4f} | lex={lex_score:.4f}")
    print("-" * 40)

    best_score = matches[0][0]
    if best_score < MIN_SCORE:
        print("\nAnswer:\n")
        print("NOT FOUND IN DOCUMENTS")
        return

    prompt = build_prompt(query, matches)

    req = Request(
        GENERATE_URL,
        data=json.dumps({
            "model": CHAT_MODEL,
            "prompt": prompt,
            "stream": True,
            "think": False,
            "keep_alive": CHAT_KEEP_ALIVE,
            "options": {
                "temperature": 0,
                "num_predict": 120
            }
        }).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    print("\nAnswer:\n")
    with urlopen(req) as resp:
        for raw_line in resp:
            if not raw_line.strip():
                continue
            obj = json.loads(raw_line.decode("utf-8"))
            if "response" in obj:
                print(obj["response"], end="", flush=True)
    print()

if __name__ == "__main__":
    main()

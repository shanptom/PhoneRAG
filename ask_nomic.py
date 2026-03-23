import argparse
import json
import math
import re
from pathlib import Path
from urllib.request import Request, urlopen

INDEX_FILE = Path.home() / "phone-rag" / "index_nomic.json"
EMBED_URL = "http://localhost:11434/api/embed"
GENERATE_URL = "http://localhost:11434/api/generate"

EMBED_MODEL = "nomic-embed-text-v2-moe"
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

BROAD_KEYWORDS = {"summarize", "summary", "summarise", "overview", "outline",
                   "main points", "key points", "recap", "brief", "briefing",
                   "explain the file", "explain the document", "tell me about",
                   "what is this about", "what does it say", "entire", "whole",
                   "all about", "full"}

def is_broad_query(query):
    q_lower = query.lower()
    for kw in BROAD_KEYWORDS:
        if kw in q_lower:
            return True
    return False

def find_target_file(query, records):
    """Try to match a filename mentioned in the query. Falls back to the top-scored file."""
    q_lower = query.lower()
    known_files = {r["file"] for r in records}
    for fname in known_files:
        name_no_ext = Path(fname).stem.replace("_", " ").replace("-", " ").lower()
        if fname.lower() in q_lower or name_no_ext in q_lower:
            return fname
    return None

def retrieve_broad(query, records):
    """For broad queries: get all chunks from the target file, ordered by chunk_id."""
    target = find_target_file(query, records)

    if target:
        file_chunks = [r for r in records if r["file"] == target]
        file_chunks.sort(key=lambda r: r.get("chunk_id", 0))
        return [(1.0, 1.0, 0.0, r) for r in file_chunks], target

    # No filename detected — use similarity to pick the best file, then return all its chunks
    qvec = embed_text(query)
    qnorm = math.sqrt(sum(x * x for x in qvec))

    scored = []
    for rec in records:
        emb_score = cosine_similarity_prenorm(qvec, qnorm, rec["embedding"], rec["norm"])
        scored.append((emb_score, rec))
    scored.sort(reverse=True, key=lambda x: x[0])

    best_file = scored[0][1]["file"]
    file_chunks = [r for r in records if r["file"] == best_file]
    file_chunks.sort(key=lambda r: r.get("chunk_id", 0))
    return [(1.0, 1.0, 0.0, r) for r in file_chunks], best_file

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

def build_prompt(query, matches, broad=False):
    context_blocks = []
    for final_score, emb_score, lex_score, rec in matches:
        context_blocks.append(
            f"[Source: {rec['file']} | chunk={rec.get('chunk_id', '?')} | final={final_score:.4f}]\n{rec['text']}"
        )

    context = "\n\n".join(context_blocks)

    if broad:
        return f"""Context:
{context}

Using ONLY the context above, provide a concise summary that covers all the main points. Organize by topic if appropriate. If the context is insufficient, say so.

Question: {query}
Answer:"""

    return f"""Context:
{context}

Using ONLY the context above, answer the question. You may combine facts from multiple passages. Be thorough — explain your answer with relevant details from the context. If the answer is not in the context, write only: NOT FOUND

Question: {query}
Answer:"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=CHAT_MODEL, help="Ollama chat model to use")
    args = parser.parse_args()
    chat_model = args.model

    query = input("Question: ").strip()
    if not query:
        print("Empty question.")
        return

    records = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
    broad = is_broad_query(query)

    if broad:
        matches, target = retrieve_broad(query, records)
        print(f"\n[Broad query detected — retrieving all {len(matches)} chunks from: {target}]\n")
        for _, _, _, rec in matches:
            print(f"  {rec['file']} | chunk {rec.get('chunk_id', '?')}")
    else:
        matches = retrieve(query, records, top_k=5)
        print("\nRetrieved context:\n")
        for final_score, emb_score, lex_score, rec in matches:
            print(f"{rec['file']} | chunk {rec.get('chunk_id', '?')} | final={final_score:.4f} | emb={emb_score:.4f} | lex={lex_score:.4f}")
    print("-" * 40)

    if not broad:
        best_score = matches[0][0]
        if best_score < MIN_SCORE:
            print("\nAnswer:\n")
            print("NOT FOUND IN DOCUMENTS")
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
                "num_predict": 512 if broad else 256
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

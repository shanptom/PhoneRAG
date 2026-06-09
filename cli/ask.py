"""CLI RAG query tool — LiteRT-LM + BM25 retrieval."""
import argparse
import json
import math
import os
import re
from pathlib import Path

import litert_lm

INDEX_FILE = Path.home() / "phone-rag" / "index_bm25.json"
# Defaults to ~/models/ (works on Termux); override with the MODEL_PATH env var.
MODEL_PATH = os.environ.get(
    "MODEL_PATH", str(Path.home() / "models" / "gemma-4-E2B-it.litertlm")
)

MIN_SCORE = 2.0
BM25_K1 = 1.5
BM25_B = 0.75

_TOKENIZE_RE = re.compile(r"[a-z0-9]+")

STOPWORDS = {
    "the", "a", "an", "is", "in", "of", "to", "and", "or", "what", "how",
    "does", "do", "it", "its", "about", "from", "would", "could", "should",
    "very", "just", "been", "have", "has", "be", "are", "was", "were",
    "this", "that", "with", "for", "not", "but", "can", "will", "than",
}


def _porter_stem(word):
    """Minimal Porter stemmer — handles the most common English suffixes."""
    if len(word) <= 2:
        return word
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


def tokenize(text):
    return [_porter_stem(w) for w in _TOKENIZE_RE.findall(text.lower()) if w not in STOPWORDS]

def build_bm25_index(records):
    N = len(records)
    df = {}
    total_dl = 0
    for rec in records:
        total_dl += rec["dl"]
        for t in rec["tf"]:
            df[t] = df.get(t, 0) + 1
    avg_dl = total_dl / N if N else 1
    idf = {}
    for t, d in df.items():
        idf[t] = math.log((N - d + 0.5) / (d + 0.5) + 1)
    return idf, avg_dl

def bm25_score(query_tokens, rec, idf, avg_dl):
    score = 0.0
    tf = rec["tf"]
    dl = rec["dl"]
    for t in query_tokens:
        if t not in idf:
            continue
        f = tf.get(t, 0)
        num = f * (BM25_K1 + 1)
        den = f + BM25_K1 * (1 - BM25_B + BM25_B * dl / avg_dl)
        score += idf[t] * num / den
    return score

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
    q_lower = query.lower()
    known_files = {r["file"] for r in records}
    for fname in known_files:
        name_no_ext = Path(fname).stem.replace("_", " ").replace("-", " ").lower()
        if fname.lower() in q_lower or name_no_ext in q_lower:
            return fname
    return None

MAX_BROAD_CHUNKS = 25

def retrieve_broad(query, records, idf, avg_dl):
    target = find_target_file(query, records)
    if not target:
        query_tokens = tokenize(query)
        scored = [(bm25_score(query_tokens, r, idf, avg_dl), r) for r in records]
        scored.sort(reverse=True, key=lambda x: x[0])
        target = scored[0][1]["file"]

    file_chunks = [r for r in records if r["file"] == target]
    file_chunks.sort(key=lambda r: r.get("chunk_id", 0))
    file_chunks = file_chunks[:MAX_BROAD_CHUNKS]
    return [(1.0, r) for r in file_chunks], target

def retrieve(query, records, idf, avg_dl, top_k=3):
    query_tokens = tokenize(query)
    scored = [(bm25_score(query_tokens, r, idf, avg_dl), r) for r in records]
    scored.sort(reverse=True, key=lambda x: x[0])
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


def _extract_text(chunk):
    """Extract text from a LiteRT-LM streaming chunk."""
    if isinstance(chunk, dict):
        if "content" in chunk:
            content = chunk["content"]
            if isinstance(content, list):
                return "".join(c.get("text", "") for c in content if isinstance(c, dict))
            if isinstance(content, str):
                return content
        if "text" in chunk:
            return chunk["text"]
    if isinstance(chunk, str):
        return chunk
    return ""


def main():
    parser = argparse.ArgumentParser(description="Ask your documents a question (LiteRT + BM25)")
    parser.add_argument("--model", default=MODEL_PATH, help="Path to .litertlm model file")
    args = parser.parse_args()

    query = input("Question: ").strip()
    if not query:
        print("Empty question.")
        return

    records = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
    idf, avg_dl = build_bm25_index(records)
    broad = is_broad_query(query)

    if broad:
        matches, target = retrieve_broad(query, records, idf, avg_dl)
        print(f"\n[Broad query — {len(matches)} chunks from: {target}]\n")
        for _, rec in matches:
            print(f"  {rec['file']} | chunk {rec.get('chunk_id', '?')}")
    else:
        matches = retrieve(query, records, idf, avg_dl, top_k=3)
        print("\nRetrieved context:\n")
        for score, rec in matches:
            print(f"  {rec['file']} | chunk {rec.get('chunk_id', '?')} | score={score:.2f}")
    print("-" * 40)

    if not broad and matches[0][0] < MIN_SCORE:
        print("\nNOT FOUND IN DOCUMENTS")
        return

    prompt = build_prompt(query, matches, broad=broad)

    print(f"\nLoading model from {args.model}...")
    litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)

    with litert_lm.Engine(args.model) as engine:
        sampler = litert_lm.SamplerConfig(temperature=0.1, top_k=20)
        conversation = engine.create_conversation(
            system_message="You are a helpful assistant that answers questions based on provided context. Be concise and accurate.",
            sampler_config=sampler,
        )

        print("\nAnswer:\n")
        try:
            for chunk in conversation.send_message_async(prompt):
                text = _extract_text(chunk)
                if text:
                    print(text, end="", flush=True)
        finally:
            conversation.close()
    print()

if __name__ == "__main__":
    main()

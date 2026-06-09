"""CLI BM25 index builder — no embeddings needed.

Reads documents from ~/phone-rag/docs/, splits into chunks,
computes BM25 term frequencies, and saves to index_bm25.json.
"""
import json
import re
from pathlib import Path
from pypdf import PdfReader

DOCS_DIR = Path.home() / "phone-rag" / "docs"
OUT_FILE = Path.home() / "phone-rag" / "index_bm25.json"

CHUNK_MAX = 768
CHUNK_OVERLAP = 96

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_SEMICOLON_RE = re.compile(r"(?<=[;:])\s+")
_COMMA_RE = re.compile(r"(?<=[,])\s+")

SEPARATORS = ["\n\n", "\n", _SENTENCE_RE, _SEMICOLON_RE, _COMMA_RE, " "]

_TOKENIZE_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "the", "a", "an", "is", "in", "of", "to", "and", "or", "what", "how",
    "does", "do", "it", "its", "about", "from", "would", "could", "should",
    "very", "just", "been", "have", "has", "be", "are", "was", "were",
    "this", "that", "with", "for", "not", "but", "can", "will", "than",
}


def _porter_stem(word):
    """Minimal Porter stemmer."""
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


def term_freq(tokens):
    tf = {}
    for t in tokens:
        tf[t] = tf.get(t, 0) + 1
    return tf


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


_HEADING_RE = re.compile(r"^(?:#{1,6}\s+.+|[A-Z][^\n]{0,80})$")


def _is_heading(line):
    line = line.strip()
    if not line or len(line) > 120:
        return False
    if line.startswith("#"):
        return True
    if len(line) < 80 and line[0].isupper() and line[-1] not in ".!?,;:)\"'":
        return True
    return False


def chunk_text(text: str):
    text = text.strip()
    if not text:
        return []
    lines = text.split("\n")
    current_heading = ""
    sections = []
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


def main():
    if not DOCS_DIR.exists():
        print(f"No docs directory found at {DOCS_DIR}")
        print("Create it and add .txt or .pdf files first.")
        return

    records = []
    if OUT_FILE.exists():
        records = json.loads(OUT_FILE.read_text(encoding="utf-8"))
    indexed_files = {r["file"] for r in records}
    new_count = 0

    all_paths = sorted(list(DOCS_DIR.glob("*.txt")) + list(DOCS_DIR.glob("*.pdf")))
    to_index = [p for p in all_paths if p.name not in indexed_files]

    if not to_index:
        print("All files already indexed.")
        print(f"Total chunks in index: {len(records)}")
        return

    print(f"Found {len(to_index)} new file(s) to index...")

    for path in to_index:
        print(f"\nReading {path.name}...")
        if path.suffix == ".pdf":
            reader = PdfReader(path)
            text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
        else:
            text = path.read_text(encoding="utf-8").strip()

        if not text:
            print(f"  Skipping {path.name} (empty)")
            continue

        chunks = chunk_text(text)
        print(f"  {len(chunks)} chunks")

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

        print(f"  Done: {path.name}")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(records), encoding="utf-8")
    print(f"\nSaved index to: {OUT_FILE}")
    print(f"New chunks indexed: {new_count}")
    print(f"Total chunks in index: {len(records)}")


if __name__ == "__main__":
    main()

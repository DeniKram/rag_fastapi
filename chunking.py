import argparse
import json
import random
import re
from pathlib import Path

DATA_DIR = Path("data")
SECTIONS_PATH = DATA_DIR / "sections.jsonl"
WORD_RE = re.compile(r"\S+")


def load_sections(path: Path = SECTIONS_PATH) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def split_fixed(text: str, max_words: int, overlap: int) -> list[str]:
    assert 0 <= overlap < max_words, "overlap must be smaller than max_words"
    words = list(WORD_RE.finditer(text))
    if len(words) <= max_words:
        return [text.strip()]
    step = max_words - overlap
    chunks = []
    for start in range(0, len(words), step):
        end = min(start + max_words, len(words))
        chunks.append(text[words[start].start():words[end - 1].end()])
        if end == len(words):
            break
    return chunks


def split_blocks(text: str) -> list[str]:
    blocks: list[str] = []
    cur: list[str] = []
    in_code = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
        if not in_code and not line.strip() and not line.strip().startswith("```"):
            if cur:
                blocks.append("\n".join(cur))
                cur = []
            continue
        cur.append(line)
    if cur:
        blocks.append("\n".join(cur))
    return blocks


def split_long_block(block: str, max_words: int) -> list[str]:
    lines = block.splitlines()
    fence = None
    if len(lines) >= 2 and lines[0].strip().startswith("```") and lines[-1].strip() == "```":
        fence = lines[0].strip()
        lines = lines[1:-1]

    pieces, cur, n = [], [], 0
    for line in lines:
        w = len(line.split())
        if cur and n + w > max_words:
            pieces.append(cur)
            cur, n = [], 0
        cur.append(line)
        n += w
    if cur:
        pieces.append(cur)

    if fence:
        return [fence + "\n" + "\n".join(p) + "\n```" for p in pieces]
    return ["\n".join(p) for p in pieces]


def split_paragraph(text: str, max_words: int) -> list[str]:
    units: list[str] = []
    for b in split_blocks(text):
        units += split_long_block(b, max_words) if len(b.split()) > max_words else [b]

    chunks, cur, n = [], [], 0
    for u in units:
        w = len(u.split())
        if cur and n + w > max_words:
            chunks.append("\n\n".join(cur))
            cur, n = [], 0
        cur.append(u)
        n += w
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


def make_chunks(sections: list[dict], strategy: str, max_words: int,
                overlap: int, prefix: bool) -> list[dict]:
    chunks: list[dict] = []
    for sec in sections:
        if strategy == "fixed":
            parts = split_fixed(sec["text"], max_words, overlap)
        elif strategy == "paragraph":
            parts = split_paragraph(sec["text"], max_words)
        else:
            raise ValueError(f"unknown strategy: {strategy}")

        for i, part in enumerate(parts):
            text = part
            if prefix:
                head = sec["page_title"]
                if sec["heading"] != sec["page_title"]:
                    head += f" > {sec['heading']}"
                text = f"{head}\n\n{part}"
            chunks.append({
                "chunk_id": f"{sec['id']}::{i}",
                "section_id": sec["id"],
                "file": sec["file"],
                "page_title": sec["page_title"],
                "heading": sec["heading"],
                "url": sec["url"],
                "text": text,
                "n_words": len(text.split()),
            })
    return chunks


def chunk_name(strategy: str, max_words: int, overlap: int, prefix: bool) -> str:
    name = f"{strategy}_{max_words}"
    if strategy == "fixed":
        name += f"_ov{overlap}"
    if prefix:
        name += "_prefix"
    return name


def print_stats(chunks: list[dict]) -> None:
    w = sorted(c["n_words"] for c in chunks)
    n = len(w)
    print(f"Chunks: {n}, words: min {w[0]}, median {w[n // 2]}, "
          f"90th percentile {w[int(n * 0.9)]}, max {w[-1]}")
    print(f"Shorter than 20 words: {sum(x < 20 for x in w)}, longer than 300 words: {sum(x > 300 for x in w)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", choices=["fixed", "paragraph"], default="paragraph")
    ap.add_argument("--max-words", type=int, default=200)
    ap.add_argument("--overlap", type=int, default=40, help="fixed strategy only")
    ap.add_argument("--prefix", action="store_true", help="add page heading at the start of each chunk")
    ap.add_argument("--show", type=int, default=2, help="how many random chunks to print")
    args = ap.parse_args()

    sections = load_sections()
    chunks = make_chunks(sections, args.strategy, args.max_words, args.overlap, args.prefix)

    name = chunk_name(args.strategy, args.max_words, args.overlap, args.prefix)
    out_path = DATA_DIR / f"chunks_{name}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"Sections: {len(sections)}, strategy: {name}")
    print_stats(chunks)
    print(f"Saved: {out_path}\n")

    random.seed(0)
    for c in random.sample(chunks, min(args.show, len(chunks))):
        print("=" * 70)
        print(f"{c['chunk_id']}  ({c['n_words']} words)\n{c['url']}\n")
        print(c["text"][:600] + ("..." if len(c["text"]) > 600 else ""))


if __name__ == "__main__":
    main()
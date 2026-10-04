# inspect_chunks.py
"""
Compare chunk files to see how the chunking strategy breaks code blocks.

Run:
    py -3.12 inspect_chunks.py data/chunks_paragraph_200.jsonl data/chunks_fixed_200_ov40.jsonl
    py -3.12 inspect_chunks.py data/chunks_fixed_200_ov40.jsonl --show 2   # show examples of broken chunks
"""
import argparse
import json
import random


def load(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def is_broken(text: str) -> bool:
    """A chunk has an odd number of ``` lines: a code block was opened but not closed (or vice versa)."""
    fences = sum(1 for line in text.splitlines() if line.strip().startswith("```"))
    return fences % 2 == 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--show", type=int, default=0, help="how many broken chunk examples to print")
    args = ap.parse_args()

    for path in args.files:
        chunks = load(path)
        broken = [c for c in chunks if is_broken(c["text"])]
        with_code = [c for c in chunks if "```" in c["text"]]
        print(f"{path}")
        print(f"  chunks: {len(chunks)}, with code blocks: {len(with_code)}")
        print(f"  broken code blocks: {len(broken)} "
              f"({100 * len(broken) / max(len(with_code), 1):.0f}% of chunks containing code)")

        random.seed(1)
        for c in random.sample(broken, min(args.show, len(broken))):
            lines = c["text"].splitlines()
            print("-" * 60)
            print(f"  {c['chunk_id']}")
            print("  ...start of chunk:", " | ".join(lines[:2])[:120])
            print("  ...end of chunk:  ", " | ".join(lines[-2:])[:120])


if __name__ == "__main__":
    main()
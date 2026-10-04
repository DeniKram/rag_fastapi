# review_eval.py
"""
Step 5 of the project: manually review questions from data/eval_questions.jsonl.

For each question, we show the section text and you decide:
    y   good question: the section answers it and it looks like a real user request
    n   bad question: the section does not answer it, the question is odd, or it is almost a copied phrase from the text
    a   ambiguous: another section answers it equally well
    s   skip
    q   save and exit

Suspicious questions are shown first (strong overlap with section text, mention of "documentation"), then random ones. Decisions are saved immediately, and you can abort at any time.

Run:
    py -3.12 review_eval.py --limit 30
"""
import argparse
import json
import random
from pathlib import Path

EVAL_PATH = Path("data/eval_questions.jsonl")
SECTIONS_PATH = Path("data/sections.jsonl")
SHOW_CHARS = 1000                      # how many characters of the section to show


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def is_suspicious(r: dict) -> bool:
    return r["lexical_overlap"] >= 0.8 or r["mentions_doc"]


def review_order(rows: list[dict], limit: int, seed: int) -> list[int]:
    """Indexes of unchecked questions: suspicious first, then random."""
    todo = [i for i, r in enumerate(rows) if r["status"] == "unchecked"]
    random.Random(seed).shuffle(todo)
    todo.sort(key=lambda i: not is_suspicious(rows[i]))       # suspicious first; random order within groups
    return todo[:limit]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=30, help="how many questions to review in this run")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = read_jsonl(EVAL_PATH)
    sections = {s["id"]: s for s in read_jsonl(SECTIONS_PATH)}
    order = review_order(rows, args.limit, args.seed)
    print(f"Total questions: {len(rows)}, unchecked: {sum(r['status'] == 'unchecked' for r in rows)}, "
          f"reviewing now: {len(order)}\n")

    for k, i in enumerate(order, start=1):
        r = rows[i]
        text = sections[r["section_id"]]["text"]
        print("=" * 78)
        print(f"[{k}/{len(order)}] {r['section_id']}"
              + ("   <- SUSPICIOUS" if is_suspicious(r) else ""))
        print(f"EN: {r['question_en']}")
        print(f"RU: {r['question_ru']}")
        print(f"(word overlap with section text: {r['lexical_overlap']})\n")
        print(text[:SHOW_CHARS] + (" ..." if len(text) > SHOW_CHARS else ""))
        print()

        while True:
            ans = input("y good / n bad / a ambiguous / s skip / q quit > ").strip().lower()
            if ans in {"y", "n", "a", "s", "q"}:
                break
        if ans == "q":
            break
        if ans != "s":
            r["status"] = {"y": "ok", "n": "bad", "a": "ambiguous"}[ans]
            write_jsonl(EVAL_PATH, rows)                      # save after every decision

    counts = {s: sum(r["status"] == s for r in rows) for s in ("ok", "ambiguous", "bad", "unchecked")}
    print("\nSummary:", counts)


if __name__ == "__main__":
    main()
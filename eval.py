import argparse
import json
from pathlib import Path

import embedder
from index import Index

EVAL_PATH = Path("data/eval_questions.jsonl")
KS = (1, 3, 5, 10)
TOP_K = max(KS)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def rank_of_gold(results: list[tuple[dict, float]], gold_section: str) -> int | None:
    for pos, (chunk, _score) in enumerate(results, start=1):
        if chunk["section_id"] == gold_section:
            return pos
    return None


def rank_of_page(results: list[tuple[dict, float]], gold_file: str) -> int | None:
    for pos, (chunk, _score) in enumerate(results, start=1):
        if chunk["file"] == gold_file:
            return pos
    return None


def metrics(ranks: list[int | None]) -> dict:
    n = len(ranks)
    out = {f"recall@{k}": sum(r is not None and r <= k for r in ranks) / n for k in KS}
    out["MRR"] = sum(1 / r for r in ranks if r is not None) / n
    return out


def format_metrics(m: dict) -> str:
    return "  ".join(f"{k}={v:.3f}" for k, v in m.items())


def evaluate(index: Index, rows: list[dict], lang: str, model: str) -> tuple[list, list]:
    field = "question_en" if lang == "en" else "question_ru"
    q_vecs = embedder.embed_queries([r[field] for r in rows], model)
    section_ranks, page_ranks = [], []
    for r, q_vec in zip(rows, q_vecs):
        results = index.search(q_vec, TOP_K)
        section_ranks.append(rank_of_gold(results, r["section_id"]))
        page_ranks.append(rank_of_page(results, r["section_id"].split("#")[0]))
    return section_ranks, page_ranks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=Path, required=True)
    ap.add_argument("--model", choices=list(embedder.MODELS), default="e5")
    ap.add_argument("--lang", choices=["en", "ru"], default="en")
    ap.add_argument("--show-misses", type=int, default=10, help="how many misses to print")
    args = ap.parse_args()

    rows = read_jsonl(EVAL_PATH)
    if any(r["status"] == "unchecked" for r in rows):
        print("Warning: there are unchecked questions; they are skipped.\n")
    if args.lang == "ru":
        rows = [r for r in rows if r.get("ru_ok", True)]
    ok = [r for r in rows if r["status"] == "ok"]
    ambiguous = [r for r in rows if r["status"] == "ambiguous"]

    index = Index.load(args.chunks, args.model)
    print(f"Chunks: {args.chunks.name}, model: {args.model}, question language: {args.lang}, chunks in index: {len(index.chunks)}")

    ranks, page_ranks = evaluate(index, ok, args.lang, args.model)
    print(f"\nOK questions: {len(ok)}")
    print("  section:  " + format_metrics(metrics(ranks)))
    print("  page:     " + format_metrics(metrics(page_ranks)))

    if ambiguous:
        a_ranks, _ = evaluate(index, ambiguous, args.lang, args.model)
        print(f"\nAmbiguous (separate): {len(ambiguous)}")
        print("  section:  " + format_metrics(metrics(a_ranks)))

    misses = [(r, rk) for r, rk in zip(ok, ranks) if rk is None or rk > 5]
    print(f"\nMisses (correct section not in top-5): {len(misses)} out of {len(ok)}")
    field = "question_en" if args.lang == "en" else "question_ru"
    for r, rk in misses[:args.show_misses]:
        print(f"  rank {rk or '>10'}: {r[field]}\n      correct section: {r['section_id']}")


if __name__ == "__main__":
    main()
import argparse
from pathlib import Path

import embedder
from index import Index

DEMO_QUESTIONS = [
    "How to make a query parameter optional?",
    "How do I enable CORS?",
    "How do I run an application in Docker?",
    "How do I run a background task after returning a response?",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=Path, required=True)
    ap.add_argument("--model", choices=list(embedder.MODELS), default="e5")
    ap.add_argument("-q", "--question", action="append", help="question text can be provided multiple times")
    ap.add_argument("-k", type=int, default=5)
    args = ap.parse_args()

    index = Index.load(args.chunks, args.model)
    questions = args.question or DEMO_QUESTIONS
    q_vecs = embedder.embed_queries(questions, args.model)

    for question, q_vec in zip(questions, q_vecs):
        print("=" * 78)
        print(f"QUESTION: {question}")
        for rank, (chunk, score) in enumerate(index.search(q_vec, args.k), start=1):
            first_line = chunk["text"].strip().splitlines()[0][:70]
            print(f"  {rank}. {score:.3f}  {chunk['chunk_id']}")
            print(f"       {first_line}")


if __name__ == "__main__":
    main()
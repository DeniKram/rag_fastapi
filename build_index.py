import argparse
import time
from pathlib import Path

import numpy as np

import embedder
from index import emb_path_for, load_chunks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=Path, required=True)
    ap.add_argument("--model", choices=list(embedder.MODELS), default="e5")
    args = ap.parse_args()

    chunks = load_chunks(args.chunks)
    texts = [c["text"] for c in chunks]
    print(f"Chunk count: {len(texts)}, model: {embedder.MODELS[args.model]['hf_name']}, device: {embedder.DEVICE}")

    n_cut = embedder.count_truncated(texts, args.model)
    print(f"Longer than {embedder.MAX_SEQ_LENGTH} tokens (tail will be truncated): {n_cut}")

    t0 = time.time()
    emb = embedder.embed_passages(texts, args.model)
    print(f"Done in {time.time() - t0:.0f}s, matrix shape: {emb.shape}, dtype: {emb.dtype}")
    print(f"Norm of first vector (should be ~1.0): {np.linalg.norm(emb[0]):.3f}")

    out = emb_path_for(args.chunks, args.model)
    out.parent.mkdir(exist_ok=True)
    np.save(out, emb)
    print(f"Saved: {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
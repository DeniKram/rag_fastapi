import json
from pathlib import Path

import numpy as np

DATA_DIR = Path("data")


def load_chunks(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def emb_path_for(chunks_path: Path, model_key: str) -> Path:
    stem = Path(chunks_path).stem.removeprefix("chunks_")
    return DATA_DIR / f"emb_{stem}__{model_key}.npy"


class Index:
    def __init__(self, emb: np.ndarray, chunks: list[dict]):
        assert emb.shape[0] == len(chunks), f"embeddings {emb.shape[0]}, chunks {len(chunks)}"
        self.emb = emb
        self.chunks = chunks

    @classmethod
    def load(cls, chunks_path: Path, model_key: str) -> "Index":
        emb = np.load(emb_path_for(chunks_path, model_key))
        return cls(emb, load_chunks(chunks_path))

    def search(self, q_vec: np.ndarray, k: int = 5) -> list[tuple[dict, float]]:
        scores = self.emb @ q_vec
        top = np.argpartition(-scores, k)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self.chunks[i], float(scores[i])) for i in top]
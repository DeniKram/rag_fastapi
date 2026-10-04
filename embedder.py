import numpy as np

DEVICE = "cpu"
MAX_SEQ_LENGTH = 512

MODELS = {
    "e5": {
        "hf_name": "intfloat/multilingual-e5-small",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
    },
    "bge": {
        "hf_name": "BAAI/bge-small-en-v1.5",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "passage_prefix": "",
    },
}

_loaded: dict = {}


def get_model(key: str):
    if key not in _loaded:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(MODELS[key]["hf_name"], device=DEVICE)
        model.max_seq_length = MAX_SEQ_LENGTH
        _loaded[key] = model
    return _loaded[key]


def _encode(texts: list[str], key: str, prefix: str, batch_size: int, show_progress: bool) -> np.ndarray:
    model = get_model(key)
    vectors = model.encode(
        [prefix + t for t in texts],
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=show_progress,
    )
    return vectors.astype(np.float32)


def embed_passages(texts: list[str], key: str, batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
    return _encode(texts, key, MODELS[key]["passage_prefix"], batch_size, show_progress)


def embed_queries(texts: list[str], key: str, batch_size: int = 32) -> np.ndarray:
    return _encode(texts, key, MODELS[key]["query_prefix"], batch_size, show_progress=False)


def count_truncated(texts: list[str], key: str) -> int:
    model = get_model(key)
    prefix = MODELS[key]["passage_prefix"]
    ids = model.tokenizer([prefix + t for t in texts], add_special_tokens=True)["input_ids"]
    return sum(len(x) > MAX_SEQ_LENGTH for x in ids)
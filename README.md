# FastAPI Documentation RAG

This project implements a retrieval-augmented generation (RAG) pipeline over the FastAPI documentation. It chunks the docs, builds embeddings, retrieves the most relevant passages for a user question, and passes them to a local LLM for grounded answer generation.

## Overview

The system is designed for question answering over a documentation corpus. Instead of relying on the LLM to memorize the full documentation, it first retrieves the most relevant sections and then generates an answer grounded in that context.

This project is useful for:

- semantic search over technical documentation
- retrieval-based QA on a domain-specific corpus
- experimentation with vector search and local LLMs
- evaluation of retrieval quality on a curated dataset

## Architecture

- `chunking.py` — splits documentation sections into chunks
- `embedder.py` — encodes text passages and queries into embeddings
- `index.py` — builds a vector index and performs similarity search
- `build_index.py` — creates the embedding matrix and stores it in `data/`
- `generator.py` — sends retrieved chunks to the local model via LM Studio
- `search.py` — quick manual retrieval checks for sample questions
- `eval.py` — evaluates retrieval quality with recall@k and MRR

## Workflow

1. The FastAPI docs are split into chunks.
2. Each chunk is embedded with a sentence-transformer model.
3. A user query is embedded in the same space.
4. The most relevant chunks are retrieved by cosine similarity.
5. The retrieved chunks are passed to a local LLM.
6. The LLM answers using only the supplied context.

## Tech stack

- Python 3.12+
- SentenceTransformers
- NumPy
- OpenAI-compatible client
- LM Studio
- FastAPI documentation corpus

## Requirements

- Python 3.12+
- LM Studio running locally
- A compatible model available through the LM Studio OpenAI-compatible endpoint
- Internet access for downloading embedding model weights from Hugging Face

## Setup

1. Clone the repository.
2. Create and activate a virtual environment:

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Copy the example environment file and configure your local settings:

```bash
copy .env.example .env
```

5. Start LM Studio and confirm the endpoint is reachable.

## Build the index

```bash
python build_index.py --chunks data/chunks_paragraph_200.jsonl --model e5
```

## Run a sample query

```bash
python search.py --chunks data/chunks_paragraph_200.jsonl --model e5 -q "How do I enable CORS?" -k 5
```

## Run evaluation

```bash
python eval.py --chunks data/chunks_paragraph_200.jsonl --model e5 --lang en
```

## Environment variables

The project uses environment variables for model and endpoint configuration.

Example configuration from `.env.example`:

```env
LM_BASE_URL=http://localhost:1234/v1
LM_API_KEY=lm-studio
LM_MODEL=qwen2.5-14b-instruct-1m
LM_TIMEOUT_S=900
LM_MAX_TOKENS=1024
LM_EXTRA_BODY=
TOP_K_CONTEXT=5
```

## Evaluation results

The project includes retrieval evaluation over curated FastAPI questions. English-language results are the strongest and show that the retrieval pipeline is functional:

- recall@1: approximately 0.44
- recall@3: approximately 0.68
- recall@5: approximately 0.72
- MRR: approximately 0.57

This demonstrates that the system can find the correct documentation sections for many technical queries, though there is still room for improvement.

## Project structure

```text
.
├── build_index.py
├── chunking.py
├── config.py
├── corpus.py
├── embedder.py
├── eval.py
├── eval_set.py
├── generator.py
├── index.py
├── inspect_chunks.py
├── list_models.py
├── markup.py
├── search.py
├── test_stream.py
├── data/
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
└── ...
```

## Limitations

- The project is built for experimentation and portfolio/demo use.
- Retrieval quality is stronger for English-language questions than for Russian ones.
- The answer generation depends on the availability and performance of the local LM Studio endpoint.
- The current pipeline is not a full production system with a web frontend or API layer.

## Future work

- add a small API layer
- add a reranker for better retrieval quality
- improve multilingual query handling
- add automated smoke tests
- extend evaluation coverage and benchmark reporting


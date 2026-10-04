import re
import time
from typing import Iterator

from openai import OpenAI

import config

SYSTEM_PROMPT = (
    "You are a FastAPI documentation assistant. Answer only using the provided fragments. "
    "After each factual statement, add a citation in the form [1], [2]. "
    "If the fragments do not contain the answer, say: 'No answer was found in the documentation.' "
    "Answer in the same language as the question."
)

_client = OpenAI(
    base_url=config.LM_BASE_URL,
    api_key=config.LM_API_KEY,
    timeout=config.LM_TIMEOUT_S,
    max_retries=0,
)


def build_messages(question: str, chunks: list[dict]) -> list[dict]:
    context = "\n\n".join(
        f"[{i}] ({c['source']})\n{c['text']}" for i, c in enumerate(chunks, start=1)
    )
    user_msg = f"Documentation fragments:\n\n{context}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]


def _request_kwargs(question: str, chunks: list[dict]) -> dict:
    kwargs = dict(
        model=config.LM_MODEL,
        messages=build_messages(question, chunks),
        temperature=0.1,
        max_tokens=config.LM_MAX_TOKENS,
    )
    if config.LM_EXTRA_BODY:
        kwargs["extra_body"] = config.LM_EXTRA_BODY
    return kwargs


def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def generate(question: str, chunks: list[dict]) -> str:
    resp = _client.chat.completions.create(**_request_kwargs(question, chunks))
    text = _strip_thinking(resp.choices[0].message.content or "")
    if not text:
        raise RuntimeError(
            "The model returned an empty response. It may be stuck in a reasoning mode. "
            "Increase LM_MAX_TOKENS, disable thinking with LM_EXTRA_BODY, or use a model without reasoning mode."
        )
    return text


def generate_stream(question: str, chunks: list[dict]) -> Iterator[str]:
    stream = _client.chat.completions.create(**_request_kwargs(question, chunks), stream=True)
    for event in stream:
        delta = event.choices[0].delta.content
        if delta:
            yield delta


if __name__ == "__main__":
    demo_chunks = [
        {"source": "tutorial/query-params.md#defaults",
         "text": "Query parameters can be optional and can have default values. "
                 "Set the default to None to make a parameter optional: q: str | None = None."},
        {"source": "tutorial/cors.md",
         "text": "CORS refers to situations when a frontend running in a browser "
                 "has JavaScript code that communicates with a backend on a different origin."},
    ]
    t0 = time.time()
    answer = generate("How do I make a query parameter optional?", demo_chunks)
    print(answer)
    print(f"\n[response time: {time.time() - t0:.1f} s]")
# test_stream.py
import time

import generator

demo_chunks = [
    {"source": "tutorial/query-params.md#defaults",
     "text": "Query parameters can be optional and can have default values. "
             "Set the default to None to make a parameter optional: q: str | None = None."},
    {"source": "tutorial/cors.md",
     "text": "CORS refers to situations when a frontend running in a browser "
             "has JavaScript code that communicates with a backend on a different origin."},
]

question = "How do I make a query parameter optional?"
t0 = time.time()
n_events = 0

stream = generator._client.chat.completions.create(
    **generator._request_kwargs(question, demo_chunks), stream=True
)
for event in stream:
    delta = event.choices[0].delta
    # Some models provide reasoning in a separate reasoning_content field.
    reasoning = getattr(delta, "reasoning_content", None)
    if reasoning:
        print(reasoning, end="", flush=True)
    if delta.content:
        print(delta.content, end="", flush=True)
    n_events += 1

print(f"\n\n[received events: {n_events}, time: {time.time() - t0:.0f} s]")
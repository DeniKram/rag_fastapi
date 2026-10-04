import argparse
import json
import random
import re
import time
from pathlib import Path

from openai import OpenAI

import config

SECTIONS_PATH = Path("data/sections.jsonl")
OUT_PATH = Path("data/eval_questions.jsonl")

MIN_WORDS, MAX_WORDS = 40, 400
MAX_PER_FILE = 3
PROMPT_WORDS = 350
EVAL_MAX_TOKENS = 2500
GENERIC_HEADINGS = {"recap", "summary", "check it", "check the docs", "learn more",
                    "more info", "run it", "fastapi cloud", "review"}
BAD_WORDS = ("documentation", "docs", "section", "the text", "the passage", "the page", "according to")

PROMPT = """Below is a fragment of the FastAPI documentation.

Write ONE question that a developer who has NOT read this documentation would type into a search box or ask an assistant, and whose answer is contained in this fragment.

Rules:
- Do not copy phrases from the fragment; paraphrase in everyday words. Use names of functions or classes only if the question really needs them.
- Never mention "the documentation", "the text", "the section" or "the fragment".
- The question must make sense on its own and be answerable from this fragment alone.
- Give the question in English and in Russian.

Answer with JSON only, in exactly this form:
{{"en": "...", "ru": "..."}}

Fragment (page: {page}, heading: {heading}):
\"\"\"
{text}
\"\"\"
"""

_client = OpenAI(base_url=config.LM_BASE_URL, api_key=config.LM_API_KEY,
                 timeout=config.LM_TIMEOUT_S, max_retries=0)


def load_sections() -> list[dict]:
    with SECTIONS_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def pick_sections(sections: list[dict], n: int, seed: int) -> list[dict]:
    pool = [s for s in sections
            if MIN_WORDS <= s["n_words"] <= MAX_WORDS and s["heading"].lower() not in GENERIC_HEADINGS]
    random.Random(seed).shuffle(pool)
    per_file: dict[str, int] = {}
    picked = []
    for s in pool:
        if per_file.get(s["file"], 0) >= MAX_PER_FILE:
            continue
        per_file[s["file"]] = per_file.get(s["file"], 0) + 1
        picked.append(s)
        if len(picked) == n:
            break
    return picked


def short_text(text: str, max_words: int) -> str:
    words = text.split()
    return text if len(words) <= max_words else " ".join(words[:max_words]) + " ..."


def parse_json_answer(raw: str) -> dict | None:
    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not all(isinstance(data.get(k), str) and data[k].strip() for k in ("en", "ru")):
        return None
    return {"en": data["en"].strip(), "ru": data["ru"].strip()}


def lexical_overlap(question: str, section_text: str) -> float:
    q = {w for w in re.findall(r"[a-z]+", question.lower()) if len(w) > 3}
    t = set(re.findall(r"[a-z]+", section_text.lower()))
    return round(len(q & t) / len(q), 2) if q else 0.0


def ask_model(section: dict) -> dict | None:
    prompt = PROMPT.format(page=section["page_title"], heading=section["heading"],
                           text=short_text(section["text"], PROMPT_WORDS))
    kwargs = dict(model=config.LM_MODEL, messages=[{"role": "user", "content": prompt}],
                  temperature=0.5, max_tokens=EVAL_MAX_TOKENS)
    if config.LM_EXTRA_BODY:
        kwargs["extra_body"] = config.LM_EXTRA_BODY
    resp = _client.chat.completions.create(**kwargs)
    return parse_json_answer(resp.choices[0].message.content or "")


def load_done() -> set[str]:
    if not OUT_PATH.exists():
        return set()
    with OUT_PATH.open(encoding="utf-8") as f:
        return {json.loads(line)["section_id"] for line in f}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="how many questions to generate total")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sections = pick_sections(load_sections(), args.n, args.seed)
    done = load_done()
    todo = [s for s in sections if s["id"] not in done]
    print(f"Questions needed: {args.n}, already ready: {len(done)}, remaining: {len(todo)}")

    OUT_PATH.parent.mkdir(exist_ok=True)
    t_start = time.time()
    n_ok = n_fail = 0
    with OUT_PATH.open("a", encoding="utf-8") as out:
        for i, sec in enumerate(todo, start=1):
            t0 = time.time()
            try:
                answer = ask_model(sec)
            except Exception as e:
                print(f"[{i}/{len(todo)}] Request error: {type(e).__name__}: {e}")
                break
            if answer is None:
                n_fail += 1
                print(f"[{i}/{len(todo)}] {sec['id']}: model returned invalid JSON, skipping")
                continue

            record = {
                "section_id": sec["id"],
                "url": sec["url"],
                "question_en": answer["en"],
                "question_ru": answer["ru"],
                "lexical_overlap": lexical_overlap(answer["en"], sec["text"]),
                "mentions_doc": any(w in answer["en"].lower() for w in BAD_WORDS),
                "status": "unchecked",
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            n_ok += 1

            dt = time.time() - t0
            eta_min = (time.time() - t_start) / i * (len(todo) - i) / 60
            print(f"[{i}/{len(todo)}] {dt:.0f}s, remaining ~{eta_min:.0f} min | {sec['id']}")
            print(f"    EN: {answer['en']}\n    RU: {answer['ru']}  (overlap {record['lexical_overlap']})")

    print(f"\nDone: +{n_ok}, failures: {n_fail}. File: {OUT_PATH}")


if __name__ == "__main__":
    main()
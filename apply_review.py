# apply_review.py
"""
Applies verdicts from data/review_verdicts.json to data/eval_questions.jsonl.

  - questions missing from the exception list get status ok;
  - for the others, the status and note come from the file (ok / ambiguous / bad);
  - if fix_section_id is present, the correct section is replaced with the corrected one;
  - ru_ok=false means the Russian version of the question is poor and should be skipped in Russian evaluation.
The previous file is saved as data/eval_questions.before_review.jsonl.
If you already set a manual status that differs, the script will print it separately.

Run:
    py -3.12 apply_review.py
"""
import json
import shutil
from collections import Counter
from pathlib import Path

EVAL_PATH = Path("data/eval_questions.jsonl")
BACKUP_PATH = Path("data/eval_questions.before_review.jsonl")
VERDICTS_PATH = Path("data/review_verdicts.json")
SECTIONS_PATH = Path("data/sections.jsonl")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main() -> None:
    verdicts = json.loads(VERDICTS_PATH.read_text(encoding="utf-8"))["exceptions"]
    sections = {s["id"]: s for s in read_jsonl(SECTIONS_PATH)}
    rows = read_jsonl(EVAL_PATH)
    shutil.copyfile(EVAL_PATH, BACKUP_PATH)

    conflicts = []
    for r in rows:
        v = verdicts.get(r["section_id"])
        new_status = v["status"] if v else "ok"
        if r["status"] not in ("unchecked", new_status):
            conflicts.append((r["section_id"], r["status"], new_status, v["reason"] if v else ""))
        r["status"] = new_status
        r["review_note"] = v["reason"] if v else ""
        r["ru_ok"] = not (v and v.get("ru_ok") is False)
        r["reviewed_by"] = "claude"
        if v and v.get("fix_section_id"):
            r["gold_fixed_from"] = r["section_id"]
            r["section_id"] = v["fix_section_id"]
            r["url"] = sections[v["fix_section_id"]]["url"]

    with EVAL_PATH.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("Summary:", dict(Counter(r["status"] for r in rows)), "| with poor Russian:", sum(not r["ru_ok"] for r in rows))
    if conflicts:
        print("\nConflicts with your manual judgments (file verdicts were applied):")
        for sid, old, new, why in conflicts:
            print(f"  {sid}: was {old}, now {new}. {why}")
    print(f"\nBackup copy: {BACKUP_PATH}")


if __name__ == "__main__":
    main()

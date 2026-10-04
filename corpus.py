# corpus.py
"""
Step 1 of the project: turn the FastAPI documentation into a list of sections.

What the script does:
  1. Downloads the FastAPI repository (ZIP from GitHub, no git required) and extracts
     the English markdown pages and code example files (docs_src).
  2. For each page:
       - inserts code snippets instead of lines such as {* ../../docs_src/xxx.py hl[9] *}
       - removes metadata markup (///, <div>, heading anchors)
       - splits the page into sections by ## headings (the first section is the page title and intro)
  3. Saves everything to data/sections.jsonl (one JSON object per section).

Run:
    py -3.12 corpus.py                  # download if needed, then build the corpus
    py -3.12 corpus.py --raw PATH       # use an already downloaded repository folder (with docs/ and docs_src/)
"""
import argparse
import io
import json
import re
import urllib.request
import zipfile
from pathlib import Path
from markup import clean_inline

# ---------- settings ----------
ZIP_URL = "https://github.com/fastapi/fastapi/archive/refs/heads/master.zip"
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"                 # extracted content goes here
OUT_PATH = DATA_DIR / "sections.jsonl"
SITE_URL = "https://fastapi.tiangolo.com"

DOCS_SUBDIR = Path("docs/en/docs")         # English pages
INCLUDE_DIRS = ["tutorial", "advanced", "deployment", "how-to"]
INCLUDE_FILES = ["async.md", "python-types.md", "virtual-environments.md",
                 "environment-variables.md"]   # pages from the docs root

# ---------- regular expressions ----------
CODE_INCLUDE_RE = re.compile(r"^\{\*\s+(\S+)(.*?)\*\}\s*$")          # {* path ln[1:11] hl[9] *}
LN_RE = re.compile(r"\bln\[([^\]]*)\]")                               # ln[1:2,12:16] = line numbers
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*(?:\{\s*#([\w\-.]+)\s*\})?\s*$")
ADMONITION_RE = re.compile(r"^///\s*(\w+)?\s*(?:\|\s*(.*))?$")          # /// note | Title
HTML_TAG_RE = re.compile(r"</?(?:div|details|summary)[^>]*>")


# ---------- download ----------
def download_raw() -> None:
    """Downloads the repository ZIP and keeps only the needed docs: docs/en/docs/*.md, docs_src/*.py, and fastapi/*.py"""
    print(f"Downloading {ZIP_URL} (this may take a minute) ...")
    with urllib.request.urlopen(ZIP_URL, timeout=300) as resp:
        data = resp.read()
    n_files = 0
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for member in zf.namelist():
            rel = member.split("/", 1)[1] if "/" in member else ""      # remove the "fastapi-master/" prefix
            is_doc = rel.startswith("docs/en/docs/") and rel.endswith(".md")
            is_code = (rel.startswith("docs_src/") or rel.startswith("fastapi/")) and rel.endswith(".py")
            if not (is_doc or is_code):
                continue
            target = RAW_DIR / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zf.read(member))
            n_files += 1
    print(f"Extracted files: {n_files} -> {RAW_DIR}")


# ---------- clean one page ----------
def select_lines(code: str, ln_spec: str) -> str:
    """ln[1:2,12:16,29] -> keep lines 1-2, 12-16, and 29 (1-based indexing; inclusive). 
    Segments separated by gaps are joined with a '# ...' separator."""
    lines = code.splitlines()
    pieces: list[str] = []
    for part in ln_spec.split(","):
        part = part.strip()
        if not part:
            continue
        a, _, b = part.partition(":")
        start, end = int(a), int(b or a)
        pieces.append("\n".join(lines[start - 1:end]))
    return "\n# ...\n".join(pieces)


def read_code(include_path: str, options: str, base_dir: Path) -> str | None:
    """Finds the .py file referenced in {* ... *}. Paths are relative to docs/en/. If the line contains ln[...], only those lines are included."""
    candidate = (base_dir / include_path).resolve()
    if not candidate.is_file():
        return None
    code = candidate.read_text(encoding="utf-8").rstrip()
    m = LN_RE.search(options)
    return select_lines(code, m.group(1)) if m else code


def clean_page(text: str, raw_dir: Path) -> tuple[str, int, int]:
    """Returns (cleaned_text, inserted_code_count, missing_code_count)."""
    base_dir = raw_dir / "docs/en"          # paths such as ../../docs_src/... are resolved relative to this folder
    out: list[str] = []
    n_inserted = n_missing = 0
    skip_tab = False                         # inside a legacy code tab (non-Annotated)

    for line in text.splitlines():
        stripped = line.strip()

        # tabs like //// tab | Python 3.10+ non-Annotated ... ////  (duplicate examples; skip them)
        if stripped.startswith("////"):
            if skip_tab and stripped == "////":
                skip_tab = False
            elif stripped.startswith("//// tab") and "non-Annotated" in stripped:
                skip_tab = True
            continue
        if skip_tab:
            continue

        # admonition blocks /// tip, /// note | Technical Details, closing ///
        m = ADMONITION_RE.match(stripped)
        if m and stripped.startswith("///"):
            kind, title = m.group(1), m.group(2)
            if kind:                         # opening line
                label = kind.capitalize() + (f": {title}" if title else "")
                out.append(f"**{label}**")
            continue                         # skip the closing /// line

        # insert code
        m = CODE_INCLUDE_RE.match(stripped)
        if m:
            code = read_code(m.group(1), m.group(2), base_dir)
            if code is None:
                n_missing += 1
            else:
                n_inserted += 1
                out.append("```python\n" + code + "\n```")
            continue

        line = HTML_TAG_RE.sub("", line)
        out.append(line)

    cleaned = clean_inline("\n".join(out))
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned, n_inserted, n_missing


# ---------- split a page into sections ----------
def page_url(rel_md: Path, anchor: str | None) -> str:
    """tutorial/query-params.md + 'defaults' -> https://.../tutorial/query-params/#defaults"""
    parts = list(rel_md.with_suffix("").parts)
    if parts and parts[-1] == "index":
        parts = parts[:-1]
    url = f"{SITE_URL}/" + "/".join(parts) + ("/" if parts else "")
    return url + (f"#{anchor}" if anchor else "")


def split_sections(cleaned: str, rel_md: Path) -> list[dict]:
    """Split by ## headings; lines with # inside code blocks are not treated as headings."""
    sections: list[dict] = []
    page_title = rel_md.stem
    page_anchor = None
    cur = {"heading": None, "anchor": None, "lines": []}
    in_code = False

    def flush():
        body = "\n".join(cur["lines"]).strip()
        if body:
            sections.append({"heading": cur["heading"], "anchor": cur["anchor"], "text": body})

    for line in cleaned.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
        m = None if in_code else HEADING_RE.match(line)
        if m and len(m.group(1)) == 1:       # page title (#)
            page_title, page_anchor = m.group(2), m.group(3)
            cur["heading"], cur["anchor"] = page_title, page_anchor
            continue
        if m and len(m.group(1)) == 2:       # new section (##)
            flush()
            cur = {"heading": m.group(2), "anchor": m.group(3), "lines": []}
            continue
        cur["lines"].append(line)
    flush()

    result = []
    for s in sections:
        # for the intro section (before the first ##), the link points to the page itself
        is_intro = (s["heading"] == page_title and s["anchor"] == page_anchor)
        anchor = None if is_intro else s["anchor"]
        result.append({
            "id": f"{rel_md.as_posix()}#{anchor or 'intro'}",
            "file": rel_md.as_posix(),
            "page_title": page_title,
            "heading": s["heading"] or page_title,
            "anchor": anchor,
            "url": page_url(rel_md, anchor),
            "text": s["text"],
            "n_words": len(s["text"].split()),
        })
    return result


# ---------- main pass ----------
def collect_files(docs_root: Path) -> list[Path]:
    files: list[Path] = []
    for d in INCLUDE_DIRS:
        files += sorted((docs_root / d).rglob("*.md"))
    for f in INCLUDE_FILES:
        if (docs_root / f).is_file():
            files.append(docs_root / f)
    return files


def build(raw_dir: Path) -> list[dict]:
    docs_root = raw_dir / DOCS_SUBDIR
    all_sections: list[dict] = []
    total_inserted = total_missing = 0
    for path in collect_files(docs_root):
        rel_md = path.relative_to(docs_root)
        cleaned, ins, miss = clean_page(path.read_text(encoding="utf-8"), raw_dir)
        total_inserted += ins
        total_missing += miss
        all_sections += split_sections(cleaned, rel_md)
    print(f"Code inserted: {total_inserted} times, not found: {total_missing}")
    return all_sections


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=None,
                    help="folder with an already downloaded repository (with docs/ and docs_src/ inside)")
    args = ap.parse_args()

    raw_dir = args.raw or RAW_DIR
    # download if pages are missing or the fastapi/ source tree is missing (it is referenced by one of the pages)
    if not (raw_dir / DOCS_SUBDIR).is_dir() or not (raw_dir / "fastapi").is_dir():
        download_raw()

    sections = build(raw_dir)
    DATA_DIR.mkdir(exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for s in sections:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    words = [s["n_words"] for s in sections]
    files = {s["file"] for s in sections}
    print(f"Files: {len(files)}, sections: {len(sections)}, total words: {sum(words)}")
    print(f"Words per section: min {min(words)}, median {sorted(words)[len(words)//2]}, max {max(words)}")
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
# markup.py
"""
Remove embedded markup from documentation text: images, links, HTML tags.
The goal is to keep meaningful words in the chunk text instead of distracting formatting noise.
We do not touch code blocks except for console color tags.
"""
import re

IMG_TAG_RE = re.compile(r"<img\b[^>]*>")
MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")           # [text](url) -> text
HTML_TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>")              # <abbr title="..">word</abbr> -> word
SUBHEADING_ANCHOR_RE = re.compile(r"^(#{3,6}\s+.*?)\s*\{\s*#[\w\-.]+\s*\}\s*$")  # "### Text { #anchor }" -> "### Text"
CONSOLE_COLOR_RE = re.compile(r"</?(?:font|span|b|u)\b[^>]*>")


def clean_prose(line: str) -> str:
    line = IMG_TAG_RE.sub("", line)
    line = MD_IMAGE_RE.sub("", line)
    line = MD_LINK_RE.sub(r"\1", line)
    line = HTML_TAG_RE.sub("", line)
    line = SUBHEADING_ANCHOR_RE.sub(r"\1", line)   # keep heading anchors untouched: corpus.py uses them for links
    return line


def clean_inline(text: str) -> str:
    out: list[str] = []
    fence = None                                  # None or the language for the current code block
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            fence = None if fence is not None else (stripped[3:].strip() or "plain")
            out.append(line)
            continue
        if fence is None:
            line = clean_prose(line)
            if not line.strip() and stripped:     # the line contained only an image
                continue
        elif fence == "console":
            line = CONSOLE_COLOR_RE.sub("", line)
        out.append(line)
    return "\n".join(out)
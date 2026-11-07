"""Turn Claude's markdown replies into text that sounds natural when spoken.

Kept free of heavy imports (no numpy) so the Claude Code hooks stay fast."""
from __future__ import annotations

import re

SENTENCE_END = re.compile(r"(?<!\b(?:Mr|Ms|Dr|St|vs))[.!?;:]\s+|\n+")

# Common file extensions, spoken as "dot <ext>"; anything else keeps the plain dot.
_EXT = (r"py|pyc|js|jsx|ts|tsx|mjs|cjs|json|jsonl|md|txt|toml|yaml|yml|ini|cfg|lock|sh|zsh|"
        r"bash|html|css|scss|sql|go|rs|rb|java|kt|swift|c|h|cpp|hpp|cs|php|xml|csv|env|log|"
        r"wav|mp3|png|jpg|jpeg|gif|svg|pdf|dart|vue|svelte|lua|r|ipynb|onnx|bin|zip|gz|tar")
_FILE = re.compile(rf"\b([\w-]+)\.({_EXT})\b(?::(\d+))?", re.I)
# A path: two or more segments joined by "/" (optionally starting with ~/ ./ or /).
_PATH = re.compile(r"(?<![\w:/])((?:~|\.{1,2})?/?)((?:[\w.@-]+/)+)([\w.@-]+)")


def _speak_path(m: re.Match) -> str:
    """Say only the last part of a path; leave "read/write" or "50/50" alone."""
    prefix, dirs, last = m.groups()
    if prefix or dirs.count("/") >= 2 or _FILE.fullmatch(last):
        return last
    return m.group(0)
_SLASH_CMD = re.compile(r"(?<![\w/])/([a-z][\w-]*)(?::([\w-]+))?\b")
_VERSION = re.compile(r"\bv?(\d+(?:\.\d+){1,3})\b(?=(?:[^\d.]|\.\s|\.$|$))")
_UNITS = {
    "ms": "milliseconds", "s": "seconds", "sec": "seconds", "min": "minutes", "h": "hours",
    "kb": "kilobytes", "mb": "megabytes", "gb": "gigabytes", "tb": "terabytes",
    "kib": "kibibytes", "mib": "mebibytes", "gib": "gibibytes",
    "khz": "kilohertz", "hz": "hertz", "px": "pixels",
}
_UNIT_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s?(" + "|".join(_UNITS) + r")\b", re.I)
_ABBREV = [
    (r"\be\.g\.", "for example"), (r"\bi\.e\.", "that is"), (r"\betc\.", "et cetera"),
    (r"\bvs\.?(?=\s)", "versus"), (r"\bapprox\.", "approximately"),
]
_SYMBOLS = [
    (r"\s*(?:→|->|=>)\s*", " to "), (r"\s*(?:←|<-)\s*", " from "), (r"\s&\s", " and "),
    (r"(?<=\s)~(?=\d)", "about "), (r"(?<=\d)\s?×\s?(?=\d)", " by "), (r"≈", "about "),
    (r"≥", "at least "), (r"≤", "at most "), (r"\s+/\s+", " or "),
]
# Emoji and pictographs (incl. variation selectors and joiners) are silently dropped.
_EMOJI = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF\uFE0F\u200D\u20E3]+")


def _split_identifier(word: str) -> str:
    """speak_detached -> speak detached; parseConfig -> parse config."""
    word = word.replace("_", " ").replace("-", " ") if "_" in word else word
    word = re.sub(r"(?<=[a-z0-9])(?=[A-Z][a-z])", " ", word)
    return word.strip()


def _speak_file(m: re.Match) -> str:
    name, ext, line = m.group(1), m.group(2).lower(), m.group(3)
    out = f"{_split_identifier(name)} dot {ext}"
    return f"{out}, line {line}" if line else out


def _speak_version(m: re.Match) -> str:
    whole, num = m.group(0), m.group(1)
    parts = num.split(".")
    if not whole.startswith("v") and len(parts) == 2:
        return whole  # 3.5 is an ordinary decimal
    spoken = " point ".join(parts)
    return f"version {spoken}" if whole.startswith("v") else spoken


def _speak_unit(m: re.Match) -> str:
    num, unit = m.group(1), m.group(2).lower()
    if unit == "s" and re.fullmatch(r"1[5-9]\d0|20\d0", num):
        return m.group(0)  # "the 1990s"
    word = _UNITS[unit]
    if num == "1":
        word = word.removesuffix("s")
    return f"{num} {word}"


def clean_for_speech(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.S)           # drop code blocks
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)          # [label](url) -> label
    text = re.sub(r"https?://\S+", "a link", text)
    text = _EMOJI.sub("", text)
    # inline code: read identifiers as words
    text = re.sub(r"`([^`\n]+)`", lambda m: " " + m.group(1) + " ", text)
    text = _PATH.sub(_speak_path, text)                           # src/a/hooks.py -> hooks.py
    text = _SLASH_CMD.sub(lambda m: " ".join(filter(None, m.groups())), text)  # /jarvis:talk
    text = _FILE.sub(_speak_file, text)
    text = re.sub(r"\b[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+\b", lambda m: _split_identifier(m.group(0)), text)
    text = re.sub(r"\b[a-z]{2,}(?:[A-Z][a-z0-9]+)+\b",lambda m: _split_identifier(m.group(0)), text)
    for pat, rep in _ABBREV:
        text = re.sub(pat, rep, text, flags=re.I)
    for pat, rep in _SYMBOLS:
        text = re.sub(pat, rep, text)
    text = _VERSION.sub(_speak_version, text)
    text = _UNIT_RE.sub(_speak_unit, text)
    text = re.sub(r"[`*_#>|~]", "", text)
    text = re.sub(r"^\s*(?:[-•]|\d+[.)])\s+", "", text, flags=re.M)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str) -> list[str]:
    out, buf = [], text
    while (m := SENTENCE_END.search(buf)):
        out.append(buf[: m.end()])
        buf = buf[m.end():]
    if buf.strip():
        out.append(buf)
    return [s for s in out if s.strip()]

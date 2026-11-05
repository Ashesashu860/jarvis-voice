"""Claude Code hook handlers. Each reads the hook JSON from stdin, never blocks
Claude Code for long, and always exits 0 so a voice problem can't break a session."""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import tempfile

from . import config
from .brain import claude_code_note
from .voice import clean_for_speech, split_sentences


def _read_input() -> dict:
    try:
        return json.load(sys.stdin)
    except Exception:
        return {}


def _active(cfg: dict) -> bool:
    # Sessions started by `jarvis talk` speak for themselves.
    return cfg["speak_replies"] and not os.environ.get("JARVIS_VOICE_CHILD")


# ----------------------------------------------------------------------------- speaking process control
def hush() -> None:
    """Stop whatever Jarvis is currently saying."""
    try:
        pid = int(config.PID_FILE.read_text().strip())
        os.killpg(pid, signal.SIGTERM)
    except (FileNotFoundError, ValueError, ProcessLookupError, PermissionError):
        pass
    try:
        config.PID_FILE.unlink()
    except FileNotFoundError:
        pass


def speak_detached(text: str) -> None:
    """Speak `text` in a background process so the hook returns immediately."""
    hush()
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fd, path = tempfile.mkstemp(prefix="say-", suffix=".txt", dir=config.CACHE_DIR)
    with os.fdopen(fd, "w") as f:
        f.write(text)
    proc = subprocess.Popen([sys.executable, "-m", "jarvis_voice", "say", "--file", path, "--delete"],
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            start_new_session=True)
    config.PID_FILE.write_text(str(proc.pid))


# ----------------------------------------------------------------------------- reply extraction
def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
    return ""


def last_reply_from_transcript(path: str) -> str:
    """Assistant text written since the user's last real prompt."""
    texts: list[str] = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("isSidechain"):
                    continue
                kind = entry.get("type")
                msg = entry.get("message") or {}
                if kind == "user":
                    content = msg.get("content")
                    is_tool_result = isinstance(content, list) and any(
                        isinstance(b, dict) and b.get("type") == "tool_result" for b in content)
                    if not is_tool_result and not entry.get("isMeta"):
                        texts = []
                elif kind == "assistant":
                    t = _text_of(msg.get("content"))
                    if t.strip():
                        texts.append(t)
    except OSError:
        return ""
    # The final text block is the answer; earlier ones are mid-task narration.
    return texts[-1] if texts else ""


def spoken_excerpt(reply: str, max_chars: int) -> str:
    """The part of a reply worth saying aloud: prose before the first code block/table,
    cut at a sentence boundary."""
    reply = re.split(r"\n```|\n\|", reply, maxsplit=1)[0]
    sentences = split_sentences(clean_for_speech(reply))
    out, total = [], 0
    for s in sentences:
        if out and total + len(s) > max_chars:
            out.append("The rest is on screen.")
            break
        out.append(s.strip())
        total += len(s)
    return " ".join(out)


# ----------------------------------------------------------------------------- hook entry points
def on_stop() -> None:
    data = _read_input()
    cfg = config.load()
    if not _active(cfg):
        return
    reply = data.get("last_assistant_message") or ""
    if not reply and data.get("transcript_path"):
        reply = last_reply_from_transcript(data["transcript_path"])
    text = spoken_excerpt(reply, cfg["max_spoken_chars"])
    if text:
        speak_detached(text)


def on_prompt() -> None:
    _read_input()
    hush()  # you started talking: Jarvis stops
    cfg = config.load()
    if _active(cfg) and cfg["persona_in_claude_code"]:
        print(json.dumps({"hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": claude_code_note(cfg["address"]),
        }}))


def run(event: str) -> int:
    try:
        {"stop": on_stop, "prompt": on_prompt}[event]()
    except Exception as e:  # never break the user's Claude Code session
        print(f"jarvis hook {event}: {type(e).__name__}: {e}", file=sys.stderr)
    return 0

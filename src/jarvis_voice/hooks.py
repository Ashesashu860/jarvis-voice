"""Claude Code hook handlers. Each reads the hook JSON from stdin, never blocks
Claude Code for long, and always exits 0 so a voice problem can't break a session."""
from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time

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


def speak_detached(text: str, listen: bool = False) -> None:
    """Speak `text` in a background process so the hook returns immediately.
    With `listen`, it also hears the user talking over it (see converse.speak_listening)."""
    hush()
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fd, path = tempfile.mkstemp(prefix="say-", suffix=".txt", dir=config.CACHE_DIR)
    with os.fdopen(fd, "w") as f:
        f.write(text)
    cmd = [sys.executable, "-m", "jarvis_voice", "say", "--file", path, "--delete"]
    if listen:
        cmd.append("--listen")
    proc = subprocess.Popen(cmd,
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
    from . import converse

    data = _read_input()
    cfg = config.load()
    if os.environ.get("JARVIS_VOICE_CHILD"):
        return
    conversing = converse.claim(data.get("session_id"))
    if not (conversing or cfg["speak_replies"]):
        return
    reply = data.get("last_assistant_message") or ""
    if not reply and data.get("transcript_path"):
        reply = last_reply_from_transcript(data["transcript_path"])
    text = spoken_excerpt(reply, cfg["max_spoken_chars"])
    if not conversing:
        if text:
            speak_detached(text)
        return
    # Conversation mode: speak, listen, and hand the user's reply straight back to Claude.
    converse.wait_for_capture()  # let a progress update finish catching what they said
    hush()
    pending = converse.pop_all()
    if pending:  # they spoke over a progress update and Claude hasn't seen it yet
        print(json.dumps({"decision": "block",
                          "reason": converse.continuation(" ".join(pending), cfg["address"])}))
        return
    heard = converse.turn(cfg, text)
    if heard is None:
        converse.end()
        return
    print(json.dumps({"decision": "block", "reason": converse.continuation(heard, cfg["address"])}))


def on_prompt() -> None:
    _read_input()
    hush()  # you started talking: Jarvis stops
    cfg = config.load()
    from .converse import is_on
    voiced = _active(cfg) or (is_on() and not os.environ.get("JARVIS_VOICE_CHILD"))
    if voiced and cfg["persona_in_claude_code"]:
        print(json.dumps({"hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": claude_code_note(cfg["address"]),
        }}))


# ----------------------------------------------------------------------------- progress narration
NARRATE_FILE = config.CACHE_DIR / "narrate.json"
TOOL_GAP_S = 12.0  # at most one "Editing cli.py"-style update this often


def _base(path) -> str:
    return os.path.basename(str(path or "")).replace("_", " ") or "a file"


def describe_tool(name: str, inp: dict) -> str | None:
    """A short spoken line for what a tool call is doing, or None to stay quiet."""
    inp = inp or {}
    if name == "Bash":
        desc = (inp.get("description") or "").strip().rstrip(".")
        return f"{desc}." if desc else "Running a command."
    if name in ("Edit", "MultiEdit", "NotebookEdit"):
        return f"Editing {_base(inp.get('file_path') or inp.get('notebook_path'))}."
    if name == "Write":
        return f"Writing {_base(inp.get('file_path'))}."
    if name == "Read":
        return f"Reading {_base(inp.get('file_path'))}."
    if name in ("Grep", "Glob"):
        return "Searching the code."
    if name in ("WebSearch", "WebFetch"):
        return "Looking it up online."
    if name in ("Agent", "Task"):
        desc = (inp.get("description") or "").strip().rstrip(".")
        return f"Delegating: {desc}." if desc else "Handing part of this to a helper."
    return None


def _speaking() -> bool:
    """Is a detached `jarvis say` still talking? (Tool updates never cut it off.)"""
    try:
        os.kill(int(config.PID_FILE.read_text().strip()), 0)
        return True
    except (FileNotFoundError, ValueError, ProcessLookupError):
        return False
    except PermissionError:
        return True


def _narrate_state() -> dict:
    try:
        return json.loads(NARRATE_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def on_tool() -> None:
    """PreToolUse: say what Claude is doing. New narration from Claude ("Now I'll fix
    the tests") is always spoken; otherwise a brief description of the tool, rate-limited."""
    from . import converse

    data = _read_input()
    cfg = config.load()
    if os.environ.get("JARVIS_VOICE_CHILD") or not cfg["narrate"]:
        return
    conversing = converse.claim(data.get("session_id"))
    if not (cfg["speak_replies"] or conversing):
        return
    if converse.capturing():  # the user is talking to us: don't speak over them
        return
    st = _narrate_state()
    session = data.get("session_id")
    if st.get("session") != session:
        st = {"session": session}
    now = time.time()

    line = None
    note = last_reply_from_transcript(data["transcript_path"]) if data.get("transcript_path") else ""
    note = spoken_excerpt(note, 220) if note else ""
    key = hashlib.sha1(note.encode()).hexdigest() if note else None
    if note and key != st.get("note"):
        line, st["note"] = note, key
    elif now - st.get("tool_at", 0) >= TOOL_GAP_S and not _speaking():
        line = describe_tool(data.get("tool_name", ""), data.get("tool_input") or {})
    if not line:
        return
    st["tool_at"] = now
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    NARRATE_FILE.write_text(json.dumps(st))
    # In a conversation, keep the mic open while speaking so the user can cut in.
    speak_detached(line, listen=conversing)


def on_tool_done() -> None:
    """PostToolUse: pass anything the user said over a progress update on to Claude."""
    from . import converse

    data = _read_input()
    if os.environ.get("JARVIS_VOICE_CHILD") or not converse.claim(data.get("session_id")):
        return
    pending = converse.pop_all()
    if pending:
        print(json.dumps({"hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "additionalContext": converse.interjection(pending, config.load()["address"]),
        }}))


def run(event: str) -> int:
    try:
        {"stop": on_stop, "prompt": on_prompt, "tool": on_tool, "tool-done": on_tool_done}[event]()
    except Exception as e:  # never break the user's Claude Code session
        print(f"jarvis hook {event}: {type(e).__name__}: {e}", file=sys.stderr)
    return 0

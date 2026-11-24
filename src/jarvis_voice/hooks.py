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
from .text import clean_for_speech, split_sentences


def _read_input() -> dict:
    try:
        return json.load(sys.stdin)
    except Exception:
        return {}


def _active(cfg: dict) -> bool:
    # Sessions started by `jarvis talk` speak for themselves.
    return cfg["speak_replies"] and not os.environ.get("JARVIS_VOICE_CHILD")


# ----------------------------------------------------------------------------- speaking process control
# Replies and updates go to the speech daemon (fast, session-aware). With the daemon
# turned off, each runs in its own process, tracked by PID_FILE as {"pid": ..., "session": ...}.
# During /jarvis:talk a separate watcher process listens while Claude works (converse.watch).
NARRATION_TTL_S = 15.0  # a progress update that can't start this soon is dropped


def _speaker_proc() -> tuple[int | None, str | None]:
    try:
        raw = config.PID_FILE.read_text().strip()
    except FileNotFoundError:
        return None, None
    try:
        st = json.loads(raw)
        if isinstance(st, int):
            return st, None
        return int(st["pid"]), st.get("session")
    except (ValueError, KeyError, TypeError):
        return None, None


def hush(session: str | None = None) -> None:
    """Stop what Jarvis is saying: for one Claude Code session, or (None) everywhere."""
    from . import daemon
    daemon.request("hush", session=session)
    pid, owner = _speaker_proc()
    if pid is None or (session is not None and owner not in (None, session)):
        return
    try:
        os.killpg(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        config.PID_FILE.unlink()
    except FileNotFoundError:
        pass


def speak_detached(text: str, session: str | None = None, ttl: float | None = None) -> None:
    """Speak `text` without blocking the hook."""
    if config.load()["speech_daemon"]:
        from . import daemon
        pid, owner = _speaker_proc()
        if pid is not None and owner in (None, session):
            hush(session)  # a listening update from this session is still going
        daemon.speak(text, session=session, ttl=ttl)
        return
    hush(session)
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fd, path = tempfile.mkstemp(prefix="say-", suffix=".txt", dir=config.CACHE_DIR)
    with os.fdopen(fd, "w") as f:
        f.write(text)
    cmd = [sys.executable, "-m", "jarvis_voice", "say", "--file", path, "--delete"]
    proc = subprocess.Popen(cmd,
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            start_new_session=True)
    config.PID_FILE.write_text(json.dumps({"pid": proc.pid, "session": session}))


# ----------------------------------------------------------------------------- reply extraction
def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
    return ""


def _lines_backwards(path: str, chunk: int = 1 << 16):
    """Yield a file's lines last-first, reading from the end in chunks, so the cost
    depends on how far back we look rather than on how long the transcript is."""
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        pos, tail = f.tell(), b""
        while pos > 0:
            step = min(chunk, pos)
            pos -= step
            f.seek(pos)
            parts = (f.read(step) + tail).split(b"\n")
            tail = parts.pop(0)  # may be cut off: complete it with the next chunk
            for line in reversed(parts):
                if line.strip():
                    yield line.decode("utf-8", "replace")
        if tail.strip():
            yield tail.decode("utf-8", "replace")


def last_reply_from_transcript(path: str) -> str:
    """The latest assistant text written since the user's last real prompt.

    Called on every tool use, so it scans from the end of the transcript and stops at
    the first answer (or at the user's prompt) instead of parsing the whole file."""
    try:
        for line in _lines_backwards(path):
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
                    return ""  # reached the user's prompt: nothing said since
            elif kind == "assistant":
                t = _text_of(msg.get("content"))
                if t.strip():
                    return t  # the final text block is the answer
    except OSError:
        return ""
    return ""


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
            speak_detached(text, session=data.get("session_id"))
        return
    # Conversation mode: speak, listen, and hand the user's reply straight back to Claude.
    from . import daemon
    converse.wait_for_capture()  # let the watcher finish catching what they said
    converse.stop_watcher()      # the turn below opens the mic itself
    stopped = converse.halted(data.get("session_id"))
    converse.clear_halt()
    daemon.settle()              # finish the sentence being said, then go quiet
    hush()
    pending = converse.pop_all()
    if pending:  # they spoke over a progress update and Claude hasn't seen it yet
        print(json.dumps({"decision": "block",
                          "reason": converse.continuation(" ".join(pending), cfg["address"])}))
        return
    if stopped:  # they asked Claude to stop: skip reading the reply, just listen
        text = ""
    heard = converse.turn(cfg, text)
    if heard is None:
        converse.end()
        return
    print(json.dumps({"decision": "block", "reason": converse.continuation(heard, cfg["address"])}))


def on_prompt() -> None:
    data = _read_input()
    hush(data.get("session_id"))  # you started typing here: this session's Jarvis stops
    cfg = config.load()
    from .converse import TALK_COMMAND, is_on, note_owner
    if TALK_COMMAND.match(data.get("prompt") or "") and not os.environ.get("JARVIS_VOICE_CHILD"):
        note_owner(data.get("session_id"))  # this window owns the conversation
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
    """Is Jarvis talking in any session? (Tool updates never cut it off.)"""
    from . import daemon
    st = daemon.request("status")
    if st and (st.get("busy") or st.get("queued")):
        return True
    pid, _ = _speaker_proc()
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _narrate_states() -> dict:
    """Per-session narration state: {session_id: {"note": ..., "tool_at": ...}}."""
    try:
        st = json.loads(NARRATE_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    if not isinstance(st, dict) or "session" in st:  # old single-session format
        return {}
    cutoff = time.time() - 6 * 3600
    return {k: v for k, v in st.items() if isinstance(v, dict) and v.get("seen", 0) > cutoff}


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
    session = data.get("session_id")
    if conversing:
        stop_text = converse.halted(session)
        if stop_text:  # the user said "stop" while Claude was working
            print(json.dumps({"hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": (
                    f"The user said aloud \"{stop_text}\" (voice conversation): they want you to stop "
                    "this task now. Do not call any more tools. End your turn with one short "
                    "sentence saying where you stopped."),
            }}))
            return
        if cfg["listen_while_working"]:
            converse.ensure_watcher(session)
    if converse.capturing():  # the user is talking to us: don't speak over them
        return
    states = _narrate_states()
    st = states.setdefault(str(session), {})
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
    st["tool_at"] = st["seen"] = now
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    NARRATE_FILE.write_text(json.dumps(states))
    speak_detached(line, session=session, ttl=NARRATION_TTL_S)


def on_tool_done() -> None:
    """PostToolUse: pass anything the user said over a progress update on to Claude."""
    from . import converse

    data = _read_input()
    if os.environ.get("JARVIS_VOICE_CHILD") or not converse.claim(data.get("session_id")):
        return
    converse.heartbeat()  # a long task is still going: keep the watcher listening
    pending = converse.pop_all()
    if pending:
        print(json.dumps({"hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "additionalContext": converse.interjection(pending, config.load()["address"]),
        }}))


def _protect_stdout() -> None:
    """Claude Code parses a hook's stdout as JSON. C libraries (PortAudio prints
    "||PaMacCore (AUHAL)|| Error ..." when playback is cut off) write to fd 1 directly,
    which would corrupt it and silently drop the decision. Point fd 1 at stderr and
    give Python's print() a private copy of the real stdout."""
    try:
        real = os.dup(1)
        os.dup2(2, 1)
        sys.stdout = os.fdopen(real, "w", buffering=1)
    except OSError:
        pass


def run(event: str) -> int:
    _protect_stdout()
    try:
        {"stop": on_stop, "prompt": on_prompt, "tool": on_tool, "tool-done": on_tool_done}[event]()
    except Exception as e:  # never break the user's Claude Code session
        print(f"jarvis hook {event}: {type(e).__name__}: {e}", file=sys.stderr)
    finally:
        sys.stdout.flush()
    return 0

"""Hands-free conversation inside an interactive Claude Code session.

`/jarvis:talk` arms conversation mode and takes the first spoken prompt. From then on,
when Claude finishes a reply the Stop hook speaks it (you can talk over it), listens for
a follow-up, and hands what you said back to Claude by blocking the stop with your words
as the reason. Silence for `converse_timeout` seconds (default 60), or "that's all", ends the conversation
and the session goes back to normal typing."""
from __future__ import annotations

import json
import time

from . import config

STATE_FILE = config.CACHE_DIR / "converse.json"
STALE_S = 6 * 3600  # an armed conversation nobody claimed for this long is forgotten


# ----------------------------------------------------------------------------- state
def start() -> None:
    """Arm conversation mode; the next Claude Code session to finish a reply claims it."""
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps({"session": None, "since": time.time()}))


def end() -> None:
    for f in (STATE_FILE, INBOX_FILE):
        try:
            f.unlink()
        except FileNotFoundError:
            pass


def is_on() -> bool:
    return _state() is not None


def _state() -> dict | None:
    try:
        st = json.loads(STATE_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    if st.get("session") is None and time.time() - st.get("since", 0) > STALE_S:
        end()
        return None
    return st


def claim(session_id: str | None) -> bool:
    """True if conversation mode belongs to this session (claiming it if unclaimed)."""
    st = _state()
    if st is None:
        return False
    if st.get("session") is None:
        st["session"] = session_id
        STATE_FILE.write_text(json.dumps(st))
        return True
    return st["session"] == session_id


# ----------------------------------------------------------------------------- inbox
# Things the user said over Jarvis while Claude was still working. The next tool hook
# (or the Stop hook) hands them to Claude.
INBOX_FILE = config.CACHE_DIR / "inbox.jsonl"


def push(text: str) -> None:
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(INBOX_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps({"text": text, "at": time.time()}) + "\n")


def pop_all() -> list[str]:
    try:
        lines = INBOX_FILE.read_text(encoding="utf-8").splitlines()
        INBOX_FILE.unlink()
    except FileNotFoundError:
        return []
    out = []
    for line in lines:
        try:
            out.append(json.loads(line)["text"])
        except (json.JSONDecodeError, KeyError):
            pass
    return out


def interjection(texts: list[str], address: str) -> str:
    said = " / ".join(f"\"{t}\"" for t in texts)
    return (
        "While you were working, the user spoke to you aloud (speech-to-text, may contain "
        f"recognition errors): {said}. Take this into account right away: if it changes or adds "
        "to the task, adjust; if it's a question, answer it briefly in your next update. "
        f"Address the user as '{address}'."
    )


def speak_listening(cfg: dict, text: str) -> None:
    """Speak `text` (a progress update) while listening; if the user talks over it,
    stop, catch what they say and leave it in the inbox for Claude."""
    from .cli import is_pause_phrase
    from .ears import BargeIn, Listener, Transcriber, is_noise
    from .voice import chime, make_speaker, split_sentences

    speaker = make_speaker(cfg)
    listener = Listener(cfg["silence"], 60.0, cfg["sensitivity"], calibrate_s=0.4)
    listener.start_continuous()
    try:
        barge = BargeIn(listener, speaker, speaker.stop)
        with barge.watch():
            for s in split_sentences(text):
                speaker.say(s)
            speaker.wait()
        if not barge.triggered:
            return
        speaker.stop()
        CAPTURE_FILE.write_text(str(time.time()))  # don't let the next update cut us off
        audio = listener.listen(prefix=barge.captured)
        if audio is None:
            return
        chime("Pop", cfg)
        heard = Transcriber(cfg["whisper"])(audio)
        if heard and not is_noise(heard) and not is_pause_phrase(heard):
            push(heard)
            speaker.say(f"Noted, {cfg['address']}.")
            speaker.wait()
    finally:
        speaker.stop()
        listener.close()
        capture_done()


CAPTURE_FILE = config.CACHE_DIR / "capturing"


def capturing() -> bool:
    """Is a background update busy recording/transcribing the user right now?"""
    try:
        return time.time() - float(CAPTURE_FILE.read_text()) < 90
    except (FileNotFoundError, ValueError):
        return False


def capture_done() -> None:
    try:
        CAPTURE_FILE.unlink()
    except FileNotFoundError:
        pass


def wait_for_capture(limit_s: float = 90) -> None:
    end = time.time() + limit_s
    while capturing() and time.time() < end:
        time.sleep(0.2)


# ----------------------------------------------------------------------------- one turn
def turn(cfg: dict, text: str) -> str | None:
    """Speak `text`, then listen for the user's reply. Returns what they said, or None
    when the conversation is over (silence, or a sleep/exit phrase)."""
    from .cli import EXIT_PHRASES, is_pause_phrase, is_sleep_phrase
    from .ears import BargeIn, Listener, Transcriber, is_noise, normalize
    from .voice import chime, make_speaker

    address = cfg["address"]
    speaker = make_speaker(cfg)
    listener = Listener(cfg["silence"], 60.0, cfg["sensitivity"], calibrate_s=0.6)
    transcriber = Transcriber(cfg["whisper"])
    listener.start_continuous()

    def say(line: str) -> None:
        speaker.say(line)
        speaker.wait()

    barged = None
    try:
        if text:
            if cfg["barge_in"]:
                barge = BargeIn(listener, speaker, speaker.stop)
                with barge.watch():
                    speaker.say(text)
                    speaker.wait()
                barged = barge.captured
            else:
                say(text)

        deadline = time.time() + float(cfg["converse_timeout"])
        while True:
            if barged is not None:  # they're mid-sentence: keep recording, no chime
                audio, barged, interrupted = listener.listen(prefix=barged), None, True
            else:
                interrupted = False
                chime("Tink", cfg, wait=True)
                listener.flush()
                remaining = deadline - time.time()
                if remaining <= 0:
                    audio = None
                    listener.timed_out = True
                else:
                    audio = listener.listen(start_timeout=remaining)
            if audio is None:
                if listener.timed_out:
                    chime("Bottle", cfg, wait=True)
                    return None
                continue
            chime("Pop", cfg)
            heard = transcriber(audio)
            if is_noise(heard):
                continue
            if interrupted and is_pause_phrase(heard):
                deadline = time.time() + float(cfg["converse_timeout"])
                continue
            if is_sleep_phrase(heard) or normalize(heard) in EXIT_PHRASES:
                say(f"Very good, {address}. Standing by.")
                return None
            return heard
    finally:
        speaker.stop()
        listener.close()


def continuation(heard: str, address: str) -> str:
    """The Stop-hook reason that feeds the user's spoken reply back to Claude."""
    from .brain import claude_code_note
    return (
        "The user replied aloud in voice conversation mode. Treat this speech-to-text transcript "
        "exactly as their next prompt (it may contain recognition errors; infer the meaning) and "
        f"respond to it now: \"{heard}\"\n\n{claude_code_note(address)}"
    )

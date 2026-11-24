"""Hands-free conversation inside an interactive Claude Code session.

`/jarvis:talk` arms conversation mode and takes the first spoken prompt. From then on,
when Claude finishes a reply the Stop hook speaks it (you can talk over it), listens for
a follow-up, and hands what you said back to Claude by blocking the stop with your words
as the reason. Silence for `converse_timeout` seconds (default 60), or "that's all", ends the conversation
and the session goes back to normal typing."""
from __future__ import annotations

import json
import re
import time

from . import config

STATE_FILE = config.CACHE_DIR / "converse.json"
STALE_S = 6 * 3600  # an armed conversation nobody claimed for this long is forgotten


# ----------------------------------------------------------------------------- state
def start() -> None:
    """Arm conversation mode; the next Claude Code session to finish a reply claims it."""
    stop_watcher()
    clear_halt()
    config.CACHE_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    config.write_json(STATE_FILE, {"session": _owner_hint(), "since": time.time()})


def end() -> None:
    stop_watcher()
    for f in (STATE_FILE, INBOX_FILE, HALT_FILE, OWNER_FILE):
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
    """True if conversation mode belongs to this session (claiming it if unclaimed).

    An unclaimed conversation goes only to the session the user typed /jarvis:talk in
    (see `note_owner`), so a busy second window's tool hooks can't take it over."""
    st = _state()
    if st is None:
        return False
    if st.get("session") is None:
        owner = _owner_hint()
        if owner is not None and owner != session_id:
            return False
        st["session"] = session_id
        config.write_json(STATE_FILE, st)
        return True
    return st["session"] == session_id


# The UserPromptSubmit hook sees which session /jarvis:talk was typed in; the command's
# own `jarvis converse` doesn't. Either may run first, so both sides check this hint.
OWNER_FILE = config.CACHE_DIR / "converse-owner.json"
OWNER_TTL_S = 120.0
TALK_COMMAND = re.compile(r"^\s*/(?:jarvis:)?talk\b")


def note_owner(session_id: str | None) -> None:
    """Record that /jarvis:talk was typed in this session, and claim an unclaimed
    conversation for it straight away."""
    if not session_id:
        return
    config.CACHE_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    config.write_json(OWNER_FILE, {"session": session_id, "at": time.time()})
    st = _state()
    if st is not None and st.get("session") is None:
        st["session"] = session_id
        config.write_json(STATE_FILE, st)


def _owner_hint() -> str | None:
    try:
        st = json.loads(OWNER_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    if time.time() - st.get("at", 0) > OWNER_TTL_S:
        return None
    return st.get("session")


# When the user's spoken reply was handed to Claude, so the next Stop can log how long
# Claude took to answer it.
def _asked_file():
    return config.CACHE_DIR / "asked.json"


def mark_asked(session: str | None) -> None:
    config.write_json(_asked_file(), {"session": session, "at": time.time()})


def take_asked(session: str | None) -> float | None:
    """Seconds since this session's last spoken request was handed over (and forget it)."""
    try:
        st = json.loads(_asked_file().read_text())
        _asked_file().unlink()
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return time.time() - st["at"] if st.get("session") == session else None


# ----------------------------------------------------------------------------- inbox
# Things the user said over Jarvis while Claude was still working. The next tool hook
# (or the Stop hook) hands them to Claude.
INBOX_FILE = config.CACHE_DIR / "inbox.jsonl"


def push(text: str) -> None:
    config.CACHE_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    with open(INBOX_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps({"text": text, "at": time.time()}) + "\n")


def pop_all() -> list[str]:
    """Take everything in the inbox. The file is renamed away before it's read, so
    text the watcher appends meanwhile lands in a fresh inbox instead of being lost."""
    import os
    taken = INBOX_FILE.with_name(f".{INBOX_FILE.name}.{os.getpid()}.taken")
    try:
        os.replace(INBOX_FILE, taken)
    except FileNotFoundError:
        return []
    try:
        lines = taken.read_text(encoding="utf-8").splitlines()
    finally:
        taken.unlink(missing_ok=True)
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


# ----------------------------------------------------------------------------- listening while Claude works
# While Claude works on a spoken request, a background "watcher" process keeps the mic
# open. Anything the user says goes to the inbox (handed to Claude by the next tool hook);
# "stop" / "cancel" sets a halt that makes the tool hooks refuse further tools, so Claude
# wraps up and the conversation goes back to listening.
WATCH_FILE = config.CACHE_DIR / "watch.json"
WATCH_LOCK = config.CACHE_DIR / "watch.lock"
HALT_FILE = config.CACHE_DIR / "halt.json"
WATCH_IDLE_S = 15 * 60  # no tool activity for this long: the watcher gives up

# Said while Claude works, these mean "stop the task" (plain "stop"/"wait" only do when
# Jarvis isn't talking; over Jarvis they just mean "stop talking").
STOP_TASK_PHRASES = (
    "stop the task", "stop that", "stop it", "stop working", "stop everything", "stop all",
    "stop what youre doing", "stop doing that", "cancel", "cancel that", "cancel it",
    "cancel the task", "abort", "abort that", "abort it", "halt", "never mind stop",
)


def is_stop_task(text: str) -> bool:
    from .ears import normalize
    n = normalize(text)
    for filler in ("jarvis ", "ok ", "okay ", "no ", "please ", "hey jarvis "):
        n = n.removeprefix(filler)
    n = n.removesuffix(" please").removesuffix(" jarvis").removesuffix(" now").strip()
    return n in STOP_TASK_PHRASES


def is_echo(heard: str, recent: list[str]) -> bool:
    """Did the mic just pick up Jarvis's own voice rather than the user?"""
    from .ears import normalize
    words = normalize(heard).split()
    if not words:
        return False
    for said in recent:
        spoken = set(normalize(said).split())
        if spoken and sum(w in spoken for w in words) >= 0.7 * len(words):
            return True
    return False


def halt(session: str | None, text: str) -> None:
    config.CACHE_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    config.write_json(HALT_FILE, {"session": session, "text": text, "at": time.time()})


def halted(session: str | None) -> str | None:
    """What the user said to stop this session's task, if they did."""
    try:
        st = json.loads(HALT_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    if st.get("session") not in (None, session):
        return None
    return st.get("text") or "stop"


def clear_halt() -> None:
    try:
        HALT_FILE.unlink()
    except FileNotFoundError:
        pass


def _watch_state() -> dict | None:
    try:
        return json.loads(WATCH_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _alive(pid: int) -> bool:
    import os
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def ensure_watcher(session: str | None) -> None:
    """Start the background listener for this session unless it's already running."""
    import subprocess
    import sys
    config.CACHE_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    st = _watch_state()
    if st and st.get("session") == session and _alive(int(st.get("pid", 0))):
        config.write_json(WATCH_FILE, {**st, "beat": time.time()})
        return
    cmd = [sys.executable, "-m", "jarvis_voice", "watch"] + (["--session", session] if session else [])
    subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, start_new_session=True)


def heartbeat() -> None:
    st = _watch_state()
    if st:
        config.write_json(WATCH_FILE, {**st, "beat": time.time()})


def stop_watcher() -> None:
    import os
    import signal
    st = _watch_state()
    if st and st.get("pid"):
        try:
            os.kill(int(st["pid"]), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        end = time.time() + 2
        while _alive(int(st["pid"])) and time.time() < end:
            time.sleep(0.05)
    try:
        WATCH_FILE.unlink()
    except FileNotFoundError:
        pass


class _DaemonVoice:
    """Stands in for a speaker in BargeIn: `busy` mirrors whether the speech daemon is
    talking (polled), and `recent` holds what it said lately."""

    def __init__(self):
        import threading
        self.busy = threading.Event()
        self.recent: list[str] = []
        threading.Thread(target=self._poll, daemon=True).start()

    def _poll(self):
        from . import daemon
        while True:
            st = daemon.request("status", timeout=0.3) or {}
            self.recent = st.get("recent") or []
            if st.get("busy") or st.get("queued"):
                self.busy.set()
            else:
                self.busy.clear()
            time.sleep(0.15)

    def output_level(self, t: float) -> float | None:
        """Playback loudness from the daemon, so BargeIn can learn our own echo
        instead of falling back to a high fixed threshold."""
        from . import daemon
        return daemon.level(t)


def watch(cfg: dict, session: str | None) -> int:
    """The background listener (`jarvis watch`). Runs until stopped, the conversation
    ends, or Claude has been idle for WATCH_IDLE_S."""
    import fcntl
    import os

    from . import daemon
    from .cli import is_pause_phrase
    from .ears import BargeIn, Listener, is_noise, make_fast_transcriber
    from .voice import chime

    config.CACHE_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock = open(WATCH_LOCK, "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return 0  # another watcher is already listening
    config.write_json(WATCH_FILE, {"pid": os.getpid(), "session": session, "beat": time.time()})

    address = cfg["address"]
    listener = Listener(cfg["silence"], 60.0, cfg["sensitivity"], calibrate_s=0.4)
    listener.start_continuous()
    voice = _DaemonVoice()
    transcriber = make_fast_transcriber(cfg)

    def wanted() -> bool:
        st = _watch_state()
        if not st or st.get("pid") != os.getpid():
            return False
        if time.time() - st.get("beat", 0) > WATCH_IDLE_S:
            return False
        live = _state()
        return live is not None and live.get("session") in (None, session)

    def capturing_now():
        config.write_text_atomic(CAPTURE_FILE, str(time.time()))  # no progress update over the user

    try:
        while wanted():
            barged = False
            if voice.busy.is_set():
                barge = BargeIn(listener, voice, lambda: daemon.request("hush", session=None))
                with barge.watch():
                    while voice.busy.is_set() and not barge.triggered:
                        time.sleep(0.05)
                if not barge.triggered:
                    continue
                barged = True
                capturing_now()
                audio = listener.listen(prefix=barge.captured)
            else:
                audio = listener.listen(start_timeout=0.5, on_start=capturing_now)
                if audio is None and listener.timed_out:
                    continue
            try:
                if audio is None:
                    continue
                chime("Pop", cfg)
                heard = transcriber(audio)
                if not heard or is_noise(heard) or is_echo(heard, voice.recent):
                    continue
                if is_stop_task(heard) or (not barged and is_pause_phrase(heard)):
                    halt(session, heard)
                    daemon.speak(f"Stopping, {address}.", session=session)
                elif barged and is_pause_phrase(heard):
                    continue  # "stop talking": silence was all they wanted
                else:
                    push(heard)
                    daemon.speak(f"Noted, {address}.", session=session)
            finally:
                capture_done()
                listener.flush()
    finally:
        listener.close()
        st = _watch_state()
        if st and st.get("pid") == os.getpid():
            try:
                WATCH_FILE.unlink()
            except FileNotFoundError:
                pass
    return 0


CAPTURE_FILE = config.CACHE_DIR / "capturing"


def capturing() -> bool:
    """Is the background listener busy recording/transcribing the user right now?
    A capture file left behind by a listener that has since died doesn't count."""
    try:
        if time.time() - float(CAPTURE_FILE.read_text()) >= CAPTURE_STALE_S:
            return False
    except (FileNotFoundError, ValueError):
        return False
    st = _watch_state()
    return bool(st and st.get("pid") and _alive(int(st["pid"])))


def capture_done() -> None:
    try:
        CAPTURE_FILE.unlink()
    except FileNotFoundError:
        pass


CAPTURE_STALE_S = 30.0  # a capture can't sensibly still be going after this long
CAPTURE_WAIT_S = 6.0    # the Stop hook waits at most this long for one to finish


def wait_for_capture(limit_s: float = CAPTURE_WAIT_S) -> None:
    end = time.time() + limit_s
    while capturing() and time.time() < end:
        time.sleep(0.2)


# ----------------------------------------------------------------------------- one turn
def _open_turn_local(cfg: dict):
    """Speaker, listener and transcriber loaded in this process (speech daemon off):
    about 4 s of model loading per turn, partly hidden by doing it in parallel."""
    import threading

    from .ears import Listener, make_transcriber
    from .voice import make_speaker

    mic: dict = {}
    calib = threading.Thread(target=lambda: mic.setdefault(
        "l", Listener(cfg["silence"], 60.0, cfg["sensitivity"], calibrate_s=0.6)), daemon=True)
    calib.start()
    speaker = make_speaker(cfg)
    calib.join()
    model: dict = {}
    loader = threading.Thread(target=lambda: model.setdefault("t", make_transcriber(cfg)), daemon=True)
    loader.start()

    def transcriber(audio):
        loader.join()
        return model["t"](audio)
    return speaker, mic["l"], transcriber


def _open_turn_daemon(cfg: dict, session: str | None, text: str = ""):
    """Speaker and transcriber borrowed from the speech daemon, whose models are
    already loaded: the reply starts in well under a second instead of ~3 s.

    Returns (speaker, listener, transcriber, started). When the room's noise level is
    already known, the reply is sent before the mic opens (`started`): the voice takes
    ~0.3 s to render its first words, and the mic's 0.15 s check fits inside that, so
    it doesn't hear Jarvis as background noise."""
    from .daemon import DaemonSpeaker
    from .ears import Listener, _noise_hint, make_fast_transcriber

    transcriber = make_fast_transcriber(cfg)  # first, so the daemon warms up meanwhile
    speaker = DaemonSpeaker(session)
    started = bool(text.strip()) and _noise_hint() is not None
    if started:
        speaker.say(text)
    listener = Listener(cfg["silence"], 60.0, cfg["sensitivity"], calibrate_s=0.6)
    return speaker, listener, transcriber, started


def turn(cfg: dict, text: str, session: str | None = None) -> str | None:
    """Speak `text`, then listen for the user's reply. Returns what they said, or None
    when the conversation is over (silence, or a sleep/exit phrase)."""
    from .cli import is_exit_phrase, is_pause_phrase, is_sleep_phrase
    from .ears import BargeIn, is_noise
    from .text import split_sentences
    from .voice import chime

    address = cfg["address"]
    started = False  # reply already sent to the daemon
    if cfg.get("speech_daemon", True):
        speaker, listener, transcriber, started = _open_turn_daemon(cfg, session, text)
    else:
        speaker, listener, transcriber = _open_turn_local(cfg)
    listener.start_continuous()

    def say(line: str) -> None:
        speaker.say(line)
        speaker.wait()

    def speak_all() -> None:
        if started:
            pass
        elif getattr(speaker, "splits_itself", False):
            speaker.say(text)
        else:
            for s in split_sentences(text):  # sentence by sentence: the first plays while the rest render
                speaker.say(s)
        speaker.wait()

    barged = None
    try:
        if text:
            if cfg["barge_in"]:
                barge = BargeIn(listener, speaker, speaker.stop)
                with barge.watch():
                    speak_all()
                barged = barge.captured
            else:
                speak_all()

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
            t_stt = time.monotonic()
            heard = transcriber(audio)
            if is_noise(heard):
                continue
            config.timing(f"you said ({len(audio) / 16000:.1f}s of speech, transcribed in "
                          f"{time.monotonic() - t_stt:.2f}s): \"{heard}\"")
            if interrupted and is_pause_phrase(heard):
                deadline = time.time() + float(cfg["converse_timeout"])
                continue
            if is_sleep_phrase(heard) or is_exit_phrase(heard):
                say(f"Very good, {address}. Standing by.")
                return None
            return heard
    finally:
        speaker.stop()
        getattr(speaker, "close", lambda: None)()
        listener.close()


def continuation(heard: str, address: str) -> str:
    """The Stop-hook reason that feeds the user's spoken reply back to Claude."""
    from .brain import claude_code_note
    return (
        "The user replied aloud in voice conversation mode. Treat this speech-to-text transcript "
        "exactly as their next prompt (it may contain recognition errors; infer the meaning) and "
        f"respond to it now: \"{heard}\"\n\n{claude_code_note(address)}"
    )

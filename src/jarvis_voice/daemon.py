"""Background speech daemon: keeps the voice model loaded so Jarvis starts talking in
well under a second instead of ~3 s, and arbitrates between Claude Code sessions.

It also keeps Whisper loaded, so a spoken reply is transcribed without a ~2 s model load
each time, and serves `/jarvis:talk` turns, which used to load both models every turn.

Protocol: one JSON object per line over a Unix socket in the cache dir.
  {"op": "say", "text": ..., "session": ..., "ttl": seconds | null, "append": bool}
                                              append = queue after this session's speech
  {"op": "hush", "session": ... | null, "gentle": bool}   null = everything;
                                              gentle = finish the current sentence first
  {"op": "level", "t": monotonic time}        how loud playback is around t (barge-in)
  {"op": "transcribe", "audio": base64 float32 16 kHz mono}  -> {"text": ...}
  {"op": "warm", "stt": bool}                 load the voice (and Whisper) ahead of use
  {"op": "status"}  /  {"op": "quit"}

Rules:
  * a new utterance from a session replaces whatever that session was saying, at the
    end of the sentence being spoken (never mid-word);
  * an utterance from another session waits its turn (two voices at once is no help);
  * an utterance with a `ttl` (progress narration) is dropped if it can't start in time;
  * hushing a session only silences that session.

The client half (request/speak) imports nothing heavy, so hooks stay fast."""
from __future__ import annotations

import base64
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time

from . import config

SOCK = config.CACHE_DIR / "speechd.sock"
LOCK = config.CACHE_DIR / "speechd.lock"
LOG = config.CACHE_DIR / "speechd.log"
RECENT_S = 60.0  # how long status reports what was said (for echo filtering)
VOICE_KEYS = ("engine", "blend", "speed", "fx", "edge_voice", "edge_rate", "edge_pitch", "mac_voice", "mac_rate")


# ============================================================================= client
def request(op: str, timeout: float = 0.5, **fields) -> dict | None:
    """Send one request; None if no daemon is running."""
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            s.connect(str(SOCK))
            s.sendall((json.dumps({"op": op, **fields}) + "\n").encode())
            buf = b""
            while not buf.endswith(b"\n"):
                chunk = s.recv(4096)
                if not chunk:
                    break
                buf += chunk
        return json.loads(buf or b"{}")
    except (OSError, ValueError):
        return None


def speak(text: str, session: str | None = None, ttl: float | None = None, append: bool = False) -> None:
    """Hand text to the daemon, starting it (with the text) if it isn't running."""
    job = {"text": text, "session": session, "ttl": ttl, "append": append}
    if request("say", **job) is None:
        _spawn({"op": "say", **job})


def warm(stt: bool = False) -> None:
    """Have the voice (and, with `stt`, Whisper) loaded before it's needed, starting the
    daemon if necessary. Returns at once; loading happens in the background."""
    if request("warm", stt=stt) is None:
        _spawn({"op": "warm", "stt": stt})


def transcribe(audio, timeout: float = 90.0) -> str | None:
    """Transcribe 16 kHz mono float32 audio with the daemon's Whisper. None if the
    daemon isn't there (the caller then loads Whisper itself)."""
    import numpy as np
    data = base64.b64encode(np.ascontiguousarray(audio, dtype="<f4").tobytes()).decode()
    resp = request("transcribe", timeout=timeout, audio=data)
    if not resp or not resp.get("ok"):
        return None
    return str(resp.get("text") or "")


def level(t: float) -> float | None:
    """How loud the daemon is playing around monotonic time t (None if unknown)."""
    resp = request("level", timeout=0.2, t=t)
    return None if not resp else resp.get("level")


def _spawn(first: dict) -> None:
    """Start the daemon with `first` as its first request."""
    config.CACHE_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, path = tempfile.mkstemp(prefix="first-", suffix=".json", dir=config.CACHE_DIR)
    with os.fdopen(fd, "w") as f:
        json.dump(first, f)
    try:
        if LOG.stat().st_size > 1 << 20:
            LOG.unlink()
    except FileNotFoundError:
        pass
    with open(LOG, "ab") as log:
        subprocess.Popen([sys.executable, "-m", "jarvis_voice", "daemon", "--first", path],
                         stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)


def settle(limit_s: float = 8.0) -> None:
    """Drop queued speech, let the current sentence finish, and wait until quiet."""
    if request("hush", session=None, gentle=True) is None:
        return
    end = time.time() + limit_s
    while time.time() < end:
        st = request("status")
        if not st or not st.get("busy"):
            return
        time.sleep(0.1)
    request("hush", session=None)


def running() -> bool:
    return request("status") is not None


class DaemonSpeaker:
    """A speaker (say / wait / stop / trim / busy / output_level, as BargeIn expects)
    that talks through the daemon's already-loaded voice instead of loading its own.

    `splits_itself`: hand it whole replies; the daemon splits them into sentences and
    renders the next one while the current one plays."""

    splits_itself = True
    GRACE_S = 0.6  # right after say(), trust our own "busy" over a status that predates it

    def __init__(self, session: str | None = None):
        self.session = session
        self.busy = threading.Event()
        self._said_at = 0.0
        self._closed = threading.Event()
        threading.Thread(target=self._poll, daemon=True).start()

    def say(self, text: str) -> None:
        if not text.strip():
            return
        self._said_at = time.monotonic()
        self.busy.set()
        speak(text, session=self.session, append=True)

    def wait(self) -> None:
        while self.busy.is_set() and not self._closed.is_set():
            time.sleep(0.05)

    def stop(self) -> None:
        self._said_at = 0.0
        request("hush", session=self.session)
        self.busy.clear()

    def trim(self) -> None:
        request("hush", session=self.session, gentle=True)

    def output_level(self, t: float) -> float | None:
        return level(t)

    def close(self) -> None:
        self._closed.set()
        self.busy.clear()

    def _poll(self) -> None:
        started = time.monotonic()
        while not self._closed.is_set():
            st = request("status", timeout=0.3)
            fresh = time.monotonic() - self._said_at < self.GRACE_S
            # A just-spawned daemon takes a moment to answer: give it time before giving up.
            booting = st is None and time.monotonic() - started < 20 and self._said_at
            if st and (st.get("busy") or st.get("queued")):
                self.busy.set()
            elif not fresh and not booting:
                self.busy.clear()
            time.sleep(0.1)


# ============================================================================= server
class _Job:
    def __init__(self, text: str, session, ttl):
        self.text, self.session = text, session
        self.deadline = time.monotonic() + ttl if ttl else None
        self.cancelled = False


class SpeechDaemon:
    def __init__(self, idle_s: float):
        self.idle_s = idle_s
        self.lock = threading.Lock()
        self.wake = threading.Condition(self.lock)
        self.queue: list[_Job] = []
        self.current: _Job | None = None
        self.recent: list[tuple[float, str]] = []  # (wall time, text) recently spoken
        self.speaker = None
        self.voice_sig = None
        self.last_active = time.monotonic()
        self.stopping = False
        self.voice_lock = threading.Lock()  # creating the speaker (worker vs warm-up)
        self.stt = None
        self.stt_sig = None
        self.stt_lock = threading.Lock()    # one transcription at a time

    # -- requests (called from the socket threads)
    def say(self, text: str, session, ttl, append: bool = False) -> dict:
        with self.lock:
            if session is not None and not append:
                self._cancel(session, gentle=True)
            self.queue.append(_Job(text, session, ttl))
            self.last_active = time.monotonic()
            self.wake.notify()
            return {"ok": True, "position": len(self.queue) - 1 + (self.current is not None)}

    def hush(self, session=None, gentle: bool = False) -> dict:
        with self.lock:
            self._cancel(session, gentle)
            return {"ok": True}

    def status(self) -> dict:
        with self.lock:
            cur = self.current
            return {"ok": True, "pid": os.getpid(), "busy": cur is not None,
                    "session": cur.session if cur else None, "queued": len(self.queue),
                    "engine": (self.voice_sig or {}).get("engine"),
                    "recent": [t for at, t in self.recent if time.time() - at < RECENT_S]}

    def level(self, t: float) -> dict:
        with self.lock:
            speaker, busy = self.speaker, self.current is not None
        out = speaker.output_level(float(t)) if speaker and busy else 0.0
        return {"ok": True, "level": out, "busy": busy}

    def warm(self, stt: bool = False) -> dict:
        def run():
            try:
                with self.voice_lock:
                    self._ensure_speaker(0.0)
                    render = getattr(self.speaker, "render", None)
                    if render:
                        render("Ready.")  # the first render after loading is ~3x slower
            except Exception as e:  # noqa: BLE001 - report and carry on
                print(f"warm-up failed: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            if stt:
                try:
                    self._ensure_stt()
                except Exception as e:  # noqa: BLE001
                    print(f"whisper unavailable: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        self.touch()
        threading.Thread(target=run, daemon=True).start()
        return {"ok": True}

    def transcribe(self, audio_b64: str) -> dict:
        import numpy as np
        audio = np.frombuffer(base64.b64decode(audio_b64), dtype="<f4")
        self.touch()
        try:
            stt = self._ensure_stt()
            text = stt(audio)
        except Exception as e:  # noqa: BLE001 - the client falls back to its own Whisper
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}
        finally:
            self.touch()
        return {"ok": True, "text": text}

    def touch(self) -> None:
        with self.lock:
            self.last_active = time.monotonic()

    def _ensure_stt(self):
        """The loaded Whisper, (re)loaded if the model or vocabulary setting changed."""
        from .ears import make_transcriber
        cfg = config.load()
        sig = (cfg["whisper"], cfg.get("vocabulary", ""))
        with self.stt_lock:
            if self.stt is None or sig != self.stt_sig:
                self.stt, self.stt_sig = make_transcriber(cfg), sig
            return _Locked(self.stt, self.stt_lock)

    def _cancel(self, session, gentle: bool = False) -> None:
        """Drop queued jobs and stop the current one, for one session or (None) for all.
        `gentle` lets the sentence being spoken finish first."""
        match = (lambda j: True) if session is None else (lambda j: j.session == session)
        self.queue = [j for j in self.queue if not match(j)]
        if self.current and match(self.current):
            self.current.cancelled = True
            if self.speaker:
                trim = getattr(self.speaker, "trim", None) if gentle else None
                (trim or self.speaker.stop)()

    # -- speaking (worker thread)
    def _ensure_speaker(self, idle_for: float) -> None:
        from .voice import make_speaker
        cfg = config.load()
        sig = {k: cfg.get(k) for k in VOICE_KEYS}
        if self.speaker is None or sig != self.voice_sig:
            if self.speaker:
                self.speaker.stop()
            self.speaker, self.voice_sig = make_speaker(cfg), sig
        elif idle_for > 2.0:
            _refresh_audio_devices()  # follow headphones plugged in since we last spoke

    def worker(self) -> None:
        from .text import split_sentences
        idle_since = time.monotonic()
        while True:
            with self.lock:
                while not self.queue and not self.stopping:
                    self.wake.wait(timeout=1.0)
                if self.stopping:
                    return
                job = self.queue.pop(0)
                if job.deadline and time.monotonic() > job.deadline:
                    continue  # stale progress update: no longer worth saying
                self.current = job  # from here a hush marks it cancelled
                self.recent = [(at, t) for at, t in self.recent if time.time() - at < RECENT_S][-7:]
                self.recent.append((time.time(), job.text))
            try:
                with self.voice_lock:
                    self._ensure_speaker(time.monotonic() - idle_since)
            except Exception as e:
                print(f"speaker unavailable: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            with self.lock:
                if job.cancelled or self.speaker is None:
                    self.current = None
                    continue
                for s in split_sentences(job.text):
                    self.speaker.say(s)
            self.speaker.wait()
            with self.lock:
                self.current = None
                self.last_active = idle_since = time.monotonic()

    # -- lifecycle
    def idle_expired(self) -> bool:
        with self.lock:
            return (not self.queue and self.current is None
                    and time.monotonic() - self.last_active > self.idle_s)

    def stop(self) -> None:
        with self.lock:
            self.stopping = True
            self._cancel(None)
            self.wake.notify_all()


class _Locked:
    """Calls a transcriber while holding its lock (Whisper isn't re-entrant)."""

    def __init__(self, fn, lock):
        self.fn, self.lock = fn, lock

    def __call__(self, audio):
        with self.lock:
            return self.fn(audio)


def _refresh_audio_devices() -> None:
    """PortAudio caches the device list at start-up; re-read it so the default output
    follows the system (e.g. new headphones)."""
    try:
        import sounddevice as sd
        sd._terminate()
        sd._initialize()
    except Exception:
        pass


def _handle(conn: socket.socket, d: SpeechDaemon) -> bool:
    """Serve one connection. Returns False when asked to quit."""
    with conn:
        conn.settimeout(10.0)  # a transcribe request carries up to a few MB of audio
        buf = b""
        try:
            while not buf.endswith(b"\n"):
                chunk = conn.recv(65536)
                if not chunk:
                    break
                buf += chunk
            req = json.loads(buf or b"{}")
        except (OSError, ValueError):
            return True
        op = req.get("op")
        if op == "say":
            resp = d.say(str(req.get("text") or ""), req.get("session"), req.get("ttl"),
                         bool(req.get("append")))
        elif op == "hush":
            resp = d.hush(req.get("session"), bool(req.get("gentle")))
        elif op == "status":
            resp = d.status()
        elif op == "level":
            resp = d.level(req.get("t") or 0.0)
        elif op == "warm":
            resp = d.warm(bool(req.get("stt")))
        elif op == "transcribe":
            resp = d.transcribe(str(req.get("audio") or ""))
        elif op == "quit":
            resp = {"ok": True}
        else:
            resp = {"ok": False, "error": f"unknown op {op!r}"}
        try:
            conn.sendall((json.dumps(resp) + "\n").encode())
        except OSError:
            pass
        return op != "quit"


def serve(first: str | None = None) -> int:
    """Run the daemon (only one per user; a second start hands its text to the first)."""
    import fcntl

    config.CACHE_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    first_job = None
    if first:
        try:
            with open(first, encoding="utf-8") as f:
                first_job = json.load(f)
        except (OSError, ValueError):
            pass
        finally:
            try:
                os.unlink(first)
            except OSError:
                pass

    lock = open(LOCK, "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:  # another daemon is starting or running: pass our request on
        if first_job:
            op = first_job.pop("op", "say")
            for _ in range(100):
                if request(op, **first_job) is not None:
                    break
                time.sleep(0.1)
        return 0

    cfg = config.load()
    d = SpeechDaemon(idle_s=float(cfg["daemon_idle_minutes"]) * 60)
    if first_job:
        if first_job.get("op") == "warm":
            d.warm(bool(first_job.get("stt")))
        else:  # "say" (or the pre-0.5 format, which had no op)
            d.say(first_job.get("text") or "", first_job.get("session"), first_job.get("ttl"),
                  bool(first_job.get("append")))
    threading.Thread(target=d.worker, daemon=True).start()

    try:
        SOCK.unlink()
    except FileNotFoundError:
        pass
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    old_umask = os.umask(0o077)  # the socket is owner-only from the moment it exists
    try:
        srv.bind(str(SOCK))
    finally:
        os.umask(old_umask)
    os.chmod(SOCK, 0o600)
    srv.listen(16)
    srv.settimeout(0.5)
    print(f"[{time.strftime('%H:%M:%S')}] speech daemon {os.getpid()} up", file=sys.stderr, flush=True)
    quit_ = threading.Event()

    def serve_one(conn):
        if not _handle(conn, d):
            quit_.set()

    try:
        # Each connection gets its own thread: a transcription taking a second must not
        # hold up a hush or a barge-in level check.
        while not quit_.is_set():
            try:
                conn, _ = srv.accept()
            except TimeoutError:
                if d.idle_expired():
                    break
                continue
            threading.Thread(target=serve_one, args=(conn,), daemon=True).start()
    finally:
        d.stop()
        srv.close()
        try:
            SOCK.unlink()
        except FileNotFoundError:
            pass
        print(f"[{time.strftime('%H:%M:%S')}] speech daemon {os.getpid()} down", file=sys.stderr, flush=True)
    return 0

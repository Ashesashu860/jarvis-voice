"""Background speech daemon: keeps the voice model loaded so Jarvis starts talking in
well under a second instead of ~3 s, and arbitrates between Claude Code sessions.

Protocol: one JSON object per line over a Unix socket in the cache dir.
  {"op": "say", "text": ..., "session": ..., "ttl": seconds | null}
  {"op": "hush", "session": ... | null, "gentle": bool}   null = everything;
                                              gentle = finish the current sentence first
  {"op": "status"}  /  {"op": "quit"}

Rules:
  * a new utterance from a session replaces whatever that session was saying, at the
    end of the sentence being spoken (never mid-word);
  * an utterance from another session waits its turn (two voices at once is no help);
  * an utterance with a `ttl` (progress narration) is dropped if it can't start in time;
  * hushing a session only silences that session.

The client half (request/speak) imports nothing heavy, so hooks stay fast."""
from __future__ import annotations

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


def speak(text: str, session: str | None = None, ttl: float | None = None) -> None:
    """Hand text to the daemon, starting it (with the text) if it isn't running."""
    if request("say", text=text, session=session, ttl=ttl) is not None:
        return
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fd, path = tempfile.mkstemp(prefix="say-", suffix=".json", dir=config.CACHE_DIR)
    with os.fdopen(fd, "w") as f:
        json.dump({"text": text, "session": session, "ttl": ttl}, f)
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

    # -- requests (called from the socket thread)
    def say(self, text: str, session, ttl) -> dict:
        with self.lock:
            if session is not None:
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
        conn.settimeout(2.0)
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
            resp = d.say(str(req.get("text") or ""), req.get("session"), req.get("ttl"))
        elif op == "hush":
            resp = d.hush(req.get("session"), bool(req.get("gentle")))
        elif op == "status":
            resp = d.status()
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

    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
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
    except OSError:  # another daemon is starting or running: give it our text
        if first_job:
            for _ in range(100):
                if request("say", **first_job) is not None:
                    break
                time.sleep(0.1)
        return 0

    cfg = config.load()
    d = SpeechDaemon(idle_s=float(cfg["daemon_idle_minutes"]) * 60)
    if first_job:
        d.say(first_job.get("text") or "", first_job.get("session"), first_job.get("ttl"))
    threading.Thread(target=d.worker, daemon=True).start()

    try:
        SOCK.unlink()
    except FileNotFoundError:
        pass
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(str(SOCK))
    os.chmod(SOCK, 0o600)
    srv.listen(16)
    srv.settimeout(5.0)
    print(f"[{time.strftime('%H:%M:%S')}] speech daemon {os.getpid()} up", file=sys.stderr, flush=True)
    try:
        while True:
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                if d.idle_expired():
                    break
                continue
            if not _handle(conn, d):
                break
    finally:
        d.stop()
        srv.close()
        try:
            SOCK.unlink()
        except FileNotFoundError:
            pass
        print(f"[{time.strftime('%H:%M:%S')}] speech daemon {os.getpid()} down", file=sys.stderr, flush=True)
    return 0

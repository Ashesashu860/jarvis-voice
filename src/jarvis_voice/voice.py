"""Speech output: Kokoro (offline) / edge-tts (online) / macOS `say`, plus the
"refined AI" processing chain. Nothing here models or clones any real person's
voice: the Kokoro voice is a weighted blend of stock open-source voices."""
from __future__ import annotations

import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request

import numpy as np

from . import config
from .text import clean_for_speech, split_sentences  # noqa: F401  (re-exported)

SR = 24000

KOKORO_FILES = {
    "kokoro-v1.0.onnx": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
    "voices-v1.0.bin": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
}

DOWNLOAD_TIMEOUT_S = 60
KOKORO_VOICES = ("bm_daniel", "bm_george", "bm_fable", "bm_lewis",
                 "bf_alice", "bf_emma", "bf_isabella", "bf_lily")  # British voices in voices-v1.0.bin

MAC_VOICE_PREFERENCE = [
    "Jamie (Premium)", "Daniel (Premium)", "Daniel (Enhanced)",
    "Oliver (Enhanced)", "Arthur (Enhanced)", "Daniel",
]


def log(msg: str) -> None:
    print(f"\033[2m{msg}\033[0m", file=sys.stderr, flush=True)


# ----------------------------------------------------------------------------- models
def ensure_kokoro(quiet: bool = False) -> None:
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in KOKORO_FILES.items():
        dest = config.MODELS_DIR / name
        if dest.exists() and dest.stat().st_size > 0 and not any(
                p.startswith(name) for p in model_problems()):
            continue
        if not quiet:
            log(f"Downloading voice model {name} (one time)...")
        tmp = dest.with_suffix(dest.suffix + ".part")
        # A timeout so a stalled connection fails instead of hanging setup forever.
        with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_S) as r, open(tmp, "wb") as f:
            total = int(r.headers.get("Content-Length") or 0)
            done = 0
            while chunk := r.read(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total and not quiet:
                    print(f"\r  {done * 100 // total:3d}%  {done >> 20} / {total >> 20} MB", end="",
                          file=sys.stderr, flush=True)
        if not quiet:
            print(file=sys.stderr)
        if total and done != total:  # cut off mid-way: don't keep a broken model
            tmp.unlink(missing_ok=True)
            raise OSError(f"download of {name} was incomplete ({done} of {total} bytes); "
                          "run `jarvis setup` again")
        tmp.rename(dest)
        (config.MODELS_DIR / f"{name}.size").write_text(str(done))


def model_problems() -> list[str]:
    """Model files that are missing or don't match the size they were downloaded at."""
    problems = []
    for name in KOKORO_FILES:
        path = config.MODELS_DIR / name
        if not path.exists():
            problems.append(f"{name} missing")
            continue
        try:
            expected = int((config.MODELS_DIR / f"{name}.size").read_text())
        except (FileNotFoundError, ValueError):
            continue  # downloaded by an older version: nothing to compare against
        if path.stat().st_size != expected:
            problems.append(f"{name} is damaged ({path.stat().st_size} of {expected} bytes)")
    return problems


def kokoro_ready() -> bool:
    return not model_problems()


# ----------------------------------------------------------------------------- macOS say
def pick_mac_voice(requested: str | None) -> str:
    if requested:
        return requested
    if not shutil.which("say"):
        return ""
    out = subprocess.run(["say", "-v", "?"], capture_output=True, text=True).stdout
    names = {re.split(r"\s{2,}", line.strip())[0] for line in out.splitlines() if line.strip()}
    return next((v for v in MAC_VOICE_PREFERENCE if v in names), "Daniel")


def system_voice_command(voice: str, rate: int) -> list[str] | None:
    """The built-in speech command for this OS: macOS `say`, else espeak-ng / espeak
    (common on Linux) in a British voice. None if there is none."""
    if shutil.which("say"):
        return ["say", "-v", voice, "-r", str(rate)]
    for prog in ("espeak-ng", "espeak"):
        if shutil.which(prog):
            return [prog, "-v", "en-gb", "-s", str(rate)]
    return None


class SaySpeaker:
    """The system voice (macOS `say`, or espeak-ng on Linux); silently does nothing
    where neither is available."""

    def __init__(self, voice: str | None = None, rate: int = 185):
        self.voice, self.rate = pick_mac_voice(voice), rate
        self.q: queue.Queue[tuple[str, str | None]] = queue.Queue()
        self.proc: subprocess.Popen | None = None
        self.busy = threading.Event()
        self.on_play = None  # callback(event: "start" | "end", tag) as each piece plays
        self.cmd = system_voice_command(self.voice, rate)
        self.available = self.cmd is not None
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            text, tag = self.q.get()
            if self.on_play:
                self.on_play("start", tag)
            if self.available:
                self.proc = subprocess.Popen([*self.cmd, text],
                                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                self.proc.wait()
                self.proc = None
            if self.on_play:
                self.on_play("end", tag)
            if self.q.empty():
                self.busy.clear()

    def say(self, text: str, tag: str | None = None) -> bool:
        """Queue one piece; False if nothing in it is speakable."""
        text = clean_for_speech(text)
        if not text:
            return False
        self.busy.set()
        self.q.put((text, tag))
        return True

    def wait(self):
        while self.busy.is_set():
            time.sleep(0.05)

    def stop(self):
        try:
            while True:
                self.q.get_nowait()
        except queue.Empty:
            pass
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
        self.busy.clear()

    def trim(self):
        """Finish the sentence being spoken, drop the rest."""
        try:
            while True:
                self.q.get_nowait()
        except queue.Empty:
            pass
        if self.proc is None:
            self.busy.clear()

    def output_level(self, t: float) -> float | None:
        return None  # unknown: `say` plays the audio itself


# ----------------------------------------------------------------------------- processed neural voices
ENV_HOP_S = 0.01            # resolution of the playback loudness envelope
ECHO_WINDOW = (0.3, 0.05)   # mic frame at t may hold echo of audio played in [t-0.3, t+0.05]


def loudness_envelope(audio: np.ndarray, hop_s: float = ENV_HOP_S) -> np.ndarray:
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    hop = max(1, int(SR * hop_s))
    n = len(mono) // hop
    if n == 0:
        return np.zeros(1, np.float32)
    return np.sqrt((mono[: n * hop].reshape(n, hop) ** 2).mean(axis=1)).astype(np.float32)


def level_at(ref, t: float) -> float:
    """Loudest playback level that could be echoing in the mic at monotonic time t."""
    start, env = ref
    i0 = max(0, int((t - ECHO_WINDOW[0] - start) / ENV_HOP_S))
    i1 = min(len(env), int((t + ECHO_WINDOW[1] - start) / ENV_HOP_S) + 1)
    return float(env[i0:i1].max()) if i1 > i0 else 0.0
class ProcessedSpeaker:
    """Base for neural voices. Pipeline: text queue -> synth thread (prepares the
    next sentence while the current one plays) -> audio queue -> player thread."""

    def __init__(self, fx: float, fallback: SaySpeaker | None):
        import sounddevice as sd
        self.sd, self.fx, self.fallback = sd, fx, fallback
        self.text_q: queue.Queue = queue.Queue()
        self.audio_q: queue.Queue = queue.Queue()
        self.pending = 0
        self.lock = threading.Lock()
        self.generation = 0
        self.playing_gen = None  # generation of the sentence playing right now
        self.kept_gen = None     # an older generation trim() let finish
        self.busy = threading.Event()
        self.playing = None  # (monotonic start, loudness envelope), or "fallback"
        self.first_start = None  # when audio first played since this was last reset (timing log)
        self.on_play = None  # callback(event: "start" | "end", tag) as each piece plays
        self.ir = _impulse_response()
        threading.Thread(target=self._synth_loop, daemon=True).start()
        threading.Thread(target=self._play_loop, daemon=True).start()

    def synth(self, text: str) -> np.ndarray:  # mono float32 at SR
        raise NotImplementedError

    def render(self, text: str) -> np.ndarray:
        return process(self.synth(text), self.fx, self.ir)

    # -- public
    def say(self, text: str, tag: str | None = None) -> bool:
        """Queue one piece; False if nothing in it is speakable. `tag` is handed back
        through `on_play` when the piece starts and finishes playing."""
        text = clean_for_speech(text)
        if not text:
            return False
        with self.lock:
            self.pending += 1
            self.busy.set()
        self.text_q.put((self.generation, text, tag))
        return True

    def wait(self):
        while self.busy.is_set():
            time.sleep(0.05)

    def stop(self):
        with self.lock:
            self.generation += 1
            for q in (self.text_q, self.audio_q):
                try:
                    while True:
                        q.get_nowait()
                except queue.Empty:
                    pass
            self.pending = 0
            self.kept_gen = None
            self.busy.clear()
        self.sd.stop()
        if self.fallback:
            self.fallback.stop()

    def trim(self):
        """Let the sentence that is playing finish, but drop everything after it, so a
        newer utterance takes over at a sentence boundary instead of mid-word."""
        with self.lock:
            playing = self.playing_gen
            self.generation += 1
            for q in (self.text_q, self.audio_q):
                try:
                    while True:
                        q.get_nowait()
                except queue.Empty:
                    pass
            self.kept_gen = playing
            self.pending = 1 if playing is not None else 0
            if playing is None:
                self.busy.clear()

    def output_level(self, t: float) -> float | None:
        """How loud the audio playing around monotonic time t is (None if unknown)."""
        ref = self.playing
        if ref == "fallback":
            return None
        return 0.0 if ref is None else level_at(ref, t)

    # -- internals
    def _done_one(self):
        with self.lock:
            self.pending = max(0, self.pending - 1)
            if self.pending == 0:
                self.busy.clear()

    def _synth_loop(self):
        while True:
            gen, text, tag = self.text_q.get()
            if gen != self.generation:
                continue
            try:
                self.audio_q.put((gen, self.render(text), text, tag))
            except Exception as e:  # offline / model hiccup -> macOS voice
                log(f"[voice engine error: {type(e).__name__}: {e}; using fallback voice]")
                self.audio_q.put((gen, None, text, tag))

    def _play_loop(self):
        while True:
            gen, audio, text, tag = self.audio_q.get()
            with self.lock:
                if gen != self.generation:
                    continue
                self.playing_gen = gen
            if self.on_play:
                self.on_play("start", tag)
            if audio is None:
                if self.fallback:
                    self.playing = "fallback"
                    self.fallback.say(text)
                    self.fallback.wait()
            else:
                env = loudness_envelope(audio)
                self.playing = (time.monotonic(), env)
                if self.first_start is None:
                    self.first_start = self.playing[0]
                self.sd.play(audio, SR)
                self.sd.wait()
            with self.lock:
                self.playing_gen = None
                counted = gen == self.generation or gen == self.kept_gen
                if gen == self.kept_gen:
                    self.kept_gen = None
            if self.on_play:
                self.on_play("end", tag)
            if counted:
                self._done_one()


class KokoroSpeaker(ProcessedSpeaker):
    """Offline Kokoro voice; `blend` mixes stock voices, e.g. 'bm_daniel:0.6,bm_george:0.4'."""

    def __init__(self, blend: str, speed: float, fx: float, fallback: SaySpeaker | None):
        from kokoro_onnx import Kokoro
        ensure_kokoro()
        self.kokoro = Kokoro(str(config.MODELS_DIR / "kokoro-v1.0.onnx"), str(config.MODELS_DIR / "voices-v1.0.bin"))
        parts = config.parse_blend(blend)
        weights = np.array([w for _, w in parts], dtype=np.float32)
        weights /= weights.sum()
        self.style = sum(w * self.kokoro.get_voice_style(name) for w, (name, _) in zip(weights, parts, strict=True))
        self.speed = speed
        super().__init__(fx, fallback)

    def synth(self, text: str) -> np.ndarray:
        audio, sr = self.kokoro.create(text, voice=self.style, speed=self.speed, lang="en-gb")
        audio = audio.astype(np.float32)
        if sr != SR:
            from scipy.signal import resample_poly
            audio = resample_poly(audio, SR, sr).astype(np.float32)
        return audio


class EdgeSpeaker(ProcessedSpeaker):
    """Microsoft neural voices via edge-tts (online; `pip install jarvis-voice[online]`)."""

    def __init__(self, voice: str, rate: str, pitch: str, fx: float, fallback: SaySpeaker | None):
        import asyncio

        import edge_tts  # noqa: F401  (fail early if the extra is missing)
        self.voice, self.rate, self.pitch = voice, rate, pitch
        self.loop = asyncio.new_event_loop()
        super().__init__(fx, fallback)

    def synth(self, text: str) -> np.ndarray:
        return self.loop.run_until_complete(self._synth(text))

    async def _synth(self, text: str) -> np.ndarray:
        import io

        import av
        import edge_tts
        mp3 = bytearray()
        async for chunk in edge_tts.Communicate(text, self.voice, rate=self.rate, pitch=self.pitch).stream():
            if chunk["type"] == "audio":
                mp3 += chunk["data"]
        pcm = []
        with av.open(io.BytesIO(bytes(mp3))) as c:
            res = av.AudioResampler(format="flt", layout="mono", rate=SR)
            for frame in c.decode(audio=0):
                for f in res.resample(frame):
                    pcm.append(f.to_ndarray().reshape(-1))
        return np.concatenate(pcm).astype(np.float32) if pcm else np.zeros(16, np.float32)


def make_speaker(cfg: dict, **overrides):
    """Build the configured speaker; falls back to macOS `say` if the engine can't start."""
    cfg = {**cfg, **{k: v for k, v in overrides.items() if v is not None}}
    fallback = SaySpeaker(cfg["mac_voice"], cfg["mac_rate"])
    engine = cfg["engine"]
    try:
        if engine == "kokoro":
            return KokoroSpeaker(cfg["blend"], cfg["speed"], cfg["fx"], fallback)
        if engine == "edge":
            return EdgeSpeaker(cfg["edge_voice"], cfg["edge_rate"], cfg["edge_pitch"], cfg["fx"], fallback)
    except Exception as e:
        log(f"[{engine} voice unavailable ({type(e).__name__}: {e}); using the system voice]")
    return fallback


def describe(cfg: dict) -> str:
    if cfg["engine"] == "kokoro":
        return f"kokoro {cfg['blend']} speed {cfg['speed']} fx {cfg['fx']}"
    if cfg["engine"] == "edge":
        return f"{cfg['edge_voice']} fx {cfg['fx']}"
    return f"macOS {pick_mac_voice(cfg['mac_voice'])}"


# macOS system sound -> the closest freedesktop sound theme file (Linux).
_FREEDESKTOP_SOUNDS = {"Tink": "message", "Pop": "complete", "Bottle": "bell"}


def chime_command(name: str) -> list[str] | None:
    """How to play a listening/heard chime on this OS, or None if we can't."""
    if shutil.which("afplay"):
        return ["afplay", f"/System/Library/Sounds/{name}.aiff"]
    sound = f"/usr/share/sounds/freedesktop/stereo/{_FREEDESKTOP_SOUNDS.get(name, 'message')}.oga"
    if shutil.which("paplay") and os.path.exists(sound):
        return ["paplay", sound]
    return None


def chime(name: str, cfg: dict | None = None, wait: bool = False) -> None:
    if cfg is not None and not cfg.get("chimes", True):
        return
    cmd = chime_command(name)
    if cmd:
        p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if wait:
            p.wait()


# ----------------------------------------------------------------------------- processing chain
# An original "refined AI" timbre: clean, close, crisp, with a faint synthetic
# shimmer and width. Mono voice in -> stereo (n, 2) out.

def _biquad(kind: str, f0: float, sr: int, gain_db: float = 0.0, q: float = 0.707):
    """RBJ audio-EQ-cookbook biquad -> (b, a)."""
    A = 10 ** (gain_db / 40)
    w = 2 * np.pi * f0 / sr
    cw, sw = np.cos(w), np.sin(w)
    alpha = sw / (2 * q)
    if kind == "hp":
        b = [(1 + cw) / 2, -(1 + cw), (1 + cw) / 2]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "bp":
        b = [alpha, 0, -alpha]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "peak":
        b = [1 + alpha * A, -2 * cw, 1 - alpha * A]
        a = [1 + alpha / A, -2 * cw, 1 - alpha / A]
    elif kind == "shelf":  # high shelf
        sa = 2 * np.sqrt(A) * alpha
        b = [A * ((A + 1) + (A - 1) * cw + sa), -2 * A * ((A - 1) + (A + 1) * cw),
             A * ((A + 1) + (A - 1) * cw - sa)]
        a = [(A + 1) - (A - 1) * cw + sa, 2 * ((A - 1) - (A + 1) * cw), (A + 1) - (A - 1) * cw - sa]
    else:
        raise ValueError(kind)
    return np.array(b) / a[0], np.array(a) / a[0]


def _impulse_response():
    """Two decorrelated short, bright 'glass room' tails (L, R)."""
    from scipy.signal import lfilter
    rng = np.random.default_rng(7)
    n = int(0.28 * SR)
    t = np.arange(n) / SR
    b, a = _biquad("hp", 900, SR)
    irs = []
    for _ in range(2):
        ir = rng.standard_normal(n) * np.exp(-t / 0.055)
        ir[: int(0.009 * SR)] = 0  # pre-delay keeps the voice up front
        ir = lfilter(b, a, ir)
        irs.append(ir / np.sqrt((ir ** 2).sum()))
    return irs


def _mod_delay(x, base_ms, depth_ms, rate_hz, phase=0.0):
    n = len(x)
    idx = np.arange(n, dtype=np.float64)
    d = (base_ms + depth_ms * np.sin(2 * np.pi * rate_hz * idx / SR + phase)) * SR / 1000
    src = np.clip(idx - d, 0, n - 1)
    lo = src.astype(np.int64)
    fr = src - lo
    return x[lo] * (1 - fr) + x[np.minimum(lo + 1, n - 1)] * fr


def process(x: np.ndarray, fx: float, ir=None) -> np.ndarray:
    from scipy.signal import fftconvolve, lfilter
    ir = ir if ir is not None else _impulse_response()
    x = x.astype(np.float64)
    if fx > 0:
        # 1. tone: tidy lows, remove boxiness, add precise diction and air
        for kind, f, g, q in (("hp", 95, 0, 0.707), ("peak", 280, -2.5 * fx, 1.0),
                              ("peak", 3300, 2.5 * fx, 0.9), ("shelf", 7500, 3.0 * fx, 0.707)):
            b, a = _biquad(kind, f, SR, g, q)
            x = lfilter(b, a, x)
        # 2. gentle compression for an even, close-mic delivery
        k = np.exp(-1 / (0.02 * SR))
        env = lfilter([1 - k], [1, -k], np.abs(x))
        env_db = 20 * np.log10(env + 1e-9)
        peak_db = 20 * np.log10(np.abs(x).max() + 1e-9)
        over = np.maximum(env_db - (peak_db - 14), 0)
        x = x * 10 ** (-(over * (1 - 1 / 2.5)) / 20)
    dry = np.stack([x, x], axis=1)
    if fx <= 0:
        out = dry
    else:
        # 3. shimmer: upper band doubled by two slowly drifting short delays, panned L/R
        b, a = _biquad("bp", 3500, SR, q=0.6)
        band = lfilter(b, a, x)
        sheen = np.stack([_mod_delay(band, 1.3, 0.35, 0.21),
                          _mod_delay(band, 2.1, 0.35, 0.27, phase=1.7)], axis=1) * (0.22 * fx)
        # 3b. exciter: soft-saturated upper band adds faint 'digital' harmonics
        norm = np.abs(band).max() + 1e-9
        dry = dry + (np.tanh(band / norm * 3) * norm / 3 * 0.12 * fx)[:, None]
        # 4. tiny bright room, decorrelated per channel (with its tail)
        room = np.stack([fftconvolve(x, r) for r in ir], axis=1) * (0.08 * fx)
        out = np.concatenate([dry + sheen, np.zeros((len(room) - len(dry), 2))]) + room
    out = np.tanh(out / (np.abs(out).max() + 1e-9) * 1.1) / np.tanh(1.1) * 0.9
    return out.astype(np.float32)

"""Speech input: energy-based voice activity detection + local Whisper."""
from __future__ import annotations

import queue
import re
import threading
import time
from contextlib import contextmanager

import numpy as np

SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_LEN = SAMPLE_RATE * FRAME_MS // 1000

# Whisper tends to hallucinate these on noise/silence (exact match after normalizing).
NOISE_PHRASES = {
    "", "you", "thank you", "thank you so much", "thanks", "thanks for watching",
    "thank you for watching", "thanks for listening", "thank you for listening",
    "bye", "bye bye", "okay", "uh", "um", "hmm", "so", "oh", "ah", "the end",
    "please subscribe", "like and subscribe", "see you next time", "see you in the next video",
}

# Caption and credit boilerplate from Whisper's training data. Real prompts never contain these.
HALLUCINATION_MARKERS = (
    "subtitles by", "subtitled by", "captions by", "captioned by", "amaraorg",
    "thanks for watching", "thank you for watching", "please subscribe",
    "like and subscribe", "dont forget to subscribe",
)

# Per-segment confidence limits. A segment is dropped if Whisper thinks it's probably
# not speech and is unsure of the words, or if the text is a repeating loop.
NO_SPEECH_PROB = 0.6
MIN_AVG_LOGPROB = -1.0
MAX_COMPRESSION_RATIO = 2.4


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x * x)))


def normalize(s: str) -> str:
    s = re.sub(r"[^a-z0-9 ]", "", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def is_repetitive(text: str) -> bool:
    """True for loops like 'I'm going to I'm going to I'm going to...'."""
    words = normalize(text).split()
    if len(words) < 6:
        return False
    for n in range(1, 5):
        grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
        top = max(grams.count(g) for g in set(grams))
        if top >= 4 and top * n >= 0.6 * len(words):
            return True
    return False


def is_noise(text: str) -> bool:
    t = normalize(text)
    if t in NOISE_PHRASES or is_repetitive(t):
        return True
    squashed = t.replace(" ", "")
    return any(m.replace(" ", "") in squashed for m in HALLUCINATION_MARKERS)


def keep_segment(seg) -> bool:
    """Drop the low-confidence segments Whisper writes over silence or noise."""
    if seg.no_speech_prob > NO_SPEECH_PROB and seg.avg_logprob < MIN_AVG_LOGPROB:
        return False
    if seg.compression_ratio > MAX_COMPRESSION_RATIO:
        return False
    return not is_noise(seg.text)


class Listener:
    """Captures one utterance at a time from the default microphone."""

    def __init__(self, silence_s: float = 1.0, max_s: float = 30.0, sensitivity: float = 3.0,
                 calibrate_s: float = 1.0, noise_cache: bool = True):
        import sounddevice as sd
        self.silence_frames = int(silence_s * 1000 / FRAME_MS)
        self.max_frames = int(max_s * 1000 / FRAME_MS)
        self.sensitivity = sensitivity
        self.q: queue.Queue[tuple[float, np.ndarray]] = queue.Queue()  # (monotonic time, frame)
        self.enabled = threading.Event()
        self.continuous = False
        self.timed_out = False
        self.stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                                     blocksize=FRAME_LEN, callback=self._cb)
        self.stream.start()
        self.noise_cache = noise_cache
        hint = _noise_hint() if noise_cache else None
        self.threshold = self._calibrate(calibrate_s, hint)

    def start_continuous(self):
        """Keep recording between listen() calls, so speech that starts while the
        previous utterance is being transcribed isn't lost. Call flush() after
        Jarvis speaks or chimes, so it never hears itself."""
        self.continuous = True
        self.enabled.set()

    def flush(self):
        self._drain()

    def close(self):
        self.stream.stop()
        self.stream.close()
        if self.noise_cache:
            _save_noise(self.noise)

    def _cb(self, indata, frames, t, status):
        if self.enabled.is_set():
            self.q.put((time.monotonic(), indata[:, 0].copy()))

    def _drain(self):
        try:
            while True:
                self.q.get_nowait()
        except queue.Empty:
            pass

    def _calibrate(self, seconds: float, hint: float | None = None) -> float:
        """Measure the room's background level. With a recent measurement (`hint`) a
        short check is enough: it only has to catch the room having got louder, and
        listen() keeps adapting to it anyway."""
        if hint is not None:
            seconds = min(seconds, HINT_CALIBRATE_S)
        else:
            time.sleep(0.3)  # the first frames after opening a mic are often near-silent
        self.enabled.set()
        levels = []
        end = time.time() + seconds
        while (left := end - time.time()) > 0:
            try:
                levels.append(rms(self.q.get(timeout=min(0.5, left))[1]))
            except queue.Empty:
                break
        self.enabled.clear()
        self._drain()
        measured = float(np.percentile(levels, 75)) if levels else None
        if hint is not None:
            self.noise = max(hint, measured or 0.0)
        else:
            self.noise = measured if measured is not None else 0.005
        return max(self.noise * self.sensitivity, 0.008)

    def listen(self, start_timeout: float | None = None,
               prefix: list[np.ndarray] | None = None, on_start=None) -> np.ndarray | None:
        """Block until one utterance is captured. Returns None for a too-short sound,
        or when start_timeout passes with no speech (then self.timed_out is True).
        `prefix` is speech already under way (caught by BargeIn); recording carries
        on from it until the speaker falls silent. `on_start()` is called the moment
        speech begins."""
        self.timed_out = False
        if not self.continuous:
            self._drain()
            self.enabled.set()
        preroll: list[np.ndarray] = []
        voiced, frames, silent, started = 0, [], 0, False
        if prefix:
            frames, started = list(prefix), True
        deadline = time.time() + start_timeout if start_timeout else None
        try:
            while True:
                if not started and deadline and time.time() > deadline:
                    self.timed_out = True
                    return None
                try:
                    _, f = self.q.get(timeout=0.5)
                except queue.Empty:
                    continue
                level = rms(f)
                loud = level > self.threshold
                if not started:
                    if not loud:  # keep tracking the room's noise while waiting
                        self.noise = 0.95 * self.noise + 0.05 * level
                        self.threshold = max(self.noise * self.sensitivity, 0.008)
                    preroll = (preroll + [f])[-10:]
                    voiced = voiced + 1 if loud else 0
                    if voiced >= 3:
                        started, frames = True, list(preroll)
                        if on_start:
                            on_start()
                    continue
                frames.append(f)
                silent = 0 if loud else silent + 1
                if silent >= self.silence_frames or len(frames) >= self.max_frames:
                    break
        finally:
            if not self.continuous:
                self.enabled.clear()
        if len(frames) < 12:  # < ~0.35 s: a click, not speech
            return None
        return np.concatenate(frames)


# The room's noise level is remembered between turns, saving ~0.7 s of calibration.
HINT_TTL_S = 10 * 60
HINT_CALIBRATE_S = 0.15


def _noise_file():
    from . import config
    return config.CACHE_DIR / "noise.json"


def _noise_hint() -> float | None:
    import json
    try:
        st = json.loads(_noise_file().read_text())
        if time.time() - st["at"] < HINT_TTL_S and 0 < st["noise"] < 1:
            return float(st["noise"])
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def _save_noise(noise: float) -> None:
    from . import config
    try:
        config.write_json(_noise_file(), {"noise": float(noise), "at": time.time()})
    except OSError:
        pass


class BargeIn:
    """Notices the user talking over Jarvis, so Jarvis can stop and listen.

    The hard part is that the mic also hears Jarvis through the speakers. When the
    speaker can report what it is playing (`output_level(t)`), we learn how loud
    that echo is in the mic (the echo "gain") and only count frames well above it.
    Otherwise (macOS `say`) we fall back to a much higher fixed threshold.
    Headphones make this near-perfect; laptop speakers work at normal volume."""

    MARGIN = 2.5          # mic must be this many times louder than the expected echo
    FLOOR = 1.5           # ... and this many times the normal speech threshold
    NO_REF_FACTOR = 4.0   # threshold multiplier when the playing audio is unknown
    TRIGGER = 8           # leaky count of loud frames (~240 ms) that means "user is talking"
    WARMUP = 15           # echo samples needed before trusting the echo estimate
    QUIET_OUT = 1e-3      # playback level treated as silence

    def __init__(self, listener, speaker, on_trigger):
        from collections import deque
        self.listener, self.speaker, self.on_trigger = listener, speaker, on_trigger
        self.ratios: deque[float] = deque(maxlen=300)   # mic/playback level while only Jarvis talks
        self.preroll: deque[np.ndarray] = deque(maxlen=20)
        self.score = 0
        self.captured: list[np.ndarray] | None = None

    @property
    def triggered(self) -> bool:
        return self.captured is not None

    def echo_gain(self) -> float | None:
        if len(self.ratios) < self.WARMUP:
            return None
        return float(np.percentile(self.ratios, 90))

    def feed(self, ts: float, frame: np.ndarray) -> bool:
        """Process one mic frame captured at monotonic time `ts`; True means barge-in."""
        self.preroll.append(frame)
        if not self.speaker.busy.is_set():
            self.score = 0
            return False
        level = rms(frame)
        out = getattr(self.speaker, "output_level", lambda t: None)(ts)
        floor = self.listener.threshold * self.FLOOR
        learn = False
        if out is None:
            limit = self.listener.threshold * self.NO_REF_FACTOR
        elif out < self.QUIET_OUT:  # a pause between sentences
            limit = floor
        elif (gain := self.echo_gain()) is None:
            limit, learn = float("inf"), True  # still learning how loud our own echo is
        else:
            limit = max(floor, gain * out * self.MARGIN)
            learn = level <= limit
        if learn:
            self.ratios.append(level / out)
        self.score = self.score + 1 if level > limit else max(0, self.score - 1)
        return self.score >= self.TRIGGER

    @contextmanager
    def watch(self):
        """While inside, consume mic frames and call on_trigger() if the user barges in.
        Afterwards `captured` holds the start of what they said (or None)."""
        self.captured, self.score = None, 0
        self.preroll.clear()
        stop = threading.Event()
        t = threading.Thread(target=self._run, args=(stop,), daemon=True)
        t.start()
        try:
            yield self
        finally:
            stop.set()
            t.join()

    def _run(self, stop: threading.Event):
        while not stop.is_set():
            try:
                ts, f = self.listener.q.get(timeout=0.1)
            except queue.Empty:
                continue
            if self.captured is not None:
                self.captured.append(f)
            elif self.feed(ts, f):
                self.captured = list(self.preroll)
                self.on_trigger()


def vocabulary_prompt(vocabulary: str) -> str | None:
    """Whisper's `initial_prompt`: a short glossary that nudges it towards the spelling
    of names and jargon ("GNSC", "Next.js") it would otherwise guess at."""
    words = [w.strip() for w in (vocabulary or "").split(",") if w.strip()]
    return f"Glossary: {', '.join(words)}." if words else None


def echoes_prompt(text: str, prompt: str | None) -> bool:
    """Whisper sometimes just repeats its prompt over silence: treat that as noise."""
    if not prompt:
        return False
    t, p = normalize(text), normalize(prompt)
    return bool(t) and len(t.split()) >= 3 and t in p


def whisper_devices(requested: str = "auto") -> list[tuple[str, str]]:
    """(device, compute type) to try, best first. On Apple Silicon, ctranslate2 4.9+
    runs Whisper on the GPU ("mps"): small.en went 0.90 s -> 0.57 s for a 4 s clip."""
    import platform
    cpu = ("cpu", "int8")
    if requested == "cpu":
        return [cpu]
    mps_ok = platform.system() == "Darwin" and platform.machine() == "arm64"
    if requested == "mps" or (requested == "auto" and mps_ok):
        try:
            import ctranslate2
            if "float16" in ctranslate2.get_supported_compute_types("mps"):
                return [("mps", "float16"), cpu]
        except Exception:  # noqa: BLE001 - older ctranslate2 has no mps: use the CPU
            pass
    return [cpu]


class Transcriber:
    def __init__(self, model: str = "base.en", vocabulary: str = "", device: str = "auto"):
        import os
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        from faster_whisper import WhisperModel
        self.prompt = vocabulary_prompt(vocabulary)
        last_error = None
        for dev, compute in whisper_devices(device):
            try:
                self.model = WhisperModel(model, device=dev, compute_type=compute)
                self.device = dev
                self(np.zeros(SAMPLE_RATE // 2, np.float32))  # the first call on a device is slow
                return
            except Exception as e:  # noqa: BLE001 - try the next device
                last_error = e
        raise last_error

    def __call__(self, audio: np.ndarray) -> str:
        segs, _ = self.model.transcribe(
            audio, language="en", beam_size=1, vad_filter=True,
            condition_on_previous_text=False, temperature=0.0, initial_prompt=self.prompt,
            no_speech_threshold=NO_SPEECH_PROB, log_prob_threshold=MIN_AVG_LOGPROB,
            compression_ratio_threshold=MAX_COMPRESSION_RATIO,
        )
        text = " ".join(s.text.strip() for s in segs if keep_segment(s)).strip()
        return "" if echoes_prompt(text, self.prompt) else text


def make_transcriber(cfg: dict) -> Transcriber:
    return Transcriber(cfg["whisper"], cfg.get("vocabulary", ""), cfg.get("whisper_device", "auto"))


DAEMON_BOOT_S = 6.0


class FastTranscriber:
    """Transcribes with the speech daemon's already-loaded Whisper (asking it to load
    now, so it's ready by the time you stop talking). Falls back to loading Whisper in
    this process if the daemon is off or unavailable."""

    def __init__(self, cfg: dict):
        self.cfg, self.local = cfg, None
        self.use_daemon = bool(cfg.get("speech_daemon", True))
        self.t0 = time.time()
        if self.use_daemon:
            from . import daemon
            daemon.warm(stt=True)

    def __call__(self, audio: np.ndarray) -> str:
        if self.use_daemon:
            from . import daemon
            text = daemon.transcribe(audio)
            # A daemon we just started takes a few seconds to come up; don't wait longer than that.
            boot_left = DAEMON_BOOT_S - (time.time() - self.t0)
            if text is None and boot_left > 0 and _wait_for(daemon.running, boot_left):
                text = daemon.transcribe(audio)
            if text is not None:
                return text
            self.use_daemon = False  # don't wait on it again this time
        if self.local is None:
            self.local = make_transcriber(self.cfg)
        return self.local(audio)


def _wait_for(check, limit_s: float) -> bool:
    end = time.time() + limit_s
    while time.time() < end:
        if check():
            return True
        time.sleep(0.1)
    return False


def make_fast_transcriber(cfg: dict):
    return FastTranscriber(cfg)

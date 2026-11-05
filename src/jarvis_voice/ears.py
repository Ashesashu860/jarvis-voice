"""Speech input: energy-based voice activity detection + local Whisper."""
from __future__ import annotations

import queue
import re
import threading
import time

import numpy as np

SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_LEN = SAMPLE_RATE * FRAME_MS // 1000

# Whisper tends to hallucinate these on noise/silence.
NOISE_PHRASES = {
    "", "you", "thank you", "thanks for watching", "thank you for watching",
    "bye", "okay", "uh", "um", "hmm", "so",
}


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x * x)))


def normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def is_noise(text: str) -> bool:
    return normalize(text) in NOISE_PHRASES


class Listener:
    """Captures one utterance at a time from the default microphone."""

    def __init__(self, silence_s: float = 1.0, max_s: float = 30.0, sensitivity: float = 3.0,
                 calibrate_s: float = 1.0):
        import sounddevice as sd
        self.silence_frames = int(silence_s * 1000 / FRAME_MS)
        self.max_frames = int(max_s * 1000 / FRAME_MS)
        self.sensitivity = sensitivity
        self.q: queue.Queue[np.ndarray] = queue.Queue()
        self.enabled = threading.Event()
        self.stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                                     blocksize=FRAME_LEN, callback=self._cb)
        self.stream.start()
        self.threshold = self._calibrate(calibrate_s)

    def close(self):
        self.stream.stop()
        self.stream.close()

    def _cb(self, indata, frames, t, status):
        if self.enabled.is_set():
            self.q.put(indata[:, 0].copy())

    def _drain(self):
        try:
            while True:
                self.q.get_nowait()
        except queue.Empty:
            pass

    def _calibrate(self, seconds: float) -> float:
        time.sleep(0.3)  # the first frames after opening a mic are often near-silent
        self.enabled.set()
        levels = []
        end = time.time() + seconds
        while time.time() < end:
            try:
                levels.append(rms(self.q.get(timeout=0.5)))
            except queue.Empty:
                break
        self.enabled.clear()
        self._drain()
        self.noise = float(np.percentile(levels, 75)) if levels else 0.005
        return max(self.noise * self.sensitivity, 0.008)

    def listen(self, start_timeout: float | None = None) -> np.ndarray | None:
        """Block until one utterance is captured (or start_timeout passes with no speech)."""
        self._drain()
        self.enabled.set()
        preroll: list[np.ndarray] = []
        voiced, frames, silent, started = 0, [], 0, False
        deadline = time.time() + start_timeout if start_timeout else None
        try:
            while True:
                if not started and deadline and time.time() > deadline:
                    return None
                try:
                    f = self.q.get(timeout=0.5)
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
                    continue
                frames.append(f)
                silent = 0 if loud else silent + 1
                if silent >= self.silence_frames or len(frames) >= self.max_frames:
                    break
        finally:
            self.enabled.clear()
        if len(frames) < 12:  # < ~0.35 s: a click, not speech
            return None
        return np.concatenate(frames)


class Transcriber:
    def __init__(self, model: str = "base.en"):
        import os
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        from faster_whisper import WhisperModel
        self.model = WhisperModel(model, device="cpu", compute_type="int8")

    def __call__(self, audio: np.ndarray) -> str:
        segs, _ = self.model.transcribe(audio, language="en", beam_size=1, vad_filter=True,
                                        condition_on_previous_text=False)
        return " ".join(s.text for s in segs).strip()

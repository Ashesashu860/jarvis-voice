"""Paths and user settings (~/.config/jarvis-voice/config.json)."""
from __future__ import annotations

import json
import os
from pathlib import Path

DEFAULT_BLEND = "bm_daniel:0.6,bm_george:0.25,bm_fable:0.15"

DEFAULTS: dict = {
    # voice
    "engine": "kokoro",            # kokoro (offline) | edge (Microsoft online) | say (macOS)
    "blend": DEFAULT_BLEND,        # kokoro voice blend
    "speed": 1.05,                 # kokoro speed
    "fx": 0.5,                     # AI voice processing amount, 0 = clean
    "edge_voice": "en-GB-ThomasNeural",
    "edge_rate": "+8%",
    "edge_pitch": "-2Hz",
    "mac_voice": None,             # None = best British voice installed
    "mac_rate": 185,
    "address": "sir",              # how Jarvis addresses you
    "chimes": True,
    # listening
    "wake": True,                  # jarvis talk: wait for "Hey Jarvis" (False = always listening)
    "wake_timeout": 10.0,          # seconds of silence before going back to standby
    "whisper": "base.en",
    "whisper_device": "auto",      # auto (Apple GPU when available, else CPU) | cpu | mps
    # comma-separated names/jargon Whisper should spell right
    "vocabulary": "Jarvis, Claude, Claude Code, Next.js, React, TypeScript, JavaScript, Python, Flutter, GitHub, npm",
    "listen_while_working": True,  # /jarvis:talk: keep listening while Claude works, so you can cut in
    "silence": 0.8,                # seconds of quiet that end your turn (lower = snappier, may cut you off)
    "sensitivity": 3.0,
    "converse_timeout": 60.0,      # /jarvis:talk: seconds of silence before the conversation ends
    "barge_in": True,             # jarvis talk: talking over Jarvis stops it and it listens
    # Claude Code integration
    "speak_replies": True,         # read Claude Code replies aloud (Stop hook)
    "narrate": True,               # speak progress updates while Claude works (PreToolUse hook)
    "persona_in_claude_code": True,  # give Claude Code the Jarvis manner while voice is on
    "max_spoken_chars": 600,       # longer replies are cut at a sentence; the rest stays on screen
    "speech_daemon": True,         # keep the voice loaded in the background for instant replies
    "daemon_idle_minutes": 30.0,   # the background voice exits after this long unused (frees ~600 MB)
}


def _xdg(var: str, fallback: str) -> Path:
    return Path(os.environ.get(var) or Path.home() / fallback)


CONFIG_DIR = Path(os.environ.get("JARVIS_VOICE_CONFIG") or _xdg("XDG_CONFIG_HOME", ".config") / "jarvis-voice")
CACHE_DIR = Path(os.environ.get("JARVIS_VOICE_CACHE") or _xdg("XDG_CACHE_HOME", ".cache") / "jarvis-voice")
MODELS_DIR = CACHE_DIR / "models"
CONFIG_FILE = CONFIG_DIR / "config.json"
PID_FILE = CACHE_DIR / "speaking.pid"


# Allowed values, checked by `jarvis config set` and when the file is loaded.
CHOICES = {"engine": ("kokoro", "edge", "say"), "whisper_device": ("auto", "cpu", "mps")}
RANGES = {"speed": (0.5, 2.0), "fx": (0.0, 3.0), "silence": (0.2, 5.0), "sensitivity": (1.0, 20.0),
          "wake_timeout": (1.0, 600.0), "converse_timeout": (5.0, 3600.0),
          "max_spoken_chars": (50, 10_000), "mac_rate": (80, 400), "daemon_idle_minutes": (1.0, 1440.0)}


def validate(key: str, value) -> None:
    """Raise ValueError if `value` isn't acceptable for `key`."""
    default = DEFAULTS[key]
    if value is None:
        if default is not None:
            raise ValueError(f"{key} can't be empty")
        return
    if isinstance(default, bool) and not isinstance(value, bool):
        raise ValueError(f"{key} expects true/false")
    if isinstance(default, (int, float)) and not isinstance(default, bool):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{key} expects a number")
    if isinstance(default, str) and not isinstance(value, str):
        raise ValueError(f"{key} expects text")
    if key in CHOICES and value not in CHOICES[key]:
        raise ValueError(f"{key} must be one of: {', '.join(CHOICES[key])}")
    if key in RANGES:
        lo, hi = RANGES[key]
        if not lo <= value <= hi:
            raise ValueError(f"{key} must be between {lo:g} and {hi:g}")
    if key == "blend":
        parse_blend(value)
    if key == "address" and not value.strip():
        raise ValueError("address can't be empty")


def parse_blend(blend: str) -> list[tuple[str, float]]:
    """'bm_daniel:0.6,bm_george:0.4' -> [('bm_daniel', 0.6), ('bm_george', 0.4)]."""
    import re
    out = []
    for part in (p.strip() for p in blend.split(",")):
        if not part:
            continue
        name, _, weight = part.partition(":")
        name = name.strip()
        if not re.fullmatch(r"[a-z]{2}_[a-z]+", name):
            raise ValueError(f"blend: '{name}' isn't a Kokoro voice name (like bm_daniel)")
        try:
            w = float(weight) if weight.strip() else 1.0
        except ValueError:
            raise ValueError(f"blend: weight for {name} must be a number") from None
        if w <= 0:
            raise ValueError(f"blend: weight for {name} must be above 0")
        out.append((name, w))
    if not out:
        raise ValueError("blend needs at least one voice")
    return out


def load() -> dict:
    """Settings, with defaults for anything missing. A bad value in the file (edited by
    hand) is ignored with a warning rather than crashing a hook later."""
    cfg = dict(DEFAULTS)
    try:
        saved = json.loads(CONFIG_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return cfg
    for k, v in (saved.items() if isinstance(saved, dict) else ()):
        if k in DEFAULTS:
            try:
                validate(k, v)
            except ValueError as e:
                import sys
                print(f"jarvis: ignoring setting in {CONFIG_FILE}: {e}", file=sys.stderr)
                continue
        cfg[k] = v
    return cfg


def save(cfg: dict) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    changed = {k: v for k, v in cfg.items() if DEFAULTS.get(k, object()) != v}
    write_text_atomic(CONFIG_FILE, json.dumps(changed, indent=2) + "\n")


def write_text_atomic(path: Path, text: str) -> None:
    """Replace a file in one step, so another process (hooks run concurrently across
    Claude Code sessions) never reads half of it."""
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def write_json(path: Path, data) -> None:
    write_text_atomic(path, json.dumps(data))


def coerce(key: str, value: str):
    """Parse a CLI string into the type of the default for `key`."""
    if key not in DEFAULTS:
        raise KeyError(f"unknown setting '{key}'. Known: {', '.join(DEFAULTS)}")
    default = DEFAULTS[key]
    if value.lower() == "default":
        return default
    if value.lower() in ("none", "null") and default is None:
        return None
    if isinstance(default, bool):
        if value.lower() in ("1", "true", "yes", "on"):
            return True
        if value.lower() in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"{key} expects true/false")
    try:
        parsed = int(value) if isinstance(default, int) else float(value) if isinstance(default, float) else value
    except ValueError:
        raise ValueError(f"{key} expects a number") from None
    validate(key, parsed)
    return parsed

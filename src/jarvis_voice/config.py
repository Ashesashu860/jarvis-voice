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
    "silence": 1.0,
    "sensitivity": 3.0,
    "converse_timeout": 60.0,      # /jarvis:talk: seconds of silence before the conversation ends
    "barge_in": True,             # jarvis talk: talking over Jarvis stops it and it listens
    # Claude Code integration
    "speak_replies": True,         # read Claude Code replies aloud (Stop hook)
    "narrate": True,               # speak progress updates while Claude works (PreToolUse hook)
    "persona_in_claude_code": True,  # give Claude Code the Jarvis manner while voice is on
    "max_spoken_chars": 600,       # longer replies are cut at a sentence; the rest stays on screen
}


def _xdg(var: str, fallback: str) -> Path:
    return Path(os.environ.get(var) or Path.home() / fallback)


CONFIG_DIR = Path(os.environ.get("JARVIS_VOICE_CONFIG") or _xdg("XDG_CONFIG_HOME", ".config") / "jarvis-voice")
CACHE_DIR = Path(os.environ.get("JARVIS_VOICE_CACHE") or _xdg("XDG_CACHE_HOME", ".cache") / "jarvis-voice")
MODELS_DIR = CACHE_DIR / "models"
CONFIG_FILE = CONFIG_DIR / "config.json"
PID_FILE = CACHE_DIR / "speaking.pid"


def load() -> dict:
    cfg = dict(DEFAULTS)
    try:
        cfg.update(json.loads(CONFIG_FILE.read_text()))
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return cfg


def save(cfg: dict) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    changed = {k: v for k, v in cfg.items() if DEFAULTS.get(k, object()) != v}
    CONFIG_FILE.write_text(json.dumps(changed, indent=2) + "\n")


def coerce(key: str, value: str):
    """Parse a CLI string into the type of the default for `key`."""
    if key not in DEFAULTS:
        raise KeyError(f"unknown setting '{key}'. Known: {', '.join(DEFAULTS)}")
    default = DEFAULTS[key]
    if value.lower() in ("none", "null", "default"):
        return DEFAULTS[key] if value.lower() == "default" else None
    if isinstance(default, bool):
        if value.lower() in ("1", "true", "yes", "on"):
            return True
        if value.lower() in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"{key} expects true/false")
    if isinstance(default, int):
        return int(value)
    if isinstance(default, float):
        return float(value)
    return value

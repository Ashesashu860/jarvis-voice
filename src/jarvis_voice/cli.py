"""`jarvis` command line."""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time

from . import __version__, config

C_DIM, C_GRN, C_YEL, C_RST = "\033[2m", "\033[32m", "\033[33m", "\033[0m"

EXIT_PHRASES = ("goodbye jarvis", "shut down", "power down", "exit jarvis", "goodbye")
RESET_PHRASES = ("new conversation", "reset conversation", "start over", "clear context")
SAMPLE_LINE = ("Good evening. All systems are operational. I've taken the liberty of reviewing "
               "your schedule, and I'm afraid it's rather ambitious. Shall I clear the afternoon?")


def time_of_day() -> str:
    h = time.localtime().tm_hour
    return "morning" if h < 12 else "afternoon" if h < 18 else "evening"


def voice_overrides(args) -> dict:
    return {k: getattr(args, k, None) for k in ("engine", "blend", "speed", "fx", "address")}


# ----------------------------------------------------------------------------- talk
def extract_wake(text: str) -> tuple[str | None, bool]:
    """(prompt, woke): only text after 'Jarvis' counts in wake-word mode."""
    m = re.search(r"\b(jarvis|jervis|travis|jarvus|javis)\b[,.!?]?\s*(.*)", text, re.I | re.S)
    if not m:
        return None, False
    return (m.group(2).strip() or None), True


def cmd_talk(args, cfg) -> int:
    from .brain import Brain
    from .ears import Listener, Transcriber, is_noise, normalize
    from .voice import chime, describe, make_speaker

    if not shutil.which("claude"):
        sys.exit("The `claude` CLI was not found on PATH. Install Claude Code first.")
    cfg = {**cfg, **{k: v for k, v in voice_overrides(args).items() if v is not None}}
    address = cfg["address"]
    speaker = make_speaker(cfg)
    brain = Brain(address, args.model, args.cwd or os.getcwd(), args.tools, args.permission_mode)
    listener = transcriber = None
    if not args.type:
        print(f"{C_DIM}Loading speech recognition ({cfg['whisper']})...{C_RST}")
        transcriber = Transcriber(cfg["whisper"])
        print(f"{C_DIM}Calibrating microphone, stay quiet for a second...{C_RST}")
        listener = Listener(cfg["silence"], 30.0, cfg["sensitivity"])

    mode = ("type your request and press Enter" if args.type
            else "say 'Jarvis' followed by your request" if args.wake else "just speak")
    print(f"{C_GRN}JARVIS online{C_RST} {C_DIM}(voice: {describe(cfg)}; {mode}; Ctrl+C interrupts, "
          f"say 'goodbye' or Ctrl+C twice to quit){C_RST}")
    speaker.say(f"Good {time_of_day()}, {address}. JARVIS online and at your service.")
    speaker.wait()

    awaiting_followup, last_interrupt = False, 0.0
    while True:
        try:
            if args.type:
                text = input(f"{C_GRN}You:{C_RST} ").strip()
            else:
                chime("Tink", cfg)
                print(f"{C_DIM}listening...{C_RST}", end="\r", flush=True)
                audio = listener.listen()
                if audio is None:
                    continue
                print(f"{C_DIM}transcribing...{C_RST}", end="\r", flush=True)
                text = transcriber(audio)
                if is_noise(text):
                    continue
                print(f"{C_GRN}You:{C_RST} {text}            ")
            if not text:
                continue

            if args.wake and not awaiting_followup:
                prompt, woke = extract_wake(text)
                if woke and not prompt:
                    speaker.say(f"Yes, {address}?")
                    speaker.wait()
                    awaiting_followup = True
                    continue
                if not prompt:
                    continue
            else:
                prompt = text
            awaiting_followup = False

            n = normalize(prompt)
            if any(n == p or n.startswith(p) for p in EXIT_PHRASES):
                speaker.say(f"Very good, {address}. Powering down.")
                speaker.wait()
                return 0
            if any(p in n for p in RESET_PHRASES):
                brain.session_id = None
                speaker.say(f"Of course, {address}. Starting a fresh conversation.")
                speaker.wait()
                continue

            if not args.type:
                chime("Pop", cfg)
            brain.ask(prompt, speaker)
            speaker.wait()
        except KeyboardInterrupt:
            now = time.time()
            busy = speaker.busy.is_set() or brain.proc is not None
            brain.cancel()
            speaker.stop()
            if not busy and now - last_interrupt < 1.5:
                print("\nGoodbye.")
                return 0
            last_interrupt = now
            print(f"\n{C_YEL}[interrupted; Ctrl+C again to quit]{C_RST}")
        except EOFError:
            return 0


# ----------------------------------------------------------------------------- listen / say
def cmd_listen(args, cfg) -> int:
    """Record one spoken prompt and print its transcript (used by /jarvis:listen)."""
    from .ears import Listener, Transcriber, is_noise
    from .hooks import hush
    from .voice import chime

    hush()
    transcriber = Transcriber(cfg["whisper"])
    listener = Listener(cfg["silence"], args.max_seconds, cfg["sensitivity"], calibrate_s=0.8)
    chime("Tink", cfg, wait=True)  # finish the chime so it isn't recorded as speech
    print("Listening... speak now.", file=sys.stderr, flush=True)
    t0 = time.time()
    audio = listener.listen(start_timeout=args.timeout)
    if os.environ.get("JARVIS_DEBUG"):
        print(f"[threshold {listener.threshold:.4f}, waited {time.time() - t0:.1f}s, "
              f"captured {0 if audio is None else len(audio) / 16000:.1f}s]", file=sys.stderr)
    listener.close()
    if audio is None:
        print("(no speech detected)")
        return 1
    chime("Pop", cfg)
    text = transcriber(audio)
    if is_noise(text):
        print("(no speech detected)")
        return 1
    print(text)
    return 0


def cmd_say(args, cfg) -> int:
    from .voice import make_speaker, split_sentences

    if args.file:
        with open(args.file, encoding="utf-8") as f:
            text = f.read()
        if args.delete:
            os.unlink(args.file)
    elif args.text:
        text = " ".join(args.text)
    else:
        text = sys.stdin.read()
    cfg = {**cfg, **{k: v for k, v in voice_overrides(args).items() if v is not None}}
    speaker = make_speaker(cfg)
    for s in split_sentences(text):
        speaker.say(s)
    speaker.wait()
    return 0


def cmd_demo(args, cfg) -> int:
    from .voice import describe, make_speaker
    cfg = {**cfg, **{k: v for k, v in voice_overrides(args).items() if v is not None}}
    print(f"{C_DIM}voice: {describe(cfg)}{C_RST}")
    speaker = make_speaker(cfg)
    speaker.say(SAMPLE_LINE.replace("Good evening.", f"Good {time_of_day()}, {cfg['address']}."))
    speaker.wait()
    return 0


def cmd_samples(args, cfg) -> int:
    """Render voice variants to WAV files for side-by-side comparison."""
    import wave

    import numpy as np

    from .config import DEFAULT_BLEND
    from .voice import SR, KokoroSpeaker

    os.makedirs(args.dir, exist_ok=True)
    variants = {
        "A_default": (DEFAULT_BLEND, 1.05, 0.5),
        "B_more_fx": (DEFAULT_BLEND, 1.05, 1.0),
        "C_clean": (DEFAULT_BLEND, 1.05, 0.0),
        "D_daniel_pure": ("bm_daniel", 1.05, 0.5),
        "E_daniel_george_even": ("bm_daniel:0.5,bm_george:0.5", 1.08, 0.5),
        "F_daniel_fable_warm": ("bm_daniel:0.55,bm_fable:0.45", 1.05, 0.5),
        "G_deeper_daniel_lewis": ("bm_daniel:0.6,bm_lewis:0.4", 1.05, 0.5),
        "H_george_crisp": ("bm_george:0.7,bm_daniel:0.3", 1.1, 0.5),
    }
    for name, (blend, speed, fx) in variants.items():
        sp = KokoroSpeaker(blend, speed, fx, None)
        audio = sp.render(SAMPLE_LINE)
        pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
        path = os.path.join(args.dir, f"{name}.wav")
        with wave.open(path, "wb") as w:
            w.setnchannels(2)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes(pcm.tobytes())
        print(f"  {path}   --blend '{blend}' --speed {speed} --fx {fx}")
    return 0


def cmd_hush(args, cfg) -> int:
    from .hooks import hush
    hush()
    return 0


def cmd_hook(args, cfg) -> int:
    from .hooks import run
    return run(args.event)


# ----------------------------------------------------------------------------- settings
def cmd_toggle(on: bool):
    def run(args, cfg) -> int:
        from .hooks import hush
        cfg["speak_replies"] = on
        config.save(cfg)
        if not on:
            hush()
        print(f"Jarvis voice replies in Claude Code: {'ON' if on else 'OFF'}")
        return 0
    return run


def cmd_status(args, cfg) -> int:
    from .voice import describe, kokoro_ready
    print(f"jarvis-voice {__version__}")
    print(f"  voice replies in Claude Code: {'ON' if cfg['speak_replies'] else 'OFF'}")
    print(f"  voice: {describe(cfg)}")
    print(f"  voice model downloaded: {'yes' if kokoro_ready() else 'no (run: jarvis setup)'}")
    print(f"  config: {config.CONFIG_FILE}")
    print(f"  claude CLI: {shutil.which('claude') or 'not found'}")
    return 0


def cmd_config(args, cfg) -> int:
    if args.action == "show" or not args.action:
        print(json.dumps(cfg, indent=2))
    elif args.action == "set":
        if not args.key or args.value is None:
            sys.exit("usage: jarvis config set <key> <value>")
        try:
            cfg[args.key] = config.coerce(args.key, args.value)
        except (KeyError, ValueError) as e:
            sys.exit(str(e))
        config.save(cfg)
        print(f"{args.key} = {cfg[args.key]!r}")
    elif args.action == "reset":
        config.save(dict(config.DEFAULTS))
        print("Settings reset to defaults.")
    elif args.action == "path":
        print(config.CONFIG_FILE)
    return 0


def cmd_setup(args, cfg) -> int:
    from .voice import ensure_kokoro
    print(f"{C_GRN}Setting up Jarvis voice{C_RST}")
    print("1/4 Voice model (Kokoro, ~350 MB, one time)")
    ensure_kokoro()
    print("2/4 Speech recognition model (Whisper)")
    from .ears import Transcriber
    Transcriber(cfg["whisper"])
    print("3/4 Microphone check (your OS may ask for permission)")
    try:
        import numpy as np
        import sounddevice as sd
        rec = sd.rec(16000, samplerate=16000, channels=1, dtype="float32")
        sd.wait()
        print("    microphone OK" if float(np.abs(rec).max()) > 0 else
              f"    {C_YEL}microphone returned silence: allow mic access for your terminal app{C_RST}")
    except Exception as e:
        print(f"    {C_YEL}microphone unavailable: {e}{C_RST}")
    if sys.stdin.isatty() and not args.yes:
        ans = input(f"4/4 How should Jarvis address you? [{cfg['address']}] ").strip()
        if ans:
            cfg["address"] = ans
            config.save(cfg)
    else:
        print("4/4 Address: " + cfg["address"] + "  (change: jarvis config set address <name>)")
    cmd_demo(args, cfg)
    print(f"\n{C_GRN}Done.{C_RST} Run `jarvis talk` for a voice conversation, or in Claude Code:\n"
          "  /plugin marketplace add Ashesashu860/jarvis-voice\n"
          "  /plugin install jarvis@jarvis-voice\n"
          "then use /jarvis:listen to speak a prompt.")
    return 0


# ----------------------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="jarvis", description="A JARVIS-style voice for Claude Code.")
    ap.add_argument("--version", action="version", version=f"jarvis-voice {__version__}")
    sub = ap.add_subparsers(dest="cmd")

    def voice_opts(p):
        p.add_argument("--engine", choices=["kokoro", "edge", "say"])
        p.add_argument("--blend", help="kokoro voice blend, e.g. 'bm_daniel:0.6,bm_george:0.4'")
        p.add_argument("--speed", type=float)
        p.add_argument("--fx", type=float, help="voice processing amount, 0 = clean")
        p.add_argument("--address", help="how Jarvis addresses you")
        return p

    p = voice_opts(sub.add_parser("talk", help="hands-free voice conversation with Claude"))
    p.add_argument("--wake", action="store_true", help="only respond to 'Jarvis, ...'")
    p.add_argument("--type", action="store_true", help="type prompts, hear replies")
    p.add_argument("--model", help="Claude model, e.g. sonnet, opus, haiku")
    p.add_argument("--cwd", help="directory Claude works in (default: current)")
    p.add_argument("--tools", default="Read,Glob,Grep,WebSearch,WebFetch",
                   help="tools Claude may use without asking ('' for none)")
    p.add_argument("--permission-mode", help="e.g. acceptEdits")
    p.set_defaults(func=cmd_talk)

    p = sub.add_parser("listen", help="record one spoken prompt and print the transcript")
    p.add_argument("--timeout", type=float, default=15.0, help="seconds to wait for speech to start")
    p.add_argument("--max-seconds", type=float, default=60.0)
    p.set_defaults(func=cmd_listen)

    p = voice_opts(sub.add_parser("say", help="speak text (args, --file, or stdin)"))
    p.add_argument("text", nargs="*")
    p.add_argument("--file")
    p.add_argument("--delete", action="store_true", help=argparse.SUPPRESS)
    p.set_defaults(func=cmd_say)

    sub.add_parser("hush", help="stop speaking").set_defaults(func=cmd_hush)
    sub.add_parser("on", help="read Claude Code replies aloud").set_defaults(func=cmd_toggle(True))
    sub.add_parser("off", help="stop reading Claude Code replies aloud").set_defaults(func=cmd_toggle(False))
    sub.add_parser("status", help="show settings and readiness").set_defaults(func=cmd_status)

    p = sub.add_parser("config", help="show or change settings")
    p.add_argument("action", nargs="?", choices=["show", "set", "reset", "path"])
    p.add_argument("key", nargs="?")
    p.add_argument("value", nargs="?")
    p.set_defaults(func=cmd_config)

    p = voice_opts(sub.add_parser("setup", help="download models, check mic, pick your title"))
    p.add_argument("-y", "--yes", action="store_true", help="non-interactive")
    p.set_defaults(func=cmd_setup)

    voice_opts(sub.add_parser("demo", help="speak a sample line")).set_defaults(func=cmd_demo)

    p = sub.add_parser("samples", help="write voice variant WAVs to compare")
    p.add_argument("dir", nargs="?", default="jarvis-samples")
    p.set_defaults(func=cmd_samples)

    p = sub.add_parser("hook", help=argparse.SUPPRESS)
    p.add_argument("event", choices=["stop", "prompt"])
    p.set_defaults(func=cmd_hook)
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    if not args.cmd:
        ap.print_help()
        return 0
    return args.func(args, config.load()) or 0


if __name__ == "__main__":
    sys.exit(main())

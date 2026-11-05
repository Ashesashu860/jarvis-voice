# jarvis-voice

**Talk to Claude Code and hear it answer in a refined, JARVIS-style British AI voice.**

- 🎙️ **Speak your prompts.** In Claude Code, run `/jarvis:listen` and just talk. Speech is transcribed locally with Whisper.
- 🔊 **Hear the replies.** Claude's answers are read aloud as they finish, in a calm, crisp British voice with a subtle "AI" sheen.
- 🤵 **The manner too.** While voice mode is on, Claude opens each reply with a short spoken summary in a composed, dry-witted butler style.
- 🗣️ **Hands-free mode.** `jarvis talk` is a continuous voice conversation with Claude (optional "Jarvis, ..." wake word).
- 🔒 **Local and offline voice.** Text-to-speech (Kokoro) and speech-to-text (Whisper) both run on your machine. No audio leaves it.

> **About the voice.** This project does **not** clone, sample or imitate any actor's voice or any film audio.
> The default voice is an original blend of stock, Apache-2.0 [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M)
> British voices (60% `bm_daniel`, 25% `bm_george`, 15% `bm_fable`), run through an original processing chain.
> "JARVIS" is used here only as a generic name for a style of assistant. This project is not affiliated with
> or endorsed by Marvel, Disney or anyone involved with the films.

## Requirements

- macOS (built and tested on Apple Silicon; Linux may work, but `say` fallback and chimes are macOS-only)
- [Claude Code](https://claude.com/claude-code) installed and logged in (the `claude` command)
- Python 3.10+ (the installer handles this via `uv`)
- ~400 MB of disk for the voice and speech models (downloaded once to `~/.cache/jarvis-voice`)

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/Ashesashu860/jarvis-voice/main/install.sh | bash
```

This installs the `jarvis` command (via `uv` or `pipx`), downloads the models, checks your microphone,
asks how Jarvis should address you, and plays a sample.

Or do it by hand:

```bash
uv tool install "jarvis-voice[online] @ git+https://github.com/Ashesashu860/jarvis-voice"
# or: pipx install "jarvis-voice[online] @ git+https://github.com/Ashesashu860/jarvis-voice"
jarvis setup
```

Then enable it inside Claude Code (type these at the Claude Code prompt):

```
/plugin marketplace add Ashesashu860/jarvis-voice
/plugin install jarvis@jarvis-voice
```

Restart Claude Code if the commands don't appear straight away.

> macOS will ask for **microphone access** for your terminal app the first time. Allow it,
> or enable it later in System Settings → Privacy & Security → Microphone.

## Use

### Inside Claude Code

| Command | What it does |
|---|---|
| `/jarvis:listen` | Speak your prompt. A *tink* means it's listening and a *pop* means it heard you. It stops after a second of silence. |
| `/jarvis:on` / `/jarvis:off` | Turn spoken replies on or off |
| `/jarvis:hush` | Stop the current speech (sending any new prompt also stops it) |

With the plugin installed, every finished reply is read aloud: its opening prose, up to about 600 characters,
skipping code blocks and tables. The rest stays on screen.

### Hands-free conversation

```bash
jarvis talk            # always listening: just speak
jarvis talk --wake     # responds only to "Jarvis, ..."
jarvis talk --type     # type prompts, hear replies
```

Say "new conversation" to clear context, or "goodbye" to quit. Ctrl+C interrupts.
By default Claude may read files and search the web without asking. Add `--permission-mode acceptEdits` to let it edit files too.

### Other commands

```bash
jarvis say "Good evening."     # speak any text (or pipe text in)
jarvis demo                    # hear the current voice
jarvis samples ./samples       # render voice variants to WAV to compare
jarvis status                  # what's on, what's downloaded
jarvis config show             # all settings
jarvis config set address boss # how Jarvis addresses you
```

## Settings

Stored in `~/.config/jarvis-voice/config.json`. Change them with `jarvis config set <key> <value>`.

| Key | Default | Meaning |
|---|---|---|
| `address` | `sir` | How Jarvis addresses you: `ma'am`, `boss`, your name… |
| `engine` | `kokoro` | `kokoro` (offline), `edge` (Microsoft online neural voices), `say` (macOS) |
| `blend` | `bm_daniel:0.6,bm_george:0.25,bm_fable:0.15` | Kokoro voice mix. Voices: `bm_daniel`, `bm_george`, `bm_fable`, `bm_lewis` |
| `speed` | `1.05` | Speaking speed |
| `fx` | `0.5` | Amount of "AI" processing. `0` is a clean voice, `1`+ is more synthetic |
| `speak_replies` | `true` | Read Claude Code replies aloud |
| `persona_in_claude_code` | `true` | Give Claude the Jarvis manner while voice is on |
| `max_spoken_chars` | `600` | How much of a reply to read aloud |
| `whisper` | `base.en` | Speech model: `tiny.en` (fastest) … `small.en` (more accurate) |
| `silence` | `1.0` | Seconds of silence that end your turn |
| `sensitivity` | `3.0` | Mic threshold over background noise. Lower is more sensitive |
| `chimes` | `true` | Listening and heard sounds |

## How the voice is made

1. **Kokoro** synthesises speech offline from a weighted blend of stock British voices. `bm_daniel` gives steady,
   brisk delivery, `bm_george` crisp diction, and `bm_fable` warmth.
2. **Processing chain**, scaled by `fx`:
   - EQ: low cut at 95 Hz, −2.5 dB at 280 Hz to remove boxiness, +2.5 dB at 3.3 kHz for diction, +3 dB air above 7.5 kHz
   - gentle 2.5:1 compression for a close, even sound
   - a faint stereo shimmer: the upper band, doubled through two slowly drifting 1–2 ms delays
   - a soft high-band exciter that adds subtle "digital" harmonics
   - a tiny bright room, then a soft limiter
3. Each sentence is generated while the previous one plays, so speech starts in about a second.

## How it plugs into Claude Code

- `UserPromptSubmit` hook → `jarvis hook prompt`: stops any current speech, and while voice is on, adds a short
  style note asking Claude to open with a speakable summary in the Jarvis manner.
- `Stop` hook → `jarvis hook stop`: takes the final reply and speaks its opening in a background process,
  so Claude Code never waits on audio.
- `/jarvis:listen` runs `jarvis listen` and hands the transcript to Claude as your prompt.

The hooks do nothing if `jarvis` isn't installed and always exit cleanly, so a voice problem can't break a session.

## Uninstall

```
/plugin uninstall jarvis@jarvis-voice
```
```bash
uv tool uninstall jarvis-voice   # or: pipx uninstall jarvis-voice
rm -rf ~/.cache/jarvis-voice ~/.config/jarvis-voice
```

## Develop

```bash
git clone https://github.com/Ashesashu860/jarvis-voice && cd jarvis-voice
python3 -m venv .venv && .venv/bin/pip install -e ".[online,dev]"
.venv/bin/pytest
claude --plugin-dir ./plugin    # try the plugin without installing it
```

## Credits

- [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) (Apache-2.0) via [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (MIT), OpenAI Whisper models (MIT)
- [edge-tts](https://github.com/rany2/edge-tts) (optional online engine)

MIT licensed. See [LICENSE](LICENSE).

# Changelog

## Unreleased

- **Snappier voice turns.** Whisper runs on the Apple Silicon GPU when available (`whisper_device auto`; small.en
  0.90 s -> 0.57 s per 4 s clip), the end-of-turn pause is 0.8 s instead of 1.0 s, a long first sentence is
  spoken in two parts so the voice starts sooner, and `/jarvis:talk` sends the reply before the mic finishes
  opening. `jarvis daemon --log` shows real timings.
- **Replies are no longer read twice** in `/jarvis:talk`. A spoken follow-up is recorded as a hidden
  "Stop hook feedback" entry, which progress narration didn't treat as a new prompt, so it re-read the previous
  reply on the first step of the next turn. Narration also now skips any reply the Stop hook already spoke.
- **Faster `/jarvis:talk` turns.** Each turn used to load the voice (~2 s) and Whisper (~2 s) from scratch. Turns
  now borrow both from the speech daemon, so Jarvis starts answering in about half a second, and talking over it
  still works (the daemon reports how loud it's playing, so it can tell its own echo from you).
- **`/jarvis:listen` is ready at once.** Whisper loads in the daemon while you're talking instead of before the
  chime, and stays loaded for next time.
- **Quicker mic start.** The room's noise level is remembered for ten minutes, so the mic is ready in about
  0.15 s instead of 0.9 s.
- The background listener during `/jarvis:talk` uses the daemon's Whisper too, and tells Jarvis's echo from your
  voice better.
- The daemon handles requests in parallel, so a transcription never delays a hush.
- **`/jarvis:talk` stays in its own window.** The session you type it in owns the conversation; a busy second
  window can no longer take it over.
- **Fewer misfires in `jarvis talk`.** "Shut down the dev server" or "make it start over" are requests now, not
  commands to quit or reset; only the whole phrase counts.
- **No more hangs.** `jarvis talk` could freeze if Claude wrote a lot of error output; cancelling now force-stops
  Claude if it lingers.
- **Faster progress updates on long sessions.** The transcript is read from the end instead of in full on every
  tool call.
- **`jarvis doctor`** checks PATH, the plugin, settings, the voice model, audio devices and cache privacy, and says
  how to fix each problem (`--mic` also tests recording, `--fix` repairs what it can).
- **Settings are checked.** `jarvis config set` rejects out-of-range or malformed values (speed, blend, engine...),
  and a bad value in the settings file is ignored with a warning instead of breaking a hook.
- **Safer downloads.** Model downloads time out instead of hanging, and a cut-off or damaged model is detected
  and downloaded again.
- **Sturdier state.** State files are written atomically, and something you say while Jarvis is reading the inbox
  is no longer lost. The cache folder (which holds transcripts) and the daemon socket are owner-only.
- **Linux:** falls back to espeak-ng and freedesktop chimes when macOS `say`/`afplay` aren't there.
- A reply cut short at a code block now ends with "The details are on screen."
- CI runs ruff; the version is defined once, in `__init__.py`.

## 0.4.0

- **Instant replies.** A background speech daemon (`jarvis daemon`) keeps the voice loaded, so Jarvis starts
  speaking in under a second instead of about three. It starts on the first reply and exits after
  `daemon_idle_minutes` (default 30) unused, or on `/jarvis:off`. Set `speech_daemon false` for the old behaviour.
- **Several sessions at once.** Speech is tracked per Claude Code session: typing in one window silences only that
  window, replies from other windows wait their turn, and stale progress updates are dropped. Narration state is
  kept per session too.
- **Code read naturally.** `hooks.py:42` → "hooks dot py, line 42", `speak_detached` → "speak detached" (it used to
  come out as "speakdetached"), paths shortened to their last part, units and versions spelled out, `e.g.`/`→`/`~`
  expanded, emoji skipped instead of read aloud ("party popper").
- Hooks no longer import numpy (startup about 50 ms → 17 ms per tool call).
- CI runs the tests on macOS and Linux.

## 0.3.0

- `/jarvis:talk`: hands-free conversation inside a Claude Code session, with barge-in.
- Progress narration while Claude works; talk over it to steer the task.

## 0.2.0

- `jarvis talk`: "Hey Jarvis" wake word with a conversation window.

## 0.1.0

- First release: spoken replies in Claude Code, `/jarvis:listen`, Kokoro voice blend and processing chain.

# Changelog

## Unreleased

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

# Changelog

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

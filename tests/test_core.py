import json
import os
import threading
from types import SimpleNamespace

import numpy as np

from jarvis_voice import config, hooks
from jarvis_voice.cli import extract_wake, is_pause_phrase, is_sleep_phrase
from jarvis_voice.ears import FRAME_LEN, BargeIn, is_noise, keep_segment
from jarvis_voice.voice import SR, clean_for_speech, level_at, loudness_envelope, split_sentences


def test_split_sentences_keeps_decimals_and_titles():
    s = split_sentences("Certainly, sir. Version 3.5 is out now! Dr. Smith agrees; fine\nNext line")
    assert s == ["Certainly, sir. ", "Version 3.5 is out now! ", "Dr. Smith agrees; ", "fine\n", "Next line"]


def test_clean_for_speech_strips_markdown():
    assert clean_for_speech("**Bold** see [docs](http://x.com) and `code`\n- item") == "Bold see docs and code item"


def test_wake_word():
    assert extract_wake("Hey Jarvis, what's the time?") == ("what's the time?", True)
    assert extract_wake("Hey, Jarvis.") == (None, True)
    assert extract_wake("Jarvis.") == (None, True)
    assert extract_wake("Okay Jervis open the report") == ("open the report", True)
    assert extract_wake("Hey Travis, what's the weather?") == ("what's the weather?", True)  # common mishearing
    assert extract_wake("what's the time?") == (None, False)
    assert extract_wake("Travis Scott has a new album") == (None, False)  # plain name, no greeting
    assert extract_wake("customer service was slow") == (None, False)


def test_whisper_hallucinations_filtered():
    assert is_noise("Thank you.")
    assert is_noise("Thanks for watching!")
    assert is_noise("Subtitles by the Amara.org community")
    assert is_noise("I'm going to I'm going to I'm going to I'm going to")
    assert not is_noise("What's the weather like today?")
    assert not is_noise("Thank you, can you open the report?")
    assert not is_noise("run the tests again and again")


def test_low_confidence_segments_dropped():
    def seg(text, nsp=0.1, lp=-0.3, cr=1.2):
        return SimpleNamespace(text=text, no_speech_prob=nsp, avg_logprob=lp, compression_ratio=cr)
    assert keep_segment(seg(" Open the report."))
    assert not keep_segment(seg(" Open the report.", nsp=0.9, lp=-1.5))  # silence, unsure
    assert keep_segment(seg(" Open the report.", nsp=0.9, lp=-0.2))  # quiet but confident
    assert not keep_segment(seg(" la la la la la la la", cr=3.0))
    assert not keep_segment(seg(" Thanks for watching."))


def test_pause_phrases():
    assert is_pause_phrase("Stop.")
    assert is_pause_phrase("Wait, wait.")
    assert is_pause_phrase("Jarvis, hold on please")
    assert not is_pause_phrase("Stop the server and restart it")
    assert not is_pause_phrase("Wait for the build to finish")


class _FakeSpeaker:
    def __init__(self, out):
        self.busy = threading.Event()
        self.busy.set()
        self.out = out

    def output_level(self, t):
        return self.out


def _frames(level, n):
    return [np.full(FRAME_LEN, level, np.float32) for _ in range(n)]


def test_barge_in_ignores_own_echo_but_hears_user():
    listener = SimpleNamespace(threshold=0.01)
    barge = BargeIn(listener, _FakeSpeaker(out=0.3), on_trigger=lambda: None)
    # Jarvis talking: the mic hears its echo at 0.03 (gain 0.1). Never a barge-in.
    assert not any(barge.feed(0.0, f) for f in _frames(0.03, 200))
    assert abs(barge.echo_gain() - 0.1) < 1e-3
    # The user talks over it, well above the echo: caught within ~a quarter second.
    hits = [barge.feed(0.0, f) for f in _frames(0.2, BargeIn.TRIGGER)]
    assert hits[-1] and not any(hits[:-1])


def test_barge_in_needs_sustained_speech_and_only_while_speaking():
    listener = SimpleNamespace(threshold=0.01)
    speaker = _FakeSpeaker(out=0.0)  # pause between sentences
    barge = BargeIn(listener, speaker, on_trigger=lambda: None)
    clicks = _frames(0.2, 2) + _frames(0.0, 6)
    assert not any(barge.feed(0.0, f) for f in clicks * 5)  # a cough or a click isn't speech
    speaker.busy.clear()  # Jarvis silent: barge-in is the normal listener's job
    assert not any(barge.feed(0.0, f) for f in _frames(0.2, 40))


def test_barge_in_without_playback_reference_uses_high_threshold():
    listener = SimpleNamespace(threshold=0.01)
    barge = BargeIn(listener, _FakeSpeaker(out=None), on_trigger=lambda: None)
    assert not any(barge.feed(0.0, f) for f in _frames(0.03, 40))  # echo of macOS `say`
    assert any(barge.feed(0.0, f) for f in _frames(0.08, 20))


def test_playback_level_follows_the_audio():
    audio = np.concatenate([np.full(SR, 0.5, np.float32), np.zeros(SR, np.float32)])
    ref = (100.0, loudness_envelope(np.stack([audio, audio], axis=1)))
    assert abs(level_at(ref, 100.5) - 0.5) < 1e-3
    assert abs(level_at(ref, 101.2) - 0.5) < 1e-3  # echo tail window still covers the sound
    assert level_at(ref, 101.6) == 0.0
    assert level_at(ref, 99.0) == 0.0  # before playback started
    assert level_at(ref, 103.0) == 0.0  # long finished


def test_sleep_phrases():
    assert is_sleep_phrase("That's all.")
    assert is_sleep_phrase("Thank you, Jarvis.")
    assert is_sleep_phrase("ok stand by")
    assert not is_sleep_phrase("What's all this about?")


def test_spoken_excerpt_stops_at_code_and_length():
    reply = "Done, sir. I fixed the bug.\n```py\nprint(1)\n```\nMore text."
    assert hooks.spoken_excerpt(reply, 600) == "Done, sir. I fixed the bug. The details are on screen."
    assert hooks.spoken_excerpt(reply, 600, mention_code=False) == "Done, sir. I fixed the bug."
    assert hooks.spoken_excerpt("Done, sir.", 600) == "Done, sir."  # no code: nothing added
    long = " ".join(f"Sentence number {i}." for i in range(100))
    out = hooks.spoken_excerpt(long, 100)
    assert out.endswith("The rest is on screen.") and len(out) < 160


def test_last_reply_from_transcript(tmp_path):
    lines = [
        {"type": "user", "message": {"role": "user", "content": "old prompt"}},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "old answer"}]}},
        {"type": "user", "message": {"role": "user", "content": "new prompt"}},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "Let me check."},
                                                      {"type": "tool_use", "name": "Read"}]}},
        {"type": "user", "message": {"content": [{"type": "tool_result", "content": "..."}]}},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "All done, sir."}]}},
    ]
    p = tmp_path / "t.jsonl"
    p.write_text("\n".join(json.dumps(x) for x in lines))
    assert hooks.last_reply_from_transcript(str(p)) == "All done, sir."


def test_config_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "config.json")
    cfg = config.load()
    cfg["fx"] = config.coerce("fx", "0.8")
    cfg["speak_replies"] = config.coerce("speak_replies", "off")
    config.save(cfg)
    again = config.load()
    assert again["fx"] == 0.8 and again["speak_replies"] is False
    assert json.loads((tmp_path / "config.json").read_text()) == {"fx": 0.8, "speak_replies": False}


def test_describe_tool():
    assert hooks.describe_tool("Bash", {"description": "Run the tests"}) == "Run the tests."
    assert hooks.describe_tool("Edit", {"file_path": "/x/src/test_core.py"}) == "Editing test core.py."
    assert hooks.describe_tool("TodoWrite", {}) is None


def test_converse_session_claim(tmp_path, monkeypatch):
    from jarvis_voice import converse
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(converse, "STATE_FILE", tmp_path / "converse.json")
    assert not converse.claim("a")
    converse.start()
    assert converse.claim("a")       # first session to stop claims it
    assert converse.claim("a")
    assert not converse.claim("b")   # other sessions keep normal behaviour
    converse.end()
    assert not converse.is_on()


def test_stop_hook_blocks_with_spoken_reply(tmp_path, monkeypatch, capsys):
    from jarvis_voice import converse
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(config, "PID_FILE", tmp_path / "speaking.pid")
    monkeypatch.setattr(converse, "STATE_FILE", tmp_path / "converse.json")
    monkeypatch.setattr(converse, "INBOX_FILE", tmp_path / "inbox.jsonl")
    monkeypatch.setattr(converse, "CAPTURE_FILE", tmp_path / "capturing")
    monkeypatch.setattr(converse, "WATCH_FILE", tmp_path / "watch.json")
    monkeypatch.setattr(converse, "HALT_FILE", tmp_path / "halt.json")
    from jarvis_voice import daemon as _daemon
    monkeypatch.setattr(_daemon, "request", lambda *a, **k: None)  # never touch the real daemon
    monkeypatch.delenv("JARVIS_VOICE_CHILD", raising=False)
    spoken = []
    monkeypatch.setattr(converse, "turn", lambda cfg, text, **kw: spoken.append(text) or "what time is it")
    monkeypatch.setattr(hooks, "_read_input", lambda: {"session_id": "s1", "last_assistant_message": "Done, sir."})
    converse.start()
    hooks.on_stop()
    out = json.loads(capsys.readouterr().out)
    assert spoken == ["Done, sir."]
    assert out["decision"] == "block" and "what time is it" in out["reason"]

    # Something said over a progress update is delivered after the next tool...
    converse.push("use the other file")
    hooks.on_tool_done()
    ctx = json.loads(capsys.readouterr().out)["hookSpecificOutput"]["additionalContext"]
    assert "use the other file" in ctx and converse.pop_all() == []
    # ... or at the stop, without waiting to listen.
    converse.push("also add a test")
    hooks.on_stop()
    assert "also add a test" in json.loads(capsys.readouterr().out)["reason"]
    assert spoken == ["Done, sir."]

    monkeypatch.setattr(converse, "turn", lambda cfg, text, **kw: None)  # silence ends it
    hooks.on_stop()
    assert capsys.readouterr().out == "" and not converse.is_on()


# ----------------------------------------------------------------------------- spoken text
import pytest  # noqa: E402


@pytest.mark.parametrize("raw, spoken", [
    ("I edited `speak_detached` in src/jarvis_voice/hooks.py:42.", "I edited speak detached in hooks dot py, line 42."),
    ("Released v0.3.0 at ~60 ms, 400 MB, 10s.",
     "Released version 0 point 3 point 0 at about 60 milliseconds, 400 megabytes, 10 seconds."),
    ("Run /jarvis:talk, e.g. now → done.", "Run jarvis talk, for example now to done."),
    ("Fixed parseConfig 🎉 & config.json", "Fixed parse Config and config dot json"),
    ("Use read/write, a 50/50 split, in the 1990s; version 3.5 is out.",
     "Use read/write, a 50/50 split, in the 1990s; version 3.5 is out."),
    ("See ~/.cache/jarvis-voice/models", "See models"),
    ("1 s later, took 0.78s.", "1 second later, took 0.78 seconds."),
    ("On macOS and iPhone via GitHub ✅", "On macOS and iPhone via GitHub"),
    ("1. First\n2) Second", "First Second"),
])
def test_clean_for_speech_reads_code_naturally(raw, spoken):
    assert clean_for_speech(raw) == spoken


def test_hooks_do_not_import_numpy():
    import subprocess
    import sys
    code = "import sys, jarvis_voice.hooks; print('numpy' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True).stdout.strip()
    assert out == "False"


# ----------------------------------------------------------------------------- speech daemon
import time as _time  # noqa: E402

from jarvis_voice import daemon  # noqa: E402


class FakeSpeaker:
    """Records what it was asked to say; 'plays' until released or stopped."""

    def __init__(self):
        self.said, self.release = [], threading.Event()
        self.busy = threading.Event()

    def say(self, text):
        self.said.append(text)
        self.busy.set()

    def wait(self):
        while self.busy.is_set() and not self.release.wait(0.01):
            pass
        self.busy.clear()

    def stop(self):
        self.busy.clear()

    def trim(self):
        self.trimmed = True  # keeps "playing" the current sentence until released


def _daemon_with(fake):
    d = daemon.SpeechDaemon(idle_s=60)
    d._ensure_speaker = lambda idle_for: setattr(d, "speaker", fake)
    threading.Thread(target=d.worker, daemon=True).start()
    return d


def _until(cond, limit=2.0):
    end = _time.time() + limit
    while not cond() and _time.time() < end:
        _time.sleep(0.01)
    return cond()


def test_daemon_same_session_replaces_other_session_waits():
    fake = FakeSpeaker()
    d = _daemon_with(fake)
    d.say("Reply from A.", "A", None)
    assert _until(lambda: fake.said == ["Reply from A."])
    d.say("Reply from B.", "B", None)       # another session: waits its turn
    _time.sleep(0.05)
    assert fake.said == ["Reply from A."] and d.status()["queued"] == 1
    d.say("Newer reply from A.", "A", None)  # same session: A finishes its sentence, newer joins the line
    _time.sleep(0.05)
    assert getattr(fake, "trimmed", False) and d.status()["session"] == "A"
    fake.release.set()
    assert _until(lambda: fake.said == ["Reply from A.", "Reply from B.", "Newer reply from A."])
    assert _until(lambda: not d.status()["busy"])
    d.stop()


def test_daemon_hush_is_per_session_and_stale_updates_drop():
    fake = FakeSpeaker()
    d = _daemon_with(fake)
    d.say("A talks.", "A", None)
    assert _until(lambda: fake.said == ["A talks."])
    d.say("B waits.", "B", None)
    d.say("Old progress.", "C", 0.01)  # ttl: dropped if it can't start in time
    _time.sleep(0.05)
    d.hush("B")
    assert d.status()["session"] == "A"  # hushing B leaves A talking
    d.hush("A")
    assert _until(lambda: not d.status()["busy"])
    assert fake.said == ["A talks."]
    d.stop()


def test_daemon_socket_roundtrip(monkeypatch):
    import tempfile as tf
    tmp = tf.mkdtemp(prefix="jv")  # short path: Unix socket paths are limited to ~104 chars
    from pathlib import Path
    monkeypatch.setattr(daemon, "SOCK", Path(tmp) / "s.sock")
    monkeypatch.setattr(daemon, "LOCK", Path(tmp) / "s.lock")
    monkeypatch.setattr(config, "CACHE_DIR", Path(tmp))
    monkeypatch.setattr(config, "load", lambda: {**config.DEFAULTS})
    fake = FakeSpeaker()
    monkeypatch.setattr(daemon.SpeechDaemon, "_ensure_speaker",
                        lambda self, idle_for: setattr(self, "speaker", fake))
    assert daemon.request("status") is None
    t = threading.Thread(target=daemon.serve, daemon=True)
    t.start()
    assert _until(lambda: daemon.request("status") is not None)
    assert daemon.request("say", text="Hello there.", session="A", ttl=None)["ok"]
    assert _until(lambda: fake.said == ["Hello there."])
    assert daemon.request("status")["session"] == "A"
    assert daemon.request("hush", session=None)["ok"]
    assert daemon.request("quit")["ok"]
    t.join(3)
    assert not t.is_alive() and not daemon.SOCK.exists()


def test_hush_leaves_other_sessions_listening_process(tmp_path, monkeypatch):
    killed = []
    monkeypatch.setattr(config, "PID_FILE", tmp_path / "speaking.pid")
    monkeypatch.setattr(daemon, "request", lambda *a, **k: None)
    monkeypatch.setattr(hooks.os, "killpg", lambda pid, sig: killed.append(pid))
    config.PID_FILE.write_text(json.dumps({"pid": 4242, "session": "A"}))
    hooks.hush("B")
    assert killed == [] and config.PID_FILE.exists()
    hooks.hush("A")
    assert killed == [4242] and not config.PID_FILE.exists()


def test_daemon_reports_recent_speech_and_gentle_hush():
    fake = FakeSpeaker()
    d = _daemon_with(fake)
    d.say("Running the tests.", "A", None)
    assert _until(lambda: fake.said == ["Running the tests."])
    assert d.status()["recent"] == ["Running the tests."]
    d.hush(None, gentle=True)
    assert getattr(fake, "trimmed", False)
    fake.release.set()
    assert _until(lambda: not d.status()["busy"])
    d.stop()


def test_vocabulary_prompt_and_prompt_echo():
    from jarvis_voice.ears import echoes_prompt, vocabulary_prompt
    p = vocabulary_prompt("Jarvis, GNSC , Next.js,")
    assert p == "Glossary: Jarvis, GNSC, Next.js."
    assert vocabulary_prompt("") is None
    assert echoes_prompt("Glossary: Jarvis, GNSC.", p)
    assert not echoes_prompt("Open GNSC", p)


def test_stop_task_and_echo_detection():
    from jarvis_voice.converse import is_echo, is_stop_task
    assert is_stop_task("Jarvis, cancel that.") and is_stop_task("Stop the task please")
    assert not is_stop_task("stop the server on port 3000 and restart it")
    assert is_echo("noted sir", ["Noted, sir."])
    assert is_echo("Running the tests", ["Tests written. Running the tests now."])
    assert not is_echo("also update the readme", ["Running the tests now."])


def test_halt_denies_tools_until_stop(tmp_path, monkeypatch, capsys):
    from jarvis_voice import converse
    for name in ("STATE_FILE", "HALT_FILE", "WATCH_FILE", "INBOX_FILE", "CAPTURE_FILE"):
        monkeypatch.setattr(converse, name, tmp_path / name)
    monkeypatch.setattr(converse, "ensure_watcher", lambda s: None)
    converse.STATE_FILE.write_text(json.dumps({"session": "S", "since": _time.time()}))
    converse.halt("S", "cancel that")
    monkeypatch.setattr(hooks, "_read_input", lambda: {"session_id": "S", "tool_name": "Bash", "tool_input": {}})
    hooks.on_tool()
    out = json.loads(capsys.readouterr().out)
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert converse.halted("other") is None


def test_speaker_trim_finishes_current_sentence_only(monkeypatch):
    from jarvis_voice import voice

    played, gate = [], threading.Event()

    class FakeSD:
        def play(self, audio, sr):
            played.append(float(audio[0]))
        def wait(self):
            gate.wait(2)
        def stop(self):
            gate.set()

    class Stub(voice.ProcessedSpeaker):
        def render(self, text):
            return np.full(10, float(len(played) + len(text)), dtype=np.float32)

    monkeypatch.setattr(voice, "_impulse_response", lambda: None)
    sp = Stub(fx=0.0, fallback=None)
    sp.sd = FakeSD()
    for s in ("One.", "Two.", "Three."):
        sp.say(s)
    assert _until(lambda: len(played) == 1)   # "One." is playing
    sp.trim()
    sp.say("New.")
    gate.set()                                 # let playback run freely
    assert _until(lambda: not sp.busy.is_set())
    assert len(played) == 2                    # "One." finished, then straight to "New."


def test_talk_is_owned_by_the_session_that_typed_it(tmp_path, monkeypatch):
    from jarvis_voice import converse
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(converse, "STATE_FILE", tmp_path / "converse.json")
    monkeypatch.setattr(converse, "OWNER_FILE", tmp_path / "converse-owner.json")
    # prompt hook first, then `jarvis converse`
    converse.note_owner("a")
    converse.start()
    assert not converse.claim("b")   # a busy second window can't take it
    assert converse.claim("a")
    converse.end()
    # `jarvis converse` first, then the prompt hook
    converse.start()
    converse.note_owner("a")
    assert not converse.claim("b")
    assert converse.claim("a")
    converse.end()
    assert not (tmp_path / "converse-owner.json").exists()
    assert converse.TALK_COMMAND.match("/jarvis:talk") and not converse.TALK_COMMAND.match("/jarvis:talkative")


def test_exit_and_reset_need_the_whole_phrase():
    from jarvis_voice.cli import is_exit_phrase, is_reset_phrase
    assert is_exit_phrase("Goodbye.") and is_exit_phrase("Shut down, please.")
    assert is_exit_phrase("Goodbye, Jarvis.")
    assert not is_exit_phrase("Shut down the dev server.")
    assert not is_exit_phrase("Goodbye message should be friendlier")
    assert is_reset_phrase("New conversation.") and is_reset_phrase("Let's start over.")
    assert not is_reset_phrase("Make the animation start over at the end.")


def test_brain_survives_lots_of_stderr(tmp_path, monkeypatch):
    """A child that floods stderr used to deadlock against the unread pipe."""
    import sys

    from jarvis_voice import brain
    fake = tmp_path / "claude"
    fake.write_text(
        f"#!{sys.executable}\n"
        "import json, sys\n"
        "sys.stderr.write('x' * 500_000)\n"
        "print(json.dumps({'type': 'stream_event', 'event': {'type': 'content_block_delta',"
        " 'delta': {'type': 'text_delta', 'text': 'Done, sir.'}}}))\n"
        "print(json.dumps({'type': 'result', 'session_id': 's1'}))\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    said = []

    class Spk:
        def say(self, t):
            said.append(t)

    b = brain.Brain("sir")
    b.ask("hi", Spk())
    assert said == ["Done, sir."] and b.session_id == "s1" and b.proc is None


def test_last_reply_reads_from_the_end(tmp_path):
    import json as _json
    p = tmp_path / "t.jsonl"
    user = {"type": "user", "message": {"content": "do it"}}
    tool = {"type": "user", "message": {"content": [{"type": "tool_result"}]}}
    def say(t):
        return {"type": "assistant", "message": {"content": [{"type": "text", "text": t}]}}
    filler = [say("old " + "y" * 3000) for _ in range(200)]  # ~600 KB, many chunks
    rows = [*filler, user, say("Working on it."), tool, say("All done, sir."), tool]
    p.write_text("\n".join(_json.dumps(r) for r in rows) + "\n")
    assert hooks.last_reply_from_transcript(str(p)) == "All done, sir."
    p.write_text("\n".join(_json.dumps(r) for r in [*filler, user]))  # no trailing newline
    assert hooks.last_reply_from_transcript(str(p)) == ""


def test_inbox_keeps_text_pushed_while_popping(tmp_path, monkeypatch):
    from jarvis_voice import converse
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(converse, "INBOX_FILE", tmp_path / "inbox.jsonl")
    converse.push("first")
    real_read = type(tmp_path).read_text

    def read_then_push(self, *a, **k):  # the watcher appends while we're reading
        out = real_read(self, *a, **k)
        if self.name.endswith(".taken"):
            converse.push("second")
        return out
    monkeypatch.setattr(type(tmp_path), "read_text", read_then_push)
    assert converse.pop_all() == ["first"]
    monkeypatch.undo()
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(converse, "INBOX_FILE", tmp_path / "inbox.jsonl")
    assert converse.pop_all() == ["second"]  # not lost
    assert list(tmp_path.glob(".*taken")) == []


def test_atomic_write_leaves_no_temp_files(tmp_path):
    target = tmp_path / "sub" / "state.json"
    config.write_json(target, {"a": 1})
    config.write_json(target, {"a": 2})
    assert json.loads(target.read_text()) == {"a": 2}
    assert [p.name for p in target.parent.iterdir()] == ["state.json"]


def test_config_set_validates_values():
    assert config.coerce("speed", "1.2") == 1.2
    assert config.coerce("speed", "default") == config.DEFAULTS["speed"]
    assert config.coerce("mac_voice", "none") is None
    bad = [("speed", "9"), ("speed", "fast"), ("speed", "none"), ("engine", "siri"),
           ("blend", "Daniel"), ("blend", "bm_daniel:0"), ("blend", "bm_daniel:x"), ("address", "  ")]
    for key, value in bad:
        with pytest.raises(ValueError):
            config.coerce(key, value)
    assert config.parse_blend("bm_daniel:0.6, bm_george") == [("bm_daniel", 0.6), ("bm_george", 1.0)]


def test_config_load_ignores_bad_values(tmp_path, monkeypatch, capsys):
    f = tmp_path / "config.json"
    f.write_text(json.dumps({"speed": "fast", "fx": 0.2, "engine": "siri", "address": "boss"}))
    monkeypatch.setattr(config, "CONFIG_FILE", f)
    cfg = config.load()
    assert cfg["speed"] == config.DEFAULTS["speed"] and cfg["engine"] == "kokoro"
    assert cfg["fx"] == 0.2 and cfg["address"] == "boss"
    assert "ignoring setting" in capsys.readouterr().err


def test_damaged_model_is_detected(tmp_path, monkeypatch):
    from jarvis_voice import voice
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    for name in voice.KOKORO_FILES:
        (tmp_path / name).write_bytes(b"x" * 10)
    assert voice.kokoro_ready()  # older installs have no recorded size: trusted
    first = next(iter(voice.KOKORO_FILES))
    (tmp_path / f"{first}.size").write_text("20")
    assert not voice.kokoro_ready()
    assert "damaged" in voice.model_problems()[0]
    (tmp_path / first).unlink()
    assert voice.model_problems()[0] == f"{first} missing"


def test_system_voice_and_chimes_fall_back_on_linux(monkeypatch):
    from jarvis_voice import voice
    have = {"espeak-ng", "paplay"}
    monkeypatch.setattr(voice.shutil, "which", lambda p: f"/usr/bin/{p}" if p in have else None)
    assert voice.system_voice_command("Daniel", 185) == ["espeak-ng", "-v", "en-gb", "-s", "185"]
    monkeypatch.setattr(voice.os.path, "exists", lambda p: True)
    assert voice.chime_command("Pop")[0] == "paplay"
    have.clear()
    assert voice.system_voice_command("Daniel", 185) is None and voice.chime_command("Pop") is None


def test_doctor_runs_and_reports(tmp_path, monkeypatch, capsys):
    from jarvis_voice import cli, daemon
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(daemon, "request", lambda *a, **k: None)
    code = cli.main(["doctor"])
    out = capsys.readouterr().out
    assert "voice model problem" in out and "jarvis setup" in out
    assert code == 1


# ----------------------------------------------------------------------------- speed: daemon serves turns
def test_daemon_append_queues_without_cutting_off():
    fake = FakeSpeaker()
    d = _daemon_with(fake)
    d.say("First part.", "A", None)
    assert _until(lambda: fake.said == ["First part."])
    d.say("Second part.", "A", None, append=True)
    _time.sleep(0.05)
    assert not getattr(fake, "trimmed", False)  # not replaced
    fake.release.set()
    assert _until(lambda: fake.said == ["First part.", "Second part."])
    d.stop()


class _LevelSpeaker(FakeSpeaker):
    def output_level(self, t):
        return 0.25


@pytest.fixture
def live_daemon(monkeypatch):
    import tempfile as tf
    from pathlib import Path
    tmp = tf.mkdtemp(prefix="jv")
    monkeypatch.setattr(daemon, "SOCK", Path(tmp) / "s.sock")
    monkeypatch.setattr(daemon, "LOCK", Path(tmp) / "s.lock")
    monkeypatch.setattr(config, "CACHE_DIR", Path(tmp))
    monkeypatch.setattr(config, "load", lambda: {**config.DEFAULTS})
    fake = _LevelSpeaker()
    calls = {"stt": 0, "voice": 0, "audio": None}

    def ensure_speaker(self, idle_for):
        calls["voice"] += 1
        self.speaker = fake

    def slow_stt(audio):
        calls["audio"] = audio
        _time.sleep(0.5)
        return "turn left at the lights"

    def ensure_stt(self):
        calls["stt"] += 1
        return slow_stt
    monkeypatch.setattr(daemon.SpeechDaemon, "_ensure_speaker", ensure_speaker)
    monkeypatch.setattr(daemon.SpeechDaemon, "_ensure_stt", ensure_stt)
    t = threading.Thread(target=daemon.serve, daemon=True)
    t.start()
    assert _until(lambda: daemon.running())
    yield fake, calls
    daemon.request("quit")
    t.join(3)


def test_daemon_transcribes_without_blocking_other_requests(live_daemon):
    fake, calls = live_daemon
    audio = np.linspace(-0.5, 0.5, 16000, dtype=np.float32)
    result = {}
    worker = threading.Thread(target=lambda: result.setdefault("text", daemon.transcribe(audio)))
    worker.start()
    _time.sleep(0.1)  # transcription under way...
    t0 = _time.monotonic()
    assert daemon.request("status") is not None
    assert _time.monotonic() - t0 < 0.2  # ...and a status/hush still answers at once
    worker.join(3)
    assert result["text"] == "turn left at the lights"
    assert np.array_equal(calls["audio"], audio)


def test_daemon_reports_playback_level_and_warms_up(live_daemon):
    fake, calls = live_daemon
    assert daemon.level(_time.monotonic()) == 0.0  # silent
    daemon.speak("Hello there.", session="A")
    assert _until(lambda: fake.said == ["Hello there."])
    assert daemon.level(_time.monotonic()) == 0.25
    fake.release.set()
    daemon.warm(stt=True)
    assert _until(lambda: calls["stt"] == 1)


def test_daemon_speaker_waits_and_stops(live_daemon):
    fake, _ = live_daemon
    sp = daemon.DaemonSpeaker("A")
    sp.say("One. Two.")
    assert sp.busy.is_set()
    assert _until(lambda: fake.said == ["One. ", "Two."])  # the daemon split it into sentences
    fake.release.set()
    done = threading.Thread(target=sp.wait)
    done.start()
    done.join(3)
    assert not done.is_alive() and not sp.busy.is_set()
    sp.close()


def test_fast_transcriber_falls_back_to_local_whisper(monkeypatch):
    from jarvis_voice import ears
    monkeypatch.setattr(daemon, "warm", lambda stt=False: None)
    monkeypatch.setattr(daemon, "transcribe", lambda audio, timeout=90.0: None)
    monkeypatch.setattr(daemon, "running", lambda: False)
    loads = []
    monkeypatch.setattr(ears, "make_transcriber", lambda cfg: loads.append(1) or (lambda a: "local text"))
    ft = ears.make_fast_transcriber({**config.DEFAULTS})
    ft.t0 -= 60  # long past start-up: don't wait for a daemon that isn't coming
    assert ft(np.zeros(10, np.float32)) == "local text"
    assert ft(np.zeros(10, np.float32)) == "local text" and loads == [1]  # loaded once


def test_noise_level_is_remembered_between_turns(tmp_path, monkeypatch):
    import queue as _queue

    from jarvis_voice import ears
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    assert ears._noise_hint() is None
    ears._save_noise(0.004)
    assert ears._noise_hint() == 0.004
    lst = ears.Listener.__new__(ears.Listener)
    lst.q, lst.enabled, lst.sensitivity = _queue.Queue(), threading.Event(), 3.0
    for f in _frames(0.002, 5):
        lst.q.put((0.0, f))
    t0 = _time.monotonic()
    threshold = lst._calibrate(1.0, hint=0.004)
    assert _time.monotonic() - t0 < 0.4  # short check instead of ~1.3 s
    assert lst.noise == 0.004 and abs(threshold - 0.012) < 1e-9
    config.write_json(tmp_path / "noise.json", {"noise": 0.004, "at": _time.time() - 3600})
    assert ears._noise_hint() is None  # stale: measure properly


def test_spoken_follow_up_is_a_new_exchange(tmp_path):
    """/jarvis:talk follow-ups arrive as a hidden 'Stop hook feedback' entry. The reply
    before it was already read out, so it must not be found again (it was spoken twice)."""
    rows = [
        {"type": "user", "message": {"content": "/jarvis:talk"}},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "Good evening, sir."}]}},
        {"type": "user", "isMeta": True,
         "message": {"content": "Stop hook feedback:\nThe user replied aloud ... \"any lag?\""}},
        {"type": "attachment", "attachment": {"type": "hook_blocking_error"}},
        {"type": "system", "subtype": "stop_hook_summary"},
    ]
    p = tmp_path / "t.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    assert hooks.last_reply_from_transcript(str(p)) == ""
    # other meta entries (skill text, reminders) still don't count as prompts
    rows[2] = {"type": "user", "isMeta": True, "message": {"content": "<command-message>x</command-message>"}}
    p.write_text("\n".join(json.dumps(r) for r in rows))
    assert hooks.last_reply_from_transcript(str(p)) == "Good evening, sir."


def test_narration_never_repeats_the_reply_the_stop_hook_read(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(hooks, "NARRATE_FILE", tmp_path / "narrate.json")
    monkeypatch.delenv("JARVIS_VOICE_CHILD", raising=False)
    monkeypatch.setattr(config, "load", lambda: {**config.DEFAULTS})
    from jarvis_voice import converse
    monkeypatch.setattr(converse, "claim", lambda s: False)
    monkeypatch.setattr(converse, "capturing", lambda: False)
    monkeypatch.setattr(hooks, "_speaking", lambda: True)  # no tool description either
    spoken = []
    monkeypatch.setattr(hooks, "speak_detached", lambda text, **kw: spoken.append(text))
    p = tmp_path / "t.jsonl"
    reply = {"type": "assistant", "message": {"content": [{"type": "text", "text": "All done,  sir."}]}}
    p.write_text(json.dumps({"type": "user", "message": {"content": "go"}}) + "\n" + json.dumps(reply))
    monkeypatch.setattr(hooks, "_read_input", lambda: {"session_id": "s1", "transcript_path": str(p),
                                                       "last_assistant_message": "All done, sir."})
    hooks.on_stop()
    assert spoken == ["All done, sir."]
    hooks.on_tool()  # a tool call that still finds that reply in the transcript
    assert spoken == ["All done, sir."]  # not again

import json
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
    assert hooks.spoken_excerpt(reply, 600) == "Done, sir. I fixed the bug."
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
    monkeypatch.delenv("JARVIS_VOICE_CHILD", raising=False)
    spoken = []
    monkeypatch.setattr(converse, "turn", lambda cfg, text: spoken.append(text) or "what time is it")
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

    monkeypatch.setattr(converse, "turn", lambda cfg, text: None)  # silence ends it
    hooks.on_stop()
    assert capsys.readouterr().out == "" and not converse.is_on()

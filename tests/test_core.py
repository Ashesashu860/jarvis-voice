import json

from jarvis_voice import config, hooks
from jarvis_voice.cli import extract_wake
from jarvis_voice.voice import clean_for_speech, split_sentences


def test_split_sentences_keeps_decimals_and_titles():
    s = split_sentences("Certainly, sir. Version 3.5 is out now! Dr. Smith agrees; fine\nNext line")
    assert s == ["Certainly, sir. ", "Version 3.5 is out now! ", "Dr. Smith agrees; ", "fine\n", "Next line"]


def test_clean_for_speech_strips_markdown():
    assert clean_for_speech("**Bold** see [docs](http://x.com) and `code`\n- item") == "Bold see docs and code item"


def test_wake_word():
    assert extract_wake("Hey Jarvis, what's the time?") == ("what's the time?", True)
    assert extract_wake("Jarvis.") == (None, True)
    assert extract_wake("what's the time?") == (None, False)


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

"""Talking to Claude Code headlessly (`claude -p`, streaming) and the Jarvis persona."""
from __future__ import annotations

import json
import subprocess

from .voice import SENTENCE_END

C_DIM, C_CYAN, C_YEL, C_RST = "\033[2m", "\033[36m", "\033[33m", "\033[0m"

STYLE = (
    "Personality: calm, precise, unfailingly polite, with a dry, understated British wit. Speak like a "
    "seasoned butler who happens to be a supercomputer: composed and economical, never gushing. "
    "Favour phrasing such as 'Certainly.', 'Right away.', 'Shall I...?', 'I've taken the liberty of...', "
    "'If I may...', 'I'm afraid...', 'Indeed.' Never open with filler like 'Great question' or "
    "'Sure thing'. Deliver bad news calmly, and allow yourself an occasional deadpan remark."
)


def persona(address: str) -> str:
    """System prompt for the standalone voice loop, where every word is spoken."""
    return (
        "You are JARVIS, a voice assistant running on the user's computer, powered by Claude. "
        "Everything you write is converted to speech, so: keep replies short and conversational "
        "(one to three sentences unless asked for detail), never use markdown, bullet points, "
        "headings, tables, emoji or code blocks in your reply, and write numbers, symbols and "
        "abbreviations the way they should be spoken. If the user needs code or a long document, "
        "write it to a file with your tools and briefly say where it is. "
        f"{STYLE} Address the user as '{address}'. "
        "The user's words come from speech recognition and may contain transcription errors; "
        "infer the intended meaning, and ask a short clarifying question only if truly ambiguous."
    )


def claude_code_note(address: str) -> str:
    """Context added to each Claude Code prompt while voice mode is on."""
    return (
        "[Jarvis voice mode is on: the opening of your final reply will be read aloud.] "
        "Begin your final reply with one or two short, plain sentences that answer or summarise, "
        "written to be spoken (no markdown, symbols or code in them), then give any detail as usual. "
        f"For that spoken part, adopt the JARVIS manner. {STYLE} Address the user as '{address}'."
    )


class Brain:
    def __init__(self, address: str, model: str | None = None, cwd: str | None = None,
                 tools: str = "", permission_mode: str | None = None):
        self.address, self.model, self.cwd = address, model, cwd
        self.tools, self.permission_mode = tools, permission_mode
        self.session_id: str | None = None
        self.proc: subprocess.Popen | None = None

    def ask(self, prompt: str, speaker) -> None:
        cmd = ["claude", "-p", prompt, "--output-format", "stream-json", "--verbose",
               "--include-partial-messages", "--append-system-prompt", persona(self.address)]
        if self.model:
            cmd += ["--model", self.model]
        if self.session_id:
            cmd += ["--resume", self.session_id]
        if self.permission_mode:
            cmd += ["--permission-mode", self.permission_mode]
        if self.tools:
            cmd += ["--allowedTools", *self.tools.split(",")]

        # JARVIS_VOICE_CHILD stops the Claude Code plugin hooks from also speaking this session.
        import os
        env = {**os.environ, "JARVIS_VOICE_CHILD": "1"}
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     text=True, cwd=self.cwd, bufsize=1, env=env)
        buf, said_wait, full = "", False, []
        print(f"{C_CYAN}JARVIS:{C_RST} ", end="", flush=True)
        for line in self.proc.stdout:
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = ev.get("type")
            if t == "system" and ev.get("subtype") == "init":
                self.session_id = ev.get("session_id") or self.session_id
            elif t == "stream_event":
                if ev.get("parent_tool_use_id"):
                    continue  # subagent chatter: don't speak it
                e = ev.get("event", {})
                if e.get("type") == "content_block_delta" and e["delta"].get("type") == "text_delta":
                    chunk = e["delta"]["text"]
                    print(chunk, end="", flush=True)
                    full.append(chunk)
                    buf += chunk
                    while (m := SENTENCE_END.search(buf)):
                        speaker.say(buf[: m.end()])
                        buf = buf[m.end():]
                elif e.get("type") == "content_block_start" and e["content_block"].get("type") == "tool_use":
                    if buf.strip():
                        speaker.say(buf)
                        buf = ""
                    print(f"\n{C_DIM}  [{e['content_block'].get('name')}]{C_RST} ", end="", flush=True)
                    if not said_wait:
                        speaker.say("One moment.")
                        said_wait = True
            elif t == "result":
                self.session_id = ev.get("session_id") or self.session_id
                if ev.get("is_error") and not full:
                    speaker.say("I'm afraid something went wrong on my end.")
                    print(f"{C_YEL}{ev.get('result')}{C_RST}")
        if buf.strip():
            speaker.say(buf)
        self.proc.wait()
        if self.proc.returncode and not full:
            err = self.proc.stderr.read().strip()
            print(f"{C_YEL}claude exited {self.proc.returncode}: {err[:400]}{C_RST}")
            speaker.say("I'm afraid I couldn't reach my brain just now.")
        self.proc = None
        print()

    def cancel(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()

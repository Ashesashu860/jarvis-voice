#!/usr/bin/env bash
# jarvis-voice installer:  curl -fsSL https://raw.githubusercontent.com/Ashesashu860/jarvis-voice/main/install.sh | bash
set -euo pipefail
REPO="git+https://github.com/Ashesashu860/jarvis-voice"
SPEC="jarvis-voice[online] @ ${REPO}"

say() { printf '\033[32m==>\033[0m %s\n' "$*"; }
die() { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }

[ "$(uname)" = "Darwin" ] || printf '\033[33mnote:\033[0m built and tested on macOS; other systems are untested.\n'

if command -v uv >/dev/null 2>&1; then
  say "Installing with uv"
  uv tool install --force --python 3.12 "$SPEC"
elif command -v pipx >/dev/null 2>&1; then
  say "Installing with pipx"
  pipx install --force "$SPEC"
else
  say "Neither uv nor pipx found; installing uv first (https://docs.astral.sh/uv/)"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  uv tool install --force --python 3.12 "$SPEC"
fi

command -v jarvis >/dev/null 2>&1 || export PATH="$HOME/.local/bin:$PATH"
command -v jarvis >/dev/null 2>&1 || die "'jarvis' isn't on your PATH. Add ~/.local/bin to PATH and run: jarvis setup"

say "Downloading models and checking your microphone"
# Interactive setup needs a real terminal; under `curl | bash` stdin is the script, so use /dev/tty if it opens.
if [ -t 0 ]; then jarvis setup
elif (exec </dev/tty) 2>/dev/null; then jarvis setup </dev/tty
else jarvis setup --yes; fi

cat <<'MSG'

Next, enable it inside Claude Code (run these at the Claude Code prompt):
  /plugin marketplace add Ashesashu860/jarvis-voice
  /plugin install jarvis@jarvis-voice

Then: /jarvis:listen to speak a prompt.  Or run `jarvis talk` for hands-free conversation.
MSG

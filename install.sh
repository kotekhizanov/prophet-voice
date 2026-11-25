#!/usr/bin/env bash
# Install prophet-voice on this Mac: the `prophet-voice` command with its audio
# packages (uv tool, editable: a `git pull` here updates it), then its models.
#
#   ./install.sh
#
# Then `prophet-voice run`, and add the voice in Omniscient (Settings → Plugins).
# Needs uv (https://docs.astral.sh/uv/).
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
command -v uv >/dev/null || { echo "uv is needed: brew install uv" >&2; exit 1; }

uv tool install --force --python 3.12 --editable "${here}[audio]"
# Where uv puts its tools may not be on PATH yet.
uv tool update-shell >/dev/null 2>&1 || true
bin="$(uv tool dir --bin)"

"${bin}/prophet-voice" install
echo
echo "Next: prophet-voice run   (then in Omniscient: Settings → Plugins → Add plugin,"
echo "                           the folder src/prophet_voice of this repository on GitHub;"
echo "                           set the voice on its page)"

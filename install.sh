#!/usr/bin/env bash
# Install prophet-voice on this Mac: the `prophet-voice` command with its audio
# packages (uv tool, editable: a `git pull` here updates it), then its models.
#
#   ./install.sh
#
# Then `prophet-voice settings` (the core's address and a client key) and
# `prophet-voice run`. Needs uv (https://docs.astral.sh/uv/).
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
command -v uv >/dev/null || { echo "uv is needed: brew install uv" >&2; exit 1; }

uv tool install --force --python 3.12 --editable "${here}[audio]"
# Where uv puts its tools may not be on PATH yet.
uv tool update-shell >/dev/null 2>&1 || true
bin="$(uv tool dir --bin)"

"${bin}/prophet-voice" install
echo
echo "Next: prophet-voice settings   (open the page, set the core and the key)"
echo "      prophet-voice run        (the first run asks to use the microphone)"

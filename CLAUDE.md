# CLAUDE.md

Instructions for Claude Code sessions working on prophet-voice.

prophet-voice is a client of a Prophet core (github.com/kotekhizanov/prophet_v2):
it hears the wake word, transcribes the phrase after it and talks to the core
over `/api/v1` with a client key. It runs on the owner's Mac.

## Rules

1. **A client, nothing more.** No imports from Prophet and no changes to
   Prophet for this repository's sake: what the core offers clients is in
   its README, "Clients: plugins and other models". The voice does only what
   a key with the `chat` scope lets it do.
2. **Nothing leaves the machine before the wake word.** Audio stays in the
   ears until Vosk hears it; only the phrase after it goes to Whisper (and
   to Groq only when the owner chose it).
3. **No credentials in the repo or in output.** Do not read the owner's
   `~/.config/prophet-voice/settings.json`: it keeps the client key and the
   Groq key. The settings page never sends a key back.
4. **Tests use their own settings folder** (`tests/conftest.py` sets
   `PROPHET_VOICE_HOME`) and fakes for Vosk, Whisper, Silero, the speakers
   and the core. Never point a test at a real core.
5. **Commit, push or open a PR only when asked**, in small commits that each
   do one reviewable thing.

## Commands

```
uv sync                       # the package without its audio extra, and the dev tools
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run --extra audio prophet-voice run   # for real, with a key from your own core
```

## Layout

| What | Where |
|---|---|
| The command (`install`, `settings`, `check`, `say`, `run`) | `src/prophet_voice/cli.py` |
| Settings (one JSON file, checked) | `src/prophet_voice/config.py` |
| Models: Vosk, Silero, Whisper | `src/prophet_voice/install.py`, `install.sh` |
| The core: /api/v1, events (SSE) | `src/prophet_voice/core.py` |
| Wake word, phrases, stop words, the microphone | `src/prophet_voice/ears.py` |
| Speech to text: faster-whisper, Groq | `src/prophet_voice/hearing.py` |
| Reading aloud: Silero, the speakers | `src/prophet_voice/mouth.py` |
| The conversation | `src/prophet_voice/voice.py` |
| The words it hears and says | `src/prophet_voice/words.py` |
| The settings page | `src/prophet_voice/settings_page.py`, `settings.html` |

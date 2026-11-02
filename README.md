# prophet-voice

Talk to [Prophet](https://github.com/kotekhizanov/prophet_v2) aloud. Say
«Пророк» and your question; Prophet answers in its voice.

prophet-voice is a client of a Prophet core, like any plugin: it lives in its
own repository, gets a key from Omniscient's Settings → Clients and talks to
the core only over `/api/v1`. The core knows nothing about it, and it can do
only what the key's scope lets a client do: chat with Prophet and answer its
questions. It cannot approve an agent's calls or change any setting.

## How it works

Everything runs on this machine (a Mac with a microphone and speakers):

- **Vosk** hears the wake word with a grammar of that one word. Until it is
  heard, no audio leaves the ears. Its confidence must reach the threshold
  in the settings.
- The phrase from the wake word to a pause (1.4 s of quiet) goes to
  **Whisper**: faster-whisper here, or Groq's hosted Whisper first when the
  settings say so. The text must begin with the wake word (Vosk may hear it
  inside other speech); the rest goes to Prophet's chat as
  `POST /api/v1/chat`, on the channel `client:<key's name>`. «Понял» says
  it was sent.
- The voice follows the run's events (`GET /api/v1/events?run=`). The answer
  is read aloud by **Silero** (v5_ru). When Prophet asks you something, the
  question is read and your next phrase, without the wake word, is the
  answer.
- While Prophet speaks it does not hear the wake word (it would be its own).
  A stop word («хватит», «понятно», «понял») stops it, and what you say at
  once is the next question. Headphones keep Prophet's voice out of the
  microphone.
- A cancel word («Пророк, отмена») sends nothing and cancels the answer
  being prepared.

Tasks Prophet starts from the chat are not read aloud: their reports are in
Omniscient.

## Install

Needs [uv](https://docs.astral.sh/uv/) (`brew install uv`).

```
git clone https://github.com/kotekhizanov/prophet-voice.git
cd prophet-voice
./install.sh
```

`install.sh` installs the `prophet-voice` command with its audio packages
(PyTorch, faster-whisper, Vosk, sounddevice) as a uv tool, editable, so a
`git pull` here updates it. Then it runs `prophet-voice install`, which
downloads the models into `~/.config/prophet-voice/models`, each once:

| Model | Size | For |
|---|---|---|
| Vosk `vosk-model-small-ru-0.22` (checked by SHA-256) | 45 MB | the wake word, the stop words |
| Silero `v5_ru` | 140 MB | Prophet's voice |
| faster-whisper `large-v3-turbo` (or the one in the settings) | 1.6 GB | speech to text |

Run `prophet-voice install` again after choosing another Whisper model. With
speech going to Groq, no Whisper model is downloaded (and Groq then has no
fallback).

## Settings

```
prophet-voice settings
```

opens a page at http://127.0.0.1:8771/. Set there:

1. **Core's address**: the core as Omniscient opens it, e.g.
   `http://192.168.1.10:8770` (on the LAN or over ZeroTier).
2. **Client key**: in Omniscient, Settings → Clients → New key, named e.g.
   `voice`, with the `chat` scope. The key (`pk_…`) is shown once: paste it,
   save, then **Check**.
3. The chat (the general one, or a project's), Prophet's voice (**Listen**
   plays it), the microphone, the wake word's threshold, Whisper, and the
   words: wake, cancel and stop words, the greeting and «Понял».

The settings are in `~/.config/prophet-voice/settings.json`
(`$PROPHET_VOICE_HOME` moves the whole folder). Only you can read the file:
it keeps the client key and the Groq key. The page never shows them again,
only their last four characters.

## Run

```
prophet-voice run
```

The first run asks macOS for the microphone (for the terminal it runs in).
`prophet-voice check` tells whether the core knows the key;
`prophet-voice say "Привет"` reads a text aloud.

To write your own client, see "Clients: plugins and other models" in
Prophet's README.

## Develop

```
uv sync
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

The tests need neither the audio packages nor a core: recognizers, Whisper,
Silero, the speakers and the core are replaced by fakes. To try it for real,
run your own core (`make dev` in Prophet), make a key there and run
`uv run --extra audio prophet-voice run`.

# prophet-voice

Talk to [Prophet](https://github.com/kotekhizanov/prophet_v2) aloud. Begin a
Private dialog in Omniscient's chat and speak; Prophet answers in its voice.

prophet-voice is a Prophet plugin: it lives in its own repository, is added
in Omniscient's Settings → Plugins, gets its client key from there and talks
to the core only over `/api/v1`. The core knows only its manifest, and it can
do only what the key's scope lets a client do: chat with Prophet and answer
its questions. It cannot approve an agent's calls or change any setting. In
Omniscient it adds the dialog (below) and its settings, but for the
secrets and this Mac's own (below).

## How it works

Everything runs on this machine (a Mac with a microphone and speakers):

- The microphone is open only in a Private dialog (below): it opens when
  you begin one in Omniscient's chat and closes when you end it. Its audio
  goes through a noise suppressor first: the room's steady noise (a fan,
  the street's hum) is lowered by about 12 dB, band by band, and a voice
  passes.
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
  is read aloud by **Silero** (v5_ru), in the voice of baya. When Prophet asks you something, the
  question is read and your next phrase, without the wake word, is the
  answer.
- While Prophet speaks the voice does not listen: the microphone would hear
  Prophet, not you (its own «Пророк» would wake it). Only a phrase you have
  begun goes on under the tone or the cue.
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
| Vosk `vosk-model-small-ru-0.22` (checked by SHA-256) | 45 MB | the wake word |
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
   `http://192.168.1.10:8770` (on the LAN or over ZeroTier). A node that
   runs the voice (below) gives it its own core's address; one set here
   wins.
2. **Client key**: with the settings page open, open Omniscient in a
   browser on this Mac, Settings → Plugins → Add plugin, address
   `http://127.0.0.1:8771`. Omniscient reads the voice's manifest
   (`/prophet-plugin.json`: the name `voice`, the `chat` scope), and after
   you add it hands the new key to the voice (`/prophet/connect`). Reload the
   settings page and press **Check**. The voice takes a key only from
   Omniscient at the core's address saved in step 1 (it is the page's
   origin, so open Omniscient by that same address).

   If a client key named `voice` exists already (made by hand in Settings →
   Clients), revoke it first: the plugin's key has the plugin's name. A key
   made by hand still works too: paste it into the field.
3. The chat (the general one, or a project's), **Listen** to hear Prophet's
   voice, the microphone, the wake word's threshold, Whisper, and the
   words: wake and cancel words, the greeting and «Понял».

Omniscient looks at the plugin's address each time its Settings open; with
the settings page closed it finds nothing and keeps the version it has.
A voice switched off in Omniscient does not start, and a node that runs it
stops it; removing it revokes the key, and the voice waits for a new one.

Some settings can be set in Omniscient too: on the voice's own page in the
menu (Plugins → Voice), or with its **Settings** button in Settings →
Plugins. Both write the same values: speech to text, the chat, the wake
word's threshold and Whisper's model. The form opens with the defaults
filled in, and once the voice is a plugin what is set there wins over this
page. Speech to text changes at the next dialog; the rest when the voice
starts again. The words, the core's address, the client key, the Groq key
and the microphone are only here: the secrets never go to the core, and the
rest belongs to this Mac.

The settings are in `~/.config/prophet-voice/settings.json`
(`$PROPHET_VOICE_HOME` moves the whole folder). Only you can read the file:
it keeps the client key and the Groq key. The page never shows them again,
only their last four characters.

## Run

```
prophet-voice run
```

The first Private dialog asks macOS for the microphone (for the terminal
the voice runs in). Only one voice runs on this machine: a second `run`
refuses to start, or both would hear and answer the same phrase. With no
client key yet, or one the core no longer knows, `run` serves the plugin and
waits: add the voice in Omniscient (above) and it goes on with the new key.
`prophet-voice check` tells whether the core knows the key;
`prophet-voice say "Привет"` reads a text aloud.

`run` serves the settings page itself (and the plugin, for Omniscient), so
`prophet-voice settings` then only opens it.

### On a node

A Prophet node on this Mac can run the voice while it is connected to the
core, so it needs no terminal of its own. In the node's config
(`~/.config/prophet-node/config.json`), under `plugins`, by the plugin's id,
then restart the node:

```json
"plugins": {
  "voice": {"command": "/Users/<you>/.local/bin/prophet-voice", "args": ["run"],
            "env": {"GROQ_API_KEY": "…"}}
}
```

The command is the one `./install.sh` installed (`which prophet-voice`). The
node starts the voice when it connects to the core and stops it when you
switch the voice off in Omniscient or 30 seconds after the core is gone; a
voice that exits is started again. It gives the voice the core's address
(`PROPHET_URL`), used unless the settings page sets one. `GROQ_API_KEY` is
needed only for speech through Groq, and only when the settings page has no
Groq key: either way it stays on this Mac. What the voice says goes to the
node's log. macOS asks for the microphone for the program that runs the
node (its terminal, say): allow it in System Settings → Privacy & Security
→ Microphone if the dialog never opens it.

To write your own client, see "Clients: plugins and other models" in
Prophet's README.

## The dialog in Omniscient

The message field has a **Private** button (a microphone) while
`prophet-voice run` runs (Omniscient opened on this Mac). It opens the
microphone and the voice's page in the menu (Plugins → Voice): a cloud of
points that breathes while waiting, turns faster while Prophet thinks, and
widens with your voice and with Prophet's, and beside it what is said in the
chat from the dialog's start. The caption under the cloud says what the
voice does; while Whisper turns your phrase into text it reads «Processing
voice…» beside a turning circle. The microphone and the speakers are this
Mac's: the page only shows the dialog and tells the voice when it begins and
ends. The script uses only Omniscient's plugin API: it does not change
Omniscient's own page.

There is no wake word in it. Any speech begins a question; Prophet says
«Ага?» softly once Vosk, with its whole vocabulary, hears a word in it (a
passing bus or a slammed door has none and is dropped here), and «Окей…»
when the pause ends it, with no «Понял». A television or another voice in
the room still starts a question.

The question goes to the chat open in Omniscient, and Prophet never talks
over you: an answer that comes while you speak waits for the end of your
phrase. A button again (by the message
field or on the voice's page), or Esc on that page, ends the dialog at once:
the microphone closes, and
the voice stops what it says, drops the phrase it was hearing and reads no
answer that comes after (the answer is still in the chat). A dialog belongs
to the page that began it: closing or reloading that page ends it, so the
microphone is never open with no page to show it. The cue words are in the
settings (Listening, Heard).

Chrome may ask once whether Omniscient may reach devices on the local
network: that is the page talking to the voice at `127.0.0.1`. After the
plugin's update Omniscient shows its new version (the script) in Settings →
Plugins: approve it and reload.

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

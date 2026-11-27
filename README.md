# prophet-voice

Talk to [Prophet](https://github.com/kotekhizanov/prophet_v2) aloud. Begin a
Private dialog in Omniscient's chat and speak; Prophet answers in its voice.
Or dictate into any text field of Omniscient with the microphone button by it.

prophet-voice is a Prophet plugin: it lives in its own repository, is added
in Omniscient's Settings → Plugins, gets its client key from there and talks
to the core only over `/api/v1`. The core knows only its manifest, and it can
do only what the key's scope lets a client do: chat with Prophet and answer
its questions. It cannot approve an agent's calls or change any setting. In
Omniscient it adds the dialog (below) and its settings, but for the
secrets and this Mac's own (below).

## How it works

Everything runs on this machine (a Mac with a microphone and speakers):

- The microphone is open only in a Private dialog or a dictation (both
  below): it opens when you begin one in Omniscient and closes when you end
  it. Its audio
  goes through a noise suppressor first: the room's steady noise (a fan,
  the street's hum) is lowered by about 12 dB, band by band, and a voice
  passes.
- The voice does not hear itself: what it plays through the speakers is
  taken out of the microphone's audio before that, band by band (the delay
  to the microphone and how much comes back are learnt as it speaks). Your
  voice over Prophet's passes.
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
- In a Private dialog you may speak over Prophet: the phrase is heard while
  it reads (what the microphone has of Prophet's own voice is told from
  yours), and goes to the chat once Prophet has finished or you stopped it.
  A phrase that is mostly what Prophet was reading then is its own voice
  heard back, and is not sent.
- A Private phrase is heard as you say it, not once you have ended. The
  dialog's page shows it over the cloud, above its caption: Vosk's words at once, rough, and
  Whisper's in their place part by part. At each pause of half a second
  the part before it goes to Whisper while you go on, so when the phrase
  ends only its last part is left to hear, and the question is sent sooner.
- A cancel word («Пророк, отмена») sends nothing and cancels the answer
  being prepared. Said over Prophet while it reads («стоп», if you added
  it), it hushes Prophet and cancels the rest of the answer: Vosk listens
  for the cancel words alone then, and that audio goes nowhere.

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
| sherpa-onnx: pyannote's segmentation and 3D-Speaker's ERes2Net (checked by SHA-256) | 46 MB | telling voices apart |
| NVIDIA Nemotron 3 Diarization (a fixed revision, checked by SHA-256) | 380 MB | telling voices apart, the other way |

Run `prophet-voice install` again after choosing another Whisper model. With
speech going to Groq, no Whisper model is downloaded (and Groq then has no
fallback).

## Settings

All of them are in Omniscient, on the voice's page in the menu (Plugins →
Voice settings), once the voice is added there:

1. **Add the voice.** In Omniscient, Settings → Plugins → Add plugin: the
   GitHub address `https://github.com/kotekhizanov/prophet-voice/tree/main/src/prophet_voice`
   (the folder with `prophet-plugin.json`) and the node on this Mac. The
   core reads the manifest there (the name `voice`, the `chat` scope, the
   voice's address `http://127.0.0.1:8772`) and the script, and after you
   approve it Omniscient hands the new key to the voice
   (`/prophet/connect`): open Omniscient in a browser on this Mac, with
   `prophet-voice run` running, or press **New key** there once it runs.
   The voice takes a key only from Omniscient at its core's address, so
   open Omniscient by that address.

   The node clones the repository into `~/plugins/voice` and runs the
   manifest's `run` there (`uv run --extra audio prophet-voice run --install`:
   the packages, then the models that are missing, about 2 GB the first
   time, so the voice is ready some minutes later), unless
   its config names a command for `voice` (below): that one wins. The
   address is `settings_port` in the settings file; another port there
   needs the same one in the manifest.

   The core's address is `http://127.0.0.1:8770`, or the one a node that
   runs the voice gives it (below), or `PROPHET_URL`; `"core"` in the
   settings file wins over them all. If a client key named `voice` exists
   already (made by hand in Settings → Clients), revoke it first: the
   plugin's key has the plugin's name.
2. **Set the rest** on the voice's page: whether the core knows the key
   (**Check**), the chat (the general one, or a project's), **Listen** to
   hear Prophet's voice, the microphone, the wake word's threshold, speech
   to text (Whisper here or Groq first), Whisper's model, the Groq key, and
   the words: wake and cancel words, the greeting and «Понял». The voice
   takes them as the next dialog begins; after a change of Whisper's model,
   run `prophet-voice install`.

The chat, the threshold, speech to text and Whisper's model are the core's
settings (the modal in Settings → Plugins → Voice → Settings has them too).
The microphone, the Groq key and the words are this Mac's: the page sets
them through the voice's own address, and only while the voice runs. The
Groq key never goes to the core, and the voice never sends it back to the
page, only its last four characters. The core's address, the client key and
the port are not set from a page at all.

Omniscient looks at the plugin's address each time its Settings open; with
the voice not running it finds nothing and keeps the version it has.
A voice switched off in Omniscient does not start, and a node that runs it
stops it; removing it revokes the key, and the voice waits for a new one.

This Mac's settings are in `~/.config/prophet-voice/settings.json`
(`$PROPHET_VOICE_HOME` moves the whole folder). Only you can read the file:
it keeps the client key and the Groq key.

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
(`PROPHET_URL`), used unless the settings file sets one. `GROQ_API_KEY` is
needed only for speech through Groq, and only when the voice's settings have
no Groq key: either way it stays on this Mac. What the voice says goes to the
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

## Dictation into a field

While the voice runs, the text field that has the cursor in Omniscient (the
message field, a goal, a task, notes: any textarea or text input, but a
password) shows a small round microphone at its right edge. Press it and
speak: the Mac's microphone opens, and what you say is typed where the
cursor is as far as it is heard; at each pause Whisper's text of the phrase
takes its place. You may go on typing meanwhile, or move the cursor to
another field: the next phrase begins where the cursor is.

The button again or Esc ends it, and the phrase you were saying still
comes. It also ends after a minute with nothing said, when the field goes
away and when the page closes. Nothing is sent to the chat and nothing is
said aloud; a dictated phrase is not written to the voice's log. There is
no dictation during a dialog (the dialog hears you already), and a dialog
that begins ends it.

This button is the one thing the script adds outside Omniscient's plugin
API: an element of its own over the page, and the text typed into the field
as the keyboard would.

### Who said what

When more than one person talks to Prophet in a Private dialog, the voice
tells them apart (sherpa-onnx, here on the Mac). The voices are numbered as
they are first heard, and once there are two, each line of the question
says whose it is:

```
Голос 1: Какая завтра погода?
Голос 2: И напомни купить хлеб.
```

A phrase two people said in turn, with no pause between them, is split by
Whisper's word times. Beside the cloud, the voices heard in the dialog have
a field each: name one («Иван») and the voice remembers him, so in the next
dialogs he is «Иван» from his first phrase. Each phrase he is heard in
refines his print. A voice left unnamed is forgotten when the dialog ends.
The prints are kept in `~/.config/prophet-voice/voices.json` (readable by
you only); the voice's settings page (Voices) lists the named voices, to
rename or forget. The setting "Tell voices apart" switches it off.

"Who speaks when" chooses what finds where each voice speaks in a phrase:
sherpa-onnx (pyannote's segmentation, lighter) or NVIDIA's Nemotron 3
Diarization (Transformers, on the Mac's CPU: about 0.05 s a phrase, and it
hears a change of voice sherpa may miss). Both are loaded as the voice
starts, so a change counts from the next dialog; the log says which split
each phrase («nemotron: 3 parts, 2 voices»). The names and the prints are
the same for both.
People speaking at the same moment are not separated: their
words go to whoever the voice hears more of.

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

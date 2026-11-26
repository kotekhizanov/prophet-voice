"""Listening: the wake word and the phrase after it.

The microphone's audio stays here. Vosk listens with a grammar of the wake
words and "unknown"; only its final results count, as its partial ones find
the word in any speech. Once the wake word is heard with enough confidence,
the audio from it on is kept until a pause and goes to Whisper (`on_phrase`).
Vosk's result comes at the end of an utterance, so a phrase said in one
breath ("Пророк, что нового?") is already in the ring of recent audio.

While Prophet speaks, the ears do not listen (`listening`): the microphone
hears Prophet, not the owner. Only a phrase already begun goes on, as the
owner goes on speaking under a tone or a cue. While Prophet reads, Vosk
listens with a grammar of the cancel words alone: one said over Prophet
(«стоп») is `on_stop`, unless Prophet is saying that word itself. That
audio goes nowhere else.
For DEAF seconds after, Prophet's echo begins nothing. `listen_free` opens
the next phrase with no wake word: after Prophet asks the owner something.

Alone with the owner (a Private dialog, `alone`), there is no wake word:
ONSET seconds of voice begin a phrase while Prophet is silent. It is only
tentative until Vosk with its whole vocabulary hears a word in it, within
CONFIRM_WITHIN seconds: a passing bus or a slammed door has none and is
dropped here, unsaid and unsent. Once a word is heard, `on_begun` says so.

With the `echo` that takes Prophet's voice out of the audio, the owner may
speak over Prophet when alone: a phrase begins while Prophet reads, from the
blocks the echo says were not Prophet's alone (`kept` of OVER_KEPT or more).
Such a phrase (`over`) has no cue, and `on_phrase` is told of it. `so_far`
is the phrase being heard, for its text to be shown as it is said.

Whether someone speaks is the loudness above VOICE_FROM against the room's
quiet: a bus's rumble and a room's hum stay mostly below it, a voice does not.
`level` is that loudness for Omniscient's picture, 0 to 1.

The microphone is open only while it is asked to be (`listen`'s `on`: a
Private dialog). Its audio comes first through `echo` (echo.py), which takes
out Prophet's own voice as the speakers play it, and then through the
denoiser (denoise.py), which takes out the room's steady noise, before
anything hears it.

The ears run in one thread: `hear` takes the microphone's blocks in order,
and the callbacks are called from it.
"""

import io
import json
import logging
import queue
import re
import time
import wave
from collections import deque
from pathlib import Path

import numpy as np

from prophet_voice.denoise import Denoiser
from prophet_voice.words import said_in

log = logging.getLogger("prophet_voice.ears")
RATE = 16000  # Hz: what Vosk's and Whisper's models hear
BLOCK = 640  # samples: 40 ms
RING = 20  # seconds of audio kept while waiting: Vosk's utterance so far
END_SILENCE = 1.4  # seconds of quiet that end the phrase
VOICE_FROM = 250  # Hz: the loudness that counts as a voice is above it
NO_PHRASE = 6  # seconds after the wake word with nothing said: back to waiting
MAX_PHRASE = 60  # seconds
DEAF = 0.5  # seconds after Prophet speaks: its echo dies out
ONSET = 0.3  # seconds of voice that begin a phrase when alone
PRE_ROLL = 0.6  # seconds kept from before it: the first syllable starts quieter
OVER_KEPT = 0.2  # of a block's power that was not Prophet's, for the owner to be in it
CONFIRM_WITHIN = 3  # seconds for Vosk to hear a word in a sound before it is dropped
VOSK_MODEL = "vosk-model-small-ru-0.22"


def wav(blocks: list[bytes], rate: int = RATE) -> bytes:
    """A 16-bit mono WAV of these blocks."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"".join(blocks))
    return buf.getvalue()


class Ears:
    def __init__(
        self,
        words: dict,
        threshold: float,
        recognizer,
        mouth,
        on_wake,
        on_drop,
        on_phrase,
        on_begun=None,
        on_stop=None,
        echo=None,
    ):
        """`recognizer(grammar)` makes a Vosk recognizer (`VoskRecognizer`); a
        grammar of None is Vosk's whole vocabulary. `mouth` says whether
        Prophet speaks (`speaking`) and what (`saying`)."""
        self.words = words
        self.threshold = threshold
        self.recognizer = recognizer
        self.mouth = mouth
        self.on_wake, self.on_drop, self.on_phrase = on_wake, on_drop, on_phrase
        self.on_begun = on_begun or (lambda: None)
        self.on_stop = on_stop or (lambda: None)
        self.stop = None  # the cancel words' grammar, while Prophet reads
        self.echo = echo  # says how much of a block was not Prophet's (`kept`)
        self.over = False  # the phrase going on began over Prophet's voice
        self.phrase_no = 0  # phrases begun so far
        self.clock = 0.0  # seconds of audio heard, as Vosk's word times count them
        self.state = "waiting"  # or "phrase"
        self.recent: deque[tuple[float, bytes]] = deque()
        self.phrase: list[bytes] = []
        self.floor = 0.0
        self.hp = (0.0, 0.0, 0.0)
        self.waiting_since = self.woke_at = self.last_voice = self.voiced = 0.0
        self.free_for = 0.0  # in a phrase with no wake word: seconds to begin it
        self.was_speaking = False
        self.deaf_until = 0.0
        self._free = None  # listen_free's request, taken by the next block
        self.alone = False  # a Private dialog: no wake word
        self.onset = 0.0  # seconds of voice so far, when alone
        self.tentative = False  # a phrase begun alone, with no word heard in it yet
        self.check = None  # the whole vocabulary, listening for that word
        self.level = 0.0
        self.wake = recognizer([*words["wake"], "[unk]"])
        self.wake_from = 0.0  # the clock when `wake` began: its word times count from it
        self.renew = False  # a new `wake` at the next block: Prophet spoke, or the words changed

    def configure(self, words: dict, threshold: float) -> None:
        """New words and a new threshold (from another thread)."""
        self.renew = self.renew or words["wake"] != self.words["wake"]
        self.words, self.threshold = words, threshold

    def listen_free(self, seconds: float) -> None:
        """Hear the next phrase without the wake word (from another thread)."""
        self._free = seconds

    def be_alone(self, on: bool) -> None:
        """A Private dialog begins or ends (from another thread): what was heard
        before does not count."""
        self.alone, self.onset, self.waiting_since = on, 0.0, self.clock

    def reset(self) -> None:
        """The microphone has closed (from the ears' thread): what was heard is gone."""
        self._wait()
        self.recent.clear()
        self._free, self.level = None, 0.0

    @property
    def listening(self) -> bool:
        """Not while Prophet speaks, but for a phrase already begun, and when
        alone with an echo that takes Prophet's voice out."""
        if self.state == "phrase" or not self.mouth.speaking:
            return True
        return self.alone and self.echo is not None

    def so_far(self) -> tuple[int, bytes, float] | None:
        """The phrase being heard (from another thread): its number, its audio so
        far and the seconds since a voice was in it. None when there is none,
        or no word was heard in it yet."""
        if self.state != "phrase" or self.tentative:
            return None
        return self.phrase_no, wav(list(self.phrase)), self.clock - self.last_voice

    def hear(self, block: bytes) -> None:
        seconds, at = len(block) / 2 / RATE, self.clock
        self.clock += seconds
        if self.mouth.speaking and (self.state == "waiting" or self.over):
            self._stops(block)
        else:
            self.stop = None
        if not self.listening:
            self.was_speaking, self.renew, self.level = True, True, 0.0
            return
        if self.renew:  # Vosk's times would skip the blocks it was not given
            self.wake = self.recognizer([*self.words["wake"], "[unk]"])
            self.wake_from, self.renew = at, False
        voice = self._voice(np.frombuffer(block, np.int16).astype(np.float32) / 32768)
        speaking = self.mouth.speaking
        if speaking and self.echo is not None and self.echo.kept < OVER_KEPT:
            voice, self.level = False, 0.0  # what is loud in this block is Prophet
        if self.was_speaking and not speaking:  # what was heard meanwhile was Prophet's own
            self.waiting_since, self.deaf_until = self.clock, self.clock + DEAF
        self.was_speaking = speaking
        found = self.wake.accept(block)
        if self.state == "waiting":
            self.recent.append((at, block))
            while self.recent[0][0] < self.clock - RING:
                self.recent.popleft()
            if self._free is not None:
                self._open(self.clock - 0.3, self._free)
                self._free = None
                return
            if self.alone:
                self._begins(voice, seconds)
                return
            if found and self.clock >= self.deaf_until:
                at = self.wake_from
                self._heard([{**w, "start": w["start"] + at, "end": w["end"] + at} for w in found])
            return
        self.phrase.append(block)
        if voice:
            self.voiced += seconds
            self.last_voice = self.clock
        since = self.clock - self.woke_at
        if self.tentative:
            self._confirm(block)
            if self.tentative and self.clock - self.last_voice > END_SILENCE:
                self._confirm(None)  # what Vosk makes of it now that it ended
            if self.tentative:
                if since > CONFIRM_WITHIN or self.clock - self.last_voice > END_SILENCE:
                    self._wait()  # a sound with no word in it: nothing was said
                return
        spoke = self.voiced > 0.25
        if not spoke and since > (self.free_for or NO_PHRASE):
            self._drop()
        elif spoke and (self.clock - self.last_voice > END_SILENCE or since > MAX_PHRASE):
            self._send()

    # ---- a cancel word over Prophet ----------------------------------------------------

    def _stops(self, block: bytes) -> None:
        """While Prophet reads, a cancel word said over it. Vosk's partial result
        finds the word at once; its final one says how sure it is."""
        cancel, saying = self.words["cancel"], self.mouth.saying
        if not saying:  # a tone, or another mouth's sound: nothing to stop
            self.stop = None
            return
        if self.stop is None:
            self.stop = self.recognizer([*cancel, "[unk]"])
        found = self.stop.accept(block)
        if found is None and said_in(self.stop.partial(), cancel):
            found = self.stop.final()
        if not found:
            return
        sure = " ".join(w["word"] for w in found if w.get("conf", 1.0) >= self.threshold)
        if said_in(sure, cancel) and not said_in(saying, cancel):  # not Prophet's own
            self.stop = None
            if self.over:  # the phrase it was said in is not a question
                self._wait()
            self.on_stop()

    # ---- the wake word -------------------------------------------------------------

    def _heard(self, words: list[dict]) -> None:
        """An utterance Vosk has finished; words heard before the last phrase are old."""
        wake = self.words["wake"]
        i = next(
            (
                i
                for i, w in enumerate(words)
                if w["word"] in wake
                and w.get("conf", 1.0) >= self.threshold
                and w["start"] >= self.waiting_since
            ),
            None,
        )
        if i is None:
            return
        self._open(words[i]["start"] - 0.2, 0)
        self.last_voice = words[-1]["end"]
        self.voiced = 1.0 if i < len(words) - 1 else 0.0  # words after it: the phrase has begun
        self.on_wake()

    def _open(self, start: float, free_for: float) -> None:
        """A phrase from `start` on: from the ring, then each new block."""
        self.phrase = [b for at, b in self.recent if at + len(b) / 2 / RATE > start]
        self.state, self.free_for, self.over = "phrase", free_for, False
        self.phrase_no += 1
        self.woke_at = self.last_voice = start
        self.voiced = 0.0
        self.recent.clear()

    def _wait(self) -> None:
        self.state, self.phrase, self.free_for = "waiting", [], 0.0
        self.waiting_since = self.clock
        self.tentative, self.check, self.onset, self.over = False, None, 0.0, False

    def _drop(self) -> None:
        free = bool(self.free_for)
        self._wait()
        if not free:  # nothing after a question: that was all
            self.on_drop()

    def _send(self) -> None:
        audio, free, over = wav(self.phrase), bool(self.free_for), self.over
        self._wait()
        self.on_phrase(audio, free, over)

    # ---- alone: speech begins a phrase ------------------------------------------------

    def _begins(self, voice: bool, seconds: float) -> None:
        """ONSET seconds of voice, with short gaps forgiven: while Prophet is silent,
        or over it (a voice then is what the echo did not take for Prophet's)."""
        if self.clock < self.deaf_until:
            self.onset = 0.0
            return
        self.onset = self.onset + seconds if voice else max(0.0, self.onset - seconds / 2)
        if self.onset < ONSET:
            return
        voiced = self.onset
        self._open(self.clock - voiced - PRE_ROLL, CONFIRM_WITHIN)
        self.voiced, self.last_voice = voiced, self.clock
        self.over = self.mouth.speaking
        self.tentative, self.check = True, self.recognizer(None)
        for b in list(self.phrase):
            if self.tentative:
                self._confirm(b)

    def _confirm(self, block: bytes | None) -> None:
        """A tentative phrase is one once Vosk hears a word in it (None: its end)."""
        if block is None:
            text = " ".join(w["word"] for w in self.check.final())
        else:
            found = self.check.accept(block)
            text = self.check.partial() if found is None else " ".join(w["word"] for w in found)
        if not is_speech(text):
            return
        self.tentative, self.check = False, None
        if not self.over:  # a cue would wait for Prophet to end, and then be late
            self.on_begun()

    # ---- loudness ---------------------------------------------------------------------

    def _voice(self, samples: np.ndarray) -> bool:
        """Someone speaks: the loudness above VOICE_FROM is well over the room's quiet.
        Two one-pole high-passes, carried from block to block."""
        a = 1 / (1 + 2 * np.pi * VOICE_FROM / RATE)
        x1, y1, y2 = self.hp
        total = 0.0
        for v in samples.tolist():
            p1 = y1
            y1 = a * (y1 + v - x1)
            x1 = v
            y2 = a * (y2 + y1 - p1)
            total += y2 * y2
        self.hp = (x1, y1, y2)
        rms = (total / max(len(samples), 1)) ** 0.5
        self.level = min(1.0, max(0.0, (rms - 2 * self.floor) * 12))
        if not self.floor:
            self.floor = rms
        if rms < self.floor:
            self.floor = 0.9 * self.floor + 0.1 * rms
        else:
            self.floor = 0.998 * self.floor + 0.002 * rms
        return rms > max(3 * self.floor, 0.002)


def is_speech(text: str) -> bool:
    """A word, not a noise Vosk made one of: a word of three letters, or two words."""
    found = re.findall(r"[а-яё]+", (text or "").lower())
    return any(len(w) >= 3 for w in found) or len(found) >= 2


# ---- Vosk and the microphone ---------------------------------------------------------


def vosk_dir(models: Path) -> Path:
    return models / "vosk" / VOSK_MODEL


def load_vosk(models: Path):
    """Vosk's small Russian model, and a function making recognizers of it."""
    import vosk

    vosk.SetLogLevel(-1)
    path = vosk_dir(models)
    if not path.exists():
        raise RuntimeError("Vosk is not installed: run `prophet-voice install`")
    model = vosk.Model(str(path))
    return lambda grammar: VoskRecognizer(vosk, model, grammar)


class VoskRecognizer:
    def __init__(self, vosk, model, grammar: list[str] | None):
        # The words as they are: Vosk reads "п…" as a word it does not know.
        words = json.dumps(grammar, ensure_ascii=False) if grammar else None
        args = (model, RATE) if words is None else (model, RATE, words)
        self.r = vosk.KaldiRecognizer(*args)
        self.r.SetWords(True)  # each word with its confidence and times

    def accept(self, block: bytes) -> list[dict] | None:
        """The words of an utterance Vosk has finished, or None while it goes on."""
        if self.r.AcceptWaveform(block):
            return json.loads(self.r.Result()).get("result", [])
        return None

    def partial(self) -> str:
        return json.loads(self.r.PartialResult()).get("partial", "")

    def final(self) -> list[dict]:
        return json.loads(self.r.FinalResult()).get("result", [])


def microphones() -> list[str]:
    import sounddevice as sd

    return [d["name"] for d in sd.query_devices() if d["max_input_channels"] > 0]


def listen(ears: Ears, device, on, stop, failed=None, echo=None) -> None:
    """The microphone's blocks, with Prophet's own voice (`echo`, told by the mouth
    what plays) and the room's noise taken out, to the ears while
    `on` is set, until `stop` is (threading.Events). While `on` is not, the
    microphone is closed and nothing is heard; `failed(exc)` when it would not
    open (no microphone, or no permission), and `on` is cleared. `device()` is
    the microphone's name each time it opens (empty: the system's)."""
    import sounddevice as sd

    blocks: queue.Queue = queue.Queue()

    def got(data, frames, _time, status):
        if status:
            log.debug("microphone: %s", status)
        blocks.put((bytes(data), time.monotonic()))  # and when it ended

    while not stop.is_set():
        if not on.wait(0.5):
            continue
        while not blocks.empty():  # left from when it was open before
            blocks.get_nowait()
        quiet = Denoiser()
        try:
            stream = sd.RawInputStream(
                samplerate=RATE,
                blocksize=BLOCK,
                channels=1,
                dtype="int16",
                device=device() or None,
                callback=got,
            )
            stream.start()
        except Exception as exc:
            on.clear()
            log.error("the microphone would not open: %s", exc)
            if failed is not None:
                failed(exc)
            continue
        log.info("the microphone is open")
        try:
            while on.is_set() and not stop.is_set():
                try:
                    block, at = blocks.get(timeout=0.2)
                except queue.Empty:
                    continue
                try:
                    ears.hear(quiet(block if echo is None else echo(block, at)))
                except Exception:
                    log.exception("the ears failed on a block")
        finally:
            stream.close()
            ears.reset()
            log.info("the microphone is closed")

"""Speaking: Prophet's answers read aloud by Silero, on this machine.

Silero's Russian model runs on the CPU: a sentence takes a few hundredths of
a second. The model file (about 140 MB) is downloaded by `prophet-voice
install` into the models folder and loaded once.

Silero reads Cyrillic letters only: Markdown is cleared away, code is not
read, numbers become words (num2words), a Latin word is spelled in Cyrillic
as it sounds and an abbreviation letter by letter, so "SQL в ClickHouse" is
read, not skipped.

The mouth reads one text after another, a piece at a time, in its own
thread. `hush` stops it at once (a stop word). While it speaks, the ears do
not listen for the wake word, or Prophet's own «Пророк» would wake it;
`saying` is the piece being read, for the ears to know its own words.
"""

import logging
import queue
import re
import threading
import time
import warnings
from pathlib import Path

import numpy as np

log = logging.getLogger("prophet_voice.mouth")
MODEL = "v5_ru"
MODEL_URL = "https://models.silero.ai/models/tts/ru/{}.pt"
RATE = 24000  # Hz: clear speech at half the bytes of 48 kHz
PIECE = 300  # characters Silero reads at once; longer sentences are cut at a comma
PAUSE = 0.25  # seconds of silence between pieces

# The English names of letters, for abbreviations ("SQL": эс-кью-эл).
LETTERS = dict(
    zip(
        "abcdefghijklmnopqrstuvwxyz",
        (
            "эй би си ди и эф джи эйч ай джей кей эл эм эн оу пи кью ар эс ти ю ви"
            " дабл-ю экс уай зед"
        ).split(),
        strict=True,
    )
)
# A Latin word as it sounds, roughly: letter pairs first, then letters.
PAIRS = {
    "sh": "ш", "ch": "ч", "th": "т", "ph": "ф", "ck": "к", "qu": "кв", "oo": "у", "ee": "и",
    "ou": "ау", "ea": "и", "ai": "эй", "ay": "эй", "oa": "оу", "kh": "х", "zh": "ж", "ts": "ц",
}  # fmt: skip
SOUNDS = {
    "a": "а", "b": "б", "c": "к", "d": "д", "e": "е", "f": "ф", "g": "г", "h": "х", "i": "и",
    "j": "дж", "k": "к", "l": "л", "m": "м", "n": "н", "o": "о", "p": "п", "q": "к", "r": "р",
    "s": "с", "t": "т", "u": "у", "v": "в", "w": "в", "x": "кс", "y": "и", "z": "з",
}  # fmt: skip
SYMBOLS = {"%": " процентов ", "+": " плюс ", "=": " равно ", "&": " и ", "№": " номер "}


def unmark(text: str) -> str:
    """Markdown as it is said: code and links are not read out."""
    text = re.sub(r"```.*?(```|$)", " (код — в чате) ", text, flags=re.S)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)  # [text](url): the text
    text = re.sub(r"https?://\S+", " ссылка ", text)
    text = re.sub(r"^\s*(?:#{1,6}|>|[-*+]|\d+[.)])\s+", "", text, flags=re.M)  # heads, lists
    text = re.sub(r"^\s*\|?[\s:|-]*-[\s:|-]*$", "", text, flags=re.M)  # a table's rule
    text = re.sub(r"\s*\|\s*", ", ", text)  # its cells
    text = re.sub(r"(\*\*|__|\*|_|~~)(?=\S)(.+?)(?<=\S)\1", r"\2", text)
    # A line that ends without a stop is a sentence of its own.
    return re.sub(r"(?<![.!?…:;,])\n+", ".\n", text.strip())


def spell_latin(word: str) -> str:
    if (word.isupper() and len(word) <= 5) or len(word) == 1:
        return "-".join(LETTERS[c] for c in word.lower())
    out, w, i = [], word.lower(), 0
    if len(w) > 3 and w.endswith("e") and w[-2] not in "aeiouy":
        w = w[:-1]  # a silent e: "house", "code"
    while i < len(w):
        if w[i : i + 2] in PAIRS:
            out.append(PAIRS[w[i : i + 2]])
            i += 2
            continue
        out.append(SOUNDS[w[i]])
        i += 1
    return "".join(out)


def number_words(match: re.Match) -> str:
    from num2words import num2words

    number = match.group().replace(",", ".")
    if len(number) > 15:  # an id or a hash, not a number to say: digit by digit
        return " " + " ".join(num2words(int(d), lang="ru") for d in number if d.isdigit()) + " "
    return " " + num2words(float(number) if "." in number else int(number), lang="ru") + " "


def prepare(text: str) -> str:
    """What Silero can read: Cyrillic, spaces and punctuation."""
    text = unmark(text)
    for symbol, said in SYMBOLS.items():
        text = text.replace(symbol, said)
    text = re.sub(r"\d+(?:[.,]\d+)?", number_words, text)
    text = re.sub(r"[A-Za-z]+", lambda m: spell_latin(m.group()), text)
    text = re.sub(r"[()]", ", ", text)
    text = re.sub(r"[^А-Яа-яЁё.,!?;:—–\-… \n]+", " ", text)
    text = re.sub(r"\s+([.,!?;:…])", r"\1", re.sub(r"\s+", " ", text))
    text = re.sub(r"([.,!?;:…])[.,;:]+", r"\1", text)  # «., » left by the steps above
    return re.sub(r"[,;:]+(?=[.!?…])|^[\s.,!?;:…]+", "", text).strip()


def pieces(text: str) -> list[str]:
    """Sentences joined up to PIECE characters; a longer one cut at commas or spaces."""
    out: list[str] = []
    for sentence in re.split(r"(?<=[.!?…;:])\s+", text):
        while len(sentence) > PIECE:
            cut = sentence.rfind(", ", PIECE // 2, PIECE) + 1  # after the comma
            if cut <= 0:
                cut = sentence.rfind(" ", 0, PIECE)
            cut = cut if cut > 0 else PIECE
            out.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if out and len(out[-1]) + len(sentence) < PIECE:
            out[-1] += " " + sentence
        elif sentence:
            out.append(sentence)
    return [p for p in out if re.search(r"[А-Яа-яЁё]", p)]


def model_file(models: Path) -> Path:
    return models / "silero" / f"{MODEL}.pt"


def load_silero(models: Path):
    """Silero as a function: a piece of text and a speaker to 16-bit samples."""
    import torch

    path = model_file(models)
    if not path.exists():
        raise RuntimeError("Silero is not installed: run `prophet-voice install`")
    torch.set_num_threads(4)
    with warnings.catch_warnings():  # Silero's package is older than this torch
        warnings.filterwarnings("ignore", message="TypedStorage is deprecated")
        model = torch.package.PackageImporter(str(path)).load_pickle("tts_models", "model")

    def say(text: str, speaker: str) -> np.ndarray:
        audio = model.apply_tts(
            text=text, speaker=speaker, sample_rate=RATE, put_accent=True, put_yo=True
        )
        return (audio.clamp(-1, 1) * 32767).to(torch.int16).numpy()

    return say


class Speakers:
    """The speakers of this machine, through sounddevice: one sound at a time."""

    def play(self, samples: np.ndarray, rate: int) -> None:
        import sounddevice as sd

        sd.play(samples, rate)
        sd.wait()  # until the end, or until `stop`

    def stop(self) -> None:
        import sounddevice as sd

        sd.stop()


def tone(up: bool) -> np.ndarray:
    """A short tone: up when the wake word is heard, down when a phrase is dropped."""
    t = np.arange(int(RATE * 0.15)) / RATE
    start, end = (660, 880) if up else (520, 390)
    freq = start + (end - start) * np.minimum(t / 0.12, 1)
    phase = 2 * np.pi * np.cumsum(freq) / RATE
    gain = 0.25 * np.exp(-t * 30)
    return (np.sin(phase) * gain * 32767).astype(np.int16)


class Mouth:
    def __init__(self, speaker: str, models: Path, load=None, out=None):
        self.speaker = speaker
        self.models = models
        self._load = load or load_silero
        self.out = out or Speakers()
        self._say = None
        self._model_lock = threading.Lock()
        self._queue: queue.Queue = queue.Queue()
        self._gen = 0  # a hush makes the texts before it old
        self._busy = 0  # texts queued or being read
        self._lock = threading.Lock()
        self.saying = ""
        self.quiet_since = time.monotonic()
        threading.Thread(target=self._work, name="mouth", daemon=True).start()

    @property
    def speaking(self) -> bool:
        return self._busy > 0

    def warm(self) -> None:
        """Load the model now, so the first answer does not wait for it."""
        self._ready()

    def say(self, text: str) -> None:
        with self._lock:
            self._busy += 1
            self._queue.put((self._gen, "text", text))

    def tone(self, up: bool) -> None:
        with self._lock:
            self._busy += 1
            self._queue.put((self._gen, "tone", up))

    def hush(self) -> None:
        with self._lock:
            self._gen += 1
        self.out.stop()

    def wait(self, timeout: float = 120) -> None:
        """Until everything said so far has been read."""
        end = time.monotonic() + timeout
        while self.speaking and time.monotonic() < end:
            time.sleep(0.05)

    def _ready(self):
        with self._model_lock:
            if self._say is None:
                self._say = self._load(self.models)
            return self._say

    def _work(self) -> None:
        while True:
            gen, kind, what = self._queue.get()
            try:
                if gen == self._gen:
                    if kind == "tone":
                        self.out.play(tone(what), RATE)
                    else:
                        self._read(gen, what)
            except Exception:
                log.exception("could not speak")
            finally:
                self.saying = ""
                with self._lock:
                    self._busy -= 1
                    if not self._busy:
                        self.quiet_since = time.monotonic()

    def _read(self, gen: int, text: str) -> None:
        say = self._ready()
        for i, piece in enumerate(pieces(prepare(text))):
            samples = say(piece, self.speaker)
            if gen != self._gen:
                return
            if i:
                samples = np.concatenate([np.zeros(int(RATE * PAUSE), np.int16), samples])
            self.saying = piece
            self.out.play(samples, RATE)
            if gen != self._gen:
                return

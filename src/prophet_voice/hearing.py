"""Speech to text: the phrase after the wake word becomes words.

"local": faster-whisper on this machine, from the models folder (`prophet-voice
install` downloads it), loaded once and kept in memory. "groq": Groq's hosted
Whisper first, one HTTP call with the owner's Groq key; when it fails, the
local model does it, if it is installed.

Only the phrase after the wake word goes to Whisper: until the wake word,
the audio stays in the ears.
"""

import io
import logging
import threading
from pathlib import Path

import httpx

log = logging.getLogger("prophet_voice.hearing")
GROQ_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_MODEL = "whisper-large-v3-turbo"
LANGUAGE = "ru"


def model_dir(models: Path, name: str) -> Path:
    return models / "whisper" / name


def load_faster_whisper(models: Path, name: str):
    from faster_whisper import WhisperModel

    path = model_dir(models, name)
    if not (path / "model.bin").exists():
        raise RuntimeError(f"Whisper {name} is not installed: run `prophet-voice install`")
    return WhisperModel(str(path), device="auto", compute_type="auto")


class Hearing:
    def __init__(self, settings: dict, models: Path, load=None, post=None):
        self.stt = settings["stt"]
        self.groq_key = settings["groq_key"]
        self.model_name = settings["whisper_model"]
        self.models = models
        self._load = load or load_faster_whisper
        self._post = post or httpx.post
        self._model = None
        self._lock = threading.Lock()  # the model is big: one phrase at a time

    def configure(self, settings: dict) -> None:
        """New settings while it runs: another local model is loaded when next needed."""
        self.stt, self.groq_key = settings["stt"], settings["groq_key"]
        if settings["whisper_model"] != self.model_name:
            with self._lock:
                self.model_name, self._model = settings["whisper_model"], None
        log.info(
            "speech to text: %s",
            self.stt if self.groq_key or self.stt == "local" else "groq, with no key: local",
        )

    def local(self) -> bool:
        return (model_dir(self.models, self.model_name) / "model.bin").exists()

    def warm(self) -> None:
        """Load the local model now, so the first phrase does not wait for it."""
        if self.stt == "local" or self.local():
            self._ready()

    def transcribe(self, wav: bytes) -> str:
        if self.stt == "groq" and self.groq_key:
            try:
                return self._groq(wav)
            except Exception as exc:
                if not self.local():
                    raise RuntimeError(f"Groq could not transcribe: {exc}") from exc
                log.warning("Groq could not transcribe (%s); the local model does it", exc)
        return self._local(wav)

    def _groq(self, wav: bytes) -> str:
        r = self._post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {self.groq_key}"},
            files={"file": ("phrase.wav", wav, "audio/wav")},
            data={"model": GROQ_MODEL, "language": LANGUAGE, "response_format": "json"},
            timeout=30,
        )
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code}")
        return str(r.json().get("text") or "").strip()

    def _ready(self):
        with self._lock:
            if self._model is None:
                self._model = self._load(self.models, self.model_name)
            return self._model

    def _local(self, wav: bytes) -> str:
        model = self._ready()
        with self._lock:
            # vad_filter skips silence, where Whisper tends to make up words.
            segments, _ = model.transcribe(io.BytesIO(wav), language=LANGUAGE, vad_filter=True)
            return " ".join(s.text.strip() for s in segments).strip()

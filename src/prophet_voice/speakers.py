"""Who speaks: the voices in a Private phrase, told apart and named.

When two people talk to Prophet in turn, one phrase may hold both: the ears
end a phrase only at a pause. sherpa-onnx, here on the Mac, finds where each
voice speaks in it (pyannote's segmentation) and makes a print of each voice
(3D-Speaker's ERes2Net, a vector: the closer two prints point, the likelier
one voice). Nothing of it leaves the Mac.

A dialog's voices are numbered as they are first heard («Голос 1»,
«Голос 2»), so the same voice keeps its label from phrase to phrase. The
owner may name one in the dialog's page: its print is remembered, in
voices.json beside the settings (readable by the owner only), and the voice
is known by its name in the next dialogs too. Forgetting a name removes its
print.
"""

import io
import json
import logging
import os
import re
import tempfile
import threading
import wave
from pathlib import Path

import numpy as np

from prophet_voice import config

log = logging.getLogger("prophet_voice.speakers")
RATE = 16000
SEGMENTATION = "sherpa-onnx-pyannote-segmentation-3-0"
EMBEDDING = "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
CLUSTER = 0.5  # sherpa's clustering threshold: parts of a phrase farther apart are two voices
SAME = 0.45  # cosine of two prints of one voice: 3D-Speaker gives 0.6–0.9, two voices 0.1–0.3
MIN_PRINT = 0.4  # seconds: a print of less says little
MAX_VOICES = 50  # remembered
NAME = re.compile(r"^[^\W\d_](?:[\w .'-]{0,38}[^\W_])?$")


def models_dir(models: Path) -> Path:
    return models / "speakers"


def installed(models: Path) -> bool:
    here = models_dir(models)
    return (here / SEGMENTATION / "model.onnx").exists() and (here / EMBEDDING).exists()


def voices_file() -> Path:
    return config.home() / "voices.json"


def samples(wav: bytes) -> np.ndarray:
    with wave.open(io.BytesIO(wav)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768


def load_sherpa(models: Path):
    """sherpa-onnx's diarization and voice prints: `diarize(samples)` is a list of
    (start, end, cluster), `embed(samples)` a print."""
    import sherpa_onnx as so

    here = models_dir(models)
    if not installed(models):
        raise RuntimeError("telling voices apart is not installed: run `prophet-voice install`")
    embedding = so.SpeakerEmbeddingExtractorConfig(model=str(here / EMBEDDING))
    diarization = so.OfflineSpeakerDiarization(
        so.OfflineSpeakerDiarizationConfig(
            segmentation=so.OfflineSpeakerSegmentationModelConfig(
                pyannote=so.OfflineSpeakerSegmentationPyannoteModelConfig(
                    model=str(here / SEGMENTATION / "model.onnx")
                )
            ),
            embedding=embedding,
            clustering=so.FastClusteringConfig(num_clusters=-1, threshold=CLUSTER),
            min_duration_on=0.3,
            min_duration_off=0.5,
        )
    )
    extractor = so.SpeakerEmbeddingExtractor(embedding)
    lock = threading.Lock()

    def diarize(x: np.ndarray) -> list[tuple[float, float, int]]:
        with lock:
            found = diarization.process(x).sort_by_start_time()
        return [(s.start, s.end, s.speaker) for s in found]

    def embed(x: np.ndarray) -> np.ndarray:
        with lock:
            stream = extractor.create_stream()
            stream.accept_waveform(RATE, x)
            stream.input_finished()
            return np.array(extractor.compute(stream), np.float32)

    return diarize, embed


def _unit(v) -> np.ndarray:
    v = np.asarray(v, np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n else v


class Voices:
    """The remembered voices, and those heard in this dialog (from any thread)."""

    def __init__(self, path: Path | None = None):
        self.path = path or voices_file()
        self.remembered: dict[str, np.ndarray] = {}
        self.heard: list[dict] = []  # {"id", "name" or None, "print"}
        self._lock = threading.Lock()
        try:
            stored = json.loads(self.path.read_text()).get("voices", [])
            self.remembered = {
                v["name"]: _unit(v["print"]) for v in stored if NAME.match(str(v.get("name", "")))
            }
        except (OSError, ValueError, AttributeError, TypeError, KeyError):
            pass

    def new_dialog(self) -> None:
        with self._lock:
            self.heard = []

    def who(self, print_) -> dict:
        """The voice of this print: one heard in the dialog, a remembered one, or a new one."""
        p = _unit(print_)
        with self._lock:
            near = max(self.heard, key=lambda v: float(v["print"] @ p), default=None)
            if near is not None and float(near["print"] @ p) >= SAME:
                near["print"] = _unit(near["print"] + p)
                return near
            name = max(self.remembered, key=lambda n: float(self.remembered[n] @ p), default=None)
            if name is not None and float(self.remembered[name] @ p) < SAME:
                name = None
            voice = {"id": len(self.heard) + 1, "name": name, "print": p}
            self.heard.append(voice)
            return voice

    def many(self) -> bool:
        """More than one voice in this dialog: what each says is labelled."""
        return len(self.heard) > 1

    def shown(self) -> list[dict]:
        with self._lock:
            return [
                {"id": v["id"], "label": label(v), "named": bool(v["name"])} for v in self.heard
            ]

    def names(self) -> list[str]:
        with self._lock:
            return sorted(self.remembered)

    def name(self, voice_id: int, name: str) -> None:
        """Name a voice of this dialog and remember it; an empty name forgets its name.
        ValueError says what is wrong."""
        name = " ".join(str(name or "").split())
        if name and not NAME.match(name):
            raise ValueError("a name is up to 40 letters, spaces, «-», «.» or «'»")
        with self._lock:
            voice = next((v for v in self.heard if v["id"] == voice_id), None)
            if voice is None:
                raise ValueError("no such voice in this dialog")
            if voice["name"]:
                self.remembered.pop(voice["name"], None)
            if name:
                if name not in self.remembered and len(self.remembered) >= MAX_VOICES:
                    raise ValueError(f"up to {MAX_VOICES} voices are remembered")
                for other in self.heard:  # one name, one voice
                    if other["name"] == name:
                        other["name"] = None
                self.remembered[name] = voice["print"]
            voice["name"] = name or None
            self._write()

    def forget(self, name: str) -> None:
        with self._lock:
            self.remembered.pop(name, None)
            for v in self.heard:
                if v["name"] == name:
                    v["name"] = None
            self._write()

    def _write(self) -> None:
        """The whole file at once, readable by the owner only."""
        stored = {"voices": [{"name": n, "print": p.tolist()} for n, p in self.remembered.items()]}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".voices-")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(stored, f, ensure_ascii=False)
            os.chmod(tmp, 0o600)
            os.replace(tmp, self.path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise


def label(voice: dict | None) -> str:
    if voice is None:
        return "Голос"
    return voice["name"] or f"Голос {voice['id']}"


class Speakers:
    def __init__(self, diarize, embed, voices: Voices | None = None):
        self.diarize, self.embed = diarize, embed
        self.voices = voices or Voices()

    def parts(self, wav: bytes) -> list[tuple[float, float, dict]]:
        """Where each voice speaks in the phrase: (start, end, voice), in order."""
        x = samples(wav)
        found = self.diarize(x)
        clusters = sorted({c for _, _, c in found})
        if len(clusters) <= 1:  # one voice, or too little for the segmentation: the whole phrase
            if len(x) < MIN_PRINT * RATE:
                return []
            return [(0.0, len(x) / RATE, self.voices.who(self.embed(x)))]
        voice_of = {}
        for c in clusters:
            mine = [x[int(s * RATE) : int(e * RATE)] for s, e, k in found if k == c]
            if sum(len(m) for m in mine) < MIN_PRINT * RATE:
                continue
            voice_of[c] = self.voices.who(self.embed(np.concatenate(mine)))
        return [(s, e, voice_of[c]) for s, e, c in found if c in voice_of]


def turns(parts: list[tuple[float, float, dict]], timed: list[tuple[float, float, str]]):
    """Whisper's words, each given to the voice speaking at its middle (or the
    nearest one): who said what, in order, as (voice, text)."""
    out: list[tuple[dict | None, str]] = []
    for start, end, word in timed:
        word = word.strip()
        if not word:
            continue
        middle = (start + end) / 2
        voice = (
            min(parts, key=lambda p: max(p[0] - middle, middle - p[1], 0.0))[2] if parts else None
        )
        if out and out[-1][0] is voice:
            out[-1] = (voice, f"{out[-1][1]} {word}")
        else:
            out.append((voice, word))
    return out

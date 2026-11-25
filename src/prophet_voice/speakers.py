"""Who speaks: the voices in a Private phrase, told apart and remembered.

When two people talk to Prophet in turn, one phrase may hold both: the ears
end a phrase only at a pause. Here on the Mac, a diarizer finds where each
voice speaks in it, as the "diarizer" setting says: sherpa-onnx (pyannote's
segmentation, then its prints clustered) or NVIDIA's Nemotron 3 Diarization
(Transformers, on the CPU), which hears a change of voice sherpa misses.
Then 3D-Speaker's ERes2Net, through sherpa-onnx, makes a print of each voice
(a vector: the closer two prints point, the likelier one voice). Nothing of
it leaves the Mac.

A dialog's voices are numbered as they are first heard in it («Голос 2»),
so the same voice keeps its label from phrase to phrase. The owner names one
in the dialog's page («Иван»): its print is remembered, in voices.json beside
the settings (readable by the owner only), and in the next dialogs the voice
is «Иван» at once. Each phrase a named voice is heard in moves its print a
little, so it is known better with time. A voice left unnamed is forgotten
when the dialog ends. The voice's settings list the named ones, to rename or
forget.
"""

import io
import json
import logging
import os
import re
import tempfile
import threading
import wave
from datetime import date
from pathlib import Path

import numpy as np

from prophet_voice import config

log = logging.getLogger("prophet_voice.speakers")
RATE = 16000
SEGMENTATION = "sherpa-onnx-pyannote-segmentation-3-0"
NEMOTRON = "Nemotron-3-Diarization"
NEMOTRON_FILES = ("config.json", "processor_config.json", "model.safetensors")
MIN_PART = 0.3  # seconds: a diarizer's shorter part is dropped, its words go to a neighbour
OVER = 0.6  # a part this much inside another voice's turn was said over it: left to that turn
PAD = 0.15  # seconds around a turn cut out for Whisper: a word's edge is not lost
EMBEDDING = "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
# One voice through the Mac's microphone and the denoiser, in phrases of a few seconds,
# prints less alike than in a clean recording (0.6–0.9): 0.45 made one person two, 0.3
# made several one. The log says how alike each part is to the voices heard, to set it by.
SAME = 0.38  # cosine of two prints of one voice
# sherpa cuts a phrase into parts finely; whether two parts are one voice is SAME's to say.
CLUSTER = 0.5
MIN_PRINT = 0.4  # seconds: a phrase shorter is too little to say whose it is
MAX_VOICES = 50  # remembered
LEARN = 0.1  # how far a phrase moves its voice's print, once the voice has a few
NAME = re.compile(r"^[^\W\d_](?:[\w .'-]{0,38}[^\W_])?$")


def models_dir(models: Path) -> Path:
    return models / "speakers"


def installed(models: Path) -> bool:
    here = models_dir(models)
    return (here / SEGMENTATION / "model.onnx").exists() and (here / EMBEDDING).exists()


def nemotron_installed(models: Path) -> bool:
    return all((models_dir(models) / NEMOTRON / f).exists() for f in NEMOTRON_FILES)


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


def load_nemotron(models: Path):
    """Nemotron 3 Diarization from the models folder, on the CPU: `diarize(samples)`
    is a list of (start, end, speaker)."""
    import torch
    from transformers import AutoModelForAudioFrameClassification, AutoProcessor

    here = models_dir(models) / NEMOTRON
    if not nemotron_installed(models):
        raise RuntimeError("Nemotron is not installed: run `prophet-voice install`")
    processor = AutoProcessor.from_pretrained(here, local_files_only=True)
    model = AutoModelForAudioFrameClassification.from_pretrained(here, local_files_only=True)
    model.eval()
    lock = threading.Lock()

    def diarize(x: np.ndarray) -> list[tuple[float, float, int]]:
        with lock, torch.inference_mode():
            inputs = processor(x, sampling_rate=RATE)
            logits = model(**inputs).logits
            found = processor.extract_speaker_dict(logits, inputs.get("attention_mask"))[0]
        return [(float(s["Start"]), float(s["End"]), int(s["Speaker"])) for s in found]

    return diarize


def _unit(v) -> np.ndarray:
    v = np.asarray(v, np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n else v


class Voices:
    """The named voices, remembered, and the unnamed ones of this dialog (from any thread)."""

    def __init__(self, path: Path | None = None):
        self.path = path or voices_file()
        self.voices: list[dict] = []  # {"id", "name" or None, "print", "prints", "heard"}
        self.dialog: list[dict] = []  # those heard in this dialog, as first heard
        self._lock = threading.Lock()
        self._load()
        self.next = max([v["id"] for v in self.voices] + [0]) + 1

    def _load(self) -> None:
        try:
            found = json.loads(self.path.read_text()).get("voices", [])
        except (OSError, ValueError, AttributeError):
            return
        for v in found if isinstance(found, list) else []:
            try:
                voice = {
                    "id": v.get("id"),
                    "name": str(v["name"]),
                    "print": _unit(v["print"]),
                    "prints": max(1, int(v.get("prints", 1))),
                    "heard": str(v.get("heard", "")),
                }
            except (AttributeError, KeyError, TypeError, ValueError):
                continue
            if not NAME.match(voice["name"]):
                continue
            if not isinstance(voice["id"], int) or any(o["id"] == voice["id"] for o in self.voices):
                voice["id"] = None  # an older file's: numbered below
            self.voices.append(voice)
        for v in self.voices:
            if v["id"] is None:
                v["id"] = max(o["id"] or 0 for o in self.voices) + 1

    def new_dialog(self) -> None:
        """A dialog begins: the voices left unnamed in the last one are forgotten."""
        with self._lock:
            self.dialog = []
            self.voices = [v for v in self.voices if v["name"]]

    def who(self, print_) -> dict:
        """The voice of this print, a known one or a new one; a known one's print learns."""
        p = _unit(print_)
        with self._lock:
            alike = sorted(((float(v["print"] @ p), v) for v in self.voices), key=lambda a: -a[0])
            if alike and alike[0][0] >= SAME:
                voice = alike[0][1]
                voice["prints"] += 1
                learn = max(LEARN, 1 / voice["prints"])  # the first few count alike
                voice["print"] = _unit((1 - learn) * voice["print"] + learn * p)
            else:
                voice = {"id": self.next, "name": None, "print": p, "prints": 1}
                self.next += 1
                self.voices.append(voice)
            voice["heard"] = date.today().isoformat()
            if all(v["id"] != voice["id"] for v in self.dialog):
                voice["number"] = len(self.dialog) + 1
                self.dialog.append(voice)
            if voice["name"]:
                self._write()
            shown = {label(v): round(a, 2) for a, v in alike[:3]}
            log.info("a voice: %s; alike %s", label(voice), shown or "none known yet")
            return voice

    def many(self) -> bool:
        """More than one voice in this dialog: what each says is labelled."""
        return len(self.dialog) > 1

    def shown(self) -> list[dict]:
        """This dialog's voices, for its page."""
        with self._lock:
            return [
                {"id": v["id"], "label": label(v), "named": bool(v["name"])} for v in self.dialog
            ]

    def listed(self) -> list[dict]:
        """The named voices, for the settings."""
        with self._lock:
            named = sorted((v for v in self.voices if v["name"]), key=lambda v: v["name"])
            return [{"id": v["id"], "name": v["name"], "heard": v["heard"]} for v in named]

    def name(self, voice_id: int, name: str) -> None:
        """Name a voice and remember it; an empty name forgets it when the dialog
        ends. ValueError says what is wrong."""
        name = " ".join(str(name or "").split())
        if name and not NAME.match(name):
            raise ValueError("a name is up to 40 letters, spaces, «-», «.» or «'»")
        with self._lock:
            voice = self._voice(voice_id)
            named = [v for v in self.voices if v["name"] and v["name"] != name]
            if name and not voice["name"] and len(named) >= MAX_VOICES:
                raise ValueError(f"up to {MAX_VOICES} voices are remembered")
            for other in self.voices:  # one name, one voice
                if name and other["name"] == name:
                    other["name"] = None
            voice["name"] = name or None
            self._write()

    def forget(self, voice_id: int) -> None:
        """A named voice is forgotten: in this dialog it is a number again."""
        with self._lock:
            self._voice(voice_id)["name"] = None
            here = {v["id"] for v in self.dialog}
            self.voices = [v for v in self.voices if v["name"] or v["id"] in here]
            self._write()

    def _voice(self, voice_id: int) -> dict:
        voice = next((v for v in self.voices if v["id"] == voice_id), None)
        if voice is None:
            raise ValueError("no such voice")
        return voice

    def _write(self) -> None:
        """The named voices, the whole file at once, readable by the owner only."""
        stored = {
            "voices": [
                {
                    "id": v["id"],
                    "name": v["name"],
                    "heard": v.get("heard", ""),
                    "prints": v["prints"],
                    "print": [round(x, 5) for x in v["print"].tolist()],
                }
                for v in self.voices
                if v["name"]
            ],
        }
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
    return voice["name"] or f"Голос {voice.get('number', voice['id'])}"


class Speakers:
    def __init__(self, diarizers: dict, embed, voices: Voices | None = None):
        """`diarizers`: the ones loaded, by name (config.DIARIZERS); `use` picks one."""
        self.diarizers, self.embed = diarizers, embed
        self.voices = voices or Voices()
        self.use = config.DIARIZERS[0]

    def parts(self, wav: bytes) -> list[tuple[float, float, dict]]:
        """Where each voice speaks in the phrase: (start, end, voice), in order."""
        x = samples(wav)
        name = self.use if self.use in self.diarizers else next(iter(self.diarizers))
        found = [(s, e, c) for s, e, c in self.diarizers[name](x) if e - s >= MIN_PART]
        log.info(
            "%s: %d parts, %d voices: %s",
            name,
            len(found),
            len({c for _, _, c in found}),
            " ".join(f"{s:.1f}-{e:.1f}:{c}" for s, e, c in found) or "none",
        )
        clusters = sorted({c for _, _, c in found})
        if len(clusters) <= 1:  # one voice, or too little for the segmentation: the whole phrase
            if len(x) < MIN_PRINT * RATE:
                return []
            return [(0.0, len(x) / RATE, self.voices.who(self.embed(x)))]
        voice_of = {}
        for c in clusters:
            # Every voice the diarizer kept is printed, however little it said («да»):
            # it heard another voice there.
            mine = [x[int(s * RATE) : int(e * RATE)] for s, e, k in found if k == c]
            voice_of[c] = self.voices.who(self.embed(np.concatenate(mine)))
        return [(s, e, voice_of[c]) for s, e, c in found if c in voice_of]


def turns(parts: list[tuple[float, float, dict]], length: float) -> list[tuple[dict, float, float]]:
    """Who spoke when, a turn each: one voice's parts in a row joined, and a part
    most of which another voice's turn already holds left to it (said over it).
    The turns hold the whole phrase, `length` seconds: each runs on to the next,
    so a word between the diarizer's parts is not lost."""
    out: list[list] = []
    for start, end, voice in sorted(parts, key=lambda p: p[0]):
        if out and out[-1][0] is voice:
            out[-1][2] = max(out[-1][2], end)
            continue
        if out and min(end, out[-1][2]) - start > OVER * (end - start):
            continue
        if out:
            start = max(start, out[-1][2])
        out.append([voice, start, end])
    for i, turn in enumerate(out):
        turn[2] = out[i + 1][1] if i + 1 < len(out) else length
    if out:
        out[0][1] = 0.0
    return [(v, s, e) for v, s, e in out]


def length(wav: bytes) -> float:
    with wave.open(io.BytesIO(wav)) as w:
        return w.getnframes() / w.getframerate()


def clip(wav: bytes, start: float, end: float) -> bytes:
    """This stretch of the phrase, a little wider, as a WAV of its own."""
    with wave.open(io.BytesIO(wav)) as w:
        frames = w.readframes(w.getnframes())
    first, last = max(0, int((start - PAD) * RATE)), int((end + PAD) * RATE)
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(frames[first * 2 : last * 2])
    return out.getvalue()

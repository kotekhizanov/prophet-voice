"""`prophet-voice install`: everything the voice needs on this machine.

The Python packages come with the "audio" extra (install.sh). This puts the
models into the models folder, each once; run it again after a change of
Whisper's model, or to repair a model whose download broke:

- Vosk's small Russian model (about 45 MB, Apache-2.0), checked by its
  SHA-256: it hears the wake word;
- Silero v5_ru (about 140 MB): it reads the answers aloud;
- sherpa-onnx's speaker models (about 46 MB: pyannote's segmentation, MIT,
  and 3D-Speaker's ERes2Net, Apache-2.0), checked by their SHA-256: they
  tell the voices of a Private dialog apart;
- NVIDIA's Nemotron 3 Diarization (about 380 MB, OpenMDW), at a fixed
  revision and checked by its SHA-256: the other diarizer to choose;
- faster-whisper's model from the settings (large-v3-turbo: about 1.6 GB),
  unless speech goes to Groq: it transcribes the phrases.

A model is downloaded into a ".part" next to it and renamed when complete,
so a broken download is never taken for a model.
"""

import hashlib
import importlib.util
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path

import httpx

from prophet_voice import config, ears, hearing, mouth, speakers

VOSK_URL = f"https://alphacephei.com/vosk/models/{ears.VOSK_MODEL}.zip"
VOSK_SHA256 = "961d5ff98a17f4aa6de69864d0aa71fa5bac682301d2b5d17a3f24c5c99a46d4"
SHERPA = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
SEGMENTATION_URL = f"{SHERPA}/speaker-segmentation-models/{speakers.SEGMENTATION}.tar.bz2"
SEGMENTATION_SHA256 = "24615ee884c897d9d2ba09bb4d30da6bb1b15e685065962db5b02e76e4996488"
EMBEDDING_URL = f"{SHERPA}/speaker-recongition-models/{speakers.EMBEDDING}"  # sic
EMBEDDING_SHA256 = "1a331345f04805badbb495c775a6ddffcdd1a732567d5ec8b3d5749e3c7a5e4b"
NEMOTRON_URL = (
    f"https://huggingface.co/nvidia/{speakers.NEMOTRON}"
    "/resolve/f667ed73aee57d40cc39428eb768b4fd87a0a29e/{}"
)
NEMOTRON_SHA256 = {
    "config.json": "dd795ec07ffa09acf18d231aa3fa8642aa5c521760953754e02559a7413b6873",
    "processor_config.json": "91726746788a024a2f0f2f699a6c3667f0d74e6a0e9288fc44fe3018e23af68e",
    "model.safetensors": "c074d86335b3b794f8fa5edc25594558f128bdb3914d27806a3a5a2e44963cb6",
}
PACKAGES = {
    "sounddevice": "the microphone and the speakers",
    "vosk": "the wake word",
    "faster_whisper": "speech to text",
    "torch": "Silero's voice",
    "sherpa_onnx": "telling voices apart",
    "transformers": "Nemotron, telling voices apart",
    "librosa": "Nemotron's features",
}


def missing_packages() -> list[str]:
    return [name for name in PACKAGES if importlib.util.find_spec(name) is None]


def download(url: str, path: Path, sha256: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    digest = hashlib.sha256()
    print(f"  downloading {url}")
    with httpx.stream("GET", url, follow_redirects=True, timeout=60) as r:
        r.raise_for_status()
        total, done, shown = int(r.headers.get("content-length") or 0), 0, 0
        with part.open("wb") as f:
            for chunk in r.iter_bytes(1 << 16):
                f.write(chunk)
                digest.update(chunk)
                done += len(chunk)
                if done - shown >= 20 << 20:
                    shown = done
                    print(f"    {done >> 20} of {total >> 20 or '?'} MB")
    if sha256 and digest.hexdigest() != sha256:
        part.unlink()
        raise RuntimeError(f"wrong checksum of {url}: not installed")
    part.rename(path)


def vosk_model(models: Path) -> None:
    target = ears.vosk_dir(models)
    if target.exists():
        print(f"Vosk: {target.name} is installed.")
        return
    print("Vosk: the wake word's model")
    with tempfile.TemporaryDirectory(dir=models) as tmp:
        archive = Path(tmp) / "model.zip"
        download(VOSK_URL, archive, VOSK_SHA256)
        with zipfile.ZipFile(archive) as z:
            z.extractall(tmp)  # noqa: S202 - checked by its SHA-256 above
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(Path(tmp) / ears.VOSK_MODEL), target)
    print(f"Vosk: {target.name} is installed.")


def silero_model(models: Path) -> None:
    target = mouth.model_file(models)
    if not target.exists():
        print("Silero: the voice")
        download(mouth.MODEL_URL.format(mouth.MODEL), target)
    print(f"Silero: {mouth.MODEL} is installed.")


def speaker_models(models: Path) -> None:
    here = speakers.models_dir(models)
    if speakers.installed(models):
        print("Voices: the speaker models are installed.")
        return
    print("Voices: the speaker models")
    embedding = here / speakers.EMBEDDING
    if not embedding.exists():
        download(EMBEDDING_URL, embedding, EMBEDDING_SHA256)
    with tempfile.TemporaryDirectory(dir=models) as tmp:
        archive = Path(tmp) / "segmentation.tar.bz2"
        download(SEGMENTATION_URL, archive, SEGMENTATION_SHA256)
        with tarfile.open(archive) as t:
            t.extractall(tmp, filter="data")
        target = here / speakers.SEGMENTATION
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(Path(tmp) / speakers.SEGMENTATION), target)
    print("Voices: the speaker models are installed.")


def nemotron_model(models: Path) -> None:
    here = speakers.models_dir(models) / speakers.NEMOTRON
    if not speakers.nemotron_installed(models):
        print("Voices: Nemotron 3 Diarization")
        for name in speakers.NEMOTRON_FILES:
            if not (here / name).exists():
                download(NEMOTRON_URL.format(name), here / name, NEMOTRON_SHA256[name])
    print("Voices: Nemotron is installed.")


def whisper_model(models: Path, name: str) -> None:
    target = hearing.model_dir(models, name)
    if (target / "model.bin").exists():
        print(f"Whisper: {name} is installed.")
        return
    from faster_whisper import download_model

    print(f"Whisper: {name} (this is the big one)")
    part = target.with_name(target.name + ".part")
    download_model(name, output_dir=str(part))
    if target.exists():
        shutil.rmtree(target)
    part.rename(target)
    print(f"Whisper: {name} is installed.")


def install(settings: dict) -> int:
    missing = missing_packages()
    if missing:
        for name in missing:
            print(f"Missing {name} ({PACKAGES[name]}).")
        print("Run ./install.sh: it installs prophet-voice with its audio packages.")
        return 1
    models = config.models()
    models.mkdir(parents=True, exist_ok=True)
    vosk_model(models)
    silero_model(models)
    speaker_models(models)
    nemotron_model(models)
    if settings["stt"] == "local":
        whisper_model(models, settings["whisper_model"])
    else:
        print("Whisper: speech goes to Groq; no local model (and so no fallback).")
    found = ears.microphones()
    print(f"Microphones: {', '.join(found) or 'none found'}.")
    if not config.file().exists():
        config.write(config.load())
    print("Ready. Run `prophet-voice run` and add the voice in Omniscient: Settings → Plugins.")
    return 0

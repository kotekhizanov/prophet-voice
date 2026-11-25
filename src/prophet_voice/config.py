"""The voice's settings: one JSON file of its own, set from Omniscient.

The home is $PROPHET_VOICE_HOME, else ~/.config/prophet-voice. It holds
settings.json (only the owner may read it: it keeps the client key, and a
Groq key if one is set) and the models in models/.

A setting that is missing or no longer valid takes its default, so an old
file still loads; saving checks every value and refuses a wrong one.

A Prophet node may run the voice (its config's `plugins`). It gives the voice
the core's address in PROPHET_URL, and may keep the Groq key in the voice's
env there as GROQ_API_KEY: they stand in for settings the file does not set.
The file keeps only what the owner set, so a node's address is not written
into it by a save of something else.
"""

import json
import os
import re
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

from prophet_voice import words

WHISPER_MODELS = ("small", "medium", "large-v3", "large-v3-turbo")
STT = ("local", "groq")
DIARIZERS = ("sherpa", "nemotron")  # which finds where each voice speaks (speakers.py)
SECRETS = ("key", "groq_key")  # never shown again, only replaced

DEFAULTS = {
    "core": "http://127.0.0.1:8770",  # the core's address, as Omniscient's
    "key": "",  # pk_… from Settings → Clients, with the "chat" scope
    "project": "",  # a project's name or number; empty: the general chat
    "stt": "local",  # local: faster-whisper here; groq: Groq's hosted Whisper first
    "whisper_model": "large-v3-turbo",
    "groq_key": "",
    "threshold": 0.8,  # Vosk's confidence the wake word needs
    "microphone": "",  # an input device's name; empty: the system's
    "speakers": True,  # Private: tell voices apart and label what each says
    "diarizer": "sherpa",  # which finds where each voice speaks: sherpa or nemotron
    "settings_port": 8772,  # the manifest's `server` names it
    "words": words.defaults(),
}
KEY = re.compile(r"^pk_[A-Za-z0-9_-]{20,200}$")


class SettingsError(ValueError):
    pass


def home() -> Path:
    return Path(os.environ.get("PROPHET_VOICE_HOME") or Path.home() / ".config/prophet-voice")


def file() -> Path:
    return home() / "settings.json"


def models() -> Path:
    return home() / "models"


def _core(value) -> str:
    url = str(value or "").strip().rstrip("/")
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname or parts.path:
        raise SettingsError("the core's address is like http://192.168.1.10:8770")
    return url


def _choice(name: str, options: tuple) -> callable:
    def check(value) -> str:
        if value not in options:
            raise SettingsError(f"{name}: one of {', '.join(options)}")
        return value

    return check


def _key(value) -> str:
    value = str(value or "").strip()
    if value and not KEY.match(value):
        raise SettingsError("the key is the pk_… shown once in Settings → Clients")
    return value


def _groq_key(value) -> str:
    value = str(value or "").strip()
    if len(value) > 200 or any(c.isspace() for c in value):
        raise SettingsError("the Groq key is one word of up to 200 characters")
    return value


def _text(name: str, limit: int) -> callable:
    def check(value) -> str:
        value = str(value or "").strip()
        if len(value) > limit:
            raise SettingsError(f"{name}: up to {limit} characters")
        return value

    return check


def _threshold(value) -> float:
    try:
        value = round(float(value), 2)
    except (TypeError, ValueError):
        raise SettingsError("the threshold is a number") from None
    if not 0.3 <= value <= 0.99:
        raise SettingsError("the threshold is from 0.3 to 0.99")
    return value


def _flag(value) -> bool:
    if not isinstance(value, bool):
        raise SettingsError("on or off: true or false")
    return value


def _port(value) -> int:
    try:
        value = int(value)
    except (TypeError, ValueError):
        raise SettingsError("the port is a number") from None
    if not 1024 <= value <= 65535:
        raise SettingsError("the port is from 1024 to 65535")
    return value


def _words(value) -> dict[str, list[str]]:
    if not isinstance(value, dict):
        raise SettingsError("the words are a list for each kind")
    out = {}
    for name in words.KINDS:
        try:
            out[name] = words.check(name, value.get(name, list(words.KINDS[name].default)))
        except ValueError as exc:
            raise SettingsError(str(exc)) from None
    return out


CHECKS = {
    "core": _core,
    "key": _key,
    "project": _text("the project", 80),
    "stt": _choice("speech to text", STT),
    "whisper_model": _choice("Whisper's model", WHISPER_MODELS),
    "groq_key": _groq_key,
    "threshold": _threshold,
    "microphone": _text("the microphone", 200),
    "speakers": _flag,
    "diarizer": _choice("the diarizer", DIARIZERS),
    "settings_port": _port,
    "words": _words,
}


# The settings Omniscient sets too (Settings → Plugins → Voice → Settings). None is
# a secret or this Mac's own (the core's address, the port, the microphone). Each has
# this file's default, so the form opens filled in, and what the core gives back wins
# over this file. The words stay this Mac's, set through the voice's address: Prophet's
# modal has one line for a field and does not scroll, so it holds only these few.
def _plugin_settings() -> dict:
    def choice(title: str, options: tuple, name: str, about: str) -> dict:
        return {
            "type": "string",
            "title": title,
            "description": about,
            "enum": list(options),
            "default": DEFAULTS[name],
        }

    return {
        "type": "object",
        "properties": {
            "stt": choice(
                "Speech to text", STT, "stt", "local: Whisper on the Mac; groq: Groq first."
            ),
            "project": {
                "type": "string",
                "title": "The chat",
                "description": "A project's name or number; empty: the general chat.",
                "default": DEFAULTS["project"],
                "maxLength": 80,
            },
            "threshold": {
                "type": "number",
                "title": "Wake word's threshold",
                "description": "How sure Vosk must be of the wake word, 0.3–0.99.",
                "default": DEFAULTS["threshold"],
                "minimum": 0.3,
                "maximum": 0.99,
            },
            "speakers": {
                "type": "boolean",
                "title": "Tell voices apart",
                "description": "Private: each line says whose voice it is.",
                "default": DEFAULTS["speakers"],
            },
            "diarizer": choice(
                "Who speaks when",
                DIARIZERS,
                "diarizer",
                "sherpa: lighter; nemotron: NVIDIA's, hears more changes.",
            ),
            "whisper_model": choice(
                "Whisper's local model",
                WHISPER_MODELS,
                "whisper_model",
                "After a change, `prophet-voice install` on the Mac.",
            ),
        },
    }


PLUGIN_SETTINGS = _plugin_settings()


def with_plugin(settings: dict, values: dict) -> tuple[dict, list[str], list[str]]:
    """The settings with what the owner set in Omniscient over them: the settings,
    the names taken from Omniscient, and why any other was not."""
    out = json.loads(json.dumps(settings))
    used, refused = [], []
    for name, value in values.items():
        if name not in PLUGIN_SETTINGS["properties"] or value is None:
            continue
        try:
            out[name] = CHECKS[name](value)
        except ValueError as exc:  # SettingsError too
            refused.append(f"{name}: {exc}")
            continue
        used.append(name)
    return out, used, refused


def _stored() -> dict:
    """The file's valid values, and nothing for what it does not set."""
    try:
        stored = json.loads(file().read_text())
    except (OSError, ValueError):
        stored = {}
    stored = stored if isinstance(stored, dict) else {}
    out = {}
    for name, check in CHECKS.items():
        if name in stored:
            try:
                out[name] = check(stored[name])
            except SettingsError:
                pass
    return out


def _from_node(name: str, variable: str) -> dict:
    """A setting a node that runs the voice gives in its env, if it fits."""
    try:
        value = CHECKS[name](os.environ.get(variable) or "")
    except SettingsError:
        return {}
    return {name: value} if value else {}


def load() -> dict:
    """The settings: the file's valid values over what a node gives and the defaults."""
    out = {**DEFAULTS, **_from_node("core", "PROPHET_URL"), **_stored()}
    if not out["groq_key"]:
        out.update(_from_node("groq_key", "GROQ_API_KEY"))
    return json.loads(json.dumps(out))  # a copy: the defaults stay as they are


def save(changes: dict) -> dict:
    """Check the changed values, write them over the stored ones, and return
    the settings now.

    An empty secret keeps the stored one: the page never shows it, so it
    cannot send it back. `clear_<secret>` removes one."""
    current = _stored()
    for name, value in changes.items():
        if name.startswith("clear_") and name[6:] in SECRETS:
            if value:
                current[name[6:]] = ""
            continue
        if name not in CHECKS:
            raise SettingsError(f"no setting {name}")
        if name in SECRETS and not str(value or "").strip():
            continue
        current[name] = CHECKS[name](value)
    write(current)
    return load()


def write(values: dict) -> None:
    """The whole file at once, readable by the owner only."""
    home().mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=home(), prefix=".settings-")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(values, f, ensure_ascii=False, indent=2)
        os.chmod(tmp, 0o600)
        os.replace(tmp, file())
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def shown(values: dict) -> dict:
    """The settings for a page: secrets only as whether they are set, and their end."""
    out = {k: v for k, v in values.items() if k not in SECRETS}
    for name in SECRETS:
        out[name] = {"set": bool(values[name]), "hint": values[name][-4:] if values[name] else ""}
    return out

"""The voice's settings: one JSON file of its own, edited on the settings page.

The home is $PROPHET_VOICE_HOME, else ~/.config/prophet-voice. It holds
settings.json (only the owner may read it: it keeps the client key, and a
Groq key if one is set) and the models in models/.

A setting that is missing or no longer valid takes its default, so an old
file still loads; saving checks every value and refuses a wrong one.
"""

import json
import os
import re
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

from prophet_voice import words

SPEAKERS = ("xenia", "baya", "kseniya", "aidar", "eugene")  # Silero v5_ru
WHISPER_MODELS = ("small", "medium", "large-v3", "large-v3-turbo")
STT = ("local", "groq")
SECRETS = ("key", "groq_key")  # never shown again, only replaced

DEFAULTS = {
    "core": "http://127.0.0.1:8770",  # the core's address, as Omniscient's
    "key": "",  # pk_… from Settings → Clients, with the "chat" scope
    "project": "",  # a project's name or number; empty: the general chat
    "speaker": "xenia",
    "stt": "local",  # local: faster-whisper here; groq: Groq's hosted Whisper first
    "whisper_model": "large-v3-turbo",
    "groq_key": "",
    "threshold": 0.8,  # Vosk's confidence the wake word needs
    "microphone": "",  # an input device's name; empty: the system's
    "settings_port": 8771,
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
    "speaker": _choice("the voice", SPEAKERS),
    "stt": _choice("speech to text", STT),
    "whisper_model": _choice("Whisper's model", WHISPER_MODELS),
    "groq_key": _groq_key,
    "threshold": _threshold,
    "microphone": _text("the microphone", 200),
    "settings_port": _port,
    "words": _words,
}


def load() -> dict:
    """The settings: the file's valid values over the defaults."""
    try:
        stored = json.loads(file().read_text())
    except (OSError, ValueError):
        stored = {}
    stored = stored if isinstance(stored, dict) else {}
    out = {}
    for name, check in CHECKS.items():
        try:
            out[name] = check(stored[name]) if name in stored else DEFAULTS[name]
        except SettingsError:
            out[name] = DEFAULTS[name]
    return json.loads(json.dumps(out))  # a copy: the defaults stay as they are


def save(changes: dict) -> dict:
    """Check the changed values and write them over the current ones.

    An empty secret keeps the stored one: the page never shows it, so it
    cannot send it back. `clear_<secret>` removes one."""
    current = load()
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
    return current


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
    """The settings for the page: secrets only as whether they are set, and their end."""
    out = {k: v for k, v in values.items() if k not in SECRETS}
    for name in SECRETS:
        out[name] = {"set": bool(values[name]), "hint": values[name][-4:] if values[name] else ""}
    return out

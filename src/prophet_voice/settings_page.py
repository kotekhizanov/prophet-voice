"""`prophet-voice settings`: the settings as a page on this machine.

It listens on 127.0.0.1 only (`settings_port`, 8771) and writes settings.json
(`config.save`). The keys are never sent to the page: it shows whether one is
set and its last four characters, and an empty field keeps it.

Another site open in the browser must not change the settings (say, the
core's address, to have the key sent elsewhere): a request is served only
with this page's own Host, a change only as JSON (which a page elsewhere
cannot send here without asking first) and with no other Origin.
"""

import json
import logging
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files

import httpx

from prophet_voice import config, install, mouth, words

log = logging.getLogger("prophet_voice.settings")
SAMPLE = "Здравствуйте! Я Пророк, так я буду читать ответы."
MAX_BODY = 64 << 10


def options() -> dict:
    """What the page offers to choose from."""
    try:
        from prophet_voice.ears import microphones

        found = microphones()
    except Exception:  # no sounddevice yet, or no audio here
        found = []
    return {
        "speakers": list(config.SPEAKERS),
        "whisper_models": list(config.WHISPER_MODELS),
        "stt": list(config.STT),
        "microphones": found,
        "words": [
            {"name": n, "label": k.label, "hint": k.hint, "default": list(k.default)}
            for n, k in words.KINDS.items()
        ],
    }


def check_core(settings: dict, get=httpx.get) -> dict:
    """Whether the core knows the key: its name and scopes, or what is wrong."""
    if not settings["key"]:
        return {"error": "no key yet"}
    try:
        r = get(
            settings["core"] + "/api/v1/me",
            headers={"Authorization": f"Bearer {settings['key']}"},
            timeout=10,
        )
    except httpx.HTTPError as exc:
        return {"error": f"the core does not answer: {exc}"}
    if r.status_code == 401:
        return {"error": "the core does not know this key"}
    if r.status_code != 200:
        return {"error": f"HTTP {r.status_code}"}
    me = r.json()
    if "chat" not in me.get("scopes", []):
        return {**me, "error": 'the key has no "chat" scope: add it in Settings → Clients'}
    return me


class Page(BaseHTTPRequestHandler):
    port = config.DEFAULTS["settings_port"]
    sampler = None  # a Mouth for "Listen", made on the first one
    sampler_lock = threading.Lock()

    def log_message(self, fmt, *args):
        log.debug(fmt, *args)

    def _ours(self) -> bool:
        return self.headers.get("Host") in (f"127.0.0.1:{self.port}", f"localhost:{self.port}")

    def _send(self, status: int, body: bytes, kind: str = "application/json") -> None:
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Frame-Options", "DENY")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, data) -> None:
        self._send(status, json.dumps(data, ensure_ascii=False).encode())

    def do_GET(self):
        if not self._ours():
            return self._json(403, {"error": "not this page"})
        if self.path == "/":
            page = files("prophet_voice").joinpath("settings.html").read_bytes()
            return self._send(200, page, "text/html; charset=utf-8")
        if self.path == "/api/settings":
            return self._json(200, {"values": config.shown(config.load()), "options": options()})
        self._json(404, {"error": "no such page"})

    def do_POST(self):
        origin = self.headers.get("Origin")
        if (
            not self._ours()
            or (origin and origin != f"http://{self.headers.get('Host')}")
            or self.headers.get("Content-Type", "").split(";")[0] != "application/json"
        ):
            return self._json(403, {"error": "not this page"})
        size = int(self.headers.get("Content-Length") or 0)
        if size > MAX_BODY:
            return self._json(413, {"error": "too big"})
        try:
            body = json.loads(self.rfile.read(size) or b"{}")
        except ValueError:
            return self._json(400, {"error": "not JSON"})
        if self.path == "/api/settings":
            try:
                saved = config.save(body if isinstance(body, dict) else {})
            except config.SettingsError as exc:
                return self._json(422, {"error": str(exc)})
            return self._json(200, {"values": config.shown(saved)})
        if self.path == "/api/check":
            return self._json(200, check_core(config.load()))
        if self.path == "/api/say":
            return self._say(body)
        self._json(404, {"error": "no such page"})

    def _say(self, body) -> None:
        speaker = body.get("speaker") if isinstance(body, dict) else None
        if speaker not in config.SPEAKERS:
            return self._json(422, {"error": "unknown voice"})
        if install.missing_packages():
            return self._json(409, {"error": "the voice is not installed yet: run ./install.sh"})
        try:
            with Page.sampler_lock:
                if Page.sampler is None:
                    Page.sampler = mouth.Mouth(speaker, config.models())
                Page.sampler.hush()
                Page.sampler.speaker = speaker
                Page.sampler.warm()
            Page.sampler.say(SAMPLE)
        except Exception as exc:  # not installed yet
            return self._json(409, {"error": str(exc)})
        self._json(200, {"ok": True})


def serve(settings: dict, open_browser: bool = True) -> None:
    Page.port = settings["settings_port"]
    server = ThreadingHTTPServer(("127.0.0.1", Page.port), Page)
    url = f"http://127.0.0.1:{Page.port}/"
    print(f"Settings at {url}; Ctrl+C ends.")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

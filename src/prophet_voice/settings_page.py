"""The voice's address as a Prophet plugin, on this machine.

`prophet-voice run` serves it from its start, so a voice with no key yet can
be added: Omniscient's Settings → Plugins → Add plugin → http://127.0.0.1:8771
(`settings_port`), on 127.0.0.1 only. `/prophet-plugin.json` says what the
voice is and asks for the "chat" scope, and Omniscient hands the new key to
`/prophet/connect`. Only Omniscient at the core's address in the settings
may read the plugin (CORS) and talk to it: a key from elsewhere would send
what the owner says to another core.

The plugin's script, `/omniscient.js`, is the dialog in Omniscient's chat
(dialog.py: `POST /dialog`, `GET /dialog/events`) and the voice's settings
page there. The manifest's `settings` are the core's (config.PLUGIN_SETTINGS);
this Mac's own (MAC: the microphone, the Groq key, the words) the voice keeps
in settings.json, and the page reads and sets them here (`/api/settings`),
tries the voice (`/api/say`) and asks whether the core knows the key
(`/api/check`). A key is never sent back: only whether one is set and its
last four characters, and an empty field keeps it. The core's address, the
client key and the port are not set from a page: another core could be given
the key.

A request is served only with this server's own Host, and a change only as
JSON (which a page elsewhere cannot send here without asking first).
"""

import json
import logging
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.metadata import version
from importlib.resources import files
from urllib.parse import parse_qs, urlsplit

import httpx

from prophet_voice import config, dialog, install, mouth, words

log = logging.getLogger("prophet_voice.settings")
SAMPLE = "Здравствуйте! Я Пророк, так я буду читать ответы."
MAX_BODY = 64 << 10
MANIFEST = {  # Prophet's plugin manifest, API v1
    "id": "voice",  # also the key's name: the chats show client:voice
    "name": "Voice",
    "version": version("prophet-voice"),
    "api_version": 1,
    "description": "Talk to Prophet aloud: say «Пророк» and your question.",
    "homepage": "https://github.com/kotekhizanov/prophet-voice",
    "scopes": ["chat"],
    "script": "/omniscient.js",
    "connect": "/prophet/connect",
    "settings": config.PLUGIN_SETTINGS,
}
PAGE = re.compile(r"^[A-Za-z0-9-]{8,64}$")  # a page's name for itself
PLUGIN = (
    "/prophet-plugin.json",
    "/omniscient.js",
    "/prophet/connect",
    "/dialog",
    "/dialog/end",
    "/dialog/events",
    "/api/settings",
    "/api/check",
    "/api/say",
)
MAC = ("microphone", "groq_key", "clear_groq_key", "words")  # what Omniscient sets here
NOT_RUNNING = "the voice is not ready: is `prophet-voice run` running, with a key?"


def origin(url: str) -> str:
    """An address as a browser names its Origin: scheme://host[:port], no default port."""
    try:
        parts = urlsplit(url.strip())
        port = parts.port
    except ValueError:
        return ""
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return ""
    host = f"[{parts.hostname}]" if ":" in parts.hostname else parts.hostname
    default = 80 if parts.scheme == "http" else 443
    return f"{parts.scheme}://{host}" + (f":{port}" if port and port != default else "")


def options() -> dict:
    """What Omniscient's page offers to choose from."""
    try:
        from prophet_voice.ears import microphones

        found = microphones()
    except Exception:  # no sounddevice yet, or no audio here
        found = []
    return {
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
    dialog = None  # the running voice's Dialog, when `run` serves this page
    sampler = None  # a Mouth for "Listen", made on the first one
    sampler_lock = threading.Lock()

    def log_message(self, fmt, *args):
        log.debug(fmt, *args)

    def _ours(self) -> bool:
        return self.headers.get("Host") in (f"127.0.0.1:{self.port}", f"localhost:{self.port}")

    def _cors(self) -> None:
        """Omniscient's page, at the core's address, may read the plugin's answers."""
        self.send_header("Access-Control-Allow-Origin", origin(config.load()["core"]))
        self.send_header("Access-Control-Allow-Methods", "GET, POST")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Private-Network", "true")  # LAN page → 127.0.0.1
        self.send_header("Vary", "Origin")

    def _send(self, status: int, body: bytes, kind: str = "application/json") -> None:
        self.send_response(status)
        if urlsplit(self.path).path in PLUGIN:
            self._cors()
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Frame-Options", "DENY")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, data) -> None:
        self._send(status, json.dumps(data, ensure_ascii=False).encode())

    def _body(self):
        """The request's JSON, or None once a refusal is sent."""
        if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
            self._json(403, {"error": "not this page"})
            return None
        size = int(self.headers.get("Content-Length") or 0)
        if size > MAX_BODY:
            self._json(413, {"error": "too big"})
            return None
        try:
            return json.loads(self.rfile.read(size) or b"{}")
        except ValueError:
            self._json(400, {"error": "not JSON"})
            return None

    def do_OPTIONS(self):
        if not self._ours() or urlsplit(self.path).path not in PLUGIN:
            return self._json(403, {"error": "not this page"})
        self.send_response(204)
        self._cors()
        self.send_header("Access-Control-Max-Age", "600")
        self.end_headers()

    def do_GET(self):
        if not self._ours():
            return self._json(403, {"error": "not this page"})
        if self.path == "/prophet-plugin.json":
            return self._json(200, MANIFEST)
        if self.path == "/omniscient.js":
            script = files("prophet_voice").joinpath("omniscient.js").read_bytes()
            return self._send(200, script, "text/javascript; charset=utf-8")
        if urlsplit(self.path).path == "/dialog/events":
            page = parse_qs(urlsplit(self.path).query).get("page", [None])[0]
            return self._events(page if page and PAGE.match(page) else None)
        if self.path == "/api/settings":
            return self._json(200, {"values": config.shown(config.load()), "options": options()})
        self._json(404, {"error": "no such page"})

    def do_POST(self):
        if not self._ours():
            return self._json(403, {"error": "not this page"})
        if self.path == "/prophet/connect":
            return self._connect()
        if self.path == "/dialog":
            return self._set_dialog()
        if self.path == "/dialog/end":
            return self._end_dialog()
        if self.path not in ("/api/settings", "/api/check", "/api/say"):
            return self._json(404, {"error": "no such page"})
        if self._omniscient("the settings") is None:
            return
        body = self._body()
        if body is None:
            return
        if self.path == "/api/settings":
            return self._set(body if isinstance(body, dict) else {})
        if self.path == "/api/check":
            return self._json(200, check_core(config.load()))
        self._say()

    def _set(self, changes: dict) -> None:
        """This Mac's settings, from Omniscient's page."""
        other = next((name for name in changes if name not in MAC), None)
        if other is not None:
            return self._json(403, {"error": f"{other} is not set from a page"})
        try:
            saved = config.save(changes)
        except config.SettingsError as exc:
            return self._json(422, {"error": str(exc)})
        self._json(200, {"values": config.shown(saved)})

    def _omniscient(self, what: str) -> str | None:
        """The core's origin when the request is from Omniscient there; else None,
        once a refusal is sent."""
        core = origin(config.load()["core"])
        if self.headers.get("Origin") != core:
            log.warning("%s asked by %s, not the core %s", what, self.headers.get("Origin"), core)
            self._json(403, {"error": f"only Omniscient at {core} does this"})
            return None
        return core

    def _connect(self) -> None:
        """Omniscient hands the plugin its key: {"prophet": <core address>, "key"}."""
        core = self._omniscient("a key")
        if core is None:
            return
        body = self._body()
        if body is None:
            return
        if not isinstance(body, dict) or origin(str(body.get("prophet") or "")) != core:
            return self._json(403, {"error": f"a key for another core than {core}"})
        key = body.get("key")
        if not isinstance(key, str) or not config.KEY.match(key):
            return self._json(422, {"error": "the key is pk_…"})
        config.save({"key": key})
        log.info("got a key from %s (pk_…%s)", core, key[-4:])
        print(f"Got a key from {core}. A voice waiting for one goes on; restart one that runs.")
        self._json(200, {"ok": True})

    def _set_dialog(self) -> None:
        """{"dialog": "casual" | "private" | null, "project": the chat open in Omniscient}."""
        if self._omniscient("a dialog") is None:
            return
        body = self._body()
        if body is None:
            return
        if Page.dialog is None:
            return self._json(409, {"error": NOT_RUNNING})
        kind = body.get("dialog") if isinstance(body, dict) else ""
        project = body.get("project") if isinstance(body, dict) else None
        page = body.get("page") if isinstance(body, dict) else None
        if kind is not None and kind not in ("casual", "private"):
            return self._json(422, {"error": "a dialog is casual or private"})
        if not (
            project is None
            or (isinstance(project, int) and not isinstance(project, bool) and project > 0)
            or (isinstance(project, str) and 0 < len(project) <= 80)
        ):
            return self._json(422, {"error": "a project is its number or name"})
        if page is not None and not (isinstance(page, str) and PAGE.match(page)):
            return self._json(422, {"error": "a page names itself in letters, digits and -"})
        Page.dialog.set(kind, project, page)
        self._json(200, {"dialog": kind})

    def _end_dialog(self) -> None:
        """A leaving page's beacon (no JSON: a beacon cannot ask first). It only ends."""
        if self._omniscient("an end of the dialog") is None:
            return
        self.rfile.read(min(int(self.headers.get("Content-Length") or 0), MAX_BODY))
        if Page.dialog is not None:
            Page.dialog.set(None)
        self._json(200, {"dialog": None})

    def _events(self, page: str | None) -> None:
        """How the voice is now, as Server-Sent Events, while the page listens."""
        if Page.dialog is None:
            return self._json(503, {"error": NOT_RUNNING})
        self.send_response(200)
        self._cors()
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.close_connection = True
        last, sent = None, 0.0
        try:
            with Page.dialog.watching(page):
                while True:
                    now = Page.dialog.snapshot()
                    if now != last:
                        self.wfile.write(
                            f"data: {json.dumps(now, ensure_ascii=False)}\n\n".encode()
                        )
                        last, sent = now, time.monotonic()
                    elif time.monotonic() - sent > (
                        dialog.KEEPALIVE_IN_DIALOG if now["dialog"] else dialog.KEEPALIVE
                    ):
                        self.wfile.write(b": still here\n\n")
                        sent = time.monotonic()
                    self.wfile.flush()
                    time.sleep(dialog.FRAME if now["dialog"] else dialog.IDLE)
        except OSError:  # the page has gone
            pass

    def _say(self) -> None:
        if install.missing_packages():
            return self._json(409, {"error": "the voice is not installed yet: run ./install.sh"})
        try:
            with Page.sampler_lock:
                if Page.sampler is None:
                    Page.sampler = mouth.Mouth(config.models())
                Page.sampler.hush()
                Page.sampler.warm()
            Page.sampler.say(SAMPLE)
        except Exception as exc:  # not installed yet
            return self._json(409, {"error": str(exc)})
        self._json(200, {"ok": True})


def start(settings: dict, voice_dialog=None) -> ThreadingHTTPServer | None:
    """For `prophet-voice run`: this page, the plugin and the dialog, in a thread.
    The dialog (`Page.dialog`) comes once the voice is ready."""
    Page.port, Page.dialog = settings["settings_port"], voice_dialog
    try:
        server = ThreadingHTTPServer(("127.0.0.1", Page.port), Page)
    except OSError as exc:  # another program has the port
        log.warning("no dialog in Omniscient: port %s is taken (%s)", Page.port, exc)
        return None
    threading.Thread(target=server.serve_forever, name="settings", daemon=True).start()
    return server

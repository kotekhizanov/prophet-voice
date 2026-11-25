import importlib.metadata
import json
import socket
import threading
from http.server import ThreadingHTTPServer

import httpx
import pytest

from prophet_voice import config, settings_page
from prophet_voice.settings_page import Page

KEY = "pk_" + "b" * 43


@pytest.fixture
def page():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Page)
    Page.port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    with httpx.Client(base_url=f"http://127.0.0.1:{Page.port}") as client:
        yield client
    server.shutdown()
    server.server_close()


OMNISCIENT = "http://127.0.0.1:8770"  # the core's address in the default settings
FROM_OMNISCIENT = {"Origin": OMNISCIENT}


def test_there_is_no_page_of_its_own(page):
    assert page.get("/").status_code == 404


def test_omniscient_reads_this_macs_settings(page):
    r = page.get("/api/settings", headers=FROM_OMNISCIENT)
    assert r.headers["Access-Control-Allow-Origin"] == OMNISCIENT
    data = r.json()
    assert data["values"]["core"] == "http://127.0.0.1:8770"
    assert "speakers" not in data["options"]  # baya is the only voice
    assert [k["name"] for k in data["options"]["words"]][0] == "wake"
    r = page.options("/api/settings", headers=FROM_OMNISCIENT)
    assert r.status_code == 204 and r.headers["Access-Control-Allow-Origin"] == OMNISCIENT


def test_omniscient_sets_this_macs_settings_and_never_sees_a_key(page):
    groq = "gsk_" + "c" * 40
    words = {"wake": ["пророк", "оракул"]}
    r = page.post(
        "/api/settings",
        json={"groq_key": groq, "microphone": "USB", "words": words},
        headers=FROM_OMNISCIENT,
    )
    assert r.status_code == 200
    assert groq not in r.text
    assert r.json()["values"]["groq_key"] == {"set": True, "hint": "cccc"}
    settings = config.load()
    assert settings["groq_key"] == groq and settings["microphone"] == "USB"
    assert settings["words"]["wake"] == ["пророк", "оракул"]
    assert groq not in page.get("/api/settings").text
    # An empty field keeps the key.
    page.post("/api/settings", json={"groq_key": ""}, headers=FROM_OMNISCIENT)
    assert config.load()["groq_key"] == groq


def test_a_wrong_value_is_refused(page):
    r = page.post("/api/settings", json={"words": {"wake": ["два слова"]}}, headers=FROM_OMNISCIENT)
    assert r.status_code == 422
    assert "one Russian word" in r.json()["error"]


def test_the_core_the_key_and_the_port_are_not_set_from_a_page(page):
    for name, value in (("core", "http://evil:8770"), ("key", KEY), ("settings_port", 9000)):
        r = page.post("/api/settings", json={name: value}, headers=FROM_OMNISCIENT)
        assert r.status_code == 403
    assert config.load()["core"] == "http://127.0.0.1:8770" and config.load()["key"] == ""


def test_another_site_cannot_change_the_settings(page):
    # Not JSON: what a form elsewhere could send without asking.
    r = page.post(
        "/api/settings",
        content=json.dumps({"microphone": "evil"}),
        headers={"Content-Type": "text/plain", **FROM_OMNISCIENT},
    )
    assert r.status_code == 403
    # From a page elsewhere, or from no page at all.
    for headers in ({"Origin": "http://evil.com"}, {}):
        r = page.post("/api/settings", json={"microphone": "evil"}, headers=headers)
        assert r.status_code == 403
    # Another name for this machine (DNS rebinding).
    r = page.get("/api/settings", headers={"Host": f"evil.com:{Page.port}"})
    assert r.status_code == 403
    assert config.load()["microphone"] == ""


def test_listen_before_the_install_says_to_install(page, monkeypatch):
    monkeypatch.setattr(settings_page.install, "missing_packages", lambda: ["torch"])
    r = page.post("/api/say", json={}, headers=FROM_OMNISCIENT)
    assert r.status_code == 409 and "install.sh" in r.json()["error"]


def test_check_core_says_what_is_wrong():
    settings = {**config.load(), "key": KEY}

    def get(status, body=None):
        return lambda url, headers, timeout: httpx.Response(status, json=body or {})

    assert settings_page.check_core({**settings, "key": ""})["error"] == "no key yet"
    assert "does not know" in settings_page.check_core(settings, get(401))["error"]
    me = settings_page.check_core(settings, get(200, {"name": "voice", "scopes": ["tasks"]}))
    assert me["name"] == "voice" and "chat" in me["error"]
    me = settings_page.check_core(settings, get(200, {"name": "voice", "scopes": ["chat"]}))
    assert me == {"name": "voice", "scopes": ["chat"]}


def connect(page, origin=OMNISCIENT, **body):
    body = {"prophet": OMNISCIENT, "key": KEY, **body}
    return page.post("/prophet/connect", json=body, headers={"Origin": origin})


def test_omniscient_may_read_the_manifest(page):
    r = page.get("/prophet-plugin.json", headers={"Origin": OMNISCIENT})
    assert r.status_code == 200
    assert r.headers["Access-Control-Allow-Origin"] == OMNISCIENT
    manifest = r.json()
    assert manifest["id"] == "voice" and manifest["api_version"] == 1
    assert manifest["scopes"] == ["chat"] and manifest["connect"] == "/prophet/connect"
    assert manifest["script"] == "/omniscient.js"
    assert manifest["settings"] == config.PLUGIN_SETTINGS
    # The core reads the same file from GitHub: it is the package's, and names this address.
    assert manifest["version"] == importlib.metadata.version("prophet-voice")
    assert manifest["server"] == f"http://127.0.0.1:{config.DEFAULTS['settings_port']}"
    assert manifest["run"]["args"][-3:] == ["prophet-voice", "run", "--install"]
    r = page.get("/omniscient.js", headers={"Origin": OMNISCIENT})
    assert r.status_code == 200 and "export function activate(prophet)" in r.text
    assert r.headers["Content-Type"].startswith("text/javascript")
    assert r.headers["Access-Control-Allow-Origin"] == OMNISCIENT
    # Before handing the key, the browser asks.
    r = page.options("/prophet/connect", headers={"Origin": OMNISCIENT})
    assert r.status_code == 204
    assert r.headers["Access-Control-Allow-Origin"] == OMNISCIENT
    assert "Content-Type" in r.headers["Access-Control-Allow-Headers"]


def test_omniscient_hands_the_voice_its_key(page):
    r = connect(page)
    assert r.status_code == 200 and KEY not in r.text
    assert r.headers["Access-Control-Allow-Origin"] == OMNISCIENT
    assert config.load()["key"] == KEY


def test_only_omniscient_at_the_cores_address_hands_a_key(page):
    # Another site, naming itself as the core: speech would go there.
    assert connect(page, "http://evil.com", prophet="http://evil.com").status_code == 403
    # Omniscient's own origin, but a key for another core.
    assert connect(page, prophet="http://evil.com").status_code == 403
    # Not JSON, not a key, another name for this machine.
    r = page.post(
        "/prophet/connect",
        content=json.dumps({"prophet": OMNISCIENT, "key": KEY}),
        headers={"Origin": OMNISCIENT, "Content-Type": "text/plain"},
    )
    assert r.status_code == 403
    assert connect(page, key="pk_short").status_code == 422
    r = page.post(
        "/prophet/connect",
        json={"prophet": OMNISCIENT, "key": KEY},
        headers={"Origin": OMNISCIENT, "Host": f"evil.com:{Page.port}"},
    )
    assert r.status_code == 403
    assert config.load()["key"] == ""


def test_a_key_is_taken_from_the_core_saved_in_the_settings(page):
    config.save({"core": "http://192.168.1.10:8770/"})
    assert connect(page).status_code == 403
    r = connect(page, "http://192.168.1.10:8770", prophet="http://192.168.1.10:8770")
    assert r.status_code == 200 and config.load()["key"] == KEY


def test_origin_is_as_a_browser_names_it():
    assert settings_page.origin("http://Core.LAN:80") == "http://core.lan"
    assert settings_page.origin("https://core:8770/") == "https://core:8770"
    assert settings_page.origin("http://[::1]:8770") == "http://[::1]:8770"
    assert settings_page.origin("http://core:99999") == ""
    assert settings_page.origin("null") == ""


class FakeDialog:
    """The running voice's Dialog, as the page sees it."""

    def __init__(self):
        self.set_to = []
        self.watched = []

    def snapshot(self):
        return {"dialog": None, "state": "waiting", "mic": 0.0, "voice": 0.0,
                "wake": ["пророк"]}  # fmt: skip

    def set(self, kind, project=None, page=None):
        self.set_to.append((kind, project))
        self.page = page

    def name_voice(self, voice_id, name):
        if voice_id != 1:
            raise ValueError("no such voice in this dialog")
        self.named = (voice_id, name)

    def watching(self, page=None):
        import contextlib

        @contextlib.contextmanager
        def watch():
            self.watched.append("in")
            try:
                yield
            finally:
                self.watched.append("out")

        return watch()


@pytest.fixture
def voice(monkeypatch):
    found = FakeDialog()
    monkeypatch.setattr(Page, "dialog", found)
    return found


def dialog(client, origin=OMNISCIENT, **body):
    return client.post("/dialog", json=body, headers={"Origin": origin})


def test_without_the_voice_running_there_is_no_dialog(page):
    assert dialog(page, dialog="casual").status_code == 409
    r = page.get("/dialog/events", headers={"Origin": OMNISCIENT})
    assert r.status_code == 503 and "prophet-voice run" in r.json()["error"]
    assert r.headers["Access-Control-Allow-Origin"] == OMNISCIENT


def test_omniscient_begins_and_ends_a_dialog(page, voice):
    assert dialog(page, dialog="private", project=7, page="a1b2c3d4-e5f6").status_code == 200
    assert voice.page == "a1b2c3d4-e5f6"
    assert dialog(page, dialog="casual", project=None).status_code == 200
    assert dialog(page, dialog=None).status_code == 200
    assert voice.set_to == [("private", 7), ("casual", None), (None, None)]


def test_a_leaving_page_ends_its_dialog_with_a_beacon(page, voice):
    headers = {"Origin": OMNISCIENT, "Content-Type": "text/plain;charset=UTF-8"}
    assert page.post("/dialog/end", content=b"", headers=headers).status_code == 200
    assert voice.set_to == [(None, None)]
    elsewhere = {**headers, "Origin": "http://evil.com"}
    assert page.post("/dialog/end", content=b"", headers=elsewhere).status_code == 403
    assert voice.set_to == [(None, None)]


def test_only_omniscient_at_the_cores_address_begins_a_dialog(page, voice):
    assert dialog(page, "http://evil.com", dialog="private").status_code == 403
    assert dialog(page, dialog="loud").status_code == 422
    assert dialog(page, dialog="casual", project=True).status_code == 422
    assert dialog(page, dialog="casual", project="x" * 81).status_code == 422
    assert dialog(page, dialog="casual", page="../../etc").status_code == 422
    assert voice.set_to == []


def test_the_voice_streams_how_it_is(page, voice, monkeypatch):
    monkeypatch.setattr(settings_page.dialog, "IDLE", 0.01)
    monkeypatch.setattr(settings_page.dialog, "KEEPALIVE", 0.05)
    with socket.create_connection(("127.0.0.1", Page.port)) as s:
        s.sendall(
            f"GET /dialog/events HTTP/1.1\r\nHost: 127.0.0.1:{Page.port}\r\n"
            f"Origin: {OMNISCIENT}\r\n\r\n".encode()
        )
        got = b""
        while b"\r\n\r\ndata: " not in got or not got.endswith(b"\n\n"):
            got += s.recv(4096)
    head, body = got.split(b"\r\n\r\n", 1)
    assert b" 200 " in head and b"text/event-stream" in head
    assert f"Access-Control-Allow-Origin: {OMNISCIENT}".encode() in head
    assert json.loads(body.decode().removeprefix("data: ")) == voice.snapshot()
    for _ in range(100):  # the page has gone: the server sees it at its next write
        if voice.watched == ["in", "out"]:
            break
        threading.Event().wait(0.05)
    assert voice.watched == ["in", "out"]


def test_omniscient_names_a_voice_heard_in_the_dialog(page, voice):
    def name(origin=OMNISCIENT, **body):
        return page.post("/dialog/voice", json=body, headers={"Origin": origin})

    assert name(id=1, name="Аня").status_code == 200
    assert voice.named == (1, "Аня")
    assert name(id=2, name="Аня").json()["error"] == "no such voice in this dialog"
    assert name(id=True, name="Аня").status_code == 422
    assert name(id=1).status_code == 422
    assert name("http://evil.com", id=1, name="Ева").status_code == 403
    assert voice.named == (1, "Аня")


def test_omniscient_lists_renames_and_forgets_the_named_voices(page):
    from prophet_voice.speakers import Voices

    v = Voices()
    voice_id = v.who([1.0, 0.0])["id"]
    v.name(voice_id, "Иван")
    listed = page.get("/api/settings", headers=FROM_OMNISCIENT).json()["voices"]
    assert [(x["id"], x["name"]) for x in listed] == [(voice_id, "Иван")]

    def change(origin=OMNISCIENT, **body):
        return page.post("/api/voices", json=body, headers={"Origin": origin})

    r = change(id=voice_id, name="Ваня")
    assert r.status_code == 200 and r.json()["voices"][0]["name"] == "Ваня"
    assert change(id=voice_id, name="<x>").status_code == 422
    assert change(id=voice_id + 1, name="Ева").json()["error"] == "no such voice"
    assert change(id=voice_id).status_code == 422
    assert change("http://evil.com", id=voice_id, forget=True).status_code == 403
    assert change(id=voice_id, forget=True).json()["voices"] == []
    assert Voices().listed() == []

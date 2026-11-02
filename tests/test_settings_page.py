import json
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


def test_the_page_and_its_settings(page):
    r = page.get("/")
    assert r.status_code == 200 and "Prophet Voice" in r.text
    data = page.get("/api/settings").json()
    assert data["values"]["core"] == "http://127.0.0.1:8770"
    assert "xenia" in data["options"]["speakers"]
    assert [k["name"] for k in data["options"]["words"]][0] == "wake"


def test_saving_keeps_the_key_out_of_the_page(page):
    r = page.post("/api/settings", json={"key": KEY, "speaker": "aidar"})
    assert r.status_code == 200
    assert KEY not in r.text
    assert r.json()["values"]["key"] == {"set": True, "hint": "bbbb"}
    assert config.load()["key"] == KEY
    assert config.load()["speaker"] == "aidar"
    assert KEY not in page.get("/api/settings").text


def test_a_wrong_value_is_refused(page):
    r = page.post("/api/settings", json={"core": "nowhere"})
    assert r.status_code == 422
    assert "address" in r.json()["error"]


def test_another_site_cannot_change_the_settings(page):
    # Not JSON: what a form elsewhere could send without asking.
    r = page.post(
        "/api/settings",
        content=json.dumps({"core": "http://evil:8770"}),
        headers={"Content-Type": "text/plain"},
    )
    assert r.status_code == 403
    # From a page elsewhere.
    r = page.post(
        "/api/settings", json={"core": "http://evil:8770"}, headers={"Origin": "http://evil.com"}
    )
    assert r.status_code == 403
    # Another name for this machine (DNS rebinding).
    r = page.get("/api/settings", headers={"Host": f"evil.com:{Page.port}"})
    assert r.status_code == 403
    assert config.load()["core"] == "http://127.0.0.1:8770"


def test_listen_before_the_install_says_to_install(page, monkeypatch):
    monkeypatch.setattr(settings_page.install, "missing_packages", lambda: ["torch"])
    r = page.post("/api/say", json={"speaker": "xenia"})
    assert r.status_code == 409 and "install.sh" in r.json()["error"]
    assert page.post("/api/say", json={"speaker": "nobody"}).status_code == 422


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

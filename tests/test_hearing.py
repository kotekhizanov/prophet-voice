import httpx
import pytest

from prophet_voice import config, hearing
from prophet_voice.hearing import Hearing


class Model:
    def transcribe(self, audio, language, vad_filter):
        assert language == "ru" and vad_filter
        segment = type("Segment", (), {})
        parts = [segment(), segment()]
        parts[0].text, parts[1].text = " Пророк, ", "привет "
        return iter(parts), None


def installed(models, name="large-v3-turbo"):
    path = hearing.model_dir(models, name)
    path.mkdir(parents=True)
    (path / "model.bin").write_bytes(b"")


def test_local_whisper(tmp_path):
    h = Hearing(config.load(), tmp_path, load=lambda models, name: Model())
    assert h.transcribe(b"wav") == "Пророк, привет"


def test_groq_first_with_the_owners_key(tmp_path):
    seen = {}

    def post(url, headers, files, data, timeout):
        seen.update(url=url, auth=headers["Authorization"], data=data)
        return httpx.Response(200, json={"text": " Пророк, привет "})

    settings = {**config.load(), "stt": "groq", "groq_key": "gsk_test"}
    h = Hearing(settings, tmp_path, load=lambda *a: pytest.fail("not local"), post=post)
    assert h.transcribe(b"wav") == "Пророк, привет"
    assert seen["url"] == hearing.GROQ_URL
    assert seen["auth"] == "Bearer gsk_test"
    assert seen["data"]["language"] == "ru"


def test_groq_failing_falls_back_to_the_local_model(tmp_path):
    installed(tmp_path)
    settings = {**config.load(), "stt": "groq", "groq_key": "gsk_test"}
    post = lambda *a, **k: httpx.Response(503)  # noqa: E731
    h = Hearing(settings, tmp_path, load=lambda models, name: Model(), post=post)
    assert h.transcribe(b"wav") == "Пророк, привет"


def test_groq_failing_with_no_local_model_says_so(tmp_path):
    settings = {**config.load(), "stt": "groq", "groq_key": "gsk_test"}
    h = Hearing(settings, tmp_path, post=lambda *a, **k: httpx.Response(401))
    with pytest.raises(RuntimeError, match="Groq could not transcribe"):
        h.transcribe(b"wav")


def test_no_local_model_asks_to_install(tmp_path):
    h = Hearing(config.load(), tmp_path)
    with pytest.raises((RuntimeError, ImportError)):
        h.transcribe(b"wav")


def test_new_settings_while_it_runs(tmp_path):
    loaded = []
    h = Hearing(config.load(), tmp_path, load=lambda models, name: loaded.append(name) or Model())
    h.transcribe(b"")
    h.configure({**config.load(), "stt": "groq", "groq_key": "gsk_x", "whisper_model": "small"})
    assert (h.stt, h.groq_key) == ("groq", "gsk_x")
    h.configure({**config.load(), "whisper_model": "small"})
    h.transcribe(b"")  # the other model, loaded when needed
    assert loaded == ["large-v3-turbo", "small"]

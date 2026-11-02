import json
import stat

import pytest

from prophet_voice import config

KEY = "pk_" + "a" * 43


def test_defaults_without_a_file():
    s = config.load()
    assert s["core"] == "http://127.0.0.1:8770"
    assert s["words"]["wake"] == ["пророк"]
    assert s["key"] == ""


def test_save_writes_a_file_only_the_owner_reads():
    config.save({"core": "http://192.168.1.10:8770/", "key": KEY, "speaker": "baya"})
    s = config.load()
    assert s["core"] == "http://192.168.1.10:8770"
    assert s["key"] == KEY
    assert s["speaker"] == "baya"
    assert stat.S_IMODE(config.file().stat().st_mode) == 0o600


def test_an_empty_secret_keeps_the_saved_one_and_clear_removes_it():
    config.save({"key": KEY, "groq_key": "gsk_x"})
    config.save({"key": "", "groq_key": "  "})
    assert config.load()["key"] == KEY
    assert config.load()["groq_key"] == "gsk_x"
    config.save({"clear_groq_key": True})
    assert config.load()["groq_key"] == ""


@pytest.mark.parametrize(
    "change",
    [
        {"core": "ftp://host"},
        {"core": "http://host:8770/api"},
        {"key": "not-a-key"},
        {"speaker": "alice"},
        {"threshold": 2},
        {"settings_port": 80},
        {"words": {"wake": ["два слова"]}},
        {"unknown": 1},
    ],
)
def test_save_refuses_a_wrong_value_and_keeps_the_file(change):
    config.save({"speaker": "baya"})
    with pytest.raises(config.SettingsError):
        config.save(change)
    assert config.load()["speaker"] == "baya"


def test_a_broken_value_in_the_file_takes_its_default(home):
    home.mkdir(parents=True)
    config.file().write_text(json.dumps({"speaker": "nobody", "threshold": 0.7, "extra": 1}))
    s = config.load()
    assert s["speaker"] == "xenia"
    assert s["threshold"] == 0.7
    assert "extra" not in s


def test_shown_hides_the_secrets():
    shown = config.shown({**config.load(), "key": KEY})
    assert shown["key"] == {"set": True, "hint": "aaaa"}
    assert shown["groq_key"] == {"set": False, "hint": ""}
    assert KEY not in json.dumps(shown)

import json
import re
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
    config.save({"core": "http://192.168.1.10:8770/", "key": KEY, "stt": "groq"})
    s = config.load()
    assert s["core"] == "http://192.168.1.10:8770"
    assert s["key"] == KEY
    assert s["stt"] == "groq"
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
        {"whisper_model": "tiny"},
        {"speaker": "baya"},  # baya is the only voice: not a setting
        {"threshold": 2},
        {"settings_port": 80},
        {"words": {"wake": ["два слова"]}},
        {"unknown": 1},
    ],
)
def test_save_refuses_a_wrong_value_and_keeps_the_file(change):
    config.save({"threshold": 0.7})
    with pytest.raises(config.SettingsError):
        config.save(change)
    assert config.load()["threshold"] == 0.7


def test_a_broken_value_in_the_file_takes_its_default(home):
    home.mkdir(parents=True)
    config.file().write_text(json.dumps({"stt": "nobody", "threshold": 0.7, "speaker": "xenia"}))
    s = config.load()
    assert s["stt"] == "local"
    assert s["threshold"] == 0.7
    assert "speaker" not in s  # an old file's voice is dropped


def test_a_node_gives_the_cores_address_and_the_groq_key(monkeypatch):
    monkeypatch.setenv("PROPHET_URL", "http://192.168.0.10:8770")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_node")
    s = config.load()
    assert s["core"] == "http://192.168.0.10:8770"
    assert s["groq_key"] == "gsk_node"
    config.save({"threshold": 0.7})  # a save of something else writes neither
    assert set(json.loads(config.file().read_text())) == {"threshold"}
    config.save({"core": "http://10.0.0.2:8770", "groq_key": "gsk_own"})  # the owner's win
    s = config.load()
    assert s["core"] == "http://10.0.0.2:8770" and s["groq_key"] == "gsk_own"


def test_a_wrong_address_from_a_node_is_not_used(monkeypatch):
    monkeypatch.setenv("PROPHET_URL", "ws://core/api")
    assert config.load()["core"] == "http://127.0.0.1:8770"


def test_shown_hides_the_secrets():
    shown = config.shown({**config.load(), "key": KEY})
    assert shown["key"] == {"set": True, "hint": "aaaa"}
    assert shown["groq_key"] == {"set": False, "hint": ""}
    assert KEY not in json.dumps(shown)


def test_the_plugin_settings_fit_prophets_rules():
    # Prophet's manifest check (Plugins in Omniscient): up to 30 settings of these
    # types and fields, each default fitting its setting.
    schema = config.PLUGIN_SETTINGS
    assert schema["type"] == "object" and set(schema) == {"type", "properties"}
    assert 1 <= len(schema["properties"]) <= 30
    fields = {"type", "title", "description", "default", "enum", "minimum", "maximum", "maxLength"}
    for name, spec in schema["properties"].items():
        assert re.fullmatch(r"[a-z0-9_]+", name)
        assert spec["type"] in ("string", "number") and set(spec) <= fields
        assert len(spec["title"]) <= 300 and len(spec["description"]) <= 300
        assert spec["default"] == config.DEFAULTS[name]
        assert config.CHECKS[name](spec["default"]) == spec["default"]
        if "enum" in spec:
            assert spec["default"] in spec["enum"] and len(spec["enum"]) <= 50
    secret_or_mac = {"core", "key", "groq_key", "microphone", "settings_port"}
    assert not secret_or_mac & set(schema["properties"])


def test_the_plugin_settings_fit_prophets_form_without_a_scroll():
    # Prophet's form does not scroll: a few settings, each hint on one line.
    properties = config.PLUGIN_SETTINGS["properties"]
    assert len(properties) <= 6
    assert all(len(spec["description"]) <= 60 for spec in properties.values())


def test_what_is_set_in_omniscient_wins():
    here = config.load() | {"stt": "local", "threshold": 0.7, "project": "Дом"}
    values = {"stt": "groq", "threshold": 0.9, "project": ""}
    out, used, refused = config.with_plugin(here, values)
    assert out["stt"] == "groq" and out["threshold"] == 0.9
    assert out["project"] == ""  # the general chat, as set there
    assert out["whisper_model"] == here["whisper_model"]  # not given: this file's
    assert used == ["stt", "threshold", "project"] and refused == []
    assert here["stt"] == "local"  # the settings given stay as they are


def test_a_setting_from_omniscient_that_does_not_fit_is_not_used():
    here = config.load()
    values = {"stt": "nobody", "words_wake": "два слова", "key": "pk_x", "project": "Дом"}
    values["speaker"] = "xenia"  # what Omniscient kept from an older version
    out, used, refused = config.with_plugin(here, values)
    assert out["stt"] == here["stt"] and out["words"]["wake"] == ["пророк"]
    assert out["key"] == "" and out["project"] == "Дом" and "speaker" not in out
    assert used == ["project"]
    assert [why.split(":")[0] for why in refused] == ["stt"]

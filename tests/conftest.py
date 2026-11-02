import pytest


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    """Every test has its own settings folder, never the owner's."""
    monkeypatch.setenv("PROPHET_VOICE_HOME", str(tmp_path / "voice"))
    return tmp_path / "voice"

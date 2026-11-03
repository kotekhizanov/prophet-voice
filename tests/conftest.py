import pytest


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    """Every test has its own settings folder, never the owner's."""
    monkeypatch.setenv("PROPHET_VOICE_HOME", str(tmp_path / "voice"))
    for name in ("PROPHET_URL", "GROQ_API_KEY"):  # what a node that runs the voice gives
        monkeypatch.delenv(name, raising=False)
    return tmp_path / "voice"

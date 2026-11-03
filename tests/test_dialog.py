import asyncio

from prophet_voice.dialog import Dialog


class Voice:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.set_to = []

    def set_dialog(self, kind, project=None):
        self.set_to.append((kind, project))


def test_the_last_page_to_go_ends_the_dialog():
    voice = Voice()
    d = Dialog(voice)
    with d.watching():
        with d.watching():
            pass
        d.set("private", 3)
        voice.loop.run_until_complete(asyncio.sleep(0))
        assert voice.set_to == [("private", 3)]  # one page still shows it
    voice.loop.run_until_complete(asyncio.sleep(0))
    assert voice.set_to == [("private", 3), (None, None)]
    voice.loop.close()


def test_the_page_that_began_a_dialog_ends_it_when_it_closes():
    voice = Voice()
    d = Dialog(voice)
    with d.watching("other"):
        with d.watching("mine"):
            d.set("private", None, "mine")
        voice.loop.run_until_complete(asyncio.sleep(0))
        assert voice.set_to == [("private", None), (None, None)]  # another page still open
        assert d.owner is None
    voice.loop.close()


def test_a_private_dialog_shows_its_voices(home):
    from types import SimpleNamespace

    from prophet_voice.speakers import Speakers, Voices

    voice = SimpleNamespace(
        dialog="private",
        private=True,
        ears=SimpleNamespace(level=0.0),
        mouth=SimpleNamespace(level=0.0),
        words={"wake": ["пророк"]},
        state=lambda: "waiting",
        speakers=Speakers(None, None, Voices(home / "voices.json")),
    )
    d = Dialog(voice)
    voice.speakers.voices.who([1.0, 0.0])
    d.name_voice(1, "Аня")
    assert d.snapshot()["voices"] == [{"id": 1, "label": "Аня", "named": True}]
    voice.private, voice.dialog = False, "casual"
    assert d.snapshot()["voices"] == []

import sys
import threading
import time

import numpy as np

from prophet_voice import mouth
from prophet_voice.mouth import Mouth


def test_prepare_reads_numbers_latin_and_abbreviations():
    said = mouth.prepare("SQL в ClickHouse: 2 запроса, 15%.")
    assert said == "эс-кью-эл в кликхаус: два запроса, пятнадцать процентов."


def test_prepare_leaves_out_markdown_code_and_links():
    text = (
        "## Итог\n\n**Готово**: смотрите [отчёт](https://example.com/r).\n"
        "- первое\n- второе\n\n```sql\nSELECT 1\n```\nВсё."
    )
    said = mouth.prepare(text)
    assert "#" not in said and "*" not in said and "селект" not in said
    assert "отчёт" in said and "ссылка" not in said
    assert said.startswith("Итог. Готово: смотрите отчёт.")
    assert "первое. второе." in said
    assert "код — в чате" in said


def test_a_bare_link_is_a_link():
    assert mouth.prepare("Вот https://example.com/x?y=1 и всё") == "Вот ссылка и всё"


def test_pieces_join_short_sentences_and_cut_long_ones():
    assert mouth.pieces("Да. Нет. Может быть.") == ["Да. Нет. Может быть."]
    long = ", ".join(["слово"] * 100) + "."
    parts = mouth.pieces(long)
    assert len(parts) > 1 and all(len(p) <= mouth.PIECE for p in parts)
    assert mouth.pieces("... !") == []


class Out:
    """Speakers that record and can be held, as a long piece plays."""

    def __init__(self):
        self.played = []
        self.hold = threading.Event()
        self.hold.set()
        self.stopped = 0

    def play(self, samples, rate):
        self.played.append(len(samples))
        self.hold.wait(5)

    def stop(self):
        self.stopped += 1
        self.hold.set()


def fake_silero(models):
    return lambda text: np.ones(len(text), np.int16)


def test_mouth_reads_texts_in_order(tmp_path):
    out = Out()
    m = Mouth(tmp_path, load=fake_silero, out=out)
    m.say("Первый.")
    m.tone(True)
    m.say("Второй.")
    m.wait(5)
    assert not m.speaking
    assert out.played[0] == len("Первый.")
    assert out.played[1] == len(mouth.tone(True))
    assert out.played[2] == len("Второй.")


def test_hush_stops_now_and_drops_what_was_waiting(tmp_path):
    out = Out()
    out.hold.clear()  # the first piece plays until stopped
    m = Mouth(tmp_path, load=fake_silero, out=out)
    m.say("Длинный ответ.")
    m.say("Ещё один.")
    for _ in range(100):
        if m.saying:
            break
        threading.Event().wait(0.01)
    assert m.speaking and m.saying == "Длинный ответ."
    m.hush()
    m.wait(5)
    assert not m.speaking
    assert out.stopped == 1
    assert out.played == [len("Длинный ответ.")]  # the second never played


def test_a_cue_is_the_same_voice_quieter(tmp_path):
    played = []

    class Kept(Out):
        def play(self, samples, rate):
            played.append(samples.copy())

    m = Mouth(
        tmp_path,
        load=lambda models: lambda text: np.full(9, 1000, np.int16),
        out=Kept(),
    )
    m.cue("Ага?", 0.5)
    m.say("Ага?")
    m.wait(5)
    assert played[0].tolist() == [500] * 9 and played[1].tolist() == [1000] * 9


def test_the_level_follows_the_sound_playing_now(tmp_path):
    out = Out()
    out.hold.clear()
    loudly = np.full(mouth.RATE * 2, 16000, np.int16)
    m = Mouth(tmp_path, load=lambda models: lambda text: loudly, out=out)
    assert m.level == 0
    m.say("Длинный ответ.")
    for _ in range(100):
        if m.saying:
            break
        threading.Event().wait(0.01)
    assert m.level == 1  # about 0.49 of full scale, times five
    m.hush()
    m.wait(5)
    assert m.level == 0


def test_two_mouths_never_sound_together(monkeypatch):
    """The settings page's sample and the voice's answer: one after the other."""
    playing, overlaps = [], []

    class Sounddevice:
        def play(self, samples, rate, **kwargs):
            if playing:
                overlaps.append(len(samples))
            playing.append(len(samples))

        def wait(self):
            time.sleep(0.05)
            playing.pop()

        def stop(self):
            pass

    monkeypatch.setitem(sys.modules, "sounddevice", Sounddevice())
    speakers = [mouth.Speakers(), mouth.Speakers()]
    threads = [
        threading.Thread(target=s.play, args=(np.zeros(10, np.int16), mouth.RATE)) for s in speakers
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert overlaps == []


def test_another_mouth_sounding_is_prophet_speaking(tmp_path):
    """The settings page's sample plays: the ears must not take it for the owner."""
    m = Mouth(tmp_path, load=fake_silero, out=Out())
    assert not m.speaking
    with mouth.ONE_SOUND:
        assert m.speaking
    assert not m.speaking


def test_the_next_piece_is_made_while_one_sounds(tmp_path):
    made, out = [], Out()
    out.hold.clear()

    def silero(models):
        def say(text):
            made.append(text)
            return np.ones(len(text), np.int16)

        return say

    m = Mouth(tmp_path, load=silero, out=out)
    first, second = "Первое предложение." * 10, "Второе предложение." * 10
    m.say(first + " " + second)
    end = time.monotonic() + 3
    while len(made) < 2 and time.monotonic() < end:
        time.sleep(0.01)
    assert len(out.played) == 1 and len(made) == 2  # the first sounds, the second is ready
    out.hold.set()
    m.wait(5)
    assert len(out.played) == 2


def test_the_mouth_tells_the_echo_what_plays_and_when_it_is_cut_off(tmp_path):
    class Echo:
        def __init__(self):
            self.calls = []

        def played(self, samples, rate):
            self.calls.append(("played", len(samples), rate))

        def stopped(self):
            self.calls.append(("stopped",))

    echo, out = Echo(), Out()
    m = Mouth(tmp_path, load=fake_silero, out=out, echo=echo)
    m.say("Привет.")
    m.wait(2)
    m.hush()
    assert [c[0] for c in echo.calls] == ["played", "stopped"]
    assert echo.calls[0][1] > 0 and echo.calls[0][2] == mouth.RATE

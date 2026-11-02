import threading

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
    return lambda text, speaker: np.ones(len(text), np.int16)


def test_mouth_reads_texts_in_order(tmp_path):
    out = Out()
    m = Mouth("xenia", tmp_path, load=fake_silero, out=out)
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
    m = Mouth("xenia", tmp_path, load=fake_silero, out=out)
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

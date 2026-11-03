import os

import numpy as np
import pytest

from prophet_voice import speakers
from prophet_voice.ears import wav
from prophet_voice.speakers import Speakers, Voices

ANYA, KOSTYA, OTHER = np.eye(3, dtype=np.float32)


def audio(seconds):
    return wav([b"\0\0" * int(seconds * speakers.RATE)])


def test_a_voice_keeps_its_number_through_the_dialog(home):
    v = Voices(home / "voices.json")
    first, second = v.who(ANYA), v.who(KOSTYA)
    assert (first["id"], second["id"]) == (1, 2)
    assert v.who(ANYA + 0.1 * KOSTYA) is first  # near enough: the same voice
    assert [s["label"] for s in v.shown()] == ["Голос 1", "Голос 2"]
    assert v.many()
    v.new_dialog()
    assert v.shown() == [] and not v.many()


def test_a_named_voice_is_known_in_the_next_dialogs(home):
    path = home / "voices.json"
    v = Voices(path)
    v.who(ANYA)
    v.name(1, "Аня")
    assert v.shown() == [{"id": 1, "label": "Аня", "named": True}]
    assert os.stat(path).st_mode & 0o777 == 0o600
    again = Voices(path)
    assert again.names() == ["Аня"]
    assert again.who(ANYA)["name"] == "Аня"
    assert again.who(KOSTYA)["name"] is None


def test_a_name_is_forgotten(home):
    v = Voices(home / "voices.json")
    v.who(ANYA)
    v.name(1, "Аня")
    v.forget("Аня")
    assert v.names() == [] and v.shown()[0]["label"] == "Голос 1"
    assert Voices(home / "voices.json").names() == []
    v.name(1, "Аня")
    v.name(1, "")  # an empty name forgets it too
    assert v.names() == []


def test_a_wrong_name_or_voice_is_refused(home):
    v = Voices(home / "voices.json")
    v.who(ANYA)
    for name in ("<script>", "1st", "x" * 41):
        with pytest.raises(ValueError):
            v.name(1, name)
    with pytest.raises(ValueError, match="no such voice"):
        v.name(9, "Аня")
    assert not (home / "voices.json").exists()


def test_one_name_is_one_voice(home):
    v = Voices(home / "voices.json")
    v.who(ANYA), v.who(KOSTYA)
    v.name(1, "Аня")
    v.name(2, "Аня")  # the owner named the wrong one first
    assert [s["label"] for s in v.shown()] == ["Голос 1", "Аня"]


def test_a_broken_file_is_no_voices(home):
    home.mkdir(parents=True)
    (home / "voices.json").write_text("{not json")
    assert Voices(home / "voices.json").names() == []


def test_a_phrase_of_two_voices_is_split_between_them(home):
    prints = {0: ANYA, 1: KOSTYA}

    def diarize(x):
        return [(0.0, 1.0, 0), (1.2, 2.0, 1), (2.2, 3.0, 0)]

    def embed(x):
        return prints[0] if len(x) > 1.5 * speakers.RATE else prints[1]  # Аня's two parts

    s = Speakers(diarize, embed, Voices(home / "voices.json"))
    parts = s.parts(audio(3))
    assert [(a, b, v["id"]) for a, b, v in parts] == [(0.0, 1.0, 1), (1.2, 2.0, 2), (2.2, 3.0, 1)]


def test_a_phrase_of_one_voice_is_that_voice(home):
    s = Speakers(lambda x: [(0.2, 1.5, 0)], lambda x: OTHER, Voices(home / "voices.json"))
    parts = s.parts(audio(2))
    assert [(a, b, v["id"]) for a, b, v in parts] == [(0.0, 2.0, 1)]
    assert s.parts(audio(0.2)) == []  # too short to tell whose


def test_whispers_words_go_to_who_spoke_them():
    anya, kostya = {"id": 1, "name": "Аня"}, {"id": 2, "name": None}
    parts = [(0.0, 1.0, anya), (1.2, 2.0, kostya), (2.2, 3.0, anya)]
    timed = [(0.1, 0.4, " Какая"), (0.5, 0.9, " погода?"), (1.1, 1.3, " А"),  # between: nearest
             (1.4, 1.9, " хлеб?"), (2.3, 2.8, " Завтра."), (3.5, 3.6, " ")]  # fmt: skip
    turns = speakers.turns(parts, timed)
    assert turns == [(anya, "Какая погода?"), (kostya, "А хлеб?"), (anya, "Завтра.")]
    assert [speakers.label(v) for v, _ in turns] == ["Аня", "Голос 2", "Аня"]

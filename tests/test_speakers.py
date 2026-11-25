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
    assert v.who(ANYA + 0.1 * KOSTYA) is first  # near enough: the same voice
    assert [s["label"] for s in v.shown()] == ["Голос 1", "Голос 2"]
    assert v.many() and second["id"] != first["id"]
    v.new_dialog()
    assert v.shown() == [] and not v.many()


def test_a_named_voice_is_known_at_once_in_the_next_dialogs(home):
    path = home / "voices.json"
    v = Voices(path)
    v.who(KOSTYA), v.who(ANYA)
    v.name(v.shown()[1]["id"], "Иван")
    assert [s["label"] for s in v.shown()] == ["Голос 1", "Иван"]
    assert os.stat(path).st_mode & 0o777 == 0o600
    again = Voices(path)  # the next session
    assert [x["name"] for x in again.listed()] == ["Иван"]
    assert again.who(ANYA)["name"] == "Иван"
    assert [s["label"] for s in again.shown()] == ["Иван"]


def test_an_unnamed_voice_is_forgotten_when_the_dialog_ends(home):
    path = home / "voices.json"
    v = Voices(path)
    v.who(ANYA)
    v.new_dialog()
    assert v.who(ANYA)["name"] is None and v.shown()[0]["label"] == "Голос 1"
    assert not path.exists()  # nothing named: nothing kept
    assert Voices(path).listed() == []


def test_a_named_voice_learns_from_each_phrase(home):
    path = home / "voices.json"
    v = Voices(path)
    v.name(v.who(ANYA)["id"], "Аня")
    for _ in range(10):
        v.who(ANYA + 0.5 * KOSTYA)
    learnt = Voices(path).voices[0]
    assert learnt["prints"] == 11
    assert 0.1 < float(learnt["print"] @ KOSTYA) < 0.45  # moved, not replaced


def test_a_voice_is_renamed_and_forgotten(home):
    path = home / "voices.json"
    v = Voices(path)
    voice_id = v.who(ANYA)["id"]
    v.name(voice_id, "Аня")
    v.name(voice_id, "Анна")
    assert [x["name"] for x in Voices(path).listed()] == ["Анна"]
    v.forget(voice_id)
    assert Voices(path).listed() == []
    assert v.shown()[0]["label"] == "Голос 1"  # in this dialog a number again
    v.new_dialog()
    assert v.voices == []


def test_a_wrong_name_or_voice_is_refused(home):
    v = Voices(home / "voices.json")
    voice_id = v.who(ANYA)["id"]
    for name in ("<script>", "1st", "x" * 41):
        with pytest.raises(ValueError):
            v.name(voice_id, name)
    with pytest.raises(ValueError, match="no such voice"):
        v.name(voice_id + 9, "Аня")
    assert not (home / "voices.json").exists()


def test_one_name_is_one_voice(home):
    v = Voices(home / "voices.json")
    a, b = v.who(ANYA)["id"], v.who(KOSTYA)["id"]
    v.name(a, "Аня")
    v.name(b, "Аня")  # the owner named the wrong one first
    assert [s["label"] for s in v.shown()] == ["Голос 1", "Аня"]


def test_an_older_file_and_a_broken_one(home):
    home.mkdir(parents=True)
    path = home / "voices.json"
    path.write_text(
        '{"voices": [{"name": "Аня", "print": [1, 0, 0]}, {"name": "<x>", "print": [0, 1, 0]}]}'
    )
    v = Voices(path)
    assert [x["name"] for x in v.listed()] == ["Аня"]
    assert v.who(KOSTYA)["id"] != v.listed()[0]["id"]
    path.write_text("{not json")
    assert Voices(path).listed() == []


def test_a_phrase_of_two_voices_is_split_between_them(home):
    prints = {0: ANYA, 1: KOSTYA}

    def diarize(x):
        return [(0.0, 1.0, 0), (1.2, 2.0, 1), (2.2, 3.0, 0)]

    def embed(x):
        return prints[0] if len(x) > 1.5 * speakers.RATE else prints[1]  # Аня's two parts

    s = Speakers({"sherpa": diarize}, embed, Voices(home / "voices.json"))
    parts = s.parts(audio(3))
    assert [(a, b, v["id"]) for a, b, v in parts] == [(0.0, 1.0, 1), (1.2, 2.0, 2), (2.2, 3.0, 1)]


def test_a_phrase_of_one_voice_is_that_voice(home):
    s = Speakers(
        {"sherpa": lambda x: [(0.2, 1.5, 0)]}, lambda x: OTHER, Voices(home / "voices.json")
    )
    parts = s.parts(audio(2))
    assert [(a, b, v["id"]) for a, b, v in parts] == [(0.0, 2.0, 1)]
    assert s.parts(audio(0.2)) == []  # too short to tell whose


def test_the_diarizer_set_is_used_and_sherpa_without_it(home):
    asked = []

    def diarizer(name):
        return lambda x: asked.append(name) or [(0.0, 1.0, 0), (1.2, 2.0, 1)]

    embed = lambda x: ANYA if len(x) > 0.9 * speakers.RATE else KOSTYA  # noqa: E731
    s = Speakers({"sherpa": diarizer("sherpa")}, embed, Voices(home / "voices.json"))
    s.use = "nemotron"  # not loaded: sherpa does it
    s.parts(audio(2))
    s.diarizers["nemotron"] = diarizer("nemotron")
    assert len(s.parts(audio(2))) == 2
    assert asked == ["sherpa", "nemotron"]


def test_a_diarizers_crumb_of_a_part_is_dropped(home):
    found = [(0.0, 1.0, 0), (1.0, 1.1, 1), (1.2, 2.0, 0)]  # Nemotron leaves such crumbs
    s = Speakers({"sherpa": lambda x: found}, lambda x: ANYA, Voices(home / "voices.json"))
    assert [(a, b) for a, b, _ in s.parts(audio(2))] == [(0.0, 2.0)]  # one voice: all of it


def test_a_voice_that_said_little_is_still_its_own(home):
    found = [(0.0, 1.5, 0), (1.6, 1.95, 1)]  # «да»: 0.35 s
    embed = lambda x: ANYA if len(x) > speakers.RATE else KOSTYA  # noqa: E731
    s = Speakers({"sherpa": lambda x: found}, embed, Voices(home / "voices.json"))
    assert [v["id"] for _, _, v in s.parts(audio(2))] == [1, 2]


def test_a_phrase_is_cut_into_turns():
    anya, kostya = {"id": 1, "name": "Аня"}, {"id": 2, "name": None}
    parts = [(0.0, 1.0, anya), (1.2, 2.0, kostya), (2.2, 2.6, kostya), (2.4, 2.5, anya),
             (2.8, 3.5, anya), (3.4, 4.0, kostya)]  # fmt: skip
    assert speakers.turns(parts, 4.5) == [
        (anya, 0.0, 1.2),
        (kostya, 1.2, 2.8),  # his two parts in a row, and Аня's «м-м» said over him
        (anya, 2.8, 3.5),
        (kostya, 3.5, 4.5),  # from where she ended to the phrase's end
    ]
    assert speakers.turns([], 2.0) == []


def test_a_turn_is_cut_out_a_little_wider():
    whole = wav([b"\1\0" * speakers.RATE, b"\2\0" * speakers.RATE])  # two seconds
    with __import__("wave").open(__import__("io").BytesIO(speakers.clip(whole, 0.5, 1.5))) as w:
        frames = w.readframes(w.getnframes())
    assert len(frames) // 2 == int(1.3 * speakers.RATE)
    assert frames[:2] == b"\1\0" and frames[-2:] == b"\2\0"

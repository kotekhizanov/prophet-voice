import pytest

from prophet_voice import words

WAKE = ["пророк"]


def test_after_wake_takes_the_rest_of_the_phrase():
    assert words.after_wake("Пророк, что нового?", WAKE) == "что нового?"
    assert words.after_wake("«Пророк»: расскажи о погоде", WAKE) == "расскажи о погоде"
    assert words.after_wake("Про-рок, привет", WAKE) == "привет"  # Whisper split the word
    assert words.after_wake("Пророк.", WAKE) == ""


def test_after_wake_needs_the_wake_word_first():
    assert words.after_wake("Я говорил с пророком вчера", WAKE) is None
    assert words.after_wake("Пророки знают", WAKE) is None  # another word
    assert words.after_wake("Грок, что нового?", WAKE) is None


def test_e_and_yo_are_alike():
    assert words.after_wake("Ёжик, привет", ["ежик"]) == "привет"


def test_said_in_is_a_whole_word_anywhere():
    assert words.said_in("[unk] стоп [unk]", ["отмена", "стоп"])
    assert words.said_in("Ну всё, хватит уже!", ["хватит уже"])
    assert not words.said_in("стопка", ["стоп"])
    assert not words.said_in("", ["стоп"])


def test_cancelled_at_the_start_or_the_end():
    cancel = ["отмена"]
    assert words.cancelled("отмена", cancel)
    assert words.cancelled("Отмена, не надо", cancel)
    assert words.cancelled("покажи отчёт, отмена.", cancel)
    assert not words.cancelled("отменарий", cancel)
    assert not words.cancelled("что значит отмена закона в истории", cancel)


def test_check_trims_and_refuses():
    assert words.check("wake", [" Пророк ", "пророк", "оракул"]) == ["пророк", "оракул"]
    assert words.check("ack", ["  Понял. "]) == ["Понял."]
    with pytest.raises(ValueError, match="one Russian word"):
        words.check("wake", ["два слова"])
    with pytest.raises(ValueError, match="one Russian word"):
        words.check("wake", ["wake"])
    with pytest.raises(ValueError, match="from 1 to"):
        words.check("cancel", [])
    with pytest.raises(ValueError, match="no words"):
        words.check("nothing", ["да"])


def test_a_private_phrase_loses_prophets_cue_and_a_wake_word():
    kinds = words.defaults()
    assert words.alone("Ага? Пророк, какая погода?", kinds) == "какая погода?"
    assert words.alone("Агарь пришла", kinds) == "Агарь пришла"


def test_a_phrase_of_cues_alone_is_not_a_question():
    kinds = words.defaults()
    assert words.cue_only("Ага. Окей…", kinds) and words.cue_only("Ок.", kinds)
    assert not words.cue_only("Окей, а завтра?", kinds) and not words.cue_only("", kinds)

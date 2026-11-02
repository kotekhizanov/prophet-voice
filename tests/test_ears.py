import io
import wave

import numpy as np

from prophet_voice import ears as ears_
from prophet_voice.ears import BLOCK, RATE, Ears
from prophet_voice.words import defaults

SECONDS = BLOCK / RATE  # one block: 40 ms


def loud() -> bytes:
    t = np.arange(BLOCK) / RATE
    return (np.sin(2 * np.pi * 1000 * t) * 0.3 * 32767).astype(np.int16).tobytes()


def quiet() -> bytes:
    rng = np.random.default_rng(1)
    return (rng.normal(0, 0.0003, BLOCK) * 32767).astype(np.int16).tobytes()


class Recognizers:
    """Vosk as a script: what each recognizer says after its n-th block.

    `plan[kind][n]` is the words of an utterance it finishes there; kinds are
    "wake" (the wake word's grammar), "stops" and "whole" (no grammar, whose
    `final` is `plan["whole"]["final"]`)."""

    def __init__(self, plan):
        self.plan = plan
        self.made = []

    def __call__(self, grammar):
        kind = "whole" if grammar is None else "wake" if "пророк" in grammar else "stops"
        r = Recognizer(self.plan.get(kind, {}), kind)
        self.made.append(r)
        return r


class Recognizer:
    def __init__(self, plan, kind):
        self.plan, self.kind, self.n = plan, kind, 0

    def accept(self, block):
        self.n += 1
        return self.plan.get(self.n)

    def partial(self):
        return ""

    def final(self):
        return self.plan.get("final", [])


class Mouth:
    def __init__(self):
        self.speaking = False
        self.saying = ""


class Ears_:
    """Ears with recorded callbacks."""

    def __init__(self, plan, threshold=0.8):
        self.events = []
        self.mouth = Mouth()
        self.recognizers = Recognizers(plan)
        self.ears = Ears(
            defaults(),
            threshold,
            self.recognizers,
            self.mouth,
            on_wake=lambda: self.events.append(("wake",)),
            on_drop=lambda: self.events.append(("drop",)),
            on_phrase=lambda audio, free: self.events.append(("phrase", seconds(audio), free)),
            on_stop=self.stop,
        )

    def stop(self):
        self.events.append(("stop",))
        self.mouth.speaking = False

    def feed(self, block, n):
        for _ in range(n):
            self.ears.hear(block)


def seconds(audio: bytes) -> float:
    with wave.open(io.BytesIO(audio)) as w:
        return w.getnframes() / w.getframerate()


def word(w, start, end, conf=0.95):
    return {"word": w, "start": start, "end": end, "conf": conf}


def test_the_wake_word_and_a_phrase_after_it():
    # 1 s of quiet, then «Пророк, что нового?» from 1.0 to 2.0 s; Vosk ends the
    # utterance at block 50 (2.0 s).
    e = Ears_({"wake": {50: [word("пророк", 1.0, 1.4), word("[unk]", 1.4, 2.0)]}})
    e.feed(quiet(), 25)
    e.feed(loud(), 25)
    assert e.events == [("wake",)]
    e.feed(quiet(), 40)  # a pause of 1.4 s ends it
    assert e.events[1][0] == "phrase" and e.events[1][2] is False
    # From just before the wake word (0.8 s) to the end of the pause.
    assert 2.5 < e.events[1][1] < 2.8
    assert e.ears.state == "waiting"


def test_a_long_phrase_goes_on_until_the_pause():
    e = Ears_({"wake": {30: [word("пророк", 1.0, 1.2)]}})
    e.feed(quiet(), 25)
    e.feed(loud(), 5)  # «Пророк», then a breath
    e.feed(quiet(), 10)
    e.feed(loud(), 50)  # then the question, 2 s
    e.feed(quiet(), 30)
    assert e.events == [("wake",)]  # 1.2 s of quiet is not yet the end
    e.feed(quiet(), 10)
    assert [x[0] for x in e.events] == ["wake", "phrase"]


def test_a_doubtful_wake_word_wakes_nothing():
    e = Ears_({"wake": {50: [word("пророк", 1.0, 1.4, conf=0.6)]}})
    e.feed(quiet(), 25)
    e.feed(loud(), 25)
    e.feed(quiet(), 50)
    assert e.events == []


def test_the_wake_word_with_nothing_after_it_is_dropped():
    e = Ears_({"wake": {35: [word("пророк", 1.0, 1.4)]}})
    e.feed(quiet(), 25)
    e.feed(loud(), 10)
    assert e.events == [("wake",)]
    e.feed(quiet(), int(ears_.NO_PHRASE / SECONDS) + 5)
    assert e.events == [("wake",), ("drop",)]


def test_no_wake_word_while_prophet_speaks_nor_just_after():
    # Prophet says «Пророк» itself until 1.2 s. Vosk reports it while Prophet
    # speaks, then again just after (the echo), then an utterance that began
    # while it spoke: none wakes. The owner's own call at 2.4 s does.
    e = Ears_(
        {
            "wake": {
                25: [word("пророк", 0.5, 0.9)],
                33: [word("пророк", 1.0, 1.3)],
                45: [word("пророк", 1.1, 1.5)],
                75: [word("пророк", 2.4, 2.8), word("[unk]", 2.8, 3.0)],
            }
        }
    )
    e.mouth.speaking = True
    e.feed(quiet(), 5)
    e.feed(loud(), 25)
    e.mouth.speaking = False
    e.feed(quiet(), 30)
    assert e.events == []
    e.feed(loud(), 15)
    assert e.events == [("wake",)]


def test_a_stop_word_hushes_prophet_and_what_follows_is_the_next_question():
    # The stops' recognizer starts with the first block Prophet speaks (at 0 s);
    # it hears «хватит» from 1.0 to 1.4 s, and Vosk with its whole vocabulary too.
    e = Ears_(
        {
            "stops": {40: [word("хватит", 1.0, 1.4, conf=0.9)]},
            "whole": {"final": [word("хватит", 0.1, 0.5)]},
        }
    )
    e.mouth.speaking = True
    e.mouth.saying = "Сегодня солнечно и тепло"
    e.feed(quiet(), 25)
    e.feed(loud(), 15)
    assert e.events == [("stop",)]
    assert e.ears.state == "phrase"
    e.feed(loud(), 25)  # the next question, with no wake word
    e.feed(quiet(), 40)
    assert e.events[1][0] == "phrase" and e.events[1][2] is True


def test_a_stop_word_prophet_says_itself_is_not_a_stop():
    e = Ears_(
        {
            "stops": {40: [word("понял", 1.0, 1.4, conf=0.9)]},
            "whole": {"final": [word("понял", 0.1, 0.5)]},
        }
    )
    e.mouth.speaking = True
    e.mouth.saying = "Я понял ваш вопрос"
    e.feed(quiet(), 25)
    e.feed(loud(), 15)
    assert e.events == []


def test_a_stop_word_the_whole_vocabulary_does_not_hear_is_not_a_stop():
    e = Ears_(
        {
            "stops": {40: [word("хватит", 1.0, 1.4, conf=0.9)]},
            "whole": {"final": [word("кватро", 0.1, 0.5)]},
        }
    )
    e.mouth.speaking = True
    e.feed(quiet(), 25)
    e.feed(loud(), 15)
    assert e.events == []


def test_listen_free_hears_an_answer_without_the_wake_word():
    e = Ears_({})
    e.feed(quiet(), 10)
    e.ears.listen_free(8)
    e.feed(loud(), 25)
    e.feed(quiet(), 40)
    assert e.events[0][0] == "phrase" and e.events[0][2] is True


def test_listen_free_with_nothing_said_ends_quietly():
    e = Ears_({})
    e.feed(quiet(), 10)
    e.ears.listen_free(2)
    e.feed(quiet(), int(2 / SECONDS) + 5)
    assert e.events == []
    assert e.ears.state == "waiting"


def test_vosk_gets_its_grammar_in_cyrillic():
    made = []

    class Vosk:
        class KaldiRecognizer:
            def __init__(self, *args):
                made.append(args)

            def SetWords(self, on):  # noqa: N802 - Vosk's name
                pass

    ears_.VoskRecognizer(Vosk, "model", ["пророк", "[unk]"])
    ears_.VoskRecognizer(Vosk, "model", None)
    assert made == [("model", RATE, '["пророк", "[unk]"]'), ("model", RATE)]


def test_the_ring_keeps_only_the_latest_seconds():
    e = Ears_({})
    e.feed(quiet(), int(ears_.RING / SECONDS) + 50)
    first = e.ears.recent[0][0]
    assert e.ears.clock - first <= ears_.RING + SECONDS

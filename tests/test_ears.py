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
    "wake" (the wake word's grammar) and "whole" (no grammar, whose
    `final` is `plan["whole"]["final"]`)."""

    def __init__(self, plan):
        self.plan = plan
        self.made = []

    def __call__(self, grammar):
        kind = "whole" if grammar is None else "wake"
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
            on_begun=lambda: self.events.append(("begun",)),
        )

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


def test_no_wake_word_just_after_prophet_speaks():
    # Prophet speaks until 1.2 s; the echo of its «Пророк» that Vosk reports
    # just after does not wake. The owner's own call a second later does.
    e = Ears_(
        {
            "wake": {
                5: [word("пророк", 0.0, 0.2)],
                40: [word("пророк", 1.0, 1.4), word("[unk]", 1.4, 1.5)],
            }
        }
    )
    e.mouth.speaking = True
    e.feed(loud(), 30)
    e.mouth.speaking = False
    e.feed(loud(), 10)
    assert e.events == []
    e.feed(quiet(), 30)
    assert e.events == [("wake",)]


def test_while_prophet_speaks_the_ears_do_not_listen():
    # Vosk is given nothing then; after it, a new wake recognizer counts its
    # word times from where it began, so the owner's «Пророк» at 0.4 s of it wakes.
    e = Ears_({"wake": {20: [word("пророк", 0.4, 0.7), word("[unk]", 0.7, 0.8)]}})
    e.mouth.speaking = True
    e.feed(loud(), 50)
    assert len(e.recognizers.made) == 1 and e.recognizers.made[0].n == 0
    assert e.ears.level == 0 and not e.ears.listening
    e.mouth.speaking = False
    e.feed(quiet(), 20)
    assert e.events == [("wake",)]
    assert len(e.recognizers.made) == 2


def test_a_phrase_begun_goes_on_while_prophet_speaks():
    # The tone after the wake word sounds while the owner goes on speaking.
    e = Ears_({"wake": {35: [word("пророк", 1.1, 1.4)]}})
    e.feed(quiet(), 25)
    e.feed(loud(), 10)
    assert e.events == [("wake",)]
    e.mouth.speaking = True
    assert e.ears.listening
    e.feed(loud(), 10)
    e.mouth.speaking = False
    e.feed(loud(), 10)
    e.feed(quiet(), 40)
    assert e.events[1][0] == "phrase" and e.events[1][1] > 0.8


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


def test_alone_speech_begins_a_phrase_once_a_word_is_heard():
    # Private: no wake word. Speech from 1 s; 0.36 s of it begin a tentative phrase
    # from 0.4 s (0.6 s kept from before), whose 24 blocks the whole vocabulary hears
    # at once; it hears a word in the 30th.
    e = Ears_({"whole": {30: [word("какая", 0.2, 0.6)]}})
    e.ears.be_alone(True)
    e.feed(quiet(), 25)
    e.feed(loud(), 9)
    assert e.ears.state == "phrase" and e.ears.tentative and e.events == []
    e.feed(loud(), 20)
    assert e.events == [("begun",)] and not e.ears.tentative
    e.feed(quiet(), 40)
    assert e.events[1][0] == "phrase" and e.events[1][2] is True
    # From 0.4 s to the end of the pause, at about 3.6 s.
    assert 2.9 < e.events[1][1] < 3.4


def test_alone_a_sound_with_no_word_is_dropped_unsaid():
    e = Ears_({"whole": {"final": [word("а", 0.1, 0.2)]}})  # a door: no word Vosk believes
    e.ears.be_alone(True)
    e.feed(quiet(), 25)
    e.feed(loud(), 10)
    e.feed(quiet(), 40)
    assert e.events == [] and e.ears.state == "waiting"
    e.feed(loud(), int(ears_.CONFIRM_WITHIN / SECONDS) + 10)  # a long rumble
    assert e.events == []  # each try at a phrase in it was dropped
    assert e.ears.state == "waiting" or e.ears.tentative


def test_alone_nothing_begins_while_prophet_speaks():
    e = Ears_({"whole": {1: [word("какая", 0.2, 0.6)]}})
    e.ears.be_alone(True)
    e.mouth.speaking = True
    e.feed(quiet(), 25)
    e.feed(loud(), 25)
    assert e.ears.state == "waiting" and e.events == []


def test_the_level_is_the_voice_above_the_quiet():
    e = Ears_({})
    e.feed(quiet(), 25)
    assert e.ears.level == 0
    e.feed(loud(), 2)
    assert e.ears.level == 1


def test_is_speech():
    assert ears_.is_speech("какая") and ears_.is_speech("да да")
    assert not ears_.is_speech("а") and not ears_.is_speech("")


class Microphone:
    """sounddevice as a script: a stream that sends `blocks` loud blocks once started."""

    def __init__(self, blocks=3, fails=False):
        self.blocks, self.fails = blocks, fails
        self.opened = self.closed = 0

    def RawInputStream(self, callback, **kwargs):  # noqa: N802 - sounddevice's name
        if self.fails:
            raise OSError("no microphone")
        mic = self

        class Stream:
            def start(self):
                mic.opened += 1
                for _ in range(mic.blocks):
                    callback(loud(), BLOCK, None, None)

            def close(self):
                mic.closed += 1

        return Stream()


class Heard:
    listening = True

    def __init__(self):
        self.blocks = 0
        self.resets = 0

    def hear(self, block):
        assert len(block) == 2 * BLOCK
        self.blocks += 1

    def reset(self):
        self.resets += 1


def run_listen(mic, monkeypatch, until, on_from_start=True, failed=None):
    import sys
    import threading
    import time

    monkeypatch.setitem(sys.modules, "sounddevice", mic)
    heard, on, stop = Heard(), threading.Event(), threading.Event()
    if on_from_start:
        on.set()
    t = threading.Thread(target=ears_.listen, args=(heard, "", on, stop), kwargs={"failed": failed})
    t.start()
    end = time.monotonic() + 3
    while not until(heard, on) and time.monotonic() < end:
        time.sleep(0.01)
    on.clear()
    time.sleep(0.3)
    stop.set()
    t.join(3)
    return heard


def test_the_microphone_is_open_only_while_asked(monkeypatch):
    mic = Microphone(blocks=3)
    heard = run_listen(mic, monkeypatch, until=lambda h, on: h.blocks == 3)
    assert heard.blocks == 3  # each block cleaned, as long as it was
    assert (mic.opened, mic.closed, heard.resets) == (1, 1, 1)


def test_the_microphone_stays_closed_until_asked(monkeypatch):
    mic = Microphone()
    heard = run_listen(mic, monkeypatch, until=lambda h, on: False, on_from_start=False)
    assert (mic.opened, heard.blocks) == (0, 0)


def test_a_microphone_that_will_not_open_is_said_and_not_asked_again(monkeypatch):
    failed = []
    heard = run_listen(
        Microphone(fails=True), monkeypatch, until=lambda h, on: failed, failed=failed.append
    )
    assert [str(e) for e in failed] == ["no microphone"]
    assert heard.blocks == 0


def test_reset_forgets_what_was_heard():
    e = Ears_({})
    e.ears.be_alone(True)
    e.feed(quiet(), 25)
    e.feed(loud(), 12)
    assert e.ears.state == "phrase"
    e.ears.reset()
    assert e.ears.state == "waiting" and not e.ears.phrase and not e.ears.recent
    assert e.ears.level == 0.0

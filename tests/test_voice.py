import asyncio
import os
import signal
import time

import pytest

from prophet_voice import config, voice
from prophet_voice.core import CoreError
from prophet_voice.ears import wav as ears_wav
from prophet_voice.voice import Voice

KEY = "pk_" + "a" * 43


class Core:
    def __init__(self, events=(), plugin=None):
        self.calls = []
        self.events_of = {}
        self.default_events = list(events)
        self.plugin_settings = plugin

    async def plugin(self):
        if self.plugin_settings is None:
            return None
        return {"plugin": "voice", "enabled": True, "settings": self.plugin_settings}

    async def chat(self, text, project):
        self.calls.append(("chat", text, project))
        return 5

    async def answer(self, question_id, text):
        self.calls.append(("answer", question_id, text))

    async def cancel(self, run_id):
        self.calls.append(("cancel", run_id))

    async def events(self, run_id):
        for event in self.events_of.get(run_id, self.default_events):
            await asyncio.sleep(0)
            if isinstance(event, Exception):
                raise event
            yield event


class Mouth:
    def __init__(self):
        self.said = []
        self.speaking = False
        self.saying = ""
        self.own = []  # the pieces read lately
        self.waits = 0

    def say(self, text):
        self.said.append(text)

    def tone(self, up):
        self.said.append("↑" if up else "↓")

    def cue(self, text, volume):
        self.said.append((text, volume))

    def hush(self):
        self.said.append("hush")

    def wait(self, timeout=120):
        self.waits += 1

    def lately(self, seconds):
        return self.own


class Hearing:
    def __init__(self, *texts):
        self.texts = list(texts)
        self.settings = None

    def configure(self, settings):
        self.settings = settings

    def transcribe(self, audio):
        """The next text; a dict of them gives each the audio whose middle sample
        is its key (each second of a test phrase says which it is)."""
        said = self.texts[0]
        if isinstance(said, dict):
            middle = (len(audio) - 44) // 4 * 2 + 44
            return said[int.from_bytes(audio[middle : middle + 2], "little")]
        return self.texts.pop(0)


class Ears:
    def __init__(self):
        self.free = []
        self.alone = False
        self.state = "waiting"
        self.tentative = False
        self.threshold = None
        self.text = ""  # the phrase so far, as Vosk has it

    def configure(self, words, threshold):
        self.words, self.threshold = words, threshold

    def listen_free(self, seconds):
        self.free.append(seconds)

    def be_alone(self, on):
        self.alone = on

    def finish(self, on=True):
        self.finishing = on


def make(core, *texts, project=""):
    settings = {**config.load(), "project": project}
    v = Voice(settings, core, Mouth(), Hearing(*texts))
    v.ears = Ears()
    return v


async def settle(v):
    for _ in range(20):
        await asyncio.sleep(0)
    await asyncio.gather(*v.followers)


def test_a_phrase_goes_to_the_chat_and_the_answer_is_read():
    core = Core(
        [
            ("step", {"id": 1, "run_id": 5}),
            ("message", {"run_id": 9, "role": "assistant", "content": "Другой чат."}),
            ("message", {"run_id": 5, "role": "user", "content": "какая погода?"}),
            ("message", {"run_id": 5, "role": "assistant", "content": "Солнечно."}),
            ("run", {"run_id": 5, "status": "done"}),
        ]
    )
    v = make(core, "Пророк, какая погода?", project="дом")

    async def go():
        await v.handle(b"", False)
        await settle(v)

    asyncio.run(go())
    assert core.calls == [("chat", "какая погода?", "дом")]
    assert v.mouth.said == ["Понял.", "Солнечно."]
    assert v.run_id is None


def test_a_phrase_without_the_wake_word_is_dropped():
    core = Core()
    v = make(core, "Грок, какая погода?")
    asyncio.run(v.handle(b"", False))
    assert core.calls == []
    assert v.mouth.said == ["↓"]


def test_a_cancel_word_cancels_the_answer_being_prepared():
    core = Core()
    v = make(core, "Пророк, отмена")
    v.run_id = 5
    asyncio.run(v.handle(b"", False))
    assert core.calls == [("cancel", 5)]
    assert v.mouth.said == ["hush", "↓"]


def test_a_cancel_word_over_prophet_hushes_it_and_cancels_the_answer():
    core = Core()
    v = make(core)
    v.run_id = 5

    async def go():
        v.loop = asyncio.get_running_loop()
        await asyncio.to_thread(v.on_stop)  # from the ears' thread
        await asyncio.sleep(0.05)

    asyncio.run(go())
    assert v.mouth.said == ["hush", "↓"]
    assert core.calls == [("cancel", 5)]


def test_a_question_is_read_and_the_next_phrase_answers_it():
    core = Core(
        [
            ("question", {"id": 3, "run_id": 5, "question": "За какой месяц?"}),
            ("run", {"run_id": 5, "status": "waiting"}),
        ]
    )
    v = make(core, "Пророк, покажи продажи", "за октябрь")

    async def go():
        await v.handle(b"", False)
        await settle(v)
        assert v.ears.free == [voice.AFTER_QUESTION]
        await v.handle(b"", True)  # no wake word: the answer

    asyncio.run(go())
    assert core.calls == [("chat", "покажи продажи", ""), ("answer", 3, "за октябрь")]
    assert v.mouth.said == ["Понял.", "За какой месяц?", "Понял."]
    assert v.question is None


def test_a_free_phrase_with_no_question_is_a_new_message():
    core = Core([("run", {"run_id": 5, "status": "done"})])
    v = make(core, "а завтра?")

    async def go():
        await v.handle(b"", True)
        await settle(v)

    asyncio.run(go())
    assert core.calls == [("chat", "а завтра?", "")]


def test_a_lost_run_is_said():
    core = Core([CoreError("lost the core")])
    v = make(core, "Пророк, привет")

    async def go():
        await v.handle(b"", False)
        await settle(v)

    asyncio.run(go())
    assert v.mouth.said == ["Понял.", voice.SAY_LOST]


def test_a_refusal_is_said_and_the_voice_goes_on():
    class Refusing(Core):
        async def chat(self, text, project):
            raise CoreError('this key has no "chat" scope', 403)

    v = make(Refusing(), "Пророк, привет", "Пророк, ещё раз")

    async def go():
        server = asyncio.create_task(v.serve())
        await asyncio.sleep(0)
        v.on_phrase(b"", False)
        v.on_phrase(b"", False)
        for _ in range(50):
            await asyncio.sleep(0.01)
            if len(v.mouth.said) == 2:
                break
        server.cancel()
        with pytest.raises(asyncio.CancelledError):
            await server

    asyncio.run(go())
    assert v.mouth.said == [voice.SAY_FAILED, voice.SAY_FAILED]


def test_ctrl_c_cleans_up_and_does_not_wait_for_threads():
    cleaned = []

    async def main():
        loop = asyncio.get_running_loop()
        loop.run_in_executor(None, time.sleep, 3)  # Whisper busy on a phrase
        try:
            await asyncio.sleep(0.05)
            raise KeyboardInterrupt  # what Ctrl+C raises in the loop
        finally:
            cleaned.append("hushed")

    t = time.monotonic()
    assert voice.run_until_stopped(main()) == 0
    assert time.monotonic() - t < 1
    assert cleaned == ["hushed"]


def test_ctrl_c_while_waiting_cancels_and_cleans_up():
    cleaned = []

    async def main():
        try:
            await asyncio.sleep(60)
        finally:
            cleaned.append("hushed")

    async def interrupt():
        await asyncio.sleep(0.05)
        raise KeyboardInterrupt

    async def both():
        asyncio.get_running_loop().create_task(interrupt())
        await main()

    assert voice.run_until_stopped(both()) == 0
    assert cleaned == ["hushed"]


def test_a_refusal_to_start_is_said_with_code_1(capsys):
    async def main():
        raise SystemExit("no client key: set one with `prophet-voice settings`")

    assert voice.run_until_stopped(main()) == 1
    assert "no client key" in capsys.readouterr().err


class Door:
    """A core as run() first sees it: the key, and the plugin's state."""

    def __init__(self, scopes=("chat",), plugin=None, error=None):
        self.scopes, self.plugin_is, self.error = list(scopes), plugin, error

    async def me(self):
        if self.error:
            raise self.error
        return {"name": "voice", "scopes": self.scopes}

    async def plugin(self):
        return self.plugin_is


def test_sigterm_is_taken_as_ctrl_c():
    cleaned = []

    async def main():
        try:
            os.kill(os.getpid(), signal.SIGTERM)  # a node stops the voice
            await asyncio.sleep(5)
        finally:
            cleaned.append("hushed")

    before = signal.getsignal(signal.SIGTERM)
    assert voice.run_until_stopped(main()) == 0
    assert cleaned == ["hushed"]
    assert signal.getsignal(signal.SIGTERM) is before


def test_with_no_key_the_voice_waits_for_one(monkeypatch, capsys):
    monkeypatch.setattr(voice, "KEY_WAIT", 0.01)
    let_in = []

    class Let(Door):
        def __init__(self, url, key):
            super().__init__()
            let_in.append(key)

        async def close(self):
            pass

    monkeypatch.setattr(voice, "Core", Let)

    async def go():
        waiting = asyncio.create_task(voice.let_in(config.load()))
        await asyncio.sleep(0.05)
        assert not waiting.done()
        config.save({"key": KEY})  # Omniscient hands the plugin its key
        return await asyncio.wait_for(waiting, 2)

    settings, core, me, plugin = asyncio.run(go())
    assert settings["key"] == KEY and let_in == [KEY] and me["name"] == "voice"
    assert "Settings → Plugins → Add plugin, https://github.com/" in capsys.readouterr().out


def test_a_key_the_core_does_not_know_waits_for_a_new_one(monkeypatch, capsys):
    monkeypatch.setattr(voice, "KEY_WAIT", 0.01)
    revoked, fresh = "pk_" + "r" * 43, KEY
    config.save({"key": revoked})
    closed = []

    class Let(Door):
        def __init__(self, url, key):
            unknown = CoreError("the core does not know this key", 401)
            super().__init__(error=unknown if key == revoked else None)

        async def close(self):
            closed.append(True)

    monkeypatch.setattr(voice, "Core", Let)

    async def go():
        waiting = asyncio.create_task(voice.let_in(config.load()))
        await asyncio.sleep(0.05)
        assert not waiting.done() and closed == [True]
        config.save({"key": fresh})
        return await asyncio.wait_for(waiting, 2)

    settings, _, me, _ = asyncio.run(go())
    assert settings["key"] == fresh and me["name"] == "voice"
    assert "Waiting for a new key from Omniscient" in capsys.readouterr().out


def test_the_voice_is_let_in_with_the_chat_scope():
    me, plugin = asyncio.run(voice.admitted(Door(), "http://core"))
    assert me["name"] == "voice" and plugin is None
    on = {"plugin": "voice", "enabled": True, "settings": {"stt": "groq"}}
    me, plugin = asyncio.run(voice.admitted(Door(plugin=on), "http://core"))
    assert me["name"] == "voice" and plugin == on


def test_the_voice_does_not_start_when_it_cannot_chat():
    with pytest.raises(SystemExit, match='no "chat" scope'):
        asyncio.run(voice.admitted(Door(scopes=["tasks"]), "http://core"))
    with pytest.raises(SystemExit, match="http://core: the core does not answer"):
        error = CoreError("the core does not answer: refused")
        asyncio.run(voice.admitted(Door(error=error), "http://core"))
    with pytest.raises(CoreError):  # an unknown key: a new one may come
        error = CoreError("the core does not know this key", 401)
        asyncio.run(voice.admitted(Door(error=error), "http://core"))


def test_a_plugin_switched_off_in_omniscient_does_not_start():
    off = {"plugin": "voice", "enabled": False, "settings": {}}
    with pytest.raises(SystemExit, match="switched off in Omniscient"):
        asyncio.run(voice.admitted(Door(plugin=off), "http://core"))


def test_a_dialog_greets_and_its_end_hushes():
    v = make(Core())
    v.set_dialog("casual", 3)
    assert v.mouth.said == ["Привет!"] and not v.ears.alone
    v.set_dialog("private", 3)  # the other button: no second greeting
    assert v.mouth.said == ["Привет!"] and v.ears.alone
    v.set_dialog(None)
    assert v.mouth.said == ["Привет!", "hush"] and not v.ears.alone
    assert v.dialog is None and v.dialog_project is None


def test_a_dialog_asks_in_the_chat_open_in_omniscient():
    core = Core([("run", {"run_id": 5, "status": "done"})])
    v = make(core, "Пророк, привет", project="дом")
    v.set_dialog("casual", 7)

    async def go():
        await v.handle(b"", False)
        await settle(v)

    asyncio.run(go())
    assert core.calls == [("chat", "привет", 7)]
    assert v.mouth.said == ["Привет!", "Понял."]


def test_private_needs_no_wake_word_and_cues_instead_of_ack():
    core = Core([("run", {"run_id": 5, "status": "done"})])
    v = make(core, "Ага? Какая погода завтра?")
    v.set_dialog("private")

    async def go():
        v.loop = asyncio.get_running_loop()
        v.on_begun()
        v.on_phrase(b"", True)
        await v.handle(*await v.phrases.get())
        await settle(v)

    asyncio.run(go())
    assert core.calls == [("chat", "Какая погода завтра?", None)]
    assert v.mouth.said == ["Привет!", ("Ага?", 0.5), ("Окей…", 0.9)]


def test_private_drops_a_phrase_of_its_own_cues():
    core = Core()
    v = make(core, "Окей.")
    v.set_dialog("private")
    asyncio.run(v.handle(b"", True))
    assert core.calls == []


def test_prophet_does_not_talk_over_the_owner():
    core = Core(
        [
            ("message", {"run_id": 5, "role": "assistant", "content": "Солнечно."}),
            ("run", {"run_id": 5, "status": "done"}),
        ]
    )
    v = make(core, "Пророк, какая погода?")

    async def go():
        v.ears.state = "phrase"  # the owner speaks again
        await v.handle(b"", False)
        await asyncio.sleep(0.3)
        assert v.mouth.said == ["Понял."]
        v.ears.state = "waiting"
        await settle(v)

    asyncio.run(go())
    assert v.mouth.said == ["Понял.", "Солнечно."]


def test_the_state_the_picture_shows():
    v = make(Core())
    assert v.state() == "waiting"
    v.run_id = 5
    assert v.state() == "thinking"
    v.mouth.speaking = True
    assert v.state() == "speaking"
    v.hearing_now = 1
    assert v.state() == "hearing"
    v.ears.state = "phrase"
    v.ears.tentative = True  # a sound with no word heard in it yet
    assert v.state() == "hearing"
    v.ears.tentative = False
    assert v.state() == "listening"
    v.ears = None
    assert v.state() == "starting"


def test_the_microphone_is_open_only_in_a_private_dialog():
    v = make(Core())
    assert not v.microphone.is_set()
    v.set_dialog("casual", 3)
    assert not v.microphone.is_set()
    v.set_dialog("private", 3)
    assert v.microphone.is_set()
    v.set_dialog("private", 4)  # another chat: the dialog goes on
    assert v.microphone.is_set() and "hush" not in v.mouth.said
    v.set_dialog(None)
    assert not v.microphone.is_set()


def test_ending_the_dialog_ends_everything_at_once():
    core = Core()

    async def endless(run_id):
        await asyncio.sleep(60)
        yield "message", {"id": 1, "run_id": run_id, "role": "assistant", "content": "Поздно."}

    core.events = endless
    v = make(core, "Какая погода?")

    async def go():
        v.loop = asyncio.get_running_loop()
        v.set_dialog("private")
        v.follow(5)
        following = list(v.followers)
        v.question = 8
        v.on_phrase(b"", True)  # heard before the end, sent after it
        await asyncio.sleep(0)
        v.set_dialog(None)
        await v.handle(*await v.phrases.get())
        await asyncio.sleep(0)
        return following

    assert all(task.cancelled() for task in asyncio.run(go()))
    assert core.calls == []  # the phrase heard in the dialog is not sent
    assert v.question is None and v.run_id is None
    assert v.mouth.said[-1] == "hush" and "Поздно." not in v.mouth.said


def test_a_phrase_being_heard_when_the_dialog_ends_is_dropped():
    v = make(Core(), "Какая погода?")
    v.set_dialog("private")
    era = v.era
    v.hearing.transcribe = lambda audio: (v.set_dialog(None), "Какая погода?")[1]
    asyncio.run(v.handle(b"", True, era))
    assert v.core.calls == []


def test_a_message_is_read_once():
    said = {"id": 11, "run_id": 5, "role": "assistant", "content": "Хорошо."}
    core = Core([("message", said), ("message", said), ("run", {"run_id": 5, "status": "done"})])
    v = make(core)

    async def go():
        v.follow(5)
        await settle(v)

    asyncio.run(go())
    assert v.mouth.said == ["Хорошо."]


def test_a_microphone_that_will_not_open_ends_the_dialog():
    v = make(Core())

    async def go():
        v.loop = asyncio.get_running_loop()
        v.set_dialog("private")
        v.on_microphone_failed(OSError("no microphone"))
        await asyncio.sleep(0)

    asyncio.run(go())
    assert v.dialog is None and not v.microphone.is_set()
    assert v.mouth.said[-1] == voice.SAY_NO_MIC


def test_only_one_voice_runs_on_this_machine():
    first = voice.only_one()
    try:
        with pytest.raises(SystemExit, match="running already"):
            voice.only_one()
    finally:
        first.close()
    voice.only_one().close()  # the first one has ended: the next starts


def test_a_dialog_begins_with_what_the_owner_set_since_the_start(home):
    v = make(Core(plugin={"whisper_model": "medium", "threshold": 0.9}))
    # This Mac's settings, set in Omniscient meanwhile.
    config.save({"groq_key": "gsk_test", "microphone": "USB", "words": {"hello": ["Слушаю."]}})

    async def go():
        v.loop = asyncio.get_running_loop()
        v.set_dialog("private")
        assert v.device == "USB"  # before the microphone opens
        await v.beginning

    asyncio.run(go())
    assert v.hearing.settings["groq_key"] == "gsk_test"
    assert v.hearing.settings["whisper_model"] == "medium"  # the core's
    assert v.ears.threshold == 0.9 and v.ears.words["hello"] == ["Слушаю."]
    assert v.mouth.said == ["Слушаю."]  # greeted with the words set now


def test_a_dialog_ended_before_its_settings_are_read_is_not_greeted():
    v = make(Core(plugin={"whisper_model": "medium"}))

    async def go():
        v.loop = asyncio.get_running_loop()
        v.set_dialog("private")
        v.set_dialog(None)
        await v.beginning

    asyncio.run(go())
    assert "Привет!" not in v.mouth.said


class Speakers:
    """Who speaks, as speakers.Speakers says: the parts of each phrase, in turn."""

    def __init__(self, *phrases):
        from prophet_voice.speakers import Voices

        self.voices = Voices(config.home() / "voices.json")
        self.phrases = list(phrases)

    def parts(self, audio):
        return [(a, b, self.voices.who(p)) for a, b, p in self.phrases.pop(0)]


def private_with_voices(core, speakers, *texts):
    v = make(core, *texts)
    v.speakers = speakers
    v.set_dialog("private")
    return v


ANYA, KOSTYA = [1.0, 0.0], [0.0, 1.0]


def test_private_with_one_voice_says_nothing_of_whose_it_is():
    core = Core([("run", {"run_id": 5, "status": "done"})])
    v = private_with_voices(core, Speakers([(0.0, 2.0, ANYA)]), "Какая погода?")
    asyncio.run(v.handle(b"", True))
    assert core.calls == [("chat", "Какая погода?", None)]


def test_private_with_two_voices_says_who_said_what():
    core = Core([("run", {"run_id": 5, "status": "done"})])
    phrase = ears_wav([n.to_bytes(2, "little") * 16000 for n in (1, 2, 3, 4)])  # four seconds
    parts = [(0.0, 1.0, ANYA), (1.0, 2.0, KOSTYA), (2.0, 3.0, ANYA), (3.0, 4.0, KOSTYA)]
    # Each turn goes to Whisper on its own; the whole phrase too, while it is split.
    texts = {1: "Привет", 2: "Привет", 3: "Как дела?", 4: "Хорошо"}
    v = private_with_voices(core, Speakers(parts, [(0.0, 1.0, ANYA)]), texts)

    async def go():
        await v.handle(phrase, True)
        v.speakers.voices.name(1, "Аня")
        v.hearing.texts = ["Ага? И отмена будильника."]
        await v.handle(phrase, True)  # one voice now, but the dialog has heard two

    asyncio.run(go())
    assert core.calls[0] == (
        "chat",
        "Голос 1: Привет\nГолос 2: Привет\nГолос 1: Как дела?\nГолос 2: Хорошо",
        None,
    )
    assert core.calls[1] == ("chat", "Аня: И отмена будильника.", None)
    assert [s["label"] for s in v.speakers.voices.shown()] == ["Аня", "Голос 2"]


def test_voices_are_not_told_apart_when_switched_off():
    core = Core([("run", {"run_id": 5, "status": "done"})])
    v = private_with_voices(core, Speakers(), "Какая погода?")
    v.tell_apart = False
    asyncio.run(v.handle(b"", True))  # Speakers() has no phrase: it is not asked
    assert core.calls == [("chat", "Какая погода?", None)]


def test_the_diarizer_is_the_one_set_as_a_dialog_begins():
    v = private_with_voices(Core(), Speakers())
    v.speakers.use = "sherpa"
    v.configure({**config.load(), "diarizer": "nemotron"})
    assert v.speakers.use == "nemotron"


def test_a_new_dialog_hears_its_voices_anew():
    v = private_with_voices(Core(), Speakers())
    v.speakers.voices.who(ANYA)
    v.set_dialog(None)
    v.set_dialog("private")
    assert v.speakers.voices.shown() == []


SECOND = ears_wav([bytes(32000)])  # a second of silence, as the ears give it


def private(core, *texts):
    v = make(core, *texts)
    v.set_dialog("private")
    v.mouth.said.clear()
    return v


def test_a_phrase_said_over_prophet_goes_once_prophet_has_finished():
    core = Core([("run", {"run_id": 5, "status": "done"})])
    v = private(core, "Погоди, а что с погодой?")
    v.mouth.own = ["Сегодня курс такой-то."]

    async def go():
        v.loop = asyncio.get_running_loop()
        v.on_phrase(SECOND, True, True)
        await v.handle(*await v.phrases.get())
        await settle(v)

    asyncio.run(go())
    assert v.mouth.said == [] and v.mouth.waits == 1  # no «Окей…»: it would be late
    assert core.calls == [("chat", "Погоди, а что с погодой?", None)]
    assert v.live == ""


def test_prophets_own_words_heard_back_are_not_sent():
    core = Core()
    v = private(core, "сегодня курс доллара вырос")
    v.mouth.own = ["Сегодня курс доллара вырос на два рубля.", "Завтра будет ясно."]
    asyncio.run(v.handle(SECOND, True, None, True))
    assert core.calls == [] and v.mouth.said == [] and v.mouth.waits == 0


def test_a_phrase_is_heard_part_by_part_as_it_is_said():
    core = Core([("run", {"run_id": 5, "status": "done"})])
    v = private(core, "Какая погода", "завтра в Тбилиси?")
    v.ears.state, v.ears.text = "phrase", "ага какая"

    async def go():
        v.loop = asyncio.get_running_loop()
        assert v.showing() == "какая"  # Vosk's words at once, without Prophet's own «Ага?»
        v.on_part(SECOND, 3)  # a pause: the part before it goes to Whisper now
        v.parts[0][1].result(2)
        v.ears.text = "ага какая погода завтра"
        assert v.showing() == "Какая погода завтра"  # Whisper's, then Vosk's of the rest
        v.on_part(SECOND, 6)  # the rest of it, as the phrase ends
        v.on_phrase(SECOND + SECOND, True)
        v.ears.state, v.ears.text = "waiting", ""
        assert v.parts == [] and v.showing() == "Какая погода завтра"  # kept till it is sent
        await v.handle(*await v.phrases.get())
        await settle(v)

    asyncio.run(go())
    assert v.hearing.texts == []  # each part once, and not the whole phrase again
    assert core.calls == [("chat", "Какая погода завтра в Тбилиси?", None)]
    assert v.showing() == ""


def test_a_part_whisper_could_not_hear_has_the_whole_phrase_heard():
    core = Core([("run", {"run_id": 5, "status": "done"})])
    v = private(core, "Какая погода завтра?")
    heard, v.hearing.transcribe = v.hearing.transcribe, None  # the part's hearing fails

    async def go():
        v.loop = asyncio.get_running_loop()
        v.on_part(SECOND, 2)
        v.hearing.transcribe = heard
        v.on_phrase(SECOND, True)
        await v.handle(*await v.phrases.get())
        await settle(v)

    asyncio.run(go())
    assert core.calls == [("chat", "Какая погода завтра?", None)]


def test_a_dropped_phrase_leaves_no_parts():
    v = private(Core(), "Какая")
    v.on_part(SECOND, 1)
    v.on_part(None, 0)
    assert v.parts == []


def test_a_dictated_phrase_is_the_pages_and_not_the_chats():
    core = Core()
    v = make(core, "Найди, почему упали заказы.", "И пришли отчёт.")

    async def go():
        v.loop = asyncio.get_running_loop()
        v.set_dictation("page-one-1")
        assert v.microphone.is_set() and v.ears.alone and v.dictating
        v.on_begun()  # no «Ага?»: nothing is said aloud
        v.on_phrase(b"", True)
        await v.handle(*await v.phrases.get())
        v.set_dictation(None)  # the phrase being said is finished, and still comes
        assert not v.microphone.is_set() and v.ears.finishing and not v.dictating
        v.on_phrase(b"", True)
        await v.handle(*await v.phrases.get())
        await settle(v)

    asyncio.run(go())
    assert v.dictation["phrases"] == ["Найди, почему упали заказы.", "И пришли отчёт."]
    assert core.calls == [] and v.mouth.said == ["hush"]


def test_a_dictation_is_shown_as_it_is_said():
    v = make(Core())
    v.set_dictation("page-one-1")
    v.ears.state, v.ears.text = "phrase", "найди почему"
    assert v.showing() == "найди почему"


def test_no_dictation_in_a_dialog_and_a_dialog_ends_one():
    v = make(Core(), "Что нового?")

    async def go():
        v.loop = asyncio.get_running_loop()
        v.set_dialog("private")
        v.set_dictation("page-one-1")
        assert v.dictation is None
        v.set_dialog(None)
        v.set_dictation("page-one-1")
        assert v.dictating
        v.set_dialog("casual")  # a Casual dialog has no microphone
        assert v.dictation is None and not v.microphone.is_set() and not v.ears.alone
        v.set_dialog(None)
        v.set_dictation("page-one-1")
        was = v.dictation
        v.on_phrase(b"", True)
        v.set_dialog("private")  # what was being dictated is dropped
        await v.handle(*await v.phrases.get())
        assert was["phrases"] == [] and v.microphone.is_set() and v.ears.alone
        if v.beginning is not None:
            await v.beginning

    asyncio.run(go())


def test_a_page_dictates_again_into_the_same_dictation():
    v = make(Core())
    v.set_dictation("page-one-1")
    first = v.dictation
    v.set_dictation(None)
    v.set_dictation("page-one-1")
    assert v.dictation is first and first["on"]  # a late phrase of it still comes
    v.set_dictation(None)
    v.set_dictation("page-two-2")
    assert v.dictation is not first and v.dictation["n"] == first["n"] + 1

import asyncio
import time

import pytest

from prophet_voice import config, voice
from prophet_voice.core import CoreError
from prophet_voice.voice import Voice


class Core:
    def __init__(self, events=()):
        self.calls = []
        self.events_of = {}
        self.default_events = list(events)

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

    def say(self, text):
        self.said.append(text)

    def tone(self, up):
        self.said.append("↑" if up else "↓")

    def hush(self):
        self.said.append("hush")

    def wait(self, timeout=120):
        pass


class Hearing:
    def __init__(self, *texts):
        self.texts = list(texts)

    def transcribe(self, audio):
        return self.texts.pop(0)


class Ears:
    def __init__(self):
        self.free = []

    def listen_free(self, seconds):
        self.free.append(seconds)


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
    assert v.mouth.said == ["↓"]


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


def test_run_refuses_without_a_key():
    with pytest.raises(SystemExit, match="no client key"):
        asyncio.run(voice.run(config.load(), config.models()))

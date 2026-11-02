"""The conversation: the ears, Whisper, Prophet's chat and the mouth together.

A phrase after the wake word is transcribed; its text must begin with the
wake word (Vosk may hear it inside other speech), and the rest goes to
Prophet's chat (`POST /api/v1/chat`) on the key's channel, "client:<name>".
«Понял» says it was sent. The voice then follows the run's events: its
answer is read aloud, and a question Prophet asks the owner is read too and
the next phrase, with no wake word, is the answer.

A cancel word after the wake word sends nothing and cancels the answer being
prepared. A stop word while Prophet speaks hushes it.

Prophet is a client here like any other: it cannot approve calls or change
settings, and everything it starts is traced in the core.
"""

import asyncio
import contextlib
import logging
import sys
import threading

from prophet_voice import ears as ears_
from prophet_voice import words
from prophet_voice.core import Core, CoreError
from prophet_voice.hearing import Hearing
from prophet_voice.mouth import Mouth

log = logging.getLogger("prophet_voice.voice")
AFTER_QUESTION = 8  # seconds to begin answering a question Prophet asked aloud
FINAL = ("done", "failed", "cancelled", "paused")
SAY_FAILED = "Не получилось. Подробности — в журнале."
SAY_LOST = "Я потерял связь с Пророком."


class Voice:
    def __init__(self, settings: dict, core: Core, mouth: Mouth, hearing: Hearing):
        self.words = settings["words"]
        self.project = settings["project"]
        self.core, self.mouth, self.hearing = core, mouth, hearing
        self.ears = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.phrases: asyncio.Queue = asyncio.Queue()
        self.run_id: int | None = None  # the latest run, which a cancel word cancels
        self.question: int | None = None  # a question asked aloud, the next phrase answers
        self.followers: set[asyncio.Task] = set()

    # ---- from the ears' thread ---------------------------------------------------------

    def on_wake(self) -> None:
        self.mouth.tone(True)

    def on_drop(self) -> None:
        self.mouth.tone(False)

    def on_stop(self) -> None:
        self.mouth.hush()

    def on_phrase(self, audio: bytes, free: bool) -> None:
        self.loop.call_soon_threadsafe(self.phrases.put_nowait, (audio, free))

    # ---- here -------------------------------------------------------------------------------

    async def serve(self) -> None:
        self.loop = asyncio.get_running_loop()
        while True:
            audio, free = await self.phrases.get()
            try:
                await self.handle(audio, free)
            except CoreError as exc:
                log.error("the core refused: %s", exc)
                self.mouth.say(SAY_FAILED)
            except Exception:
                log.exception("could not handle a phrase")
                self.mouth.say(SAY_FAILED)

    async def handle(self, audio: bytes, free: bool) -> None:
        text = await asyncio.to_thread(self.hearing.transcribe, audio)
        said = text.strip() if free else words.after_wake(text, self.words["wake"])
        log.info("heard %r", text)
        if not said:  # not the wake word after all, or nothing after it
            if not free:
                self.mouth.tone(False)
            return
        if words.cancelled(said, self.words["cancel"]):
            if self.run_id is not None:
                try:
                    await self.core.cancel(self.run_id)
                except CoreError as exc:  # it ended meanwhile
                    log.info("nothing to cancel: %s", exc)
            self.mouth.tone(False)
            return
        if self.question is not None:
            question, self.question = self.question, None
            await self.core.answer(question, said)
            self.mouth.say(words.pick(self.words["ack"]))
            return
        run_id = await self.core.chat(said, self.project)
        self.mouth.say(words.pick(self.words["ack"]))
        self.follow(run_id)

    def follow(self, run_id: int) -> None:
        self.run_id = run_id
        task = asyncio.create_task(self._follow(run_id))
        self.followers.add(task)
        task.add_done_callback(self.followers.discard)

    async def _follow(self, run_id: int) -> None:
        try:
            async for name, data in self.core.events(run_id):
                if data.get("run_id") != run_id:
                    continue  # another chat's news
                if name == "question":
                    await self._asked(data["id"], data["question"])
                elif name == "message" and data.get("role") == "assistant":
                    if not data.get("task_id"):
                        self.mouth.say(data["content"])
                elif name == "run" and data.get("status") in FINAL:
                    break
        except CoreError as exc:
            log.error("lost run #%s: %s", run_id, exc)
            self.mouth.say(SAY_LOST)
        finally:
            if self.run_id == run_id:
                self.run_id = None

    async def _asked(self, question_id: int, question: str) -> None:
        """Read the question; once it is read, the next phrase is the answer."""
        self.question = question_id
        self.mouth.say(question)
        await asyncio.to_thread(self.mouth.wait)
        if self.question == question_id and self.ears is not None:
            self.ears.listen_free(AFTER_QUESTION)


def run_until_stopped(main) -> int:
    """Run the voice (`main`, a coroutine) until it ends or Ctrl+C; the exit code.

    Ctrl+C cancels it, so it hushes Prophet and lets the core go, and then
    returns at once: unlike asyncio.run, it does not wait for the work it
    lent to threads (Whisper on a phrase, waiting for Prophet to finish)."""
    loop = asyncio.new_event_loop()
    task = loop.create_task(main)
    code = 0
    try:
        loop.run_until_complete(task)
    except KeyboardInterrupt:
        # A task that ended with the KeyboardInterrupt itself is done; one more
        # run on it would never stop (asyncio leaves such a loop running).
        if not task.done():
            task.cancel()
            with contextlib.suppress(BaseException):
                loop.run_until_complete(task)
        print()
    except SystemExit as exc:  # run() says why it cannot start
        if isinstance(exc.code, str):
            print(exc.code, file=sys.stderr)
        code = exc.code if isinstance(exc.code, int) else 1
    finally:
        loop.close()  # its threads are not waited for
    return code


async def run(settings: dict, models) -> None:
    """Listen and speak until Ctrl+C."""
    if not settings["key"]:
        raise SystemExit("no client key: set one with `prophet-voice settings`")
    core = Core(settings["core"], settings["key"])
    try:
        me = await core.me()
    except CoreError as exc:
        raise SystemExit(f"{settings['core']}: {exc}") from None
    if "chat" not in me["scopes"]:
        name = me["name"]
        raise SystemExit(f'the key "{name}" has no "chat" scope: add it in Settings → Clients')
    print(f"Prophet at {settings['core']} knows this voice as client:{me['name']}.")

    mouth = Mouth(settings["speaker"], models)
    hearing = Hearing(settings, models)
    print("Loading the models…")
    recognizer = await asyncio.to_thread(ears_.load_vosk, models)
    await asyncio.gather(asyncio.to_thread(mouth.warm), asyncio.to_thread(hearing.warm))

    voice = Voice(settings, core, mouth, hearing)
    voice.ears = ears_.Ears(
        settings["words"],
        settings["threshold"],
        recognizer,
        mouth,
        voice.on_wake,
        voice.on_drop,
        voice.on_phrase,
        voice.on_stop,
    )
    stop = threading.Event()
    serving = asyncio.create_task(voice.serve())
    await asyncio.sleep(0)  # serve() has its loop before the ears hear anything
    failed = []

    def hear():
        try:
            ears_.listen(voice.ears, settings["microphone"], stop)
        except Exception as exc:  # no microphone, or no permission to use it
            failed.append(exc)
            voice.loop.call_soon_threadsafe(serving.cancel)

    threading.Thread(target=hear, name="ears", daemon=True).start()
    wake = settings["words"]["wake"][0].capitalize()
    mouth.say(words.pick(settings["words"]["hello"]))
    print(f"Listening. Say «{wake}» and speak; Ctrl+C ends.")
    try:
        await serving
    except asyncio.CancelledError:
        if not failed:
            raise
        raise SystemExit(f"the microphone failed: {failed[0]}") from None
    finally:
        stop.set()
        mouth.hush()
        await core.close()

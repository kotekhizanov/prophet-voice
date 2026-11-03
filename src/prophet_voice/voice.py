"""The conversation: the ears, Whisper, Prophet's chat and the mouth together.

A phrase after the wake word is transcribed; its text must begin with the
wake word (Vosk may hear it inside other speech), and the rest goes to
Prophet's chat (`POST /api/v1/chat`) on the key's channel, "client:<name>".
«Понял» says it was sent. The voice then follows the run's events: its
answer is read aloud, and a question Prophet asks the owner is read too and
the next phrase, with no wake word, is the answer.

A cancel word after the wake word sends nothing and cancels the answer being
prepared. While Prophet speaks the voice does not listen. Prophet never
talks over the owner: an answer that comes while a phrase is heard waits for its end.

Omniscient's dialog button (the plugin's script) sets `dialog`. The
microphone is open only in a Private dialog: it opens when the owner begins
one in Omniscient's chat and closes when it ends. Private has no wake word:
any speech is a question, «Ага?» says it is heard from its start and
«Окей…» that it has ended, and there is no «Понял». The question goes to the
chat open in Omniscient. Ending the dialog ends everything the voice does at
once: what it says, the phrases being heard, the answers it waits for (the
core's run goes on, and its answer is in the chat).

Only one voice runs on this machine: a second `run` refuses to start, or two
would hear the same phrase and answer it together.

A Prophet node may run the voice while it is connected to the core, and
stops it (SIGTERM, taken as Ctrl+C) when the owner switches the plugin off.
So `run` serves the plugin at once, and with no key, or one the core no
longer knows, it waits for Omniscient to hand it a new one: a voice that
exited would be started again and again, and could not be added meanwhile.

Prophet is a client here like any other: it cannot approve calls or change
settings, and everything it starts is traced in the core.
"""

import asyncio
import contextlib
import fcntl
import logging
import os
import signal
import sys
import threading

from prophet_voice import config, settings_page, words
from prophet_voice import ears as ears_
from prophet_voice.core import Core, CoreError
from prophet_voice.dialog import Dialog
from prophet_voice.hearing import Hearing
from prophet_voice.mouth import Mouth

log = logging.getLogger("prophet_voice.voice")
AFTER_QUESTION = 8  # seconds to begin answering a question Prophet asked aloud
FINAL = ("done", "failed", "cancelled", "paused")
DIALOGS = ("casual", "private")
CUE_VOLUME = {"begun": 0.5, "heard": 0.9}
KEY_WAIT = 1  # seconds between looks at the settings for a key Omniscient hands
SAY_FAILED = "Не получилось. Подробности — в журнале."
SAY_LOST = "Я потерял связь с Пророком."
SAY_NO_MIC = "Микрофон не открылся. Подробности — в журнале."


class Voice:
    def __init__(self, settings: dict, core: Core, mouth: Mouth, hearing: Hearing):
        self.words = settings["words"]
        self.project = settings["project"]
        self.device = settings["microphone"]  # the microphone's name; empty: the system's
        self.from_omniscient: dict = {}  # the settings Omniscient gave, as last read
        self.core, self.mouth, self.hearing = core, mouth, hearing
        self.ears = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.phrases: asyncio.Queue = asyncio.Queue()
        self.run_id: int | None = None  # the latest run, which a cancel word cancels
        self.question: int | None = None  # a question asked aloud, the next phrase answers
        self.followers: set[asyncio.Task] = set()
        self.dialog: str | None = None  # "casual" or "private" while Omniscient shows one
        self.dialog_project = None  # the chat open in Omniscient: a project's number, or None
        self.hearing_now = 0  # phrases being transcribed and sent
        self.microphone = threading.Event()  # set: the ears' microphone is open
        self.era = 0  # an ended dialog makes the phrases heard before it old
        self.said: set[int] = set()  # the messages read aloud, each once
        self.beginning: asyncio.Task | None = None  # a dialog's settings read, its greeting

    @property
    def private(self) -> bool:
        return self.dialog == "private"

    def state(self) -> str:
        """What the voice does now, for Omniscient's picture."""
        ears = self.ears
        if ears is None:
            return "starting"
        if ears.state == "phrase" and not ears.tentative:
            return "listening"
        if self.hearing_now:
            return "hearing"
        if self.mouth.speaking:
            return "speaking"
        if self.run_id is not None:
            return "thinking"
        return "waiting"

    def set_dialog(self, dialog: str | None, project=None) -> None:
        """A dialog begins, ends, changes its kind or its chat (on the loop)."""
        was, self.dialog = self.dialog, dialog
        self.dialog_project = project if dialog else None
        if self.ears is not None and (was == "private") != self.private:
            self.ears.be_alone(self.private)
        if (was == "private" and not self.private) or (was and not dialog):
            self.quiet()
        begins = dialog and not was
        if begins:  # what the owner has set since, in Omniscient, from now on
            self.configure(config.with_plugin(config.load(), self.from_omniscient)[0])
        if self.private:
            self.microphone.set()
        if begins:
            if self.loop is not None and self.loop.is_running():
                self.beginning = self.loop.create_task(self._begin())
            else:  # not serving yet: the settings it began with
                self.mouth.say(words.pick(self.words["hello"]))

    async def _begin(self) -> None:
        """A dialog begins: in the hearing set now, a greeting."""
        await self.refresh()
        if self.dialog:
            self.mouth.say(words.pick(self.words["hello"]))

    async def refresh(self) -> None:
        """The core's settings, if the owner has changed them in Omniscient since
        they were last read (this Mac's are taken as a dialog begins)."""
        try:
            plugin = await self.core.plugin()
        except CoreError as exc:
            log.warning("the settings from Omniscient are not read: %s", exc)
            return
        if plugin is None or plugin["settings"] == self.from_omniscient:
            return
        self.from_omniscient = plugin["settings"]
        settings, _, refused = config.with_plugin(config.load(), self.from_omniscient)
        for why in refused:
            log.warning("a setting from Omniscient is not used: %s", why)
        self.configure(settings)

    def configure(self, settings: dict) -> None:
        """What the owner has set since the voice started, in Omniscient: from now on."""
        self.words, self.project, self.device = (
            settings["words"],
            settings["project"],
            settings["microphone"],
        )
        if self.ears is not None:
            self.ears.configure(settings["words"], settings["threshold"])
        self.hearing.configure(settings)

    def quiet(self) -> None:
        """Everything the voice does, ended at once: the microphone, what it says,
        the phrases heard and the answers waited for."""
        self.microphone.clear()
        self.era += 1
        self.question = None
        for task in list(self.followers):
            task.cancel()
        self.mouth.hush()

    # ---- from the ears' thread ---------------------------------------------------------

    def on_wake(self) -> None:
        self.mouth.tone(True)

    def on_drop(self) -> None:
        self.mouth.tone(False)

    def on_begun(self) -> None:
        self.mouth.cue(words.pick(self.words["begun"]), CUE_VOLUME["begun"])

    def on_phrase(self, audio: bytes, free: bool) -> None:
        if self.private:
            self.mouth.cue(words.pick(self.words["heard"]), CUE_VOLUME["heard"])
        self.loop.call_soon_threadsafe(self.phrases.put_nowait, (audio, free, self.era))

    def on_microphone_failed(self, exc: Exception) -> None:
        """The dialog cannot hear: it ends, and Omniscient's page goes back to the chat."""
        self.loop.call_soon_threadsafe(self._no_microphone)

    def _no_microphone(self) -> None:
        self.set_dialog(None)
        self.mouth.say(SAY_NO_MIC)

    # ---- here -------------------------------------------------------------------------------

    async def serve(self) -> None:
        self.loop = asyncio.get_running_loop()
        while True:
            audio, free, era = await self.phrases.get()
            try:
                await self.handle(audio, free, era)
            except CoreError as exc:
                log.error("the core refused: %s", exc)
                self.mouth.say(SAY_FAILED)
            except Exception:
                log.exception("could not handle a phrase")
                self.mouth.say(SAY_FAILED)

    async def handle(self, audio: bytes, free: bool, era: int | None = None) -> None:
        """A phrase heard in `era`: none of a dialog that has ended since."""
        era = self.era if era is None else era
        if era != self.era:
            return
        self.hearing_now += 1
        try:
            await self._handle(audio, free, era)
        finally:
            self.hearing_now -= 1

    async def _handle(self, audio: bytes, free: bool, era: int) -> None:
        text = await asyncio.to_thread(self.hearing.transcribe, audio)
        if era != self.era:  # the dialog ended while Whisper heard it
            return
        if self.private:
            said = words.alone(text, self.words)
            if words.cue_only(said, self.words):
                said = ""  # the microphone heard Prophet's own cue
        elif free:
            said = text.strip()
        else:
            said = words.after_wake(text, self.words["wake"])
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
            if not self.private:
                self.mouth.say(words.pick(self.words["ack"]))
            return
        project = self.dialog_project if self.dialog else self.project
        run_id = await self.core.chat(said, project)
        if not self.private:
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
                    await self._owner_done()
                    await self._asked(data["id"], data["question"])
                elif name == "message" and data.get("role") == "assistant":
                    if data.get("id") in self.said:
                        continue
                    if data.get("id") is not None:
                        self.said.add(data["id"])
                    if not data.get("task_id"):
                        await self._owner_done()
                        self.mouth.say(data["content"])
                elif name == "run" and data.get("status") in FINAL:
                    break
        except CoreError as exc:
            log.error("lost run #%s: %s", run_id, exc)
            self.mouth.say(SAY_LOST)
        finally:
            if self.run_id == run_id:
                self.run_id = None

    async def _owner_done(self) -> None:
        """Until the owner's phrase has ended and been sent: Prophet does not talk over it."""
        while self.hearing_now or getattr(self.ears, "state", "waiting") == "phrase":
            await asyncio.sleep(0.1)

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
    lent to threads (Whisper on a phrase, waiting for Prophet to finish).
    SIGTERM, a node stopping the voice, is taken as Ctrl+C."""
    was = signal.signal(signal.SIGTERM, signal.default_int_handler)
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
        if task.done() and not task.cancelled():
            task.exception()  # it is handled here: asyncio need not warn of it
        if isinstance(exc.code, str):
            print(exc.code, file=sys.stderr)
        code = exc.code if isinstance(exc.code, int) else 1
    finally:
        loop.close()  # its threads are not waited for
        signal.signal(signal.SIGTERM, was)
    return code


async def admitted(core: Core, address: str) -> tuple[dict, dict | None]:
    """The key's name and scopes, and the plugin's state (None for a key made by
    hand), once the core lets the voice in; a CoreError of 401 when the core
    does not know the key (a new one may come); else SystemExit.

    A plugin switched off in Omniscient keeps its key: the voice itself does
    not start then."""
    try:
        me = await core.me()
        plugin = await core.plugin()
    except CoreError as exc:
        if exc.status == 401:
            raise
        raise SystemExit(f"{address}: {exc}") from None
    if "chat" not in me["scopes"]:
        name = me["name"]
        raise SystemExit(f'the key "{name}" has no "chat" scope: add it in Settings → Clients')
    if plugin is not None and not plugin["enabled"]:
        raise SystemExit("the voice is switched off in Omniscient: Settings → Plugins")
    return me, plugin


def only_one():
    """The lock that only one voice on this machine holds, while its file is open;
    else SystemExit."""
    path = config.home() / "run.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = open(path, "a+")  # held open while the voice runs
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        lock.seek(0)
        other = lock.read().strip()
        lock.close()
        raise SystemExit(f"the voice is running already (pid {other or '?'})") from None
    lock.seek(0)
    lock.truncate()
    lock.write(str(os.getpid()))
    lock.flush()
    return lock


async def new_key(old: str) -> dict:
    """The settings once their key is no longer `old`: Omniscient hands one to
    the plugin (`/prophet/connect`)."""
    while (settings := config.load())["key"] == old:
        await asyncio.sleep(KEY_WAIT)
    return settings


async def let_in(settings: dict) -> tuple[dict, Core, dict, dict | None]:
    """The settings, the core and what admitted() says, once the core lets the
    voice in; with no key or an unknown one, it waits for a new key."""
    while True:
        if not settings["key"]:
            print(
                "No client key yet: in Omniscient, Settings → Plugins → Add plugin, "
                f"http://127.0.0.1:{settings['settings_port']}"
            )
            settings = await new_key("")
            continue
        core = Core(settings["core"], settings["key"])
        try:
            me, plugin = await admitted(core, settings["core"])
        except CoreError as exc:  # 401: the key was revoked, or is another core's
            await core.close()
            print(f"{settings['core']}: {exc}. Waiting for a new key from Omniscient.")
            settings = await new_key(settings["key"])
            continue
        except BaseException:
            await core.close()
            raise
        return settings, core, me, plugin


async def run(settings: dict, models) -> None:
    """Listen and speak until Ctrl+C."""
    lock = only_one()
    page = settings_page.start(settings)  # the plugin, for Omniscient to add it and hand a key
    try:
        await _run(settings, models)
    finally:
        if page is not None:
            page.shutdown()
            page.server_close()
        lock.close()


async def _run(settings: dict, models) -> None:
    settings, core, me, plugin = await let_in(settings)
    print(f"Prophet at {settings['core']} knows this voice as client:{me['name']}.")
    if plugin is not None:  # what the owner set in Omniscient wins; it is read at the start
        settings, used, refused = config.with_plugin(settings, plugin["settings"])
        for why in refused:
            log.warning("a setting from Omniscient is not used: %s", why)
        if used:
            print(f"From Omniscient: {', '.join(used)}.")

    mouth = Mouth(models)
    hearing = Hearing(settings, models)
    print("Loading the models…")
    recognizer = await asyncio.to_thread(ears_.load_vosk, models)
    await asyncio.gather(asyncio.to_thread(mouth.warm), asyncio.to_thread(hearing.warm))

    voice = Voice(settings, core, mouth, hearing)
    voice.from_omniscient = plugin["settings"] if plugin is not None else {}
    voice.ears = ears_.Ears(
        settings["words"],
        settings["threshold"],
        recognizer,
        mouth,
        voice.on_wake,
        voice.on_drop,
        voice.on_phrase,
        voice.on_begun,
    )
    stop = threading.Event()
    serving = asyncio.create_task(voice.serve())
    await asyncio.sleep(0)  # serve() has its loop before the ears hear anything
    threading.Thread(
        target=ears_.listen,
        args=(voice.ears, lambda: voice.device, voice.microphone, stop),
        kwargs={"failed": voice.on_microphone_failed},
        name="ears",
        daemon=True,
    ).start()
    settings_page.Page.dialog = Dialog(voice)  # Omniscient's dialog talks to it
    print("Ready. Begin a Private dialog in Omniscient's chat to talk; Ctrl+C ends.")
    try:
        await serving
    finally:
        settings_page.Page.dialog = None
        stop.set()
        mouth.hush()
        await core.close()

"""The dialog in Omniscient: the plugin's script and the running voice together.

The script (`omniscient.js`) draws the dialog in Omniscient's chat, as
Prophet 1 had it: the Casual dialog and Private buttons, and a cloud of
points that breathes, thinks and speaks. It talks to `prophet-voice run`
through the settings server on this machine: `POST /dialog` begins, changes
or ends the dialog, and `GET /dialog/events` streams how the voice is now
(Server-Sent Events): what it does, how loud the owner and Prophet are, its
wake word for the captions, the phrase being said as far as it is heard
(`text`), and in a Private dialog the voices heard in
it (speakers.py), which the page may name (`POST /dialog/voice`).

A dictation (voice.py) is asked for here too: `POST /dictation` begins or
ends one for a page, and the stream's `dictation` has its phrases, numbered,
for that page to put into the field where the cursor is: `count` of them so
far and the last few (`phrases`), so none is missed between two snapshots,
and the phrase being said (`text`), which the page types as it is heard.
It ends with its page's stream as a dialog does.

A dialog belongs to the page that began it (each page names itself, `page`)
and lives only while that page is open: when its stream closes (the tab is
closed, the page reloads), the dialog ends, as when the last stream of all
closes. So a Private dialog, which hears any speech without the wake word,
never goes on with no page to show it. Other pages only follow it.
"""

import contextlib
import threading

FRAME = 0.05  # seconds between snapshots while a dialog is on
IDLE = 0.5  # seconds between looks while none is: a snapshot goes only when it changes
KEEPALIVE = 15  # seconds: a comment on a quiet stream, so a closed page is noticed
KEEPALIVE_IN_DIALOG = 1  # seconds, in a dialog: a Private one must not outlive its page
DICTATED = 5  # of a dictation's latest phrases in each snapshot


class Dialog:
    def __init__(self, voice):
        self.voice = voice
        self.watchers = 0
        self.owner: str | None = None  # the page that began the dialog
        self._lock = threading.Lock()

    def snapshot(self) -> dict:
        v = self.voice
        ears = v.ears
        return {
            "dialog": v.dialog,
            "state": v.state(),
            "mic": round(ears.level, 3) if ears is not None else 0.0,
            "voice": round(v.mouth.level, 3),
            "wake": v.words["wake"],
            "text": v.showing(),
            "voices": v.speakers.voices.shown() if v.speakers and v.private else [],
            "dictation": self.dictated(),
        }

    def dictated(self) -> dict | None:
        """The latest dictation, for its page: whether it is on, its phrases, how
        many were said (`said`: those past `count` are Whisper's still), and the
        one being said now as far as it is heard (`text`)."""
        v = self.voice
        d = v.dictation
        if d is None:
            return None
        phrases = list(d["phrases"])
        ears = v.ears
        saying = d["on"] and ears is not None and ears.state == "phrase" and not ears.tentative
        return {
            "page": d["page"],
            "n": d["n"],
            "on": d["on"],
            "said": max(d["said"], len(phrases)),
            "text": v.showing() if saying else "",
            "count": len(phrases),
            "phrases": phrases[-DICTATED:],
        }

    def dictate(self, page: str | None) -> None:
        """From the server's thread: a dictation begins for `page`, or ends (None).
        ValueError when a dialog is on: it hears the owner already."""
        if page and self.voice.dialog:
            raise ValueError("a dialog is on: end it to dictate")
        self.voice.loop.call_soon_threadsafe(self.voice.set_dictation, page)

    def name_voice(self, voice_id: int, name: str) -> None:
        """From the server's thread; ValueError says what is wrong."""
        if self.voice.speakers is None:
            raise ValueError("voices are not told apart: run `prophet-voice install`")
        self.voice.speakers.voices.name(voice_id, name)

    def set(self, dialog: str | None, project=None, page: str | None = None) -> None:
        """From the server's thread: the voice changes on its own loop."""
        with self._lock:
            self.owner = page if dialog else None
        self.voice.loop.call_soon_threadsafe(self.voice.set_dialog, dialog, project)

    @contextlib.contextmanager
    def watching(self, page: str | None = None):
        """One page's stream: closed, it ends the dialog that page began and its
        dictation, or any when it was the last one."""
        with self._lock:
            self.watchers += 1
        try:
            yield
        finally:
            with self._lock:
                self.watchers -= 1
                ends = not self.watchers or (page is not None and page == self.owner)
                d = self.voice.dictation
                dictated = d is not None and d["on"] and (not self.watchers or page == d["page"])
            if ends:
                self.set(None)
            if dictated:
                self.dictate(None)

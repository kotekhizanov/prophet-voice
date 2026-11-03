"""The dialog in Omniscient: the plugin's script and the running voice together.

The script (`omniscient.js`) draws the dialog in Omniscient's chat, as
Prophet 1 had it: the Casual dialog and Private buttons, and a cloud of
points that breathes, thinks and speaks. It talks to `prophet-voice run`
through the settings server on this machine: `POST /dialog` begins, changes
or ends the dialog, and `GET /dialog/events` streams how the voice is now
(Server-Sent Events): what it does, how loud the owner and Prophet are, and
its wake and stop words for the captions.

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
            "stop": v.words["stop"],
        }

    def set(self, dialog: str | None, project=None, page: str | None = None) -> None:
        """From the server's thread: the voice changes on its own loop."""
        with self._lock:
            self.owner = page if dialog else None
        self.voice.loop.call_soon_threadsafe(self.voice.set_dialog, dialog, project)

    @contextlib.contextmanager
    def watching(self, page: str | None = None):
        """One page's stream: closed, it ends the dialog that page began, or any
        dialog when it was the last one."""
        with self._lock:
            self.watchers += 1
        try:
            yield
        finally:
            with self._lock:
                self.watchers -= 1
                ends = not self.watchers or (page is not None and page == self.owner)
            if ends:
                self.set(None)

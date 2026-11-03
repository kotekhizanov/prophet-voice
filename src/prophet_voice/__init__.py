"""prophet-voice: talk to Prophet aloud.

A plugin of a Prophet core (`/api/v1`, a key with the "chat" scope): it hears
the wake word on this machine, transcribes the phrase after it, sends it to
Prophet's chat and reads the answer aloud. Nothing leaves the machine before
the wake word.
"""

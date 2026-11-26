"""The words of the voice: what it hears and what it says.

Each kind has one or more variants, set in the settings. Words heard before
any audio goes to Whisper (the wake word) are heard by Vosk with a grammar,
so each is one lowercase Russian word; Vosk ignores one it
does not know. Cancel words are looked for in Whisper's text, and while
Prophet speaks Vosk listens for them alone. What Prophet
says (the greeting, «Понял») is a phrase, and a variant is picked at random
each time.
"""

import random
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Kind:
    label: str
    default: tuple[str, ...]
    # "vosk": one word for Vosk's grammar; "text": looked for in Whisper's text;
    # "said": a phrase Prophet says
    form: str
    hint: str


KINDS = {
    "wake": Kind("Wake word", ("пророк",), "vosk", "Begins a phrase; the text must begin with it."),
    "cancel": Kind(
        "Cancel",
        ("отмена",),
        "text",
        "Sends nothing and cancels the answer being prepared; said over Prophet, hushes it.",
    ),
    "hello": Kind("Greeting", ("Привет!",), "said", "Said when the voice or a dialog starts."),
    "ack": Kind(
        "Question taken", ("Понял.",), "said", "Said when a question is sent (not in Private)."
    ),
    "begun": Kind(
        "Listening", ("Ага?",), "said", "Private: here and listening, said softly as you begin."
    ),
    "heard": Kind("Heard", ("Окей…",), "said", "Private: the question has ended."),
}
MAX_VARIANTS = 10
FORMS = {
    "vosk": (re.compile(r"^[а-яё]{2,30}$"), "one Russian word"),
    "text": (re.compile(r"^[а-яё][а-яё ]{0,38}[а-яё]$"), "Russian words of up to 40 letters"),
    "said": (re.compile(r"^\S(?:.{0,98}\S)?$", re.S), "a phrase of up to 100 characters"),
}


def defaults() -> dict[str, list[str]]:
    return {name: list(kind.default) for name, kind in KINDS.items()}


def check(name: str, variants) -> list[str]:
    """The variants of a kind, trimmed and without repeats; ValueError says what is wrong."""
    kind = KINDS.get(name)
    if kind is None:
        raise ValueError(f"no words {name}")
    if not isinstance(variants, list) or not all(isinstance(v, str) for v in variants):
        raise ValueError(f"{kind.label}: a list of variants")
    pattern, what = FORMS[kind.form]
    out = []
    for v in variants:
        v = " ".join(v.split()) if kind.form != "said" else v.strip()
        if kind.form != "said":
            v = v.lower()
        if not v or v in out:
            continue
        if "\n" in v or not pattern.match(v):
            raise ValueError(f"{kind.label}: «{v[:40]}» is not {what}")
        out.append(v)
    if not 1 <= len(out) <= MAX_VARIANTS:
        raise ValueError(f"{kind.label}: from 1 to {MAX_VARIANTS} variants")
    return out


def pick(variants: list[str]) -> str:
    return random.choice(variants)  # noqa: S311 - a phrase to say, not a secret


# ---- Whisper's text ----------------------------------------------------------------

EDGE = r"\s\"«»'.,!?…—-"  # the hyphen last: elsewhere it would make a range


def _variants(variants: list[str], loose: bool = False) -> str:
    """A variant in Whisper's text: е and ё alike, any spaces; `loose` lets a word
    come apart ("про рок")."""

    def one(word: str) -> str:
        chars = [r"\s+" if c == " " else "[её]" if c in "её" else re.escape(c) for c in word]
        return (r"[\s-]?" if loose else "").join(chars)

    return "|".join(one(v) for v in variants)


def after_wake(text: str, wake: list[str]) -> str | None:
    """The text after a leading wake word, or None when it does not begin with one.

    Vosk may hear the wake word inside other speech, so Whisper's text must
    begin with it for the phrase to count."""
    found = re.match(
        rf"^[{EDGE}]*(?:{_variants(wake, loose=True)})(?![а-яё])[:;{EDGE}]*(.*)$",
        text,
        re.I | re.S,
    )
    return found[1].strip() if found else None


def _leading(text: str, variants: list[str]) -> str:
    """The text after one of these variants at its start, or the text as it is."""
    found = re.match(
        rf"^[{EDGE}]*(?:{_variants(variants)})(?![а-яё])[:;{EDGE}]*(.*)$", text, re.I | re.S
    )
    return found[1] if found else text


def _letters(variants: list[str]) -> list[str]:
    """The words of phrases Prophet says, without their punctuation: «Ага?» is «ага»."""
    return [w for v in variants if (w := " ".join(re.findall(r"[а-яёa-z]+", v.lower())))]


def alone(text: str, kinds: dict[str, list[str]]) -> str:
    """A Private phrase, with no wake word: without Prophet's own «Ага» at its
    start (the microphone hears it) and a wake word said anyway."""
    rest = _leading(text.strip(), _letters(kinds["begun"]))
    after = after_wake(rest, kinds["wake"])
    return (after if after is not None else rest).strip()


MADE_UP = ("ок", "пока")  # what Whisper makes of a near-silent phrase


def cue_only(said: str, kinds: dict[str, list[str]]) -> bool:
    """A Private phrase of nothing but Prophet's cues (or Whisper's made-up
    words) is not a question: the microphone heard Prophet."""
    cues = {
        w.replace("ё", "е") for v in _letters([*kinds["begun"], *kinds["heard"]]) for w in v.split()
    }
    heard = {w.replace("ё", "е") for w in re.findall(r"[а-яёa-z]+", said.lower())}
    return bool(heard) and heard <= cues | set(MADE_UP)


def said_in(text: str, variants: list[str]) -> bool:
    """One of these variants is among the words of the text."""
    return bool(re.search(rf"(?<![а-яё])(?:{_variants(variants)})(?![а-яё])", text, re.I))


ECHOED = 0.6  # of a phrase's words that Prophet was saying: it heard Prophet


def echoed(said: str, own: list[str]) -> bool:
    """A phrase heard over Prophet that is mostly what Prophet was saying then
    (`own`, the pieces it read): the microphone heard Prophet, not the owner."""

    def plain(text: str) -> list[str]:
        return [w.replace("ё", "е") for w in re.findall(r"[а-яёa-z]+", text.lower())]

    mine = {w for text in own for w in plain(text)}
    heard = [w for w in plain(said) if len(w) >= 3]
    return bool(heard) and sum(w in mine for w in heard) >= ECHOED * len(heard)


def cancelled(said: str, cancel: list[str]) -> bool:
    """A phrase dropped: it begins or ends with a cancel word."""
    words = _variants(cancel)
    return bool(
        re.search(
            rf"^[{EDGE}]*(?:{words})(?![а-яё])|(?<![а-яё])(?:{words})[.!…]*$", said.strip(), re.I
        )
    )

"""Prophet's own voice taken out of the microphone's audio.

The speakers are in the same room as the microphone, so what Prophet says
comes back into it. The mouth tells here what it plays and when (`played`,
`stopped`); the microphone's blocks come through with the time each ended.
Band by band, the power the microphone has is compared with the power of
what was played a little before: how much of the played sound comes back
(the room's coupling) is learnt while Prophet speaks, and each band is
lowered by the share of it that is Prophet's. The owner's voice over
Prophet is in other bands, or louder in the same ones, and passes.

How long a sound takes from `played` to the microphone (the speakers' and
the microphone's buffers, up to LAGS hops) is not known: it is found by
where the loudness of the two goes up and down together. Until then every
delay is taken as possible, and more is lowered than need be.

Only the powers are compared, not the waves: a few milliseconds of error in
the time cost nothing. With nothing played, the audio passes as it is. Like
the denoiser, it works in frames of FRAME samples that overlap by half and
is FRAME - HOP samples (20 ms) late.
"""

import threading
import time
from collections import deque

import numpy as np

RATE = 16000  # Hz, as the ears hear
FRAME = 640  # samples: 40 ms
HOP = FRAME // 2
LAGS = 45  # hops (0.9 s): the longest a played sound takes to reach the microphone
AROUND = 3  # hops around the delay found: each sound starts a little sooner or later
DECAY = 0.5  # of a band's power the room still rings with, a hop later
FORGET = 0.995  # how much of what was learnt of the coupling a hop keeps
OVER = 3.0  # Prophet's share is taken as this much more: it varies from frame to frame
FLOOR = 0.05  # the least a band is let through: about -26 dB
HISTORY = 150  # hops (3 s) of loudness the delay is looked for in
SEARCH = 10  # hops between the looks
SURE = 0.5  # how alike the two loudnesses must be for a delay to count
ENOUGH = 25  # hops of played sound in the history for a look
LEARNT = 25  # hops (0.5 s) of Prophet's voice after which the coupling is known
MOST = 4.0  # of the power expected of Prophet in a band, the most the coupling learns from
KEPT = 2.0  # seconds a played sound is remembered after its end
BAND = slice(300 * FRAME // RATE, 4000 * FRAME // RATE)  # where a voice is loud


def resample(samples: np.ndarray, rate: int) -> np.ndarray:
    """The samples at the ears' rate (through their spectrum: nothing folds over)."""
    samples = np.asarray(samples, np.float32)
    if rate == RATE or not len(samples):
        return samples
    n = int(len(samples) * RATE / rate)
    spectrum = np.fft.rfft(samples)
    out = np.fft.irfft(spectrum[: n // 2 + 1], n) * (n / len(samples))
    return out.astype(np.float32)


class Echo:
    def __init__(self):
        self.window = np.sqrt(np.hanning(FRAME + 1)[:FRAME]).astype(np.float32)
        self.bins = FRAME // 2 + 1
        self.frame = np.zeros(FRAME, np.float32)
        self.tail = np.zeros(FRAME, np.float32)
        self.pending = np.zeros(0, np.float32)
        self.ready = np.zeros(0, np.float32)
        self.lock = threading.Lock()
        self.sounds: list[tuple[float, np.ndarray]] = []  # (when it began, what) played
        self.silent = np.zeros(self.bins, np.float32)
        # the played sound's power at each hop before this one: [0] is now
        self.played_at: deque[np.ndarray] = deque([self.silent] * (LAGS + 1), LAGS + 1)
        self.ring = self.silent  # Prophet's power as the room has it now, coupling aside
        self.both = np.zeros(self.bins)  # the coupling is `both / own`, band by band
        self.own = np.zeros(self.bins)
        self.learnt = 0  # hops the coupling has been learnt from
        self.delay: int | None = None  # hops from `played` to the microphone, once found
        self.mic_loud: deque[float] = deque(maxlen=HISTORY)
        self.own_loud: deque[float] = deque(maxlen=HISTORY + LAGS)
        self.hops = 0
        self.idle = LAGS + 1  # hops since anything played: past LAGS, nothing of it is left

    # ---- from the mouth ----------------------------------------------------------------

    def played(self, samples: np.ndarray, rate: int, at: float | None = None) -> None:
        """A sound that begins to play now (or `at`, on time.monotonic's clock)."""
        sound = resample(samples, rate)
        at = time.monotonic() if at is None else at
        with self.lock:
            kept = [(t, s) for t, s in self.sounds if t + len(s) / RATE > at - KEPT]
            self.sounds = [*kept, (at, sound)]

    def stopped(self, at: float | None = None) -> None:
        """What plays was cut off (a hush): the rest of it never sounded."""
        at = time.monotonic() if at is None else at
        with self.lock:
            self.sounds = [(t, s[: max(0, int((at - t) * RATE))]) for t, s in self.sounds]

    # ---- the microphone's audio --------------------------------------------------------

    def __call__(self, block: bytes, at: float | None = None) -> bytes:
        """A block of 16-bit samples that ended `at`, without Prophet's voice; as
        long as the block."""
        at = time.monotonic() if at is None else at
        samples = np.frombuffer(block, np.int16).astype(np.float32)
        x = np.concatenate([self.pending, samples])
        hops = len(x) // HOP
        done = [
            self._hop(x[i * HOP : (i + 1) * HOP], at - (len(x) - (i + 1) * HOP) / RATE)
            for i in range(hops)
        ]
        self.ready = np.concatenate([self.ready, *done])
        self.pending = x[hops * HOP :]
        if len(self.ready) < len(samples):  # the first blocks: silence while it is late
            self.ready = np.concatenate([np.zeros(len(samples) - len(self.ready)), self.ready])
        y, self.ready = self.ready[: len(samples)], self.ready[len(samples) :]
        return np.clip(np.round(y), -32768, 32767).astype(np.int16).tobytes()

    def _hop(self, samples: np.ndarray, at: float) -> np.ndarray:
        self.frame = np.concatenate([self.frame[HOP:], samples])
        spectrum = np.fft.rfft(self.frame * self.window)
        power = spectrum.real**2 + spectrum.imag**2 + 1e-3
        own = self._own(at)
        self.idle = 0 if own is not self.silent else self.idle + 1
        self.played_at.appendleft(own)
        self.mic_loud.append(float(np.sqrt(power[BAND].sum())))
        self.own_loud.append(float(np.sqrt(own[BAND].sum())))
        self.hops += 1
        if self.hops % SEARCH == 0 and self.idle <= LAGS:
            self._find()
        if self.idle <= LAGS or self.ring.max() > 1:
            spectrum = spectrum * self._gain(power)
        y = np.fft.irfft(spectrum, FRAME).astype(np.float32) * self.window
        self.tail += y
        done = self.tail[:HOP].copy()
        self.tail = np.concatenate([self.tail[HOP:], np.zeros(HOP, np.float32)])
        return done

    def _own(self, at: float) -> np.ndarray:
        """The power of what was played in the frame that ended `at`."""
        begin = at - FRAME / RATE
        with self.lock:
            sounds = list(self.sounds)
        frame = None
        for start, sound in sounds:
            i = round((begin - start) * RATE)
            a, b = max(i, 0), min(i + FRAME, len(sound))
            if a >= b:
                continue
            if frame is None:
                frame = np.zeros(FRAME, np.float32)
            frame[a - i : b - i] += sound[a:b]
        if frame is None:
            return self.silent
        spectrum = np.fft.rfft(frame * self.window)
        return (spectrum.real**2 + spectrum.imag**2).astype(np.float32)

    def _gain(self, power: np.ndarray) -> np.ndarray:
        """How much of each band is let through: what is not Prophet's."""
        lags = list(self.played_at)
        if self.delay is not None:
            lags = lags[max(0, self.delay - AROUND) : self.delay + AROUND + 1]
        self.ring = np.maximum(np.maximum.reduce(lags), self.ring * DECAY)
        coupling = self.both / (self.own + 1e-9)
        if self.learnt >= LEARNT:  # a band far louder than Prophet makes it is the owner's
            power_of_it = np.minimum(power, MOST * coupling * self.ring + 1e-3)
        else:
            power_of_it = power
        self.learnt += 1
        self.both = FORGET * self.both + power_of_it * self.ring
        self.own = FORGET * self.own + self.ring.astype(np.float64) ** 2
        coupling = self.both / (self.own + 1e-9)
        return np.sqrt(np.maximum(1 - OVER * coupling * self.ring / power, FLOOR**2))

    def _find(self) -> None:
        """The delay at which the microphone's loudness follows the played sound's."""
        n = len(self.mic_loud)
        own = np.array(self.own_loud)
        if np.count_nonzero(own[-(n + LAGS) :]) < ENOUGH:
            return
        mic = np.array(self.mic_loud)
        mic = mic - mic.mean()
        best, found = SURE, None
        for lag in range(min(LAGS, len(own) - n) + 1):
            there = own[len(own) - n - lag : len(own) - lag]
            there = there - there.mean()
            scale = float(np.linalg.norm(mic) * np.linalg.norm(there))
            if scale > 0 and (alike := float(mic @ there) / scale) > best:
                best, found = alike, lag
        if found is None or found == self.delay:
            return
        if self.delay is None or abs(found - self.delay) > 1:
            self.both, self.own = np.zeros(self.bins), np.zeros(self.bins)  # learnt anew
            self.learnt = 0
        self.delay = found

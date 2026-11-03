"""Noise suppression: the microphone's audio with the room's steady noise taken out.

A fan, a fridge, the street's hum are steady: in each band of frequencies
they are the quietest the band gets. The denoiser follows that quietest
level band by band (minimum statistics: the least of the smoothed power over
the last WINDOW seconds) and lowers each band by how far above it the sound
is (a Wiener gain, decision-directed, so the noise left does not twinkle).
A voice is far above it and passes; the noise is lowered by up to FLOOR,
never cut to silence, so Vosk and Whisper still hear a room.

It works on the microphone's blocks as they come, in frames of FRAME samples
that overlap by half, and is FRAME - HOP samples (20 ms) late.
"""

import numpy as np

RATE = 16000  # Hz, as the ears hear
FRAME = 640  # samples: 40 ms
HOP = FRAME // 2
WINDOW = 1.5  # seconds the quietest level is looked for in
PARTS = 6  # the window in parts: the quietest of each part is kept
SMOOTH = 0.8  # how much of the power before a frame its smoothed power keeps
BIAS = 2.2  # the quietest level is below the noise's average by about this, and some
KEEP = 0.96  # how much of the previous frame's estimate a band's cleanness keeps
FLOOR = 0.12  # the least a band is let through: about -18 dB


class Denoiser:
    def __init__(self):
        # sqrt-Hann in and out: its squares overlapping by half sum to one
        self.window = np.sqrt(np.hanning(FRAME + 1)[:FRAME]).astype(np.float32)
        self.bins = FRAME // 2 + 1
        self.frame = np.zeros(FRAME, np.float32)
        self.tail = np.zeros(FRAME, np.float32)  # the next frames' overlap, so far
        self.pending = np.zeros(0, np.float32)  # samples short of a hop
        self.ready = np.zeros(0, np.float32)  # cleaned samples not given back yet
        self.smooth = None
        self.lows: list[np.ndarray] = []  # the quietest of each finished part
        self.low = None  # the quietest of the part going on
        self.frames = 0
        self.clean = np.ones(self.bins, np.float32)  # the last frame's cleaned power over noise

    @property
    def noise(self) -> np.ndarray | None:
        """The noise's power in each band, as followed now."""
        if self.low is None:
            return None
        return np.minimum.reduce([self.low, *self.lows]) * BIAS

    def __call__(self, block: bytes) -> bytes:
        """A block of 16-bit samples, cleaned; as long as the block."""
        samples = np.frombuffer(block, np.int16).astype(np.float32)
        x = np.concatenate([self.pending, samples])
        hops = len(x) // HOP
        self.ready = np.concatenate(
            [self.ready, *(self._hop(x[i * HOP : (i + 1) * HOP]) for i in range(hops))]
        )
        self.pending = x[hops * HOP :]
        if len(self.ready) < len(samples):  # the first blocks: silence while it is late
            self.ready = np.concatenate([np.zeros(len(samples) - len(self.ready)), self.ready])
        y, self.ready = self.ready[: len(samples)], self.ready[len(samples) :]
        return np.clip(np.round(y), -32768, 32767).astype(np.int16).tobytes()

    def _hop(self, samples: np.ndarray) -> np.ndarray:
        self.frame = np.concatenate([self.frame[HOP:], samples])
        spectrum = np.fft.rfft(self.frame * self.window)
        power = spectrum.real**2 + spectrum.imag**2 + 1e-3
        gain = self._gain(power)
        y = np.fft.irfft(spectrum * gain, FRAME).astype(np.float32) * self.window
        self.tail += y
        done = self.tail[:HOP].copy()
        self.tail = np.concatenate([self.tail[HOP:], np.zeros(HOP, np.float32)])
        return done

    def _gain(self, power: np.ndarray) -> np.ndarray:
        self._follow(power)
        noise = self.noise
        after = power / noise  # how far above the noise the band is now
        before = KEEP * self.clean + (1 - KEEP) * np.maximum(after - 1, 0)
        gain = np.maximum(before / (1 + before), FLOOR)
        self.clean = gain**2 * after
        return gain

    def _follow(self, power: np.ndarray) -> None:
        """The quietest smoothed power of the last WINDOW seconds, in each band."""
        if self.smooth is None:
            self.smooth = self.low = power.copy()
            return
        self.smooth = SMOOTH * self.smooth + (1 - SMOOTH) * power
        self.low = np.minimum(self.low, self.smooth)
        self.frames += 1
        if self.frames >= WINDOW * RATE / HOP / PARTS:
            self.lows = [*self.lows[-(PARTS - 1) :], self.low]
            self.low, self.frames = self.smooth.copy(), 0

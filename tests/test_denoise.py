import numpy as np

from prophet_voice.denoise import HOP, RATE, Denoiser


def rms(a) -> float:
    return float(np.sqrt(np.mean(np.asarray(a, float) ** 2)))


def clean(x: np.ndarray, block: int) -> np.ndarray:
    d = Denoiser()
    out = [np.frombuffer(d(x[i : i + block].tobytes()), np.int16) for i in range(0, len(x), block)]
    return np.concatenate(out).astype(float)


def room(seconds: float = 6):
    """A steady hiss, and a voice-like tone in it from 3 to 4.5 s."""
    rng = np.random.default_rng(0)
    t = np.arange(int(RATE * seconds)) / RATE
    noise = rng.normal(0, 300, len(t))
    voice = np.where((t > 3) & (t < 4.5), 1500 * np.sin(2 * np.pi * 220 * t), 0)
    return np.clip(noise + voice, -32768, 32767).astype(np.int16), voice


def test_the_steady_noise_is_lowered_and_the_voice_kept():
    x, voice = room()
    y = clean(x, 640)
    late = HOP  # the denoiser is a hop late
    assert rms(y[RATE + late : 3 * RATE + late]) < rms(x[RATE : 3 * RATE]) / 3
    heard = slice(int(3.2 * RATE), int(4.3 * RATE))
    kept = y[heard.start + late : heard.stop + late]
    assert abs(rms(kept) - rms(voice[heard])) < 0.05 * rms(voice[heard])
    assert rms(kept - voice[heard]) < rms(x[RATE : 3 * RATE]) / 3


def test_each_block_comes_back_as_long_as_it_went_in():
    d = Denoiser()
    for size in (640, 100, 333, 1024, 1):
        assert len(d(np.zeros(size, np.int16).tobytes())) == 2 * size

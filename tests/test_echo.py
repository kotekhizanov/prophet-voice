import numpy as np

from prophet_voice.echo import HOP, RATE, Echo, resample

BLOCK = 640
DELAY = 0.21  # seconds from `played` to the microphone


def rms(a) -> float:
    return float(np.sqrt(np.mean(np.asarray(a, float) ** 2)))


def speech(seconds: float, rate: int, seed: int) -> np.ndarray:
    """Something like a voice: noise in syllables, louder low than high."""
    rng = np.random.default_rng(seed)
    n = int(seconds * rate)
    noise = np.convolve(rng.normal(0, 1, n), np.ones(6) / 6, "same")
    syllables = np.repeat(rng.uniform(0, 1, n // (rate // 8) + 1) > 0.35, rate // 8)[:n]
    loud = np.convolve(syllables.astype(float), np.ones(rate // 50) / (rate // 50), "same")
    return (noise * loud * 9000).astype(np.float32)


def room(played: np.ndarray, seconds: float) -> np.ndarray:
    """What the microphone has of a sound played at 0: later, quieter, ringing, in a hiss."""
    rng = np.random.default_rng(7)
    ringing = rng.normal(0, 1, 400) * np.exp(-np.arange(400) / 80)
    ringing[0] = 3
    back = np.convolve(played, ringing * 0.1)
    mic = rng.normal(0, 30, int(seconds * RATE))
    start = int(DELAY * RATE)
    end = min(len(mic), start + len(back))
    mic[start:end] += back[: end - start]
    return mic


def tone(seconds: float) -> np.ndarray:
    t = np.arange(int(seconds * RATE)) / RATE
    return sum(2500 / k * np.sin(2 * np.pi * 210 * k * t) for k in (1, 2, 3, 5))


def through(echo: Echo, mic: np.ndarray, t0: float = 100.0) -> np.ndarray:
    x = np.clip(mic, -32768, 32767).astype(np.int16)
    out = [
        np.frombuffer(echo(x[i : i + BLOCK].tobytes(), t0 + (i + BLOCK) / RATE), np.int16)
        for i in range(0, len(x), BLOCK)
    ]
    return np.concatenate(out).astype(float)


def test_prophets_voice_is_lowered_and_the_owners_over_it_kept():
    said = speech(10, 24000, 3)
    echo = Echo()
    echo.played(said, 24000, at=100.0)
    mic = room(resample(said, 24000), 10)
    owner = np.zeros(len(mic))
    owner[7 * RATE : 8 * RATE] = tone(1)  # the owner speaks over Prophet from 7 to 8 s
    out = through(echo, mic + owner)
    assert echo.delay is not None and abs(echo.delay * HOP / RATE - DELAY) < 0.05
    alone = slice(3 * RATE, int(6.5 * RATE))  # Prophet alone, once the room is learnt
    assert rms(out[alone.start + HOP : alone.stop + HOP]) < rms(mic[alone]) / 4
    over = slice(int(7.1 * RATE), int(7.9 * RATE))
    kept = out[over.start + HOP : over.stop + HOP]
    assert kept @ owner[over] / (owner[over] @ owner[over]) > 0.6  # the owner's is there
    assert rms(kept - owner[over]) < rms(mic[over])  # and less of Prophet than there was


def test_with_nothing_played_the_audio_passes_as_it_is():
    mic = tone(2) + np.random.default_rng(1).normal(0, 200, 2 * RATE)
    x = np.clip(mic, -32768, 32767).astype(np.int16).astype(float)
    out = through(Echo(), mic)
    assert np.max(np.abs(out[HOP + BLOCK :] - x[BLOCK:-HOP])) <= 1


def test_what_a_hush_cut_off_lowers_nothing():
    said = speech(6, 24000, 4)
    echo = Echo()
    echo.played(said, 24000, at=100.0)
    echo.stopped(at=102.0)  # hushed after 2 s: the rest never sounded
    mic = room(resample(said, 24000)[: 2 * RATE], 6)
    owner = np.zeros(len(mic))
    owner[4 * RATE : 5 * RATE] = tone(1)
    out = through(echo, mic + owner)
    heard = slice(int(4.1 * RATE), int(4.9 * RATE))
    kept = out[heard.start + HOP : heard.stop + HOP]
    assert rms(kept - owner[heard]) < 0.1 * rms(owner[heard])


def test_each_block_comes_back_as_long_as_it_went_in():
    echo = Echo()
    for size in (640, 100, 333, 1024, 1):
        assert len(echo(np.zeros(size, np.int16).tobytes(), 5.0)) == 2 * size


def test_resample_keeps_the_length_in_seconds_and_the_loudness():
    t = np.arange(24000) / 24000
    out = resample((1000 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), 24000)
    assert len(out) == RATE and abs(rms(out) - 1000 / 2**0.5) < 5

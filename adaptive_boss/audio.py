"""Quiet, procedural game sounds with no asset downloads or audio dependency.

Only pygame's mixer is initialized when needed.  An existing mixer is reused
at its actual sample rate and channel count; a missing device is harmless.
"""

from __future__ import annotations

from time import monotonic

import numpy as np
import pygame


_INTERVALS = {
    "shot": 0.08,
    "dash": 0.12,
    "player_hit": 0.20,
    "boss_hit": 0.14,
    "decision": 0.25,
    "win": 0.80,
    "lose": 0.80,
    "click": 0.06,
}


def _tone(rate: int, frequency: float, duration: float, end: float | None = None) -> np.ndarray:
    count = max(1, round(rate * duration))
    time = np.arange(count, dtype=np.float64) / rate
    end = frequency if end is None else end
    phase = 2.0 * np.pi * (frequency * time + (end - frequency) * time * time / (2.0 * duration))
    wave = np.sin(phase) + 0.18 * np.sin(phase * 2.0)
    # A short fade at both ends prevents clicks from abrupt sample boundaries.
    envelope = np.minimum(time / 0.005, 1.0) * np.minimum((duration - time) / 0.018, 1.0)
    envelope = np.clip(envelope, 0.0, 1.0) * np.exp(-2.5 * time / duration)
    return wave * envelope / 1.18


def _waves(rate: int) -> dict[str, np.ndarray]:
    damage = _tone(rate, 110.0, 0.19, 48.0)
    noise = np.random.default_rng(7).uniform(-1.0, 1.0, damage.size)
    noise *= np.linspace(1.0, 0.0, damage.size) ** 2
    noise[: min(128, damage.size)] *= np.linspace(0.0, 1.0, min(128, damage.size))
    return {
        "shot": _tone(rate, 820.0, 0.055, 390.0),
        "dash": _tone(rate, 420.0, 0.13, 980.0),
        "player_hit": damage * 0.75 + noise * 0.15,
        "boss_hit": _tone(rate, 170.0, 0.065, 85.0) * 0.7,
        "decision": np.concatenate((_tone(rate, 340.0, 0.055), _tone(rate, 440.0, 0.075))),
        "win": np.concatenate([_tone(rate, frequency, 0.12) for frequency in (523.25, 659.25, 783.99)]),
        "lose": np.concatenate([_tone(rate, frequency, 0.15) for frequency in (294.0, 220.0, 147.0)]),
        "click": _tone(rate, 660.0, 0.045),
    }


def _pcm(wave: np.ndarray, sample_format: int, channels: int) -> np.ndarray:
    # The mixer's signed 16-bit default and both 8-bit modes cover common SDL
    # configurations.  Handle other integer formats without reinitializing an
    # application's existing audio device.
    scaled = np.clip(wave * 0.24, -1.0, 1.0)
    formats = {
        -8: (np.int8, 127.0, 0.0),
        8: (np.uint8, 127.0, 128.0),
        -16: (np.int16, 32767.0, 0.0),
        16: (np.uint16, 32767.0, 32768.0),
    }
    if sample_format == -32:
        # SDL AUDIO_F32 uses normalized IEEE float samples.
        samples = scaled.astype(np.float32)
    elif sample_format in formats:
        dtype, multiplier, offset = formats[sample_format]
        samples = (scaled * multiplier + offset).astype(dtype)
    else:
        raise ValueError("unsupported mixer sample format")
    if channels > 1:
        samples = np.repeat(samples[:, np.newaxis], channels, axis=1)
    return np.ascontiguousarray(samples)


class SoundBank:
    """Procedural effects with independent mute, rate limits, and safe cleanup.

    ``enabled`` reflects the user's preference; ``available`` reports whether
    a working audio device and sound bank exist.  If audio is unavailable,
    ``play`` is a silent no-op.  Unknown event names are also ignored.
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self.available = False
        self._sounds: dict[str, pygame.mixer.Sound] = {}
        self._last_play: dict[str, float] = {}
        self._owns_mixer = False
        self._closed = False
        if self.enabled:
            self._prepare()

    def _prepare(self) -> None:
        try:
            configuration = pygame.mixer.get_init()
            if configuration is None:
                pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
                self._owns_mixer = True
                configuration = pygame.mixer.get_init()
            if configuration is None:
                return
            rate, sample_format, channels = configuration
            for name, wave in _waves(rate).items():
                sound = pygame.sndarray.make_sound(_pcm(wave, sample_format, channels))
                sound.set_volume(0.40)
                self._sounds[name] = sound
            self.available = True
        except (pygame.error, ValueError, TypeError, OverflowError):
            self._sounds.clear()
            self.available = False
            if self._owns_mixer:
                pygame.mixer.quit()
                self._owns_mixer = False

    def set_enabled(self, enabled: bool) -> None:
        if self._closed:
            return
        self.enabled = bool(enabled)
        if self.enabled and not self.available:
            self._prepare()
        elif not self.enabled:
            for sound in self._sounds.values():
                try:
                    sound.stop()
                except pygame.error:
                    pass

    def play(self, event_name: str) -> None:
        if self._closed or not self.enabled or not self.available:
            return
        sound = self._sounds.get(event_name)
        if sound is None:
            return
        now = monotonic()
        previous = self._last_play.get(event_name)
        if previous is not None and now - previous < _INTERVALS[event_name]:
            return
        try:
            sound.play()
            self._last_play[event_name] = now
        except pygame.error:
            self.available = False

    def close(self) -> None:
        if self._closed:
            return
        self.set_enabled(False)
        self._sounds.clear()
        self._last_play.clear()
        if self._owns_mixer:
            pygame.mixer.quit()
            self._owns_mixer = False
        self.available = False
        self._closed = True

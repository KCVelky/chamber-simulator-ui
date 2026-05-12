from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class BandpassSpec:
    f_low: float
    f_high: float

    def __post_init__(self) -> None:
        if not (self.f_low > 0):
            raise ValueError("f_low must be > 0 Hz")
        if not (self.f_high > self.f_low):
            raise ValueError("f_high must be > f_low")


@dataclass(frozen=True)
class ModulationSpec:

    """Slow amplitude modulation: A(t) = 1 + depth * sin(2π f_mod t)."""
    f_mod: float = 0.3     # Hz (slow)
    depth: float = 0.2     # 0..1 typically

    def __post_init__(self) -> None:
        if self.f_mod <= 0:
            raise ValueError("f_mod must be > 0 Hz")
        if not (0.0 <= self.depth):
            raise ValueError("depth must be >= 0")
        
@dataclass(frozen=True)
class FadeSpec:
    """Apply a smooth fade-in/out to reduce edge effects."""
    fade_in: float = 0.0   # seconds
    fade_out: float = 0.0  # seconds

def __post_init__(self) -> None:
    if self.fade_in < 0 or self.fade_out < 0:
        raise ValueError("fade_in and fade_out must be >= 0")


@dataclass(frozen=True)
class SourceSignalConfig:
    # --- required (no defaults) first ---
    fs: float
    duration: float
    band: BandpassSpec

    # --- optional (defaults) after ---
    rms: float = 1.0
    modulation: Optional[ModulationSpec] = None
    fade: Optional[FadeSpec] = None
    seed: Optional[int] = 0

    def __post_init__(self) -> None:
        if self.fs <= 0:
            raise ValueError("fs must be > 0")
        if self.duration <= 0:
            raise ValueError("duration must be > 0")
        if self.band.f_high >= 0.49 * self.fs:
            raise ValueError("f_high is too close to Nyquist (fs/2). Lower f_high or increase fs.")
        if self.rms <= 0:
            raise ValueError("rms must be > 0")


def _bandpass_fft_filter(x: np.ndarray, fs: float, f_low: float, f_high: float) -> np.ndarray:
    """
    Simple, dependency-free bandpass via FFT masking.
    Pros: easy, stable, no scipy.
    Cons: assumes periodic extension (edge effects). OK for simulation; can fade-in/out later.
    """
    n = x.size
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    mask = (freqs >= f_low) & (freqs <= f_high)
    X_filt = np.zeros_like(X)
    X_filt[mask] = X[mask]

    y = np.fft.irfft(X_filt, n=n)
    return y


def _set_rms(x: np.ndarray, target_rms: float) -> np.ndarray:

    rms = np.sqrt(np.mean(x**2))
    if rms == 0:
        raise ValueError("Signal RMS is zero; cannot normalize.")
    return x * (target_rms / rms)

def _apply_fade(s: np.ndarray, fs: float, fade_in_s: float, fade_out_s: float) -> np.ndarray:
        n = s.size
        w = np.ones(n, dtype=float)

        n_in = int(round(fade_in_s * fs))
        n_out = int(round(fade_out_s * fs))

        if n_in > 0:
            n_in = min(n_in, n)
            win = 0.5 * (1 - np.cos(np.linspace(0, np.pi, n_in)))
            w[:n_in] *= win

        if n_out > 0:
            n_out = min(n_out, n)
            wout = 0.5 * (1 - np.cos(np.linspace(np.pi, 0, n_out)))
            w[-n_out:] *= wout

        return s * w

def generate_source_signal(cfg: SourceSignalConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      t: time vector (N,)
      s: source signal (N,)
    """
    rng = np.random.default_rng(cfg.seed)
    n = int(round(cfg.duration * cfg.fs))
    if n < 8:
        raise ValueError("duration*fs too small; increase duration or fs.")

    t = np.arange(n) / cfg.fs

    # 1) Start from white noise
    x = rng.standard_normal(n)

    # 2) Band-limit it
    s = _bandpass_fft_filter(x, cfg.fs, cfg.band.f_low, cfg.band.f_high)

    # 3) Optional slow amplitude modulation
    if cfg.modulation is not None:
        A = 1.0 + cfg.modulation.depth * np.sin(2.0 * np.pi * cfg.modulation.f_mod * t)
        s = s * A

    if cfg.fade is not None:
        s = _apply_fade(s, cfg.fs, cfg.fade.fade_in, cfg.fade.fade_out)

    # 4) Normalize to desired RMS
    s = _set_rms(s, cfg.rms)

    return t, s

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class PropagationConfig:
    c: float = 343.0  # speed of sound [m/s]
    include_spherical_spreading: bool = True
    gain_at_1m: float = 1.0  # amplitude at 1 meter (arbitrary scaling)
    include_rigid_floor_image: bool = False
    floor_z: float = 0.0
    eps_distance: float = 1e-6  # avoid division by zero


def _fractional_delay_fft(x: np.ndarray, fs: float, delay_s: float) -> np.ndarray:
    """
    Apply a fractional delay using frequency-domain phase shift.
    y(t) = x(t - delay_s)
    """
    n = x.size
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    phase = np.exp(-1j * 2.0 * np.pi * freqs * delay_s)
    Y = X * phase
    y = np.fft.irfft(Y, n=n)
    return y


def _path_contribution(
    s: np.ndarray,
    fs: float,
    src_pos: np.ndarray,
    mic_pos: np.ndarray,
    cfg: PropagationConfig,
) -> np.ndarray:
    """
    One propagation path (direct OR image): delay + spherical spreading.
    """
    r = float(np.linalg.norm(mic_pos - src_pos))
    r = max(r, cfg.eps_distance)

    tau = r / cfg.c
    y = _fractional_delay_fft(s, fs, tau)

    if cfg.include_spherical_spreading:
        # amplitude scaling: gain_at_1m / r (relative to 1m)
        y = y * (cfg.gain_at_1m / r)

    return y


def simulate_mic_signals_free_field(
    scene,
    t: np.ndarray,
    s: np.ndarray,
    cfg: Optional[PropagationConfig] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate microphone signals for the FIRST source in the scene.

    Inputs:
      scene: must have scene.mic_array.positions (M,3), scene.sources[0].position (3,)
      t: time vector (N,)
      s: source signal (N,)
      cfg: propagation configuration

    Returns:
      y: mic signals (M, N)
      distances: direct-path distances (M,)
    """
    if cfg is None:
        cfg = PropagationConfig()

    mics = np.asarray(scene.mic_array.positions, dtype=float)
    src = np.asarray(scene.sources[0].position, dtype=float).reshape(3,)
    fs = 1.0 / float(t[1] - t[0])

    M = mics.shape[0]
    N = s.size
    y = np.zeros((M, N), dtype=float)

    # Direct path
    distances = np.linalg.norm(mics - src[None, :], axis=1)

    for i in range(M):
        y[i, :] += _path_contribution(s, fs, src, mics[i], cfg)

    # Optional rigid floor image source (semi-anechoic quick model)
    if cfg.include_rigid_floor_image:
        src_img = src.copy()
        src_img[2] = 2.0 * cfg.floor_z - src[2]  # mirror across z=floor_z

        # For rigid boundary (Neumann), pressure reflection coefficient ≈ +1
        for i in range(M):
            y[i, :] += _path_contribution(s, fs, src_img, mics[i], cfg)

    return y, distances

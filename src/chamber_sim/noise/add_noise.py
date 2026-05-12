from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class NoiseConfig:
    snr_db: float = 20.0               # target SNR in dB
    per_mic_independent: bool = True   # independent noise on each mic
    seed: Optional[int] = 0            # None => random each run


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, dtype=float) ** 2)))


def add_awgn_snr(y: np.ndarray, cfg: NoiseConfig) -> Tuple[np.ndarray, float]:
    """
    Add white Gaussian noise to multichannel signal y (M,N) to reach target SNR.

    SNR definition used:
      snr_db = 20 log10( rms(signal) / rms(noise) )

    Returns:
      y_noisy: (M,N)
      noise_rms_used: scalar (global or per-channel target depends on mode)
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 2:
        raise ValueError("y must have shape (M, N)")

    rng = np.random.default_rng(cfg.seed)

    M, N = y.shape
    y_noisy = y.copy()

    # Compute reference signal RMS
    sig_rms = _rms(y)  # global RMS across all mics/samples
    if sig_rms == 0:
        raise ValueError("Signal RMS is zero; cannot set SNR.")

    noise_rms = sig_rms / (10 ** (cfg.snr_db / 20))

    if cfg.per_mic_independent:
        n = rng.standard_normal((M, N))
        # normalize each mic noise to RMS=1 then scale to target
        for i in range(M):
            ni_rms = _rms(n[i])
            if ni_rms == 0:
                continue
            n[i] = n[i] / ni_rms
        n *= noise_rms
    else:
        n = rng.standard_normal((M, N))
        n_rms = _rms(n)
        n = (n / n_rms) * noise_rms

    y_noisy += n
    return y_noisy, noise_rms

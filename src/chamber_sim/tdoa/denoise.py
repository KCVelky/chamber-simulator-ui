from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class DenoiseConfig:
    # STFT
    nperseg: int = 1024
    noverlap: int = 768
    window: str = "hann"

    # bruit: on estime Pn(f) sur une portion "bruit seul"
    noise_head_s: float = 0.20   # secondes au début
    noise_tail_s: float = 0.20   # secondes à la fin

    # Wiener
    eps: float = 1e-12
    gain_floor: float = 0.05     # évite d'éteindre totalement (artefacts)
    use_median_noise: bool = True  # médiane des PSD (robuste)

    # option: limiter à une bande [f_low, f_high] (utile si tu connais ta bande source)
    f_low: Optional[float] = None
    f_high: Optional[float] = None


def _stft(x: np.ndarray, fs: float, nperseg: int, noverlap: int, window: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, float).ravel()
    hop = nperseg - noverlap
    if hop <= 0:
        raise ValueError("noverlap must be < nperseg")

    if window == "hann":
        win = np.hanning(nperseg)
    else:
        raise ValueError(f"Unsupported window: {window}")

    # zero-pad to fit frames
    n = x.size
    n_frames = 1 + int(np.ceil((n - nperseg) / hop)) if n > nperseg else 1
    pad = (n_frames - 1) * hop + nperseg - n
    if pad > 0:
        x = np.pad(x, (0, pad))

    frames = np.lib.stride_tricks.sliding_window_view(x, nperseg)[::hop]
    frames = frames * win[None, :]

    X = np.fft.rfft(frames, axis=1)
    freqs = np.fft.rfftfreq(nperseg, d=1.0/fs)
    times = (np.arange(frames.shape[0]) * hop + 0.5*nperseg) / fs
    return X, freqs, times


def _istft(X: np.ndarray, fs: float, nperseg: int, noverlap: int, window: str, length: int) -> np.ndarray:
    hop = nperseg - noverlap
    if window == "hann":
        win = np.hanning(nperseg)
    else:
        raise ValueError(f"Unsupported window: {window}")

    frames = np.fft.irfft(X, n=nperseg, axis=1)

    out_len = (frames.shape[0] - 1) * hop + nperseg
    y = np.zeros(out_len, dtype=float)
    wsum = np.zeros(out_len, dtype=float)

    for k in range(frames.shape[0]):
        start = k * hop
        y[start:start+nperseg] += frames[k] * win
        wsum[start:start+nperseg] += win**2

    y = y / np.maximum(wsum, 1e-12)
    return y[:length]


def denoise_wiener_stft(
    y: np.ndarray,   # (M,N) ou (N,)
    fs: float,
    cfg: DenoiseConfig = DenoiseConfig(),
) -> np.ndarray:
    """
    Débruitage Wiener STFT canal par canal.
    Bruit estimé via autospectre sur head/tail (zones "bruit seul").
    """
    y = np.asarray(y, float)
    if y.ndim == 1:
        y = y[None, :]
    M, N = y.shape

    nhead = int(cfg.noise_head_s * fs)
    ntail = int(cfg.noise_tail_s * fs)
    if nhead + ntail <= 0:
        raise ValueError("noise_head_s + noise_tail_s must be > 0 to estimate noise.")
    if nhead + ntail >= N:
        raise ValueError("Noise-only segments too long vs signal length.")

    y_out = np.zeros_like(y)

    for m in range(M):
        xm = y[m] - np.mean(y[m])  # stabilise
        X, freqs, _ = _stft(xm, fs, cfg.nperseg, cfg.noverlap, cfg.window)

        Pyy = np.abs(X)**2  # (T,F)

        # définir quelles frames sont "bruit seul"
        # On utilise une mask temps via indices échantillons approx:
        # head: frames dont centre < nhead
        # tail: frames dont centre > N-ntail
        hop = cfg.nperseg - cfg.noverlap
        centers = (np.arange(Pyy.shape[0]) * hop + 0.5*cfg.nperseg).astype(int)
        noise_mask = (centers < nhead) | (centers > (N - ntail))

        if not np.any(noise_mask):
            raise ValueError("No noise-only frames found; increase noise_head_s/noise_tail_s or adjust STFT params.")

        if cfg.use_median_noise:
            Pn = np.median(Pyy[noise_mask, :], axis=0)
        else:
            Pn = np.mean(Pyy[noise_mask, :], axis=0)

        # option: bande
        band_mask = np.ones_like(freqs, dtype=bool)
        if cfg.f_low is not None:
            band_mask &= (freqs >= cfg.f_low)
        if cfg.f_high is not None:
            band_mask &= (freqs <= cfg.f_high)

        # Wiener gain
        # Ps_hat = max(Pyy - Pn, 0)
        Ps = np.maximum(Pyy - Pn[None, :], 0.0)
        G = Ps / (Ps + Pn[None, :] + cfg.eps)

        # sécurité: floor + bande (hors bande on garde tel quel ou on atténue)
        G = np.maximum(G, cfg.gain_floor)
        if (cfg.f_low is not None) or (cfg.f_high is not None):
            G[:, ~band_mask] = cfg.gain_floor  # on écrase hors bande

        X_hat = X * G
        x_hat = _istft(X_hat, fs, cfg.nperseg, cfg.noverlap, cfg.window, length=N)
        y_out[m] = x_hat

    return y_out if y_out.shape[0] > 1 else y_out[0]

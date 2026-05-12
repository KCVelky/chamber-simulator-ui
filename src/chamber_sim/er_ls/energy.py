from __future__ import annotations

import numpy as np

from .config import EnergyConfig


def estimate_mic_energy(
    y: np.ndarray,
    fs: float,
    cfg: EnergyConfig = EnergyConfig(),
) -> np.ndarray:
    """
    Estime l'énergie reçue par chaque microphone.

    Paramètres
    ----------
    y : ndarray, shape (M, N)
        Signaux temporels des microphones.
        M = nombre de micros, N = nombre d'échantillons.

    fs : float
        Fréquence d'échantillonnage [Hz].

    cfg : EnergyConfig
        Configuration du calcul d'énergie.

    Retour
    ------
    E : ndarray, shape (M,)
        Energie moyenne estimée pour chaque micro.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 2:
        raise ValueError("y must be of shape (M, N)")

    M, N = y.shape

    # -------------------------------------------------
    # Prétraitement : retrait de la moyenne (DC)
    # -------------------------------------------------
    yy = y.copy()
    if cfg.remove_mean:
        yy = yy - np.mean(yy, axis=1, keepdims=True)

    # -------------------------------------------------
    # Cas 1 : énergie sur tout le signal
    # -------------------------------------------------
    if cfg.window_s is None:
        # E_i = mean_t ( y_i(t)^2 )
        E = np.mean(yy ** 2, axis=1)
        return E

    # -------------------------------------------------
    # Cas 2 : énergie par fenêtres temporelles
    # -------------------------------------------------
    win_len = int(round(cfg.window_s * fs))
    if win_len <= 0 or win_len > N:
        raise ValueError("window_s incompatible with signal length")

    if cfg.hop_s is None:
        hop_len = win_len // 2
    else:
        hop_len = int(round(cfg.hop_s * fs))
    hop_len = max(1, hop_len)

    # Calcul des énergies par frame
    frames_energy = []

    for start in range(0, N - win_len + 1, hop_len):
        seg = yy[:, start:start + win_len]   # (M, win_len)
        Ei = np.mean(seg ** 2, axis=1)        # (M,)
        frames_energy.append(Ei)

    if len(frames_energy) == 0:
        raise RuntimeError("No frames available for energy estimation")

    frames_energy = np.stack(frames_energy, axis=0)  # (T, M)

    # -------------------------------------------------
    # Agrégation robuste (trimmed mean)
    # -------------------------------------------------
    if cfg.trim_frac > 0.0:
        T = frames_energy.shape[0]
        k = int(np.floor(cfg.trim_frac * T))

        # On ne trim que si ça a du sens
        if 2 * k < T - 1:
            frames_energy_sorted = np.sort(frames_energy, axis=0)
            frames_energy = frames_energy_sorted[k:T - k, :]

    # Energie finale = moyenne des frames restantes
    E = np.mean(frames_energy, axis=0)

    return E

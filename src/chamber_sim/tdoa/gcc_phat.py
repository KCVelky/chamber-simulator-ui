from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass(frozen=True)
class GCCPHATConfig:
    max_tau: Optional[float] = None   # [s] limite de recherche autour de 0 (physique: d_mn/c)
    interp: int = 8                   # facteur d'interpolation (résolution sous-échantillon)
    eps: float = 1e-12                # évite division par 0 dans 1/|Cmn|
    remove_mean: bool = True          # retire la composante DC (utile en bruit/offset)


def gcc_phat_tdoa(
    x: np.ndarray,
    y: np.ndarray,
    fs: float,
    cfg: Optional[GCCPHATConfig] = None,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    GCC-PHAT suivant:
        R_xy(tau) = ∫ [ C_xy(f)/|C_xy(f)| ] e^{j2π f tau} df
    où C_xy(f) = X(f) Y*(f).

    Convention de tau_hat (cohérente avec cette implémentation):
      tau_hat est l'argument du pic de |R_xy(tau)|.
      Interprétation pratique: si tau_hat > 0, le pic est à +tau.
      (Ensuite, dans ton pipeline tu fixes une convention unique x=ref, y=mic_i.)

    Returns:
      tau_hat [s], cc (corrélation centrée), tau_axis [s]
    """
    if cfg is None:
        cfg = GCCPHATConfig()

    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    if x.size == 0 or y.size == 0:
        raise ValueError("Empty input signals.")

    # Retrait DC (stabilise fortement en bruit / offsets)
    if cfg.remove_mean:
        x = x - np.mean(x)
        y = y - np.mean(y)

    n = x.size + y.size
    nfft = int(2 ** np.ceil(np.log2(n)))

    X = np.fft.rfft(x, n=nfft)
    Y = np.fft.rfft(y, n=nfft)

    # C_xy(f) = X * conj(Y)
    C = X * np.conj(Y)

    # PHAT : W(f)=1/|C|  =>  C/|C|
    C_phat = C / (np.abs(C) + cfg.eps)

    # R_xy(tau) via iFFT, avec interpolation par longueur de sortie
    cc = np.fft.irfft(C_phat, n=cfg.interp * nfft)

    # Recentre (tau=0 au milieu)
    max_shift = int(cfg.interp * nfft // 2)
    cc = np.concatenate((cc[-max_shift:], cc[:max_shift + 1]))

    tau_axis = (np.arange(-max_shift, max_shift + 1) / (cfg.interp * fs)).astype(float)

    # Fenêtre physique ±max_tau
    if cfg.max_tau is not None:
        if cfg.max_tau <= 0:
            raise ValueError("cfg.max_tau must be > 0 when provided.")
        mask = (tau_axis >= -cfg.max_tau) & (tau_axis <= cfg.max_tau)
        if not np.any(mask):
            raise ValueError("max_tau window is empty (check fs/interp/max_tau).")
        cc_search = cc[mask]
        tau_search = tau_axis[mask]
        idx = int(np.argmax(np.abs(cc_search)))
        tau_hat = float(tau_search[idx])
        return tau_hat, cc, tau_axis

    # Sinon: recherche globale
    idx = int(np.argmax(np.abs(cc)))
    tau_hat = float(tau_axis[idx])
    return tau_hat, cc, tau_axis

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from chamber_sim.tdoa import GCCPHATConfig, gcc_phat_tdoa
from chamber_sim.tdoa import GridSearchConfig, localize_tdoa_grid
from chamber_sim.tdoa.denoise import DenoiseConfig, denoise_wiener_stft


@dataclass(frozen=True)
class MultiRefTDOAConfig:
    # GCC-PHAT
    interp: int = 8
    eps: float = 1e-12

    # Denoise
    enable_denoise: bool = True
    denoise: DenoiseConfig = DenoiseConfig()

    # Fusion
    fusion: str = "median"  # "median" (recommandé) ou "mean"


def estimate_source_multiref_tdoa(
    mic_positions: np.ndarray,      # (M,3)
    y: np.ndarray,                  # (M,N)
    fs: float,
    c: float,
    room_size: np.ndarray,          # (3,)
    grid_cfg: GridSearchConfig,
    cfg: MultiRefTDOAConfig = MultiRefTDOAConfig(),
) -> Tuple[np.ndarray, dict]:
    """
    1) (option) débruitage canal par canal via autospectre (Wiener STFT)
    2) pour chaque micro ref = m:
       - calcule tdoa relatif à m via GCC-PHAT (max_tau = dist(m,i)/c)
       - localise par grille (ref_idx=m)
    3) fusion des positions (médiane par défaut)

    Returns:
      x_fused (3,), debug dict
    """
    mics = np.asarray(mic_positions, float)
    y = np.asarray(y, float)
    if y.ndim != 2:
        raise ValueError("y must be shape (M,N)")
    M, N = y.shape
    if mics.shape != (M, 3):
        raise ValueError("mic_positions must be shape (M,3) and match y")

    y_proc = y
    if cfg.enable_denoise:
        y_proc = denoise_wiener_stft(y_proc, fs, cfg.denoise)

    x_hats = []
    costs = []
    tdoa_all = []

    for m in range(M):
        tdoa = np.zeros(M, dtype=float)
        for i in range(M):
            if i == m:
                continue
            d_mi = float(np.linalg.norm(mics[i] - mics[m]))
            max_tau = d_mi / float(c)

            gcc_cfg = GCCPHATConfig(max_tau=max_tau, interp=int(cfg.interp), eps=float(cfg.eps))
            tau_hat, _, _ = gcc_phat_tdoa(y_proc[m], y_proc[i], fs, gcc_cfg)
            tdoa[i] = -tau_hat


        x_hat_m, cost_m = localize_tdoa_grid(
            mic_positions=mics,
            tdoa=tdoa,
            room_size=np.asarray(room_size, float),
            c=float(c),
            cfg=grid_cfg,
            ref_idx=m,
        )

        x_hats.append(x_hat_m)
        costs.append(cost_m)
        tdoa_all.append(tdoa)

    X = np.vstack(x_hats)  # (M,3)
    if cfg.fusion.lower() == "median":
        x_fused = np.median(X, axis=0)
    elif cfg.fusion.lower() == "mean":
        x_fused = np.mean(X, axis=0)
    else:
        raise ValueError("fusion must be 'median' or 'mean'")

    debug = {
        "x_hats_per_ref": X,
        "costs_per_ref": np.asarray(costs, float),
        "tdoa_per_ref": np.asarray(tdoa_all, float),
        "y_used": y_proc,
    }
    return x_fused, debug

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

import numpy as np

from .config import EnergyConfig, ERLSConfig
from .energy import estimate_mic_energy
from .geometry import (
    kappa_from_energy,
    build_apollonius_spheres,
    ApolloniusSphere,
)


@dataclass(frozen=True)
class ERLSResult:
    """
    Résultat structuré (optionnel). Tu peux utiliser soit ce dataclass,
    soit directement le tuple (x_hat, err, debug) via er_ls().
    """
    x_hat: np.ndarray   # (3,)
    err: float          # RMS des résidus sphériques [m]
    debug: Dict[str, Any]


def _solve_ls_from_spheres(spheres: List[ApolloniusSphere]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Résout la position x par LS à partir d'une liste de sphères:
        ||x - c_i|| = rho_i

    Linéarisation via différence de sphères :
        ||x-c||^2 = rho^2
    Soustraction avec la sphère 0 pour éliminer x^T x :

      2(c_i - c_0)^T x = (||c_i||^2 - rho_i^2) - (||c_0||^2 - rho_0^2)

    Retour:
      x_hat (3,), A, b  (pour debug)
    """
    if len(spheres) < 2:
        raise ValueError("Need at least 2 spheres to build LS system.")

    c0 = np.asarray(spheres[0].c, float).ravel()
    r0 = float(spheres[0].rho)

    A_rows = []
    b_rows = []

    for sp in spheres[1:]:
        c = np.asarray(sp.c, float).ravel()
        rho = float(sp.rho)

        A_rows.append(2.0 * (c - c0))
        b_rows.append((c @ c - rho * rho) - (c0 @ c0 - r0 * r0))

    A = np.vstack(A_rows)  # (K,3)
    b = np.asarray(b_rows, dtype=float)  # (K,)

    x_hat, *_ = np.linalg.lstsq(A, b, rcond=None)
    return x_hat.astype(float), A, b


def er_ls(
    mic_positions: np.ndarray,          # (M,3)
    y: np.ndarray,                      # (M,N)
    fs: float,
    cfg_E: EnergyConfig = EnergyConfig(),
    cfg: ERLSConfig = ERLSConfig(),
    mic_gains: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """
    ER-LS (Energy Ratios - Least Squares)

    Entrées
    -------
    mic_positions : (M,3)
        Positions des micros
    y : (M,N)
        Signaux micro
    fs : float
        fréquence d'échantillonnage
    cfg_E : EnergyConfig
        config estimation énergie
    cfg : ERLSConfig
        config ER-LS
    mic_gains : (M,) ou None
        gains relatifs des micros (si connus). Sinon tous à 1.

    Sorties
    -------
    x_hat : (3,)
        position estimée
    err : float
        RMS des résidus sphériques ||x - c_i|| - rho_i (mètres)
        -> indicateur de cohérence du modèle (pas l'erreur vraie).
    debug : dict
        infos utiles (E, kappa, spheres, A, b, residuals, etc.)
    """
    mics = np.asarray(mic_positions, dtype=float)
    if mics.ndim != 2 or mics.shape[1] != 3:
        raise ValueError("mic_positions must be (M,3)")

    y = np.asarray(y, dtype=float)
    if y.ndim != 2 or y.shape[0] != mics.shape[0]:
        raise ValueError("y must be (M,N) and match mic_positions")

    M = mics.shape[0]
    ref_idx = int(cfg.ref_idx)
    if not (0 <= ref_idx < M):
        raise ValueError("cfg.ref_idx out of range")

    # 1) Energies
    E = estimate_mic_energy(y, fs, cfg_E)

    # 2) κ_i
    kappa = kappa_from_energy(E, ref_idx=ref_idx, mic_gains=mic_gains)

    # 3) Sphères d’Apollonius
    spheres = build_apollonius_spheres(
        mic_positions=mics,
        kappa=kappa,
        ref_idx=ref_idx,
        kappa_eps=float(cfg.kappa_eps),
    )

    if len(spheres) < cfg.min_pairs:
        raise ValueError(f"Not enough valid spheres for ER-LS (got {len(spheres)}).")

    # 4) LS
    x_hat, A, b = _solve_ls_from_spheres(spheres)

    # 5) Résidus sphériques
    residuals = []
    for sp in spheres:
        residuals.append(np.linalg.norm(x_hat - sp.c) - sp.rho)
    residuals = np.asarray(residuals, dtype=float)
    err = float(np.sqrt(np.mean(residuals**2)))

    debug: Dict[str, Any] = {
        "E": E,
        "kappa": kappa,
        "ref_idx": ref_idx,
        "spheres": spheres,      # list[ApolloniusSphere]
        "A": A,
        "b": b,
        "residuals": residuals,
        "used_mic_indices": np.array([sp.mic_idx for sp in spheres], dtype=int),
    }

    return x_hat, err, debug

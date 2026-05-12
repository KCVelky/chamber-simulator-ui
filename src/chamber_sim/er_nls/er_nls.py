from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

import numpy as np

from .config import ERNLSConfig

from chamber_sim.er_ls import EnergyConfig, ERLSConfig, er_ls
from chamber_sim.er_ls.geometry import (
    kappa_from_energy,
    build_apollonius_spheres,
    ApolloniusSphere,
)
from chamber_sim.er_ls.energy import estimate_mic_energy


@dataclass(frozen=True)
class ERNLSResult:
    x_hat: np.ndarray
    err: float
    debug: Dict[str, Any]


def _residuals_and_jacobian(
    x: np.ndarray,
    spheres: List[ApolloniusSphere],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    r_i(x) = ||x - c_i|| - rho_i
    J_i(x) = dr_i/dx = (x - c_i)/||x - c_i||

    Retour:
      r: (K,)
      J: (K,3)
    """
    x = np.asarray(x, float).ravel()
    r_list = []
    J_list = []

    for sp in spheres:
        c = np.asarray(sp.c, float).ravel()
        rho = float(sp.rho)

        v = x - c
        d = float(np.linalg.norm(v))
        d = max(d, 1e-12)  # évite division par 0

        r_i = d - rho
        J_i = v / d

        r_list.append(r_i)
        J_list.append(J_i)

    r = np.asarray(r_list, float)
    J = np.vstack(J_list)
    return r, J


def er_nls(
    mic_positions: np.ndarray,              # (M,3)
    y: np.ndarray,                          # (M,N)
    fs: float,
    x0: Optional[np.ndarray] = None,        # init (si None -> ER-LS)
    cfg_E: EnergyConfig = EnergyConfig(),
    cfg_ls: ERLSConfig = ERLSConfig(),
    cfg: ERNLSConfig = ERNLSConfig(),
    mic_gains: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """
    ER-NLS : raffinement non linéaire basé sur les sphères d’Apollonius.

    Étapes :
      1) Estime E_i (comme ER-LS)
      2) Calcule κ_i (ratios d'énergie)
      3) Construit sphères d'Apollonius (i, ref)
      4) Init x0 :
         - si fourni: utilise x0
         - sinon: utilise ER-LS
      5) Optimisation LM :
         minimise sum_i (||x - c_i|| - rho_i)^2

    Sorties :
      x_hat (3,)
      err : RMS des résidus sphériques (m)
      debug : dict (history, résidus finaux, etc.)
    """
    mics = np.asarray(mic_positions, dtype=float)
    if mics.ndim != 2 or mics.shape[1] != 3:
        raise ValueError("mic_positions must be (M,3)")

    y = np.asarray(y, dtype=float)
    if y.ndim != 2 or y.shape[0] != mics.shape[0]:
        raise ValueError("y must be (M,N) and match mic_positions")

    M = mics.shape[0]
    ref_idx = int(cfg_ls.ref_idx)
    if not (0 <= ref_idx < M):
        raise ValueError("cfg_ls.ref_idx out of range")

    # 1) Energies
    E = estimate_mic_energy(y, fs, cfg_E)

    # 2) κ_i
    kappa = kappa_from_energy(E, ref_idx=ref_idx, mic_gains=mic_gains)

    # 3) Sphères
    spheres = build_apollonius_spheres(
        mic_positions=mics,
        kappa=kappa,
        ref_idx=ref_idx,
        kappa_eps=float(cfg.kappa_eps),
    )
    if len(spheres) < 3:
        raise ValueError(f"Not enough valid spheres for ER-NLS (got {len(spheres)}).")

    # 4) Init
    if x0 is None:
        x0, _, _ = er_ls(
            mic_positions=mics,
            y=y,
            fs=fs,
            cfg_E=cfg_E,
            cfg=cfg_ls,
            mic_gains=mic_gains,
        )
    x = np.asarray(x0, float).ravel()
    if x.size != 3:
        raise ValueError("x0 must be shape (3,)")

    # 5) LM iterations
    lam = float(cfg.lam)
    history_cost: List[float] = []
    history_step: List[float] = []

    r, J = _residuals_and_jacobian(x, spheres)
    cost = float(r @ r)
    history_cost.append(cost)
    history_step.append(0.0)

    for _ in range(int(cfg.max_iter)):
        r, J = _residuals_and_jacobian(x, spheres)
        H = J.T @ J
        g = J.T @ r

        # (H + lam I) dx = -g
        dx = np.linalg.solve(H + lam * np.eye(3), -g)

        x_new = x + dx
        r_new, _ = _residuals_and_jacobian(x_new, spheres)
        cost_new = float(r_new @ r_new)

        step_norm = float(np.linalg.norm(dx))
        history_step.append(step_norm)

        if cost_new < cost:
            # accepte
            x = x_new
            prev_cost = cost
            cost = cost_new
            history_cost.append(cost)

            # diminue lambda si on progresse
            lam = max(lam / 2.0, 1e-8)

            if step_norm < cfg.tol_step:
                break
            if abs(prev_cost - cost) < cfg.tol_cost:
                break
        else:
            # refuse, augmente lambda
            lam = min(lam * 2.0, 1e6)

    # Résidu final
    r_final, _ = _residuals_and_jacobian(x, spheres)
    err = float(np.sqrt(np.mean(r_final**2)))

    debug: Dict[str, Any] = {
        "E": E,
        "kappa": kappa,
        "ref_idx": ref_idx,
        "spheres": spheres,
        "x0": np.asarray(x0, float),
        "history_cost": np.asarray(history_cost, float),
        "history_step": np.asarray(history_step, float),
        "final_residuals": r_final,
        "lambda_final": lam,
        "n_spheres": len(spheres),
        "used_mic_indices": np.array([sp.mic_idx for sp in spheres], dtype=int),
    }

    return x.astype(float), err, debug

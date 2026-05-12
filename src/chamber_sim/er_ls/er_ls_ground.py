from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

import numpy as np

from .energy import estimate_mic_energy
from .config import EnergyConfig, ERLSConfig
from .ground_model import GroundConfig, residuals_and_jacobian_energy


@dataclass(frozen=True)
class ERLSGroundConfig:
    max_iter: int = 30
    lam: float = 1e-2
    tol_step: float = 1e-6
    tol_cost: float = 1e-9

    # beta search (robuste)
    beta_min: float = 0.0
    beta_max: float = 2.0
    beta_grid: int = 41  # ex 0..2 step 0.05


def _lm_solve_x_for_beta(
    x0: np.ndarray,
    mic_positions: np.ndarray,
    E: np.ndarray,
    beta: float,
    gcfg: GroundConfig,
    cfg: ERLSGroundConfig,
):
    x = np.asarray(x0, float).ravel()
    lam = float(cfg.lam)

    history_cost: List[float] = []
    history_step: List[float] = []

    r, J, K, phi = residuals_and_jacobian_energy(x, mic_positions, E, beta, gcfg)
    cost = float(r @ r)
    history_cost.append(cost)
    history_step.append(0.0)

    for _ in range(int(cfg.max_iter)):
        r, J, K, phi = residuals_and_jacobian_energy(x, mic_positions, E, beta, gcfg)
        H = J.T @ J
        g = J.T @ r

        dx = np.linalg.solve(H + lam * np.eye(3), -g)
        x_new = x + dx

        # contrainte z > floor_z (option)
        if gcfg.enforce_z_positive and x_new[2] <= gcfg.floor_z:
            x_new[2] = gcfg.floor_z + 1e-6

        r_new, _, _, _ = residuals_and_jacobian_energy(x_new, mic_positions, E, beta, gcfg)
        cost_new = float(r_new @ r_new)

        step_norm = float(np.linalg.norm(dx))
        history_step.append(step_norm)

        if cost_new < cost:
            prev_cost = cost
            x = x_new
            cost = cost_new
            history_cost.append(cost)
            lam = max(lam / 2.0, 1e-8)

            if step_norm < cfg.tol_step:
                break
            if abs(prev_cost - cost) < cfg.tol_cost:
                break
        else:
            lam = min(lam * 2.0, 1e6)

    return x.astype(float), float(cost), {
        "history_cost": np.asarray(history_cost, float),
        "history_step": np.asarray(history_step, float),
        "lambda_final": lam,
    }


def er_ls_ground(
    mic_positions: np.ndarray,
    y: np.ndarray,
    fs: float,
    x0: Optional[np.ndarray] = None,
    cfg_E: EnergyConfig = EnergyConfig(),
    cfg_ls_free: ERLSConfig = ERLSConfig(),  # juste pour ref_idx si tu veux init ailleurs
    cfg: ERLSGroundConfig = ERLSGroundConfig(),
    gcfg: GroundConfig = GroundConfig(),
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """
    ER-LS version "sol rigide" (energy-only), via modèle direct + image.
    - calcule E_i
    - recherche beta sur une grille
    - pour chaque beta: LM sur x
    """
    mics = np.asarray(mic_positions, float)
    y = np.asarray(y, float)
    if mics.ndim != 2 or mics.shape[1] != 3:
        raise ValueError("mic_positions must be (M,3)")
    if y.ndim != 2 or y.shape[0] != mics.shape[0]:
        raise ValueError("y must be (M,N) and match mic_positions")

    # Energies
    E = estimate_mic_energy(y, fs, cfg_E)

    # init x0 : si non fourni, tu peux mettre un guess simple
    # (ici: barycentre des micros + z au dessus du sol)
    if x0 is None:
        x0 = np.mean(mics, axis=0)
        x0[2] = max(x0[2], gcfg.floor_z + 0.5)

    # beta grid search
    betas = np.linspace(cfg.beta_min, cfg.beta_max, int(cfg.beta_grid))
    best = {"cost": np.inf}

    for beta in betas:
        x_hat, cost, dbg_lm = _lm_solve_x_for_beta(x0, mics, E, float(beta), gcfg, cfg)
        if cost < best["cost"]:
            best = {
                "x_hat": x_hat,
                "beta": float(beta),
                "cost": float(cost),
                "dbg_lm": dbg_lm,
            }

    # erreur "indicateur" : RMS des résidus énergie
    r_best, _, K_best, phi_best = residuals_and_jacobian_energy(best["x_hat"], mics, E, best["beta"], gcfg)
    err = float(np.sqrt(np.mean(r_best ** 2)))

    debug: Dict[str, Any] = {
        "E": E,
        "x0": np.asarray(x0, float),
        "beta_hat": best["beta"],
        "K_hat": K_best,
        "phi": phi_best,
        "final_residuals": r_best,
        "cost": best["cost"],
        "lm_debug": best["dbg_lm"],
        "betas_tested": betas,
        "floor_z": gcfg.floor_z,
    }
    return best["x_hat"], err, debug

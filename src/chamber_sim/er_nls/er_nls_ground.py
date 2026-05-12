from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

import numpy as np

from chamber_sim.er_ls.energy import estimate_mic_energy
from chamber_sim.er_ls.config import EnergyConfig
from chamber_sim.er_ls.ground_model import GroundConfig, residuals_and_jacobian_energy
from chamber_sim.er_ls.er_ls_ground import er_ls_ground, ERLSGroundConfig


@dataclass(frozen=True)
class ERNLSGroundConfig:
    max_iter: int = 50
    lam: float = 1e-2
    tol_step: float = 1e-6
    tol_cost: float = 1e-9


def er_nls_ground(
    mic_positions: np.ndarray,
    y: np.ndarray,
    fs: float,
    x0: Optional[np.ndarray] = None,
    beta: Optional[float] = None,   # si None -> on prend beta de er_ls_ground
    cfg_E: EnergyConfig = EnergyConfig(),
    cfg_ls_g: ERLSGroundConfig = ERLSGroundConfig(),
    cfg: ERNLSGroundConfig = ERNLSGroundConfig(),
    gcfg: GroundConfig = GroundConfig(),
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    mics = np.asarray(mic_positions, float)
    y = np.asarray(y, float)
    if mics.ndim != 2 or mics.shape[1] != 3:
        raise ValueError("mic_positions must be (M,3)")
    if y.ndim != 2 or y.shape[0] != mics.shape[0]:
        raise ValueError("y must be (M,N) and match mic_positions")

    # Energies
    E = estimate_mic_energy(y, fs, cfg_E)

    # init via ER-LS-ground si besoin
    if x0 is None or beta is None:
        x_ls, _, dbg_ls = er_ls_ground(
            mic_positions=mics, y=y, fs=fs,
            x0=x0, cfg_E=cfg_E, cfg=cfg_ls_g, gcfg=gcfg
        )
        if x0 is None:
            x0 = x_ls
        if beta is None:
            beta = float(dbg_ls["beta_hat"])

    x = np.asarray(x0, float).ravel()
    beta = float(beta)

    lam = float(cfg.lam)
    history_cost: List[float] = []
    history_step: List[float] = []

    r, J, K, phi = residuals_and_jacobian_energy(x, mics, E, beta, gcfg)
    cost = float(r @ r)
    history_cost.append(cost)
    history_step.append(0.0)

    for _ in range(int(cfg.max_iter)):
        r, J, K, phi = residuals_and_jacobian_energy(x, mics, E, beta, gcfg)
        H = J.T @ J
        g = J.T @ r

        dx = np.linalg.solve(H + lam * np.eye(3), -g)
        x_new = x + dx

        if gcfg.enforce_z_positive and x_new[2] <= gcfg.floor_z:
            x_new[2] = gcfg.floor_z + 1e-6

        r_new, _, _, _ = residuals_and_jacobian_energy(x_new, mics, E, beta, gcfg)
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

    r_final, _, K_final, phi_final = residuals_and_jacobian_energy(x, mics, E, beta, gcfg)
    err = float(np.sqrt(np.mean(r_final**2)))

    debug: Dict[str, Any] = {
        "E": E,
        "x0": np.asarray(x0, float),
        "beta": beta,
        "K_hat": K_final,
        "phi": phi_final,
        "history_cost": np.asarray(history_cost, float),
        "history_step": np.asarray(history_step, float),
        "final_residuals": r_final,
        "lambda_final": lam,
        "floor_z": gcfg.floor_z,
    }
    return x.astype(float), err, debug

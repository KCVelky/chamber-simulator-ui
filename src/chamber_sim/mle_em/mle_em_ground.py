from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

import numpy as np

from chamber_sim.er_ls.energy import estimate_mic_energy
from chamber_sim.er_ls.config import EnergyConfig
from chamber_sim.er_ls.er_ls_ground import er_ls_ground, ERLSGroundConfig
from chamber_sim.er_ls.ground_model import GroundConfig as ERLSLegacyGroundConfig

from .config import MLEEMGroundConfig, MLEEMInitConfig
from .ground_model import (
    GroundGeometryConfig,
    barycenter_init,
    clamp_source_above_floor,
    residuals_energy,
    residuals_and_jacobian,
    cost_function,
    image_source,
)


@dataclass(frozen=True)
class MLEEMDebugFlags:
    """
    Options de debug / traçage.
    """
    keep_alpha_grid_details: bool = True
    keep_iteration_details: bool = True


def _validate_inputs(
    mic_positions: np.ndarray,
    y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    mics = np.asarray(mic_positions, dtype=float)
    sig = np.asarray(y, dtype=float)

    if mics.ndim != 2 or mics.shape[1] != 3:
        raise ValueError("mic_positions must be (M,3)")
    if sig.ndim != 2 or sig.shape[0] != mics.shape[0]:
        raise ValueError("y must be (M,N) and match mic_positions")
    if mics.shape[0] < 3:
        raise ValueError("At least 3 microphones are required")
    return mics, sig


def _unique_sorted_with_anchor(values: np.ndarray, anchor: float) -> np.ndarray:
    vals = np.asarray(values, dtype=float).ravel()
    vals = np.concatenate([vals, np.asarray([anchor], dtype=float)])
    vals = np.unique(np.round(vals, 12))
    vals.sort()
    return vals.astype(float)


def _build_alpha_grid(
    alpha: Optional[float],
    cfg: MLEEMGroundConfig,
) -> np.ndarray:
    """
    Construit la grille initiale des alpha testés.
    """
    if alpha is not None:
        alpha_anchor = float(np.clip(alpha, cfg.alpha_min, cfg.alpha_max))
    else:
        alpha_anchor = float(np.clip(cfg.alpha_init, cfg.alpha_min, cfg.alpha_max))

    if not cfg.estimate_alpha:
        return np.asarray([alpha_anchor], dtype=float)

    grid = np.linspace(float(cfg.alpha_min), float(cfg.alpha_max), int(cfg.alpha_grid_size))
    return _unique_sorted_with_anchor(grid, alpha_anchor)


def _init_x(
    mic_positions: np.ndarray,
    y: np.ndarray,
    fs: float,
    x0: Optional[np.ndarray],
    cfg_E: EnergyConfig,
    cfg_init: MLEEMInitConfig,
    gcfg: GroundGeometryConfig,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Initialise la position source.
    """
    if x0 is not None:
        x_init = clamp_source_above_floor(np.asarray(x0, dtype=float).ravel(), gcfg)
        if x_init.size != 3:
            raise ValueError("x0 must have shape (3,)")
        return x_init.astype(float), {"init_method_used": "user_x0"}

    if cfg_init.method == "barycenter":
        x_init = barycenter_init(
            mic_positions=mic_positions,
            gcfg=gcfg,
            z_offset=float(cfg_init.barycenter_z_offset),
        )
        return x_init.astype(float), {"init_method_used": "barycenter"}

    if cfg_init.method == "er_ls_ground":
        legacy_gcfg = ERLSLegacyGroundConfig(
            floor_z=float(gcfg.floor_z),
            enforce_z_positive=bool(gcfg.enforce_z_positive),
        )
        legacy_cfg = ERLSGroundConfig(
            max_iter=20,
            lam=1e-2,
            tol_step=1e-6,
            tol_cost=1e-9,
            beta_min=0.0,
            beta_max=2.0,
            beta_grid=31,
        )

        x_ls, _, dbg_ls = er_ls_ground(
            mic_positions=np.asarray(mic_positions, dtype=float),
            y=np.asarray(y, dtype=float),
            fs=float(fs),
            x0=None,
            cfg_E=cfg_E,
            cfg=legacy_cfg,
            gcfg=legacy_gcfg,
        )
        x_ls = clamp_source_above_floor(x_ls, gcfg)
        return x_ls.astype(float), {
            "init_method_used": "er_ls_ground",
            "init_debug_er_ls_ground": dbg_ls,
        }

    raise ValueError(f"Unsupported init method: {cfg_init.method}")


def _lm_refine_one_alpha(
    x0: np.ndarray,
    alpha0: float,
    mic_positions: np.ndarray,
    E: np.ndarray,
    cfg: MLEEMGroundConfig,
    gcfg: GroundGeometryConfig,
) -> Tuple[np.ndarray, float, float, Dict[str, Any]]:
    """
    Raffinement LM / GEM pour une graine donnée.

    Si cfg.estimate_alpha=True :
        optimisation jointe de x et alpha.

    Sinon :
        alpha reste fixe et seul x est optimisé.
    """
    x = clamp_source_above_floor(np.asarray(x0, dtype=float).ravel(), gcfg)
    if x.size != 3:
        raise ValueError("x0 must have shape (3,)")

    alpha = float(np.clip(alpha0, cfg.alpha_min, cfg.alpha_max))
    lam = float(cfg.lam)
    include_alpha = bool(cfg.estimate_alpha)

    history_cost: List[float] = []
    history_step: List[float] = []
    history_alpha: List[float] = []
    history_K: List[float] = []
    history_b: List[float] = []
    history_x: List[np.ndarray] = []

    r, J, K_hat, b_hat, phi, aux = residuals_and_jacobian(
        x=x,
        mic_positions=mic_positions,
        E=E,
        alpha=alpha,
        cfg=cfg,
        gcfg=gcfg,
        include_alpha=include_alpha,
    )
    cost = float(r @ r)

    history_cost.append(cost)
    history_step.append(0.0)
    history_alpha.append(alpha)
    history_K.append(float(K_hat))
    history_b.append(float(b_hat))
    history_x.append(x.copy())

    for _ in range(int(cfg.max_iter)):
        r, J, K_hat, b_hat, phi, aux = residuals_and_jacobian(
            x=x,
            mic_positions=mic_positions,
            E=E,
            alpha=alpha,
            cfg=cfg,
            gcfg=gcfg,
            include_alpha=include_alpha,
        )

        H = J.T @ J
        g = J.T @ r

        try:
            dx = np.linalg.solve(H + lam * np.eye(H.shape[0]), -g)
        except np.linalg.LinAlgError:
            dx = np.linalg.lstsq(H + lam * np.eye(H.shape[0]), -g, rcond=None)[0]

        if include_alpha:
            dx_x = dx[:3]
            dx_alpha = float(dx[3])
        else:
            dx_x = dx[:3]
            dx_alpha = 0.0

        x_new = clamp_source_above_floor(x + dx_x, gcfg)
        alpha_new = float(np.clip(alpha + dx_alpha, cfg.alpha_min, cfg.alpha_max))

        cost_new = cost_function(
            x=x_new,
            mic_positions=mic_positions,
            E=E,
            alpha=alpha_new,
            cfg=cfg,
            gcfg=gcfg,
        )

        step_norm = float(np.linalg.norm(dx))
        history_step.append(step_norm)

        if cost_new < cost:
            prev_cost = cost
            x = x_new
            alpha = alpha_new
            cost = float(cost_new)

            lam = max(lam / 2.0, float(cfg.lam_min))

            r_acc, K_acc, b_acc, phi_acc, aux_acc = residuals_energy(
                x=x,
                mic_positions=mic_positions,
                E=E,
                alpha=alpha,
                cfg=cfg,
                gcfg=gcfg,
            )

            history_cost.append(cost)
            history_alpha.append(alpha)
            history_K.append(float(K_acc))
            history_b.append(float(b_acc))
            history_x.append(x.copy())

            if step_norm < float(cfg.tol_step):
                break
            if abs(prev_cost - cost) < float(cfg.tol_cost):
                break
        else:
            lam = min(lam * 2.0, float(cfg.lam_max))

    r_final, K_final, b_final, phi_final, aux_final = residuals_energy(
        x=x,
        mic_positions=mic_positions,
        E=E,
        alpha=alpha,
        cfg=cfg,
        gcfg=gcfg,
    )
    cost_final = float(r_final @ r_final)

    dbg: Dict[str, Any] = {
        "history_cost": np.asarray(history_cost, dtype=float),
        "history_step": np.asarray(history_step, dtype=float),
        "history_alpha": np.asarray(history_alpha, dtype=float),
        "history_K": np.asarray(history_K, dtype=float),
        "history_b": np.asarray(history_b, dtype=float),
        "history_x": np.asarray(history_x, dtype=float) if len(history_x) > 0 else np.empty((0, 3), dtype=float),
        "lambda_final": float(lam),
        "final_residuals": r_final.astype(float),
        "K_hat": float(K_final),
        "b_hat": float(b_final),
        "phi": phi_final.astype(float),
        "cost": float(cost_final),
        "aux_final": aux_final,
    }

    return x.astype(float), float(alpha), float(cost_final), dbg


def mle_em_ground(
    mic_positions: np.ndarray,
    y: np.ndarray,
    fs: float,
    x0: Optional[np.ndarray] = None,
    alpha: Optional[float] = None,
    cfg_E: EnergyConfig = EnergyConfig(),
    cfg_init: MLEEMInitConfig = MLEEMInitConfig(),
    cfg: MLEEMGroundConfig = MLEEMGroundConfig(),
    gcfg: GroundGeometryConfig = GroundGeometryConfig(),
    dbg_flags: MLEEMDebugFlags = MLEEMDebugFlags(),
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """
    Localisation énergétique MLE/EM (au sens pratique : GEM profilé + LM)
    avec prise en compte d'un sol rigide via source image.

    Paramètres
    ----------
    mic_positions : ndarray (M,3)
        positions des microphones
    y : ndarray (M,N)
        signaux micro
    fs : float
        fréquence d'échantillonnage
    x0 : ndarray (3,) ou None
        initialisation de la position
    alpha : float ou None
        coefficient de réflexion initial/fixe
    cfg_E : EnergyConfig
        configuration d'estimation d'énergie
    cfg_init : MLEEMInitConfig
        configuration d'initialisation
    cfg : MLEEMGroundConfig
        configuration principale de l'algo
    gcfg : GroundGeometryConfig
        configuration géométrique du sol
    dbg_flags : MLEEMDebugFlags
        options de debug

    Retourne
    --------
    x_hat : (3,)
        position estimée de la source réelle
    err : float
        RMS des résidus énergétiques
    debug : dict
        informations internes utiles
    """
    if fs <= 0.0:
        raise ValueError("fs must be > 0")

    mics, y = _validate_inputs(mic_positions, y)

    # 1) Energies micro
    E = estimate_mic_energy(y, fs, cfg_E)

    # 2) Init x
    x_init, dbg_init = _init_x(
        mic_positions=mics,
        y=y,
        fs=float(fs),
        x0=x0,
        cfg_E=cfg_E,
        cfg_init=cfg_init,
        gcfg=gcfg,
    )

    # 3) Grille initiale alpha
    alpha_grid = _build_alpha_grid(alpha=alpha, cfg=cfg)

    # 4) Recherche robuste sur alpha_init + raffinement LM/GEM
    best: Dict[str, Any] = {"cost": np.inf}
    alpha_grid_details: List[Dict[str, Any]] = []

    for alpha0 in alpha_grid:
        x_hat_i, alpha_hat_i, cost_i, dbg_i = _lm_refine_one_alpha(
            x0=x_init,
            alpha0=float(alpha0),
            mic_positions=mics,
            E=E,
            cfg=cfg,
            gcfg=gcfg,
        )

        item = {
            "alpha0": float(alpha0),
            "alpha_hat": float(alpha_hat_i),
            "x_hat": x_hat_i.astype(float),
            "cost": float(cost_i),
        }
        if dbg_flags.keep_iteration_details:
            item["solver_debug"] = dbg_i

        alpha_grid_details.append(item)

        if cost_i < best["cost"]:
            best = {
                "x_hat": x_hat_i.astype(float),
                "alpha_hat": float(alpha_hat_i),
                "cost": float(cost_i),
                "solver_debug": dbg_i,
                "alpha0_best": float(alpha0),
            }

    # 5) Résidu final
    r_final, K_final, b_final, phi_final, aux_final = residuals_energy(
        x=best["x_hat"],
        mic_positions=mics,
        E=E,
        alpha=best["alpha_hat"],
        cfg=cfg,
        gcfg=gcfg,
    )
    err = float(np.sqrt(np.mean(r_final ** 2)))

    debug: Dict[str, Any] = {
        "solver_kind": "profiled_GEM_LM",
        "model_type": cfg.model_type,
        "E": E.astype(float),
        "x0": x_init.astype(float),
        "alpha_input": None if alpha is None else float(alpha),
        "alpha0_best": float(best["alpha0_best"]),
        "alpha_hat": float(best["alpha_hat"]),
        "K_hat": float(K_final),
        "b_hat": float(b_final),
        "phi": phi_final.astype(float),
        "final_residuals": r_final.astype(float),
        "cost": float(best["cost"]),
        "floor_z": float(gcfg.floor_z),
        "x_img_hat": image_source(best["x_hat"], gcfg).astype(float),
        "aux_final": aux_final,
        "init_debug": dbg_init,
        "cfg_energy": cfg_E,
        "cfg_init": cfg_init,
        "cfg_solver": cfg,
    }

    if dbg_flags.keep_iteration_details:
        debug["solver_debug"] = best["solver_debug"]
    if dbg_flags.keep_alpha_grid_details:
        debug["alpha_grid"] = alpha_grid.astype(float)
        debug["alpha_grid_details"] = alpha_grid_details

    return best["x_hat"].astype(float), err, debug


__all__ = [
    "MLEEMDebugFlags",
    "mle_em_ground",
]
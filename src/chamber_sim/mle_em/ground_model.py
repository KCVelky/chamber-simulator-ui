from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple

import numpy as np

from .config import MLEEMGroundConfig


@dataclass(frozen=True)
class GroundGeometryConfig:
    """
    Géométrie du sol rigide.

    floor_z
    -------
    Altitude du plan du sol.

    enforce_z_positive
    ------------------
    Si True, on impose z > floor_z lors des clamp / updates.

    z_margin
    --------
    Marge minimale au-dessus du sol.
    """
    floor_z: float = 0.0
    enforce_z_positive: bool = True
    z_margin: float = 1e-6

    def __post_init__(self) -> None:
        if self.z_margin <= 0.0:
            raise ValueError("z_margin must be > 0")


def _as_point3(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size != 3:
        raise ValueError("x must have exactly 3 components")
    return x.astype(float)


def _as_mics(mic_positions: np.ndarray) -> np.ndarray:
    mics = np.asarray(mic_positions, dtype=float)
    if mics.ndim != 2 or mics.shape[1] != 3:
        raise ValueError("mic_positions must be of shape (M, 3)")
    return mics.astype(float)


def clamp_source_above_floor(
    x: np.ndarray,
    gcfg: GroundGeometryConfig,
) -> np.ndarray:
    """
    Force la source à rester au-dessus du sol si demandé.
    """
    x = _as_point3(x).copy()
    if gcfg.enforce_z_positive and x[2] <= gcfg.floor_z + gcfg.z_margin:
        x[2] = gcfg.floor_z + gcfg.z_margin
    return x


def barycenter_init(
    mic_positions: np.ndarray,
    gcfg: GroundGeometryConfig,
    z_offset: float = 0.5,
) -> np.ndarray:
    """
    Initialisation simple au barycentre des micros, avec élévation au-dessus du sol.
    """
    mics = _as_mics(mic_positions)
    x0 = np.mean(mics, axis=0)
    x0[2] = max(float(x0[2]), float(gcfg.floor_z + z_offset))
    return clamp_source_above_floor(x0, gcfg)


def image_source(
    x: np.ndarray,
    gcfg: GroundGeometryConfig,
) -> np.ndarray:
    """
    Source image par symétrie par rapport au plan z = floor_z.
    """
    x = _as_point3(x)
    x_img = x.copy()
    x_img[2] = 2.0 * float(gcfg.floor_z) - x[2]
    return x_img


def direct_and_image_vectors(
    x: np.ndarray,
    mic_positions: np.ndarray,
    gcfg: GroundGeometryConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Retourne les vecteurs source->micro pour la source réelle et la source image.

    v_dir[i] = x - m_i
    v_img[i] = x_img - m_i
    """
    x = clamp_source_above_floor(x, gcfg)
    mics = _as_mics(mic_positions)
    x_img = image_source(x, gcfg)

    v_dir = x[None, :] - mics
    v_img = x_img[None, :] - mics
    return v_dir, v_img


def direct_and_image_distances(
    x: np.ndarray,
    mic_positions: np.ndarray,
    gcfg: GroundGeometryConfig,
    stable_eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Distances directe et image pour chaque micro.
    """
    if stable_eps <= 0.0:
        raise ValueError("stable_eps must be > 0")

    v_dir, v_img = direct_and_image_vectors(x, mic_positions, gcfg)
    r_dir = np.linalg.norm(v_dir, axis=1)
    r_img = np.linalg.norm(v_img, axis=1)

    r_dir = np.maximum(r_dir, stable_eps)
    r_img = np.maximum(r_img, stable_eps)
    return r_dir, r_img


def relative_delay_image_minus_direct(
    x: np.ndarray,
    mic_positions: np.ndarray,
    gcfg: GroundGeometryConfig,
    sound_speed: float = 343.0,
    stable_eps: float = 1e-12,
) -> np.ndarray:
    """
    tau_i = (r_img_i - r_dir_i) / c
    """
    if sound_speed <= 0.0:
        raise ValueError("sound_speed must be > 0")

    r_dir, r_img = direct_and_image_distances(
        x=x,
        mic_positions=mic_positions,
        gcfg=gcfg,
        stable_eps=stable_eps,
    )
    tau = (r_img - r_dir) / float(sound_speed)
    return tau.astype(float)


def gamma_band_flat(
    tau: np.ndarray,
    f_low_hz: float,
    f_high_hz: float,
    stable_eps: float = 1e-12,
) -> np.ndarray:
    """
    Approximation bande plate du terme de cohérence :

        Gamma(tau) =
            [sin(2 pi f_h tau) - sin(2 pi f_l tau)]
            / [2 pi (f_h - f_l) tau]

    avec Gamma(0) = 1.
    """
    if f_low_hz <= 0.0:
        raise ValueError("f_low_hz must be > 0")
    if f_high_hz <= f_low_hz:
        raise ValueError("f_high_hz must be > f_low_hz")
    if stable_eps <= 0.0:
        raise ValueError("stable_eps must be > 0")

    tau = np.asarray(tau, dtype=float)
    out = np.empty_like(tau, dtype=float)

    small = np.abs(tau) <= stable_eps
    out[small] = 1.0

    ts = tau[~small]
    if ts.size > 0:
        num = np.sin(2.0 * np.pi * f_high_hz * ts) - np.sin(2.0 * np.pi * f_low_hz * ts)
        den = 2.0 * np.pi * (f_high_hz - f_low_hz) * ts
        out[~small] = num / den

    return out


def gamma_term(
    tau: np.ndarray,
    cfg: MLEEMGroundConfig,
) -> np.ndarray:
    """
    Terme de cohérence fréquentielle du modèle cohérent.

    Pour l'instant :
    - soit approximation bande plate,
    - soit même approximation si use_band_integral=True
      (hook laissé ouvert pour plus tard).
    """
    tau = np.asarray(tau, dtype=float)

    # Hook pour une future intégration fréquentielle plus riche.
    return gamma_band_flat(
        tau=tau,
        f_low_hz=float(cfg.f_low_hz),
        f_high_hz=float(cfg.f_high_hz),
        stable_eps=float(cfg.stable_eps),
    )


def energy_kernel(
    x: np.ndarray,
    mic_positions: np.ndarray,
    alpha: float,
    cfg: MLEEMGroundConfig,
    gcfg: GroundGeometryConfig,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Calcule le noyau énergétique phi_i(x, alpha).

    Modèle additive :
        phi_i = 1/r_i^2 + alpha / r_img_i^2

    Modèle coherent :
        phi_i = 1/r_i^2
              + alpha^2 / r_img_i^2
              + 2 alpha Gamma(tau_i) / (r_i r_img_i)

    Retourne :
    ----------
    phi : (M,)
        noyau énergétique par micro
    aux : dict
        infos intermédiaires utiles au debug
    """
    x = clamp_source_above_floor(x, gcfg)
    alpha = float(alpha)
    mics = _as_mics(mic_positions)

    r_dir, r_img = direct_and_image_distances(
        x=x,
        mic_positions=mics,
        gcfg=gcfg,
        stable_eps=float(cfg.stable_eps),
    )

    inv_r2 = 1.0 / np.maximum(r_dir**2, cfg.stable_eps)
    inv_rimg2 = 1.0 / np.maximum(r_img**2, cfg.stable_eps)

    tau = relative_delay_image_minus_direct(
        x=x,
        mic_positions=mics,
        gcfg=gcfg,
        sound_speed=float(cfg.sound_speed),
        stable_eps=float(cfg.stable_eps),
    )

    gamma = gamma_term(tau, cfg)
    cross = 2.0 * alpha * gamma / np.maximum(r_dir * r_img, cfg.stable_eps)

    if cfg.model_type == "additive":
        phi = inv_r2 + alpha * inv_rimg2
    elif cfg.model_type == "coherent":
        phi = inv_r2 + (alpha ** 2) * inv_rimg2 + cross
    else:
        raise ValueError(f"Unsupported model_type: {cfg.model_type}")

    # Sécurité numérique : on évite un noyau négatif ou nul
    phi = np.maximum(phi, float(cfg.stable_eps))

    aux: Dict[str, Any] = {
        "x": x.copy(),
        "x_img": image_source(x, gcfg),
        "r_dir": r_dir,
        "r_img": r_img,
        "inv_r2": inv_r2,
        "inv_rimg2": inv_rimg2,
        "tau": tau,
        "gamma": gamma,
        "cross_term": cross,
        "phi": phi,
        "model_type": cfg.model_type,
        "alpha": alpha,
    }
    return phi.astype(float), aux


def _profile_source_energy_only(
    E: np.ndarray,
    phi: np.ndarray,
    cfg: MLEEMGroundConfig,
) -> Tuple[float, float, np.ndarray]:
    """
    Profile analytique de :
        E ≈ K * phi

    Retourne
    --------
    K_hat, b_hat, y_hat
    """
    E = np.asarray(E, dtype=float).reshape(-1)
    phi = np.asarray(phi, dtype=float).reshape(-1)

    den = float(phi @ phi)
    if den <= float(cfg.stable_eps):
        K_hat = float(cfg.min_source_energy)
    else:
        K_hat = float((E @ phi) / den)
        K_hat = max(K_hat, float(cfg.min_source_energy))

    b_hat = 0.0
    y_hat = K_hat * phi
    return K_hat, b_hat, y_hat.astype(float)


def _profile_source_energy_and_noise_floor(
    E: np.ndarray,
    phi: np.ndarray,
    cfg: MLEEMGroundConfig,
) -> Tuple[float, float, np.ndarray]:
    """
    Profile analytique de :
        E ≈ K * phi + b

    avec clipping simple sur K et b.
    """
    E = np.asarray(E, dtype=float).reshape(-1)
    phi = np.asarray(phi, dtype=float).reshape(-1)

    A = np.column_stack([phi, np.ones_like(phi)])
    sol, *_ = np.linalg.lstsq(A, E, rcond=None)
    K_hat = float(sol[0])
    b_hat = float(sol[1])

    K_hat = max(K_hat, float(cfg.min_source_energy))
    b_hat = max(b_hat, float(cfg.noise_floor_min))

    y_hat = K_hat * phi + b_hat
    return K_hat, b_hat, y_hat.astype(float)


def profile_linear_amplitudes(
    E: np.ndarray,
    phi: np.ndarray,
    cfg: MLEEMGroundConfig,
) -> Tuple[float, float, np.ndarray]:
    """
    Profile les paramètres linéaires du modèle énergétique.

    Si estimate_noise_floor=False :
        E ≈ K phi

    Sinon :
        E ≈ K phi + b
    """
    if cfg.profile_source_energy:
        if cfg.estimate_noise_floor:
            return _profile_source_energy_and_noise_floor(E, phi, cfg)
        return _profile_source_energy_only(E, phi, cfg)

    # fallback : utilise la config telle quelle
    K_hat = max(float(cfg.alpha_init), float(cfg.min_source_energy))
    b_hat = float(cfg.noise_floor_init) if cfg.estimate_noise_floor else 0.0
    y_hat = K_hat * phi + b_hat
    return K_hat, b_hat, y_hat.astype(float)


def predicted_energy(
    x: np.ndarray,
    mic_positions: np.ndarray,
    alpha: float,
    cfg: MLEEMGroundConfig,
    gcfg: GroundGeometryConfig,
) -> Tuple[np.ndarray, float, float, Dict[str, Any]]:
    """
    Modèle énergétique profilé :

        E_hat = K_hat * phi(x, alpha) + b_hat

    Retourne
    --------
    E_hat : (M,)
    K_hat : float
    b_hat : float
    aux   : dict
    """
    phi, aux = energy_kernel(
        x=x,
        mic_positions=mic_positions,
        alpha=alpha,
        cfg=cfg,
        gcfg=gcfg,
    )
    K_hat, b_hat, E_hat = profile_linear_amplitudes(
        E=np.ones_like(phi),  # placeholder overwritten below
        phi=phi,
        cfg=cfg,
    )

    # Ce bloc sera surchargé dans residuals_energy avec les vraies E.
    # Ici on renvoie surtout le noyau + placeholders.
    aux = dict(aux)
    aux["K_hat_placeholder"] = K_hat
    aux["b_hat_placeholder"] = b_hat
    aux["E_hat_placeholder"] = E_hat
    return E_hat.astype(float), float(K_hat), float(b_hat), aux


def residuals_energy(
    x: np.ndarray,
    mic_positions: np.ndarray,
    E: np.ndarray,
    alpha: float,
    cfg: MLEEMGroundConfig,
    gcfg: GroundGeometryConfig,
) -> Tuple[np.ndarray, float, float, np.ndarray, Dict[str, Any]]:
    """
    Résidus énergétiques profilés :

        r = E - (K_hat * phi + b_hat)

    Retourne
    --------
    r     : (M,)
    K_hat : float
    b_hat : float
    phi   : (M,)
    aux   : dict
    """
    E = np.asarray(E, dtype=float).reshape(-1)
    mics = _as_mics(mic_positions)
    if E.size != mics.shape[0]:
        raise ValueError("E must have shape (M,) with M = number of microphones")

    phi, aux = energy_kernel(
        x=x,
        mic_positions=mics,
        alpha=alpha,
        cfg=cfg,
        gcfg=gcfg,
    )

    K_hat, b_hat, E_hat = (
        _profile_source_energy_and_noise_floor(E, phi, cfg)
        if cfg.estimate_noise_floor
        else _profile_source_energy_only(E, phi, cfg)
    )

    r = E - E_hat

    aux = dict(aux)
    aux["E"] = E
    aux["E_hat"] = E_hat
    aux["K_hat"] = K_hat
    aux["b_hat"] = b_hat

    return r.astype(float), float(K_hat), float(b_hat), phi.astype(float), aux


def _finite_difference_jacobian_x(
    x: np.ndarray,
    mic_positions: np.ndarray,
    E: np.ndarray,
    alpha: float,
    cfg: MLEEMGroundConfig,
    gcfg: GroundGeometryConfig,
) -> np.ndarray:
    """
    Jacobien numérique des résidus par rapport à x = [x,y,z].
    """
    x = clamp_source_above_floor(x, gcfg)
    r0, _, _, _, _ = residuals_energy(
        x=x,
        mic_positions=mic_positions,
        E=E,
        alpha=alpha,
        cfg=cfg,
        gcfg=gcfg,
    )

    J = np.zeros((r0.size, 3), dtype=float)
    h = float(cfg.fd_eps)

    for k in range(3):
        xp = x.copy()
        xm = x.copy()

        xp[k] += h
        xm[k] -= h

        xp = clamp_source_above_floor(xp, gcfg)
        xm = clamp_source_above_floor(xm, gcfg)

        rp, _, _, _, _ = residuals_energy(
            x=xp,
            mic_positions=mic_positions,
            E=E,
            alpha=alpha,
            cfg=cfg,
            gcfg=gcfg,
        )
        rm, _, _, _, _ = residuals_energy(
            x=xm,
            mic_positions=mic_positions,
            E=E,
            alpha=alpha,
            cfg=cfg,
            gcfg=gcfg,
        )
        J[:, k] = (rp - rm) / (2.0 * h)

    return J


def _finite_difference_jacobian_alpha(
    x: np.ndarray,
    mic_positions: np.ndarray,
    E: np.ndarray,
    alpha: float,
    cfg: MLEEMGroundConfig,
    gcfg: GroundGeometryConfig,
) -> np.ndarray:
    """
    Dérivée numérique des résidus par rapport à alpha.
    """
    h = float(cfg.fd_eps)

    ap = min(float(alpha + h), float(cfg.alpha_max))
    am = max(float(alpha - h), float(cfg.alpha_min))

    if abs(ap - am) <= float(cfg.stable_eps):
        r0, _, _, _, _ = residuals_energy(
            x=x,
            mic_positions=mic_positions,
            E=E,
            alpha=alpha,
            cfg=cfg,
            gcfg=gcfg,
        )
        return np.zeros((r0.size, 1), dtype=float)

    rp, _, _, _, _ = residuals_energy(
        x=x,
        mic_positions=mic_positions,
        E=E,
        alpha=ap,
        cfg=cfg,
        gcfg=gcfg,
    )
    rm, _, _, _, _ = residuals_energy(
        x=x,
        mic_positions=mic_positions,
        E=E,
        alpha=am,
        cfg=cfg,
        gcfg=gcfg,
    )
    Ja = ((rp - rm) / (ap - am)).reshape(-1, 1)
    return Ja.astype(float)


def residuals_and_jacobian(
    x: np.ndarray,
    mic_positions: np.ndarray,
    E: np.ndarray,
    alpha: float,
    cfg: MLEEMGroundConfig,
    gcfg: GroundGeometryConfig,
    include_alpha: bool = False,
) -> Tuple[np.ndarray, np.ndarray, float, float, np.ndarray, Dict[str, Any]]:
    """
    Retourne les résidus profilés et leur jacobien numérique.

    Paramètres
    ----------
    include_alpha
        Si True, le jacobien contient 4 colonnes :
            [d/dx, d/dy, d/dz, d/dalpha]
        Sinon :
            [d/dx, d/dy, d/dz]

    Retourne
    --------
    r     : (M,)
    J     : (M,3) ou (M,4)
    K_hat : float
    b_hat : float
    phi   : (M,)
    aux   : dict
    """
    x = clamp_source_above_floor(x, gcfg)
    alpha = float(np.clip(alpha, cfg.alpha_min, cfg.alpha_max))

    r, K_hat, b_hat, phi, aux = residuals_energy(
        x=x,
        mic_positions=mic_positions,
        E=E,
        alpha=alpha,
        cfg=cfg,
        gcfg=gcfg,
    )

    Jx = _finite_difference_jacobian_x(
        x=x,
        mic_positions=mic_positions,
        E=E,
        alpha=alpha,
        cfg=cfg,
        gcfg=gcfg,
    )

    if include_alpha:
        Ja = _finite_difference_jacobian_alpha(
            x=x,
            mic_positions=mic_positions,
            E=E,
            alpha=alpha,
            cfg=cfg,
            gcfg=gcfg,
        )
        J = np.hstack([Jx, Ja])
    else:
        J = Jx

    aux = dict(aux)
    aux["include_alpha"] = bool(include_alpha)
    aux["jacobian_shape"] = J.shape

    return (
        r.astype(float),
        J.astype(float),
        float(K_hat),
        float(b_hat),
        phi.astype(float),
        aux,
    )


def cost_function(
    x: np.ndarray,
    mic_positions: np.ndarray,
    E: np.ndarray,
    alpha: float,
    cfg: MLEEMGroundConfig,
    gcfg: GroundGeometryConfig,
) -> float:
    """
    Coût quadratique non pondéré :
        cost = r^T r
    """
    r, _, _, _, _ = residuals_energy(
        x=x,
        mic_positions=mic_positions,
        E=E,
        alpha=alpha,
        cfg=cfg,
        gcfg=gcfg,
    )
    return float(r @ r)


__all__ = [
    "GroundGeometryConfig",
    "clamp_source_above_floor",
    "barycenter_init",
    "image_source",
    "direct_and_image_vectors",
    "direct_and_image_distances",
    "relative_delay_image_minus_direct",
    "gamma_band_flat",
    "gamma_term",
    "energy_kernel",
    "profile_linear_amplitudes",
    "predicted_energy",
    "residuals_energy",
    "residuals_and_jacobian",
    "cost_function",
]
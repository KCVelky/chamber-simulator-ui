from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np


@dataclass(frozen=True)
class ApolloniusSphere:
    """
    Représente une sphère d'Apollonius sous forme :
        ||x - c|| = rho
    """
    c: np.ndarray   # (3,)
    rho: float
    mic_idx: int    # i (le micro utilisé dans la paire)
    ref_idx: int    # j (micro de référence)
    kappa: float    # κ_ij = ||x-mi|| / ||x-mj||


def kappa_from_energy(
    E: np.ndarray,
    ref_idx: int,
    mic_gains: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Calcule κ_i (relatif à ref) à partir des énergies mesurées.

    Modèle (sans bruit) :
      E_i ≈ g_i * Es / d_i^2
      => E_i / E_ref ≈ (g_i/g_ref) * (d_ref^2 / d_i^2)

    Donc :
      d_i / d_ref ≈ sqrt( (g_i/g_ref) * (E_ref / E_i) )

    On définit :
      κ_i = d_i / d_ref = ||x - m_i|| / ||x - m_ref||.

    Paramètres
    ----------
    E : (M,)
        énergies estimées.
    ref_idx : int
        index du micro de référence.
    mic_gains : (M,) ou None
        gains relatifs des micros (si inconnus -> tous à 1).

    Retour
    ------
    kappa : (M,)
        κ_ref = 1, et κ_i pour i != ref.
        κ_i peut être nan si E_i <= 0.
    """
    E = np.asarray(E, dtype=float).ravel()
    M = E.size
    if not (0 <= ref_idx < M):
        raise ValueError("ref_idx out of range")
    if E[ref_idx] <= 0:
        raise ValueError("Reference energy must be > 0")

    if mic_gains is None:
        g = np.ones(M, dtype=float)
    else:
        g = np.asarray(mic_gains, dtype=float).ravel()
        if g.size != M:
            raise ValueError("mic_gains must be shape (M,)")

    gref = g[ref_idx]
    if gref <= 0:
        raise ValueError("Reference gain must be > 0")

    kappa = np.zeros(M, dtype=float)
    for i in range(M):
        if i == ref_idx:
            kappa[i] = 1.0
            continue
        if E[i] <= 0:
            kappa[i] = np.nan
        else:
            kappa[i] = np.sqrt((g[i] / gref) * (E[ref_idx] / E[i]))
    return kappa


def apollonius_sphere_from_kappa(
    mi: np.ndarray,
    mj: np.ndarray,
    kappa: float,
) -> Tuple[np.ndarray, float]:
    """
    Construit la sphère d'Apollonius associée à :
        ||x - mi|| = kappa * ||x - mj||
    (typiquement mj = micro de référence)

    Formules :
      c   = (mi - kappa^2 * mj) / (1 - kappa^2)
      rho = (kappa * ||mi - mj||) / |1 - kappa^2|

    Paramètres
    ----------
    mi, mj : (3,)
        positions des micros i et j.
    kappa : float
        ratio de distance.

    Retour
    ------
    c : (3,)
    rho : float
    """
    mi = np.asarray(mi, dtype=float).ravel()
    mj = np.asarray(mj, dtype=float).ravel()
    if mi.size != 3 or mj.size != 3:
        raise ValueError("mi and mj must be 3D vectors")

    b = float(kappa) ** 2
    denom = 1.0 - b
    if np.isclose(denom, 0.0):
        raise ValueError("kappa too close to 1 -> sphere unstable")

    c = (mi - b * mj) / denom
    rho = (float(kappa) * np.linalg.norm(mi - mj)) / abs(denom)
    return c, float(rho)


def build_apollonius_spheres(
    mic_positions: np.ndarray,
    kappa: np.ndarray,
    ref_idx: int,
    kappa_eps: float = 1e-3,
) -> List[ApolloniusSphere]:
    """
    Construit les sphères (i, ref) pour tous les i != ref,
    en filtrant les κ instables.

    Filtrage :
    - ignore κ nan
    - ignore |κ - 1| < kappa_eps (sphère numériquement instable)

    Retour : liste de ApolloniusSphere.
    """
    mics = np.asarray(mic_positions, dtype=float)
    if mics.ndim != 2 or mics.shape[1] != 3:
        raise ValueError("mic_positions must be (M,3)")

    kappa = np.asarray(kappa, dtype=float).ravel()
    M = mics.shape[0]
    if kappa.size != M:
        raise ValueError("kappa must be shape (M,)")

    if not (0 <= ref_idx < M):
        raise ValueError("ref_idx out of range")

    spheres: List[ApolloniusSphere] = []
    mj = mics[ref_idx]

    for i in range(M):
        if i == ref_idx:
            continue
        ki = kappa[i]
        if not np.isfinite(ki):
            continue
        if abs(ki - 1.0) < kappa_eps:
            continue

        ci, rho = apollonius_sphere_from_kappa(mics[i], mj, float(ki))
        spheres.append(
            ApolloniusSphere(
                c=ci.astype(float),
                rho=float(rho),
                mic_idx=i,
                ref_idx=ref_idx,
                kappa=float(ki),
            )
        )

    return spheres

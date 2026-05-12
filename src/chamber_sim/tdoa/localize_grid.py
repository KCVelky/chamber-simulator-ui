from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass(frozen=True)
class GridSearchConfig:
    dx: float = 0.10
    dy: float = 0.10
    dz: float = 0.10
    z_fixed: Optional[float] = None


def localize_tdoa_grid(
    mic_positions: np.ndarray,     # (M,3)
    tdoa: np.ndarray,              # (M,) relatif au micro ref_idx (tdoa[ref_idx]=0)
    room_size: np.ndarray,         # (3,) [Lx,Ly,Lz]
    c: float,
    cfg: GridSearchConfig = GridSearchConfig(),
    ref_idx: int = 0,
) -> Tuple[np.ndarray, float]:
    """
    Recherche grille: minimise
        Σ_{i≠ref} ( (r_i - r_ref)/c - tdoa_i )^2

    - mic_positions: positions des micros
    - tdoa: retards relatifs (en secondes) par rapport à ref_idx
    - ref_idx: index du micro de référence

    Returns:
      best_x (3,), best_cost
    """
    mics = np.asarray(mic_positions, dtype=float)
    tdoa = np.asarray(tdoa, dtype=float).ravel()
    room = np.asarray(room_size, dtype=float).ravel()

    if mics.ndim != 2 or mics.shape[1] != 3:
        raise ValueError("mic_positions must be shape (M,3).")
    M = mics.shape[0]
    if tdoa.size != M:
        raise ValueError("tdoa must have shape (M,).")
    if not (0 <= ref_idx < M):
        raise ValueError("ref_idx out of range.")
    if c <= 0:
        raise ValueError("c must be > 0.")

    Lx, Ly, Lz = map(float, room)

    xs = np.arange(0.0, Lx + 1e-12, cfg.dx)
    ys = np.arange(0.0, Ly + 1e-12, cfg.dy)
    if cfg.z_fixed is not None:
        zs = np.array([float(cfg.z_fixed)], dtype=float)
    else:
        zs = np.arange(0.0, Lz + 1e-12, cfg.dz)

    ref = mics[ref_idx]
    idxs = np.arange(M) != ref_idx  # on ignore ref dans le coût

    best_cost = float("inf")
    best_x = np.array([0.0, 0.0, 0.0], dtype=float)

    # Boucles simples (OK pour commencer; on vectorisera si besoin)
    for x in xs:
        for y in ys:
            for z in zs:
                p = np.array([x, y, z], dtype=float)

                r_ref = np.linalg.norm(p - ref)
                r_all = np.linalg.norm(p - mics, axis=1)

                pred = (r_all - r_ref) / c
                err = pred - tdoa
                cost = float(np.sum(err[idxs] ** 2))

                if cost < best_cost:
                    best_cost = cost
                    best_x = p

    return best_x, best_cost

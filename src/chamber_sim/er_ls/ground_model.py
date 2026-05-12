from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class GroundConfig:
    floor_z: float = 0.0
    enforce_z_positive: bool = True  # x[2] > floor_z
    eps_distance: float = 1e-9


def mirror_point_across_floor(x: np.ndarray, floor_z: float) -> np.ndarray:
    x = np.asarray(x, float).ravel()
    xm = x.copy()
    xm[2] = 2.0 * floor_z - x[2]
    return xm


def phi_and_dphi_dx(
    x: np.ndarray,
    mic_positions: np.ndarray,
    beta: float,
    gcfg: GroundConfig,
):
    """
    phi_i(x,beta) = 1/r_i^2 + beta * 1/r_i'^2
    returns:
      phi: (M,)
      dphi_dx: (M,3)
    """
    x = np.asarray(x, float).ravel()
    mics = np.asarray(mic_positions, float)
    M = mics.shape[0]

    # direct
    v = x[None, :] - mics
    r = np.linalg.norm(v, axis=1)
    r = np.maximum(r, gcfg.eps_distance)
    g = 1.0 / (r * r)

    # dg/dx = -2 (x-mi) / r^4
    dg_dx = -2.0 * v / (r[:, None] ** 4)

    # image
    x_img = mirror_point_across_floor(x, gcfg.floor_z)
    v_img = x_img[None, :] - mics
    r_img = np.linalg.norm(v_img, axis=1)
    r_img = np.maximum(r_img, gcfg.eps_distance)
    h = 1.0 / (r_img * r_img)

    # dh/dx : attention au miroir => dérivée sur z change de signe
    # dh/dx_img = -2 (x_img - mi) / r_img^4
    dh_dximg = -2.0 * v_img / (r_img[:, None] ** 4)

    # dx_img/dx = diag(1,1,-1) => dh/dx = dh/dximg * diag(1,1,-1)
    dh_dx = dh_dximg.copy()
    dh_dx[:, 2] *= -1.0

    phi = g + float(beta) * h
    dphi_dx = dg_dx + float(beta) * dh_dx
    return phi, dphi_dx


def khat_and_dk_dx(E: np.ndarray, phi: np.ndarray, dphi_dx: np.ndarray, eps: float = 1e-12):
    """
    K_hat = (E·phi)/(phi·phi)
    Return K_hat and dK/dx (3,)

    dK = ((E·dphi)*b - a*(2 phi·dphi))/b^2
    with a=E·phi, b=phi·phi
    """
    E = np.asarray(E, float).ravel()
    phi = np.asarray(phi, float).ravel()
    dphi_dx = np.asarray(dphi_dx, float)

    a = float(E @ phi)
    b = float(phi @ phi)
    b = max(b, eps)

    K = a / b

    # E·dphi  => (3,)
    Edphi = (E[:, None] * dphi_dx).sum(axis=0)  # (3,)
    # phi·dphi => (3,)
    phidphi = (phi[:, None] * dphi_dx).sum(axis=0)  # (3,)

    dK = (Edphi * b - a * (2.0 * phidphi)) / (b * b)
    return float(K), dK


def residuals_and_jacobian_energy(
    x: np.ndarray,
    mic_positions: np.ndarray,
    E: np.ndarray,
    beta: float,
    gcfg: GroundConfig,
):
    """
    r_i = E_i - K_hat * phi_i
    dr_i/dx = - (dK/dx)*phi_i - K_hat*dphi_i/dx
    """
    phi, dphi_dx = phi_and_dphi_dx(x, mic_positions, beta, gcfg)
    K, dK_dx = khat_and_dk_dx(E, phi, dphi_dx)

    r = E - K * phi  # (M,)

    # J (M,3)
    J = -(phi[:, None] * dK_dx[None, :] + K * dphi_dx)
    return r.astype(float), J.astype(float), float(K), phi.astype(float)

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ERNLSConfig:
    """
    Configuration de l'algorithme ER-NLS
    (Energy Ratios - Non Linear Squares).

    L'algo minimise :
        sum_i ( ||x - c_i|| - rho_i )^2

    avec (c_i, rho_i) issus des sphères d'Apollonius (ER-LS).

    Paramètres
    ----------
    max_iter :
        Nombre maximal d'itérations.

    lam :
        Paramètre d'amortissement (Levenberg-Marquardt).
        Grand -> plus stable, petit -> plus rapide.

    tol_step :
        Seuil sur la norme du pas ||dx|| pour arrêter.

    tol_cost :
        Seuil sur la variation du coût pour arrêter.

    kappa_eps :
        Même seuil que ER-LS pour ignorer les sphères instables.
    """
    max_iter: int = 50
    lam: float = 1e-2
    tol_step: float = 1e-6
    tol_cost: float = 1e-9
    kappa_eps: float = 1e-3

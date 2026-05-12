from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


MLEEMModelType = Literal["additive", "coherent"]
MLEEMInitMethod = Literal["barycenter", "er_ls_ground"]


@dataclass(frozen=True)
class MLEEMInitConfig:
    """
    Configuration de l'initialisation de l'algorithme MLE/EM sol rigide.

    method
    ------
    "barycenter" :
        initialisation simple au barycentre des micros, avec un offset en z.

    "er_ls_ground" :
        initialisation via l'algorithme ER-LS au sol rigide déjà présent
        dans ton projet. C'est l'option recommandée.

    barycenter_z_offset
    -------------------
    Si l'init est "barycenter", on place z = max(z_barycentre, floor_z + offset).

    force_init_above_floor_eps
    --------------------------
    Petite marge de sécurité pour garantir z > floor_z.
    """
    method: MLEEMInitMethod = "er_ls_ground"
    barycenter_z_offset: float = 0.50
    force_init_above_floor_eps: float = 1e-3

    def __post_init__(self) -> None:
        if self.method not in ("barycenter", "er_ls_ground"):
            raise ValueError("method must be 'barycenter' or 'er_ls_ground'")
        if self.barycenter_z_offset <= 0.0:
            raise ValueError("barycenter_z_offset must be > 0")
        if self.force_init_above_floor_eps <= 0.0:
            raise ValueError("force_init_above_floor_eps must be > 0")


@dataclass(frozen=True)
class MLEEMGroundConfig:
    """
    Configuration principale de l'algorithme MLE/EM avec sol rigide.

    Idée physique
    -------------
    On cherche UNE source réelle x = [x,y,z], et on construit sa source image
    par symétrie par rapport au sol z = floor_z.

    Deux modèles sont prévus :

    1) model_type = "additive"
       Modèle énergétique simplifié :
           h_i = 1/r_i^2 + alpha / r_img_i^2

       Ici alpha joue le rôle d'un poids énergétique effectif de la réflexion.
       Ce modèle est le plus simple pour une logique de type EM.

    2) model_type = "coherent"
       Modèle énergétique cohérent :
           h_i = 1/r_i^2
               + alpha^2 / r_img_i^2
               + 2 alpha Gamma(tau_i)/(r_i r_img_i)

       Ici alpha est un coefficient effectif de réflexion en amplitude
       (pression), et Gamma(tau) modélise la cohérence fréquentielle
       de la bande observée.

    max_iter / lam / tol_step / tol_cost
    ------------------------------------
    Paramètres du solveur type Levenberg-Marquardt / GEM.

    profile_source_energy
    ---------------------
    Si True, l'amplitude énergétique globale K est estimée analytiquement
    à chaque itération (recommandé).

    estimate_alpha
    --------------
    Si False, alpha est fixé à alpha_init.
    Si True, alpha est estimé dans [alpha_min, alpha_max].

    alpha_grid_size
    ---------------
    Utilisé pour une recherche robuste initiale de alpha.

    fd_eps
    ------
    Pas de différences finies pour approximer le jacobien si besoin.

    stable_eps
    ----------
    Petit epsilon numérique pour éviter divisions par zéro.

    use_band_integral
    -----------------
    Si False :
        Gamma(tau) est calculé avec une formule bande plate.
    Si True :
        on garde l'option ouverte pour un modèle spectral plus riche plus tard.

    f_low_hz / f_high_hz
    --------------------
    Bande utile du signal pour le modèle cohérent.

    sound_speed
    -----------
    Vitesse du son utilisée pour tau = (r_img - r_dir)/c.

    estimate_noise_floor
    --------------------
    Garde l'option ouverte pour estimer un offset énergétique.
    Dans un premier temps, on le laissera à False.

    store_history
    -------------
    Conserve l'historique coût/pas/alpha pour le debug.
    """
    model_type: MLEEMModelType = "coherent"

    # Solveur
    max_iter: int = 40
    lam: float = 1e-2
    tol_step: float = 1e-6
    tol_cost: float = 1e-9
    lam_min: float = 1e-8
    lam_max: float = 1e6

    # Profilage de l'énergie source K
    profile_source_energy: bool = True
    min_source_energy: float = 1e-12

    # Réflexion / source image
    estimate_alpha: bool = False
    alpha_init: float = 1.0
    alpha_min: float = 0.0
    alpha_max: float = 1.5
    alpha_grid_size: int = 31

    # Offset énergétique (optionnel)
    estimate_noise_floor: bool = False
    noise_floor_init: float = 0.0
    noise_floor_min: float = 0.0

    # Numérique
    fd_eps: float = 1e-5
    stable_eps: float = 1e-12

    # Modèle cohérent fréquentiel
    use_band_integral: bool = False
    f_low_hz: float = 80.0
    f_high_hz: float = 260.0
    sound_speed: float = 343.0

    # Debug
    store_history: bool = True

    def __post_init__(self) -> None:
        if self.model_type not in ("additive", "coherent"):
            raise ValueError("model_type must be 'additive' or 'coherent'")

        if self.max_iter <= 0:
            raise ValueError("max_iter must be >= 1")
        if self.lam <= 0.0:
            raise ValueError("lam must be > 0")
        if self.tol_step <= 0.0:
            raise ValueError("tol_step must be > 0")
        if self.tol_cost <= 0.0:
            raise ValueError("tol_cost must be > 0")

        if self.lam_min <= 0.0:
            raise ValueError("lam_min must be > 0")
        if self.lam_max < self.lam_min:
            raise ValueError("lam_max must be >= lam_min")

        if self.min_source_energy <= 0.0:
            raise ValueError("min_source_energy must be > 0")

        if self.alpha_min < 0.0:
            raise ValueError("alpha_min must be >= 0")
        if self.alpha_max <= self.alpha_min:
            raise ValueError("alpha_max must be > alpha_min")
        if self.alpha_grid_size < 2:
            raise ValueError("alpha_grid_size must be >= 2")

        if self.noise_floor_init < 0.0:
            raise ValueError("noise_floor_init must be >= 0")
        if self.noise_floor_min < 0.0:
            raise ValueError("noise_floor_min must be >= 0")

        if self.fd_eps <= 0.0:
            raise ValueError("fd_eps must be > 0")
        if self.stable_eps <= 0.0:
            raise ValueError("stable_eps must be > 0")

        if self.f_low_hz <= 0.0:
            raise ValueError("f_low_hz must be > 0")
        if self.f_high_hz <= self.f_low_hz:
            raise ValueError("f_high_hz must be > f_low_hz")

        if self.sound_speed <= 0.0:
            raise ValueError("sound_speed must be > 0")


__all__ = [
    "MLEEMModelType",
    "MLEEMInitMethod",
    "MLEEMInitConfig",
    "MLEEMGroundConfig",
]
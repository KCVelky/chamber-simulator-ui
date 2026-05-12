from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class EnergyConfig:
    """
    Paramètres pour estimer l'énergie reçue par chaque micro.

    - remove_mean:
        Retire la composante continue (offset/DC) avant de calculer l'énergie.

    - window_s / hop_s:
        Si window_s est None: énergie calculée sur tout le signal.
        Sinon: énergie calculée par fenêtres (type STFT mais en énergie),
        puis agrégée (moyenne robuste).

    - trim_frac:
        Si on utilise des fenêtres: on retire les frames extrêmes (bruit impulsionnel,
        transitions de fade, etc.). Ex: 0.10 => on retire 10% des frames les plus faibles
        et 10% des plus fortes, puis on moyenne.
    """
    remove_mean: bool = True
    window_s: Optional[float] = None
    hop_s: Optional[float] = None
    trim_frac: float = 0.10


@dataclass(frozen=True)
class ERLSConfig:
    """
    Configuration de l'algo ER-LS (Energy Ratios - Least Squares).

    - ref_idx:
        Micro de référence j. On forme des rapports R_i = E_i / E_ref.

    - kappa_eps:
        Seuil pour éviter kappa ~ 1 (sphère d'Apollonius instable numériquement).

    - min_pairs:
        Nombre minimal de sphères valides (hors ref) nécessaires.
        En 3D, il faut au moins 3 contraintes non dégénérées.
    """
    ref_idx: int = 0
    kappa_eps: float = 1e-3
    min_pairs: int = 3

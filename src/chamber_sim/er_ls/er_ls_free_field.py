from __future__ import annotations

from typing import Optional, Tuple, Dict, Any

import numpy as np

from .config import EnergyConfig, ERLSConfig
from .er_ls import er_ls


def er_ls_free_field(
    mic_positions: np.ndarray,
    y: np.ndarray,
    fs: float,
    cfg_E: EnergyConfig = EnergyConfig(),
    cfg: ERLSConfig = ERLSConfig(),
    mic_gains: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """
    ER-LS champ libre strict.

    Cette variante est volontairement séparée de ER_LS adaptatif :
    elle applique toujours le modèle énergétique classique en 1/r²,
    sans modèle image, sans coefficient de réflexion et sans correction sol.

    Important : si la propagation simulée contient une réflexion au sol,
    les signaux `y` incluent quand même cette réflexion. L'algorithme,
    lui, l'ignore volontairement et estime la source comme si le champ
    était libre.
    """
    x_hat, err, debug = er_ls(
        mic_positions=mic_positions,
        y=y,
        fs=fs,
        cfg_E=cfg_E,
        cfg=cfg,
        mic_gains=mic_gains,
    )
    debug = dict(debug)
    debug.update(
        {
            "algorithm_variant": "ER_LS_FREE_FIELD",
            "energy_model": "free_field_inverse_square",
            "uses_ground_model": False,
            "uses_image_source": False,
            "ground_is_ignored_even_if_present_in_signals": True,
        }
    )
    return x_hat, err, debug

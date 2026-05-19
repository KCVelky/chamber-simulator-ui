from __future__ import annotations

import json
import sys
from html import escape
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

# Allows running this file directly from the project root without installing the package.
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from PySide6.QtCore import QObject, QPoint, Qt, QThread, QTimer, Signal
    from PySide6.QtGui import QAction, QFont
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QFileDialog,
        QFormLayout,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QInputDialog,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QPlainTextEdit,
        QProgressBar,
        QScrollArea,
        QSizePolicy,
        QSpinBox,
        QSplitter,
        QTabWidget,
        QTextEdit,
        QToolButton,
        QTreeWidget,
        QTreeWidgetItem,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:  # pragma: no cover - executed only on machines without Qt
    QT_IMPORT_ERROR = exc
    Qt = object()  # type: ignore
    QObject = object  # type: ignore
    QThread = object  # type: ignore
    QTimer = object  # type: ignore
    Signal = lambda *args, **kwargs: None  # type: ignore
    QAction = object  # type: ignore
    QApplication = object  # type: ignore
    QMainWindow = object  # type: ignore
    QWidget = object  # type: ignore
    QFrame = object  # type: ignore
else:
    QT_IMPORT_ERROR = None


try:
    from chamber_sim.doe import DOEManager, DOERunner, DOE_RUN_SHEETS, DOEAutomationRunner, DOEAutomationPlan
except Exception as exc:  # pragma: no cover - DOE module is optional while editing the UI standalone
    DOE_IMPORT_ERROR = exc
    DOEManager = None  # type: ignore
    DOERunner = None  # type: ignore
    DOE_RUN_SHEETS: Dict[str, Any] = {}
    DOEAutomationRunner = None  # type: ignore
    DOEAutomationPlan = None  # type: ignore
else:
    DOE_IMPORT_ERROR = None

try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
except Exception as exc:  # pragma: no cover - optional 3D backend
    PV_IMPORT_ERROR = exc
    pv = None  # type: ignore
    QtInteractor = None  # type: ignore
else:
    PV_IMPORT_ERROR = None

try:
    from chamber_sim.experiment_runner import DEFAULT_CONFIG, deep_copy_config, run_experiment_from_config
except Exception as exc:  # pragma: no cover - clearer error when launched outside project
    CHAMBER_IMPORT_ERROR = exc
    DEFAULT_CONFIG: Dict[str, Any] = {
        "geometry": {
            "Lx": 5.0,
            "Ly": 4.0,
            "Lz": 2.7,
            "margin": 0.2,
            "source": {"x": 2.5, "y": 2.0, "z": 1.4},
            "mics_csv": "1.000,1.000,1.200\n4.000,1.000,1.200\n1.000,3.000,1.200\n4.000,3.000,1.200",
        },
        "signal": {
            "fs": 8000,
            "duration": 1.0,
            "rms": 0.1,
            "f_low": 300,
            "f_high": 3000,
            "seed": 42,
            "use_mod": False,
            "f_mod": 2.0,
            "mod_depth": 0.5,
            "use_fade": True,
            "fade_in": 0.02,
            "fade_out": 0.02,
        },
        "propagation": {
            "c": 343.0,
            "use_spreading": True,
            "gain_at_1m": 1.0,
            "use_floor_image": False,
            "floor_z": 0.0,
            "add_noise": False,
            "snr_db": 30,
            "noise_indep": True,
            "noise_seed": 123,
            "enable_denoise": False,
            "denoise_nperseg": 256,
            "denoise_noverlap": 128,
            "denoise_noise_head": 0.05,
            "denoise_noise_tail": 0.05,
            "denoise_gain_floor": 0.05,
        },
        "algorithms": {
            "enable_algorithms": True,
            "alg_choice": "TDOA",
            "plot_estimated_source": True,
            "tdoa_interp": 16,
            "tdoa_grid_dx": 0.1,
            "tdoa_grid_dy": 0.1,
            "tdoa_grid_dz": 0.1,
            "tdoa_z_fixed": "None",
            "er_ref_idx": 0,
            "er_window_s": 0.05,
            "er_hop_s": 0.01,
            "er_trim_frac": 0.1,
            "er_kappa_eps": 1e-9,
            "ernls_max_iter": 100,
            "ernls_lam": 1e-3,
            "ernls_tol_step": 1e-6,
            "ernls_tol_cost": 1e-8,
            "mleem_model_type": "additive",
            "mleem_init_method": "barycenter",
            "mleem_estimate_alpha": True,
            "mleem_alpha_init": 0.5,
            "mleem_alpha_min": 0.0,
            "mleem_alpha_max": 1.0,
            "mleem_alpha_grid_size": 21,
            "mleem_max_iter": 50,
            "mleem_lam": 1e-3,
            "mleem_tol_step": 1e-6,
            "mleem_tol_cost": 1e-8,
            "mleem_fd_eps": 1e-4,
            "mleem_barycenter_z_offset": 0.0,
        },
        "plots": {
            "plot_scene": True,
            "plot_source_time": False,
            "plot_source_spectrum": False,
            "plot_mics": False,
            "plot_mics_zoom": False,
            "plot_source_spectrogram": False,
            "plot_mic_spectrogram": False,
            "mic_spec_index": 0,
            "plot_denoise_compare": False,
            "plot_denoise_mic": 0,
            "print_delta_r": True,
            "zoom_tmax": 0.05,
        },
        "comparison": {
            "enabled": False,
            "mode": "Deux algorithmes",
            "algorithm_a": "ER_LS",
            "algorithm_b": "ER_NLS",
            "mic_algorithm": "ER_LS",
            "mics_b_csv": "1.000,1.000,1.200\n4.000,1.000,1.200\n2.500,3.300,1.200\n2.500,0.700,1.200\n2.500,2.000,1.800\n2.500,2.000,0.800",
        },
        "interference_heatmap": {
            "enabled": False,
            "freq_hz": 1000.0,
            "ref_mic_index": 1,
            "beta": 0.7,
            "grid_n": 90,
            "opacity": 0.65,
        },
    }

    def deep_copy_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
        return json.loads(json.dumps(cfg))

    def run_experiment_from_config(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError(
            "Impossible d'importer chamber_sim.experiment_runner. "
            "Place ce fichier à la racine du projet, à côté du dossier src/. "
            f"Erreur originale: {CHAMBER_IMPORT_ERROR}"
        )
else:
    CHAMBER_IMPORT_ERROR = None


ALGORITHMS = ["None", "TDOA", "ER_LS_FF", "ER_LS", "ER_NLS", "MLE_EM_GROUND"]

# Paramètres purement UI : ils peuvent ne pas exister dans le DEFAULT_CONFIG du runner.
DEFAULT_CONFIG.setdefault("interference_heatmap", {
    "enabled": False,
    "freq_hz": 1000.0,
    "ref_mic_index": 1,
    "beta": 0.7,
    "grid_n": 90,
    "opacity": 0.65,
})


PARAMETER_HELP: Dict[Tuple[str, ...], Dict[str, str]] = {
    ("geometry", "Lx"): {
        "meaning": "Longueur de la chambre selon l'axe x.",
        "effect": "Définit le domaine admissible de la source et des microphones. Une grande dimension augmente les distances, donc les retards acoustiques et la zone de recherche des algorithmes.",
    },
    ("geometry", "Ly"): {
        "meaning": "Largeur de la chambre selon l'axe y.",
        "effect": "Influence les distances source-microphones et la géométrie du problème inverse. Une mauvaise cohérence avec les positions peut fausser la localisation.",
    },
    ("geometry", "Lz"): {
        "meaning": "Hauteur de la chambre selon l'axe z.",
        "effect": "Contrôle la dimension verticale. Pour localiser correctement z, il faut des microphones avec une géométrie 3D suffisamment informative.",
    },
    ("geometry", "margin"): {
        "meaning": "Marge de sécurité utilisée pour éviter de placer automatiquement des points trop près des parois.",
        "effect": "Une marge trop grande réduit la zone utile. Une marge trop faible peut créer des configurations proches des murs, plus sensibles aux réflexions et ambiguïtés.",
    },
    ("geometry", "source", "x"): {
        "meaning": "Coordonnée x de la source vraie utilisée pour générer les signaux simulés.",
        "effect": "Déplace la vérité terrain. Les retards, niveaux reçus et erreurs de localisation sont recalculés par rapport à cette position.",
    },
    ("geometry", "source", "y"): {
        "meaning": "Coordonnée y de la source vraie utilisée pour générer les signaux simulés.",
        "effect": "Modifie la distribution des distances aux microphones et donc les TDOA et rapports d'énergie.",
    },
    ("geometry", "source", "z"): {
        "meaning": "Coordonnée z de la source vraie utilisée pour générer les signaux simulés.",
        "effect": "Affecte la localisation verticale. Si les microphones sont presque coplanaires, l'estimation de z peut devenir moins robuste.",
    },
    ("geometry", "mics_csv"): {
        "meaning": "Positions des microphones, une ligne par micro au format x,y,z.",
        "effect": "C'est un paramètre critique. L'ouverture du réseau, le nombre de micros et leur non-alignement déterminent l'observabilité et la précision des algorithmes.",
    },
    ("signal", "fs"): {
        "meaning": "Fréquence d'échantillonnage des signaux simulés.",
        "effect": "Plus fs est élevée, meilleure est la résolution temporelle des retards TDOA, mais le calcul devient plus lourd.",
    },
    ("signal", "duration"): {
        "meaning": "Durée du signal source simulé.",
        "effect": "Une durée plus longue stabilise les estimations d'énergie et de corrélation, mais augmente le coût de calcul.",
    },
    ("signal", "rms"): {
        "meaning": "Niveau RMS du signal source.",
        "effect": "Augmente ou diminue le niveau reçu. Avec bruit activé, cela influence directement le rapport signal/bruit effectif.",
    },
    ("signal", "f_low"): {
        "meaning": "Borne basse fréquentielle du signal source.",
        "effect": "Les basses fréquences sont moins sensibles au bruit fin mais offrent une résolution temporelle plus faible pour la localisation.",
    },
    ("signal", "f_high"): {
        "meaning": "Borne haute fréquentielle du signal source.",
        "effect": "Les hautes fréquences améliorent la précision temporelle, mais peuvent être plus sensibles au bruit, au filtrage et aux erreurs de modèle.",
    },
    ("signal", "seed"): {
        "meaning": "Graine aléatoire du signal source.",
        "effect": "Permet de reproduire exactement la même simulation. Changer la graine change le signal mais pas la configuration géométrique.",
    },
    ("signal", "use_mod"): {
        "meaning": "Active une modulation d'amplitude du signal source.",
        "effect": "Peut rendre le signal plus réaliste, mais modifie l'enveloppe temporelle utilisée par certains calculs d'énergie.",
    },
    ("signal", "f_mod"): {
        "meaning": "Fréquence de modulation lorsque la modulation est activée.",
        "effect": "Une modulation lente change l'enveloppe du signal et peut influencer les méthodes basées sur l'énergie temporelle.",
    },
    ("signal", "mod_depth"): {
        "meaning": "Profondeur de modulation, entre faible modulation et modulation marquée.",
        "effect": "Une profondeur forte rend l'amplitude plus variable, ce qui peut perturber ou aider les méthodes d'énergie selon le cas.",
    },
    ("signal", "use_fade"): {
        "meaning": "Active un fondu d'entrée et de sortie sur le signal.",
        "effect": "Réduit les discontinuités qui peuvent créer des artefacts spectraux et perturber les corrélations.",
    },
    ("signal", "fade_in"): {
        "meaning": "Durée du fondu d'entrée.",
        "effect": "Un fondu trop long réduit la partie utile du signal. Un fondu trop court laisse plus de transitoires.",
    },
    ("signal", "fade_out"): {
        "meaning": "Durée du fondu de sortie.",
        "effect": "Limite les discontinuités finales et stabilise les analyses spectrales.",
    },
    ("propagation", "c"): {
        "meaning": "Vitesse du son utilisée pour convertir les retards en distances.",
        "effect": "Paramètre très sensible pour TDOA. Une vitesse du son erronée introduit un biais direct sur la position estimée.",
    },
    ("propagation", "use_spreading"): {
        "meaning": "Active l'atténuation géométrique en 1/r.",
        "effect": "Indispensable pour les méthodes énergétiques. Si désactivé, les niveaux reçus ne portent plus correctement l'information de distance.",
    },
    ("propagation", "gain_at_1m"): {
        "meaning": "Gain de référence appliqué à 1 mètre de la source.",
        "effect": "Change l'échelle d'amplitude. Son effet est surtout visible avec bruit et méthodes d'énergie.",
    },
    ("propagation", "use_floor_image"): {
        "meaning": "Ajoute une source image pour modéliser une réflexion sur sol rigide.",
        "effect": "Rend la propagation plus réaliste mais plus complexe. Peut créer des interférences et des biais si l'algorithme suppose un trajet direct seul.",
    },
    ("propagation", "floor_z"): {
        "meaning": "Altitude du plan de sol utilisé pour la source image.",
        "effect": "Modifie la position de la réflexion de sol et donc le retard et l'amplitude du trajet réfléchi.",
    },
    ("propagation", "add_noise"): {
        "meaning": "Ajoute un bruit de mesure aux microphones.",
        "effect": "Permet de tester la robustesse. Plus le bruit est fort, plus la localisation devient incertaine.",
    },
    ("propagation", "snr_db"): {
        "meaning": "Rapport signal/bruit en décibels lorsque le bruit est activé.",
        "effect": "Un SNR faible dégrade les corrélations TDOA et les estimations d'énergie. Un SNR élevé donne un cas plus idéal.",
    },
    ("propagation", "noise_indep"): {
        "meaning": "Indique si le bruit est indépendant d'un micro à l'autre.",
        "effect": "Un bruit indépendant est plus réaliste pour des capteurs séparés. Il peut réduire la cohérence des corrélations inter-micros.",
    },
    ("propagation", "noise_seed"): {
        "meaning": "Graine aléatoire du bruit.",
        "effect": "Permet de reproduire exactement un même tirage de bruit pour comparer les algorithmes équitablement.",
    },
    ("propagation", "enable_denoise"): {
        "meaning": "Active un débruitage de type Wiener/STFT avant localisation.",
        "effect": "Peut améliorer les signaux bruités, mais un mauvais réglage peut supprimer de l'information utile aux retards ou à l'énergie.",
    },
    ("propagation", "denoise_nperseg"): {
        "meaning": "Taille de fenêtre STFT utilisée par le débruitage.",
        "effect": "Grande fenêtre : meilleure résolution fréquentielle. Petite fenêtre : meilleure résolution temporelle. Le compromis influence TDOA et spectres.",
    },
    ("propagation", "denoise_noverlap"): {
        "meaning": "Recouvrement entre fenêtres STFT du débruitage.",
        "effect": "Un recouvrement élevé lisse le traitement mais augmente le temps de calcul.",
    },
    ("propagation", "denoise_noise_head"): {
        "meaning": "Durée en début de signal utilisée pour estimer le bruit.",
        "effect": "Si cette zone contient du signal utile, l'estimation du bruit devient biaisée et le débruitage peut abîmer la localisation.",
    },
    ("propagation", "denoise_noise_tail"): {
        "meaning": "Durée en fin de signal utilisée pour estimer le bruit.",
        "effect": "Même logique que noise_head : utile si la fin contient principalement du bruit de fond.",
    },
    ("propagation", "denoise_gain_floor"): {
        "meaning": "Gain minimal autorisé pendant le débruitage.",
        "effect": "Évite de supprimer complètement des composantes. Trop bas : artefacts possibles. Trop haut : débruitage faible.",
    },
    ("algorithms", "enable_algorithms"): {
        "meaning": "Active ou désactive l'étape de localisation automatique.",
        "effect": "Si désactivé, la simulation peut générer les signaux mais aucune source estimée n'est calculée.",
    },
    ("algorithms", "alg_choice"): {
        "meaning": "Algorithme de localisation utilisé.",
        "effect": "TDOA exploite les retards. ER_LS_FF force le modèle 1/r² sans sol. ER_LS/ER_NLS peuvent utiliser le modèle sol/image si l’option sol est activée. MLE/EM utilise un modèle probabiliste avec source sol/image.",
    },
    ("algorithms", "plot_estimated_source"): {
        "meaning": "Affiche la source estimée dans la scène 3D après calcul.",
        "effect": "N'affecte pas l'algorithme, seulement la visualisation de la position estimée et de l'erreur.",
    },
    ("algorithms", "tdoa_interp"): {
        "meaning": "Facteur d'interpolation pour l'estimation GCC-PHAT des retards.",
        "effect": "Augmente la résolution temporelle apparente des TDOA. Plus grand peut améliorer la précision mais coûte plus cher.",
    },
    ("algorithms", "tdoa_grid_dx"): {
        "meaning": "Pas de grille en x pour la recherche TDOA.",
        "effect": "Pas petit : meilleure précision spatiale mais temps de calcul plus élevé. Pas grand : localisation plus rapide mais plus grossière.",
    },
    ("algorithms", "tdoa_grid_dy"): {
        "meaning": "Pas de grille en y pour la recherche TDOA.",
        "effect": "Contrôle la résolution de recherche latérale. Même compromis précision/temps de calcul que dx.",
    },
    ("algorithms", "tdoa_grid_dz"): {
        "meaning": "Pas de grille en z pour la recherche TDOA.",
        "effect": "Contrôle la résolution verticale. Important si z n'est pas fixé.",
    },
    ("algorithms", "tdoa_z_fixed"): {
        "meaning": "Hauteur z imposée pour une recherche TDOA en plan horizontal. Mettre None pour chercher en 3D.",
        "effect": "Fixer z accélère et stabilise la recherche si la hauteur est connue, mais crée un biais si la vraie source n'est pas à cette hauteur.",
    },
    ("algorithms", "er_ref_idx"): {
        "meaning": "Index du microphone de référence pour les méthodes de rapport d'énergie.",
        "effect": "Un mauvais micro de référence, bruité ou mal placé, peut dégrader fortement ER-LS/ER-NLS.",
    },
    ("algorithms", "er_window_s"): {
        "meaning": "Taille de fenêtre temporelle pour calculer l'énergie locale.",
        "effect": "Fenêtre longue : énergie plus stable mais moins locale. Fenêtre courte : plus sensible aux fluctuations et au bruit.",
    },
    ("algorithms", "er_hop_s"): {
        "meaning": "Pas temporel entre deux fenêtres d'énergie.",
        "effect": "Un pas faible donne plus d'échantillons d'énergie mais augmente le calcul et la redondance.",
    },
    ("algorithms", "er_trim_frac"): {
        "meaning": "Fraction d'échantillons extrêmes retirés dans les méthodes énergétiques.",
        "effect": "Rend l'estimation plus robuste aux outliers. Trop élevé peut retirer de l'information utile.",
    },
    ("algorithms", "er_kappa_eps"): {
        "meaning": "Petite constante de régularisation pour éviter les divisions ou rapports instables.",
        "effect": "Stabilise le calcul numérique. Trop grand peut biaiser les rapports d'énergie.",
    },
    ("algorithms", "ernls_max_iter"): {
        "meaning": "Nombre maximal d'itérations pour ER-NLS.",
        "effect": "Plus d'itérations donne plus de chances de convergence, mais augmente le temps de calcul.",
    },
    ("algorithms", "ernls_lam"): {
        "meaning": "Paramètre de régularisation/amortissement pour ER-NLS.",
        "effect": "Aide à stabiliser l'optimisation. Trop fort ralentit ou biaise, trop faible peut rendre l'algorithme instable.",
    },
    ("algorithms", "ernls_tol_step"): {
        "meaning": "Tolérance d'arrêt sur le déplacement de la solution ER-NLS.",
        "effect": "Plus petit : convergence plus stricte mais calcul plus long. Plus grand : arrêt plus rapide mais potentiellement moins précis.",
    },
    ("algorithms", "ernls_tol_cost"): {
        "meaning": "Tolérance d'arrêt sur la variation du coût ER-NLS.",
        "effect": "Contrôle l'arrêt de l'optimisation lorsque l'amélioration devient faible.",
    },
    ("algorithms", "mleem_model_type"): {
        "meaning": "Type de modèle utilisé par MLE/EM pour combiner trajet direct et réflexion.",
        "effect": "Le modèle additive est plus simple. Le modèle coherent peut représenter des interférences mais est plus sensible aux hypothèses de phase.",
    },
    ("algorithms", "mleem_init_method"): {
        "meaning": "Méthode d'initialisation de l'optimisation MLE/EM.",
        "effect": "Une bonne initialisation réduit le risque de minimum local et accélère la convergence.",
    },
    ("algorithms", "mleem_estimate_alpha"): {
        "meaning": "Autorise l'algorithme à estimer le coefficient alpha du modèle de réflexion/sol.",
        "effect": "Peut mieux s'adapter aux réflexions, mais ajoute une inconnue et peut rendre l'optimisation moins robuste.",
    },
    ("algorithms", "mleem_alpha_init"): {
        "meaning": "Valeur initiale du coefficient alpha.",
        "effect": "Influence le point de départ de l'optimisation MLE/EM et peut modifier la convergence.",
    },
    ("algorithms", "mleem_alpha_min"): {
        "meaning": "Borne minimale autorisée pour alpha.",
        "effect": "Contraint le modèle. Une borne trop restrictive peut empêcher d'atteindre la meilleure solution.",
    },
    ("algorithms", "mleem_alpha_max"): {
        "meaning": "Borne maximale autorisée pour alpha.",
        "effect": "Limite l'influence maximale de la composante réfléchie ou du paramètre alpha.",
    },
    ("algorithms", "mleem_alpha_grid_size"): {
        "meaning": "Nombre de valeurs testées pour alpha lors d'une recherche discrète.",
        "effect": "Plus grand : exploration plus fine mais calcul plus long.",
    },
    ("algorithms", "mleem_max_iter"): {
        "meaning": "Nombre maximal d'itérations MLE/EM.",
        "effect": "Augmente les chances de convergence au prix d'un temps de calcul plus élevé.",
    },
    ("algorithms", "mleem_lam"): {
        "meaning": "Paramètre de régularisation/amortissement de l'optimisation MLE/EM.",
        "effect": "Stabilise les mises à jour. Trop grand peut ralentir ou lisser excessivement la solution.",
    },
    ("algorithms", "mleem_tol_step"): {
        "meaning": "Tolérance d'arrêt sur le pas de mise à jour MLE/EM.",
        "effect": "Contrôle la précision de convergence sur la position estimée.",
    },
    ("algorithms", "mleem_tol_cost"): {
        "meaning": "Tolérance d'arrêt sur la variation du coût MLE/EM.",
        "effect": "Arrête l'optimisation lorsque le gain devient négligeable.",
    },
    ("algorithms", "mleem_fd_eps"): {
        "meaning": "Pas de différence finie pour approximer certaines dérivées numériques.",
        "effect": "Trop petit : bruit numérique. Trop grand : gradient imprécis. Influence la stabilité de l'optimisation.",
    },
    ("algorithms", "mleem_barycenter_z_offset"): {
        "meaning": "Décalage vertical ajouté à l'initialisation par barycentre.",
        "effect": "Peut aider si la source est attendue au-dessus ou au-dessous du plan moyen des microphones.",
    },
    ("plots", "plot_scene"): {
        "meaning": "Active l'affichage de la scène.",
        "effect": "N'affecte pas la localisation. Dans cette UI, la scène 3D reste disponible en temps réel.",
    },
    ("plots", "plot_source_time"): {
        "meaning": "Demande l'affichage temporel du signal source.",
        "effect": "Sortie de diagnostic uniquement, sans effet sur l'algorithme.",
    },
    ("plots", "plot_source_spectrum"): {
        "meaning": "Demande l'affichage du spectre du signal source.",
        "effect": "Aide à vérifier la bande fréquentielle utile, sans modifier la localisation.",
    },
    ("plots", "plot_mics"): {
        "meaning": "Demande l'affichage des signaux reçus aux microphones.",
        "effect": "Diagnostic utile pour voir les retards et niveaux, sans effet direct sur le calcul.",
    },
    ("plots", "plot_mics_zoom"): {
        "meaning": "Affiche un zoom temporel des signaux micros.",
        "effect": "Aide à inspecter les arrivées temporelles et les décalages entre microphones.",
    },
    ("plots", "plot_source_spectrogram"): {
        "meaning": "Affiche le spectrogramme de la source.",
        "effect": "Diagnostic temps-fréquence, utile pour vérifier modulation, bande utile et fade.",
    },
    ("plots", "plot_mic_spectrogram"): {
        "meaning": "Affiche le spectrogramme d'un microphone.",
        "effect": "Diagnostic du signal reçu, du bruit et du débruitage.",
    },
    ("plots", "mic_spec_index"): {
        "meaning": "Index du microphone utilisé pour le spectrogramme.",
        "effect": "Change uniquement le microphone visualisé, pas la localisation.",
    },
    ("plots", "plot_denoise_compare"): {
        "meaning": "Affiche une comparaison avant/après débruitage.",
        "effect": "Diagnostic visuel pour vérifier si le débruitage aide ou dégrade le signal utile.",
    },
    ("plots", "plot_denoise_mic"): {
        "meaning": "Index du microphone utilisé pour la comparaison de débruitage.",
        "effect": "Change uniquement le canal affiché.",
    },
    ("plots", "print_delta_r"): {
        "meaning": "Affiche les écarts de distance Δr entre source et microphones.",
        "effect": "Diagnostic utile pour comprendre les retards attendus, sans effet sur l'estimation.",
    },
    ("plots", "zoom_tmax"): {
        "meaning": "Durée maximale affichée dans les zooms temporels.",
        "effect": "Change uniquement la fenêtre de visualisation temporelle.",
    },
}


def _nested_default(path: Tuple[str, ...]) -> Any:
    current: Any = DEFAULT_CONFIG
    for part in path:
        if not isinstance(current, dict) or part not in current:
            return "—"
        current = current[part]
    return current


def _format_default(value: Any) -> str:
    if isinstance(value, bool):
        return "activé" if value else "désactivé"
    if isinstance(value, (int, float)):
        return f"{value:g}"
    if value is None:
        return "None"
    if isinstance(value, str):
        return value if value.strip() else "chaîne vide"
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


def _parameter_tooltip(path: Tuple[str, ...], display_name: str = "") -> str:
    info = PARAMETER_HELP.get(tuple(path), {})
    default_value = _format_default(_nested_default(tuple(path)))
    title = display_name or path[-1]
    meaning = info.get("meaning", "Paramètre de configuration du modèle.")
    effect = info.get("effect", "Influence la simulation ou son affichage selon le module qui l'utilise.")
    path_text = ".".join(path)
    return (
        "<div style='width: 390px; white-space: normal;'>"
        f"<b>{escape(str(title))}</b><br>"
        f"<span style='color:#64748b;'>Chemin : {escape(path_text)}</span><br><br>"
        f"<b>Ce que ça veut dire :</b><br>{escape(meaning)}<br><br>"
        f"<b>Valeur par défaut :</b> {escape(default_value)}<br><br>"
        f"<b>Effet sur la localisation / l'algorithme :</b><br>{escape(effect)}"
        "</div>"
    )


COMSOL_QSS = """
* {
    font-family: 'Aptos', 'Segoe UI Variable', 'Segoe UI', 'Inter', Arial, sans-serif;
    font-size: 12px;
}
QMainWindow, QWidget {
    background: #eef2f6;
    color: #172033;
}
QMenuBar {
    background: #f9fbfd;
    border-bottom: 1px solid #b9c6d4;
    padding: 2px 8px;
}
QMenuBar::item {
    padding: 5px 9px;
    background: transparent;
}
QMenuBar::item:selected {
    background: #dbeafe;
    border-radius: 4px;
}
QMenu {
    background: #ffffff;
    border: 1px solid #b9c6d4;
}
QMenu::item {
    padding: 6px 28px;
}
QMenu::item:selected {
    background: #dbeafe;
}
QFrame#Ribbon {
    background: #f8fafc;
    border-bottom: 1px solid #aebccd;
}
QFrame#RibbonGroup {
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 3px;
}
QLabel#RibbonGroupTitle {
    color: #526176;
    font-size: 10px;
    font-weight: 600;
}
QToolButton {
    background: #ffffff;
    border: 1px solid transparent;
    border-radius: 3px;
    padding: 5px 8px;
    color: #172033;
}
QToolButton:hover {
    background: #e0ecff;
    border-color: #8bb6e8;
}
QToolButton:pressed {
    background: #cfe2ff;
}
QToolButton:checked {
    background: #dbeafe;
    border-color: #2563eb;
    color: #1d4ed8;
    font-weight: 600;
}
QToolButton#PrimaryTool {
    background: #2563eb;
    border-color: #1d4ed8;
    color: #ffffff;
    font-weight: 600;
}
QToolButton#PrimaryTool:hover {
    background: #1d4ed8;
}
QFrame#LeftPanel, QFrame#RightPanel, QFrame#BottomPanel, QFrame#GraphicsPanel {
    background: #ffffff;
    border: 1px solid #b9c6d4;
}
QLabel#PanelTitle {
    font-size: 14px;
    font-weight: 600;
    color: #172033;
}
QLabel#SubtleText, QLabel#FieldHelp {
    color: #64748b;
    font-size: 11px;
}
QLabel#MetricValue {
    color: #172033;
    font-size: 16px;
    font-weight: 600;
}
QLabel#MetricLabel {
    color: #64748b;
    font-size: 10px;
    font-weight: 600;
}
QTreeWidget {
    background: #ffffff;
    border: none;
    outline: none;
}
QTreeWidget::item {
    padding: 4px 4px;
}
QTreeWidget::item:selected {
    background: #dbeafe;
    color: #0f172a;
}
QTabWidget::pane {
    border-top: 1px solid #cbd5e1;
    background: #ffffff;
}
QTabBar::tab {
    background: #edf2f7;
    border: 1px solid #cbd5e1;
    border-bottom: none;
    padding: 7px 12px;
    margin-right: 2px;
    color: #334155;
}
QTabBar::tab:selected {
    background: #ffffff;
    color: #0f172a;
    font-weight: 600;
}
QGroupBox {
    background: #ffffff;
    border: 1px solid #d4dde8;
    border-radius: 3px;
    margin-top: 12px;
    padding: 14px 10px 10px 10px;
    font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: #172033;
}
QLineEdit, QComboBox, QTextEdit, QPlainTextEdit {
    background: #ffffff;
    border: 1px solid #b9c6d4;
    border-radius: 2px;
    padding: 5px 7px;
    selection-background-color: #2563eb;
    selection-color: #ffffff;
}
QLineEdit:focus, QComboBox:focus, QTextEdit:focus, QPlainTextEdit:focus {
    border: 1px solid #2563eb;
}
QCheckBox {
    spacing: 8px;
}
QPushButton {
    background: #f8fafc;
    border: 1px solid #b9c6d4;
    border-radius: 3px;
    padding: 6px 10px;
}
QPushButton:hover {
    background: #e0ecff;
    border-color: #8bb6e8;
}
QPushButton#SmallAction {
    padding: 4px 8px;
}
QPlainTextEdit#Console {
    background: #0f172a;
    color: #dbeafe;
    border: 1px solid #334155;
    font-family: 'Cascadia Mono', Consolas, 'Courier New', monospace;
    font-size: 11px;
}
QPlainTextEdit#ErrorConsole {
    background: #fff7ed;
    color: #7c2d12;
    border: 1px solid #fed7aa;
}
QFrame#StatusPill {
    border-radius: 10px;
    padding: 4px 8px;
}
QScrollArea {
    border: none;
}
QSplitter::handle {
    background: #cbd5e1;
}
QSplitter::handle:hover {
    background: #94a3b8;
}
QToolTip {
    background: #ffffff;
    color: #172033;
    border: 1px solid #94a3b8;
    padding: 8px;
}
"""


def _safe_float(value: Any, default: float) -> float:
    try:
        if isinstance(value, str) and value.strip().lower() in {"", "none", "null", "nan"}:
            return default
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int) -> int:
    try:
        if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
            return default
        return int(float(value))
    except Exception:
        return default


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on", "oui", "vrai"}
    return bool(value)


def parse_source(source: Any) -> np.ndarray:
    if isinstance(source, dict):
        return np.array([
            _safe_float(source.get("x"), 0.0),
            _safe_float(source.get("y"), 0.0),
            _safe_float(source.get("z"), 0.0),
        ], dtype=float)
    try:
        arr = np.asarray(source, dtype=float).reshape(-1)
        if arr.size >= 3:
            return arr[:3]
    except Exception:
        pass
    return np.zeros(3, dtype=float)


def parse_mics_csv(text: Any) -> np.ndarray:
    points: list[list[float]] = []
    for raw in str(text or "").replace(";", "\n").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.replace("\t", ",").split(",")]
        if len(parts) < 3:
            parts = line.split()
        if len(parts) < 3:
            continue
        try:
            points.append([float(parts[0]), float(parts[1]), float(parts[2])])
        except ValueError:
            # Allows an optional header such as x,y,z.
            continue
    if not points:
        return np.empty((0, 3), dtype=float)
    return np.asarray(points, dtype=float)


def vector_to_text(vec: Any) -> str:
    if vec is None:
        return "—"
    try:
        arr = np.asarray(vec, dtype=float).reshape(-1)
        if arr.size >= 3:
            return f"x={arr[0]:.3f} m, y={arr[1]:.3f} m, z={arr[2]:.3f} m"
        return ", ".join(f"{v:.4g}" for v in arr)
    except Exception:
        return str(vec)


def _reflection_beta_from_config(cfg: Dict[str, Any], default: float = 0.7) -> float:
    """Retourne le coefficient de réflexion utilisé pour les cartes d'interférence."""
    heatmap = cfg.get("interference_heatmap", {}) if isinstance(cfg, dict) else {}
    algorithms = cfg.get("algorithms", {}) if isinstance(cfg, dict) else {}
    raw = heatmap.get("beta", algorithms.get("mleem_alpha_init", default))
    return float(np.clip(_safe_float(raw, default), -2.0, 2.0))


def interference_ratio_grid(
    *,
    source: np.ndarray,
    Lx: float,
    Ly: float,
    plane_z: float,
    floor_z: float,
    freq_hz: float,
    beta: float,
    c: float,
    grid_n: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calcule eta(x,y) dans un plan z constant pour le trajet direct + image.

    eta = terme_interferentiel / energie_additive, borné entre -1 et 1 environ.
    eta > 0 : interférence constructive ; eta < 0 : destructive.
    """
    nx = int(np.clip(grid_n, 20, 220))
    ny = max(20, int(round(nx * max(Ly, 1e-9) / max(Lx, 1e-9))))
    xs = np.linspace(0.0, Lx, nx)
    ys = np.linspace(0.0, Ly, ny)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    Z = np.full_like(X, float(plane_z), dtype=float)

    src = np.asarray(source, dtype=float).reshape(-1)[:3]
    img = src.copy()
    img[2] = 2.0 * float(floor_z) - img[2]

    eps = 1e-9
    r = np.sqrt((X - src[0]) ** 2 + (Y - src[1]) ** 2 + (Z - src[2]) ** 2) + eps
    rp = np.sqrt((X - img[0]) ** 2 + (Y - img[1]) ** 2 + (Z - img[2]) ** 2) + eps
    k = 2.0 * np.pi * max(float(freq_hz), eps) / max(float(c), eps)

    additive = 1.0 / (r ** 2) + (beta ** 2) / (rp ** 2)
    interference = 2.0 * beta * np.cos(k * (rp - r)) / (r * rp)
    eta = interference / np.maximum(additive, eps)
    eta = np.clip(eta, -1.0, 1.0)
    return X, Y, eta


class DraggableSceneLabel(QLabel):
    """Petit label déplaçable à la souris au-dessus de la vue 3D."""

    def __init__(self, text: str, move_callback: Optional[Any] = None, parent: Optional[QWidget] = None) -> None:
        super().__init__(text, parent)
        self._drag_start: Optional[QPoint] = None
        self._move_callback = move_callback
        self.setCursor(Qt.OpenHandCursor)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet(
            "QLabel {"
            "background: rgba(255, 255, 255, 220);"
            "border: 1px solid rgba(15, 23, 42, 95);"
            "border-radius: 6px;"
            "padding: 3px 7px;"
            "font-weight: 700;"
            "}"
        )
        self.adjustSize()

    def mousePressEvent(self, event: Any) -> None:  # pragma: no cover - interaction Qt
        if event.button() == Qt.LeftButton:
            self._drag_start = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: Any) -> None:  # pragma: no cover - interaction Qt
        if self._drag_start is not None and event.buttons() & Qt.LeftButton:
            new_pos = event.globalPosition().toPoint() - self._drag_start
            parent = self.parentWidget()
            if parent is not None:
                max_x = max(parent.width() - self.width(), 0)
                max_y = max(parent.height() - self.height(), 0)
                new_pos.setX(max(0, min(new_pos.x(), max_x)))
                new_pos.setY(max(0, min(new_pos.y(), max_y)))
            self.move(new_pos)
            if self._move_callback is not None:
                self._move_callback(new_pos)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: Any) -> None:  # pragma: no cover - interaction Qt
        self._drag_start = None
        self.setCursor(Qt.OpenHandCursor)
        super().mouseReleaseEvent(event)


class SceneViewWidget(QWidget):
    """Realtime 3D scene of the acoustic room.

    PyVista/VTK is used when available. If the 3D backend is missing, the widget
    stays usable and shows a textual fallback instead of crashing the UI.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._last_cfg: Optional[Dict[str, Any]] = None
        self._last_estimated_source: Optional[np.ndarray] = None
        self._camera_position: Any = None
        self._plotter: Any = None
        self._fallback_label: Optional[QLabel] = None
        self._movable_labels: Dict[str, DraggableSceneLabel] = {}
        self._movable_label_positions: Dict[str, QPoint] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        if QtInteractor is None or pv is None:
            label = QLabel(
                "Vue 3D indisponible.\n\n"
                "Installe le backend 3D avec :\n"
                "pip install pyvista pyvistaqt vtk\n\n"
                "L'interface reste fonctionnelle, mais la chambre ne sera pas rendue en 3D."
            )
            label.setAlignment(Qt.AlignCenter)
            label.setObjectName("SubtleText")
            layout.addWidget(label, 1)
            self._fallback_label = label
            return

        self._plotter = QtInteractor(self)
        self._plotter.set_background("#ffffff")
        layout.addWidget(self._plotter.interactor if hasattr(self._plotter, "interactor") else self._plotter, 1)

    def resizeEvent(self, event: Any) -> None:  # pragma: no cover - interaction Qt
        super().resizeEvent(event)
        for key, label in self._movable_labels.items():
            if key in self._movable_label_positions:
                pos = self._movable_label_positions[key]
                label.move(max(0, min(pos.x(), max(self.width() - label.width(), 0))),
                           max(0, min(pos.y(), max(self.height() - label.height(), 0))))

    def _clear_movable_labels(self) -> None:
        active_keys = set(getattr(self, "_active_movable_label_keys", set()))
        for key, label in list(self._movable_labels.items()):
            if key not in active_keys:
                label.hide()
                label.deleteLater()
                self._movable_labels.pop(key, None)

    def _world_to_widget_pos(self, point: Any, dx: int = 10, dy: int = -26) -> QPoint:
        """Projette grossièrement un point 3D vers la vue Qt pour placer le label."""
        try:
            renderer = self._plotter.renderer
            render_widget = self._plotter.interactor if hasattr(self._plotter, "interactor") else self._plotter
            x, y, z = np.asarray(point, dtype=float).reshape(-1)[:3]
            renderer.SetWorldPoint(float(x), float(y), float(z), 1.0)
            renderer.WorldToDisplay()
            display_x, display_y, _ = renderer.GetDisplayPoint()
            qt_x = int(display_x) + dx
            qt_y = int(render_widget.height() - display_y) + dy
            return render_widget.mapTo(self, QPoint(qt_x, qt_y))
        except Exception:
            return QPoint(12 + 18 * len(self._movable_labels), 12 + 28 * len(self._movable_labels))

    def _add_movable_label(self, key: str, text: str, point: Any, color: str, dx: int = 10, dy: int = -26) -> None:
        if not hasattr(self, "_active_movable_label_keys"):
            self._active_movable_label_keys = set()
        self._active_movable_label_keys.add(key)
        label = self._movable_labels.get(key)
        if label is None:
            label = DraggableSceneLabel(text, parent=self)
            self._movable_labels[key] = label
        label.setText(text)
        label.adjustSize()
        label.setStyleSheet(
            "QLabel {"
            "background: rgba(255, 255, 255, 225);"
            f"border: 2px solid {color};"
            "border-radius: 6px;"
            "padding: 3px 7px;"
            f"color: {color};"
            "font-weight: 700;"
            "}"
        )

        def remember(pos: QPoint, label_key: str = key) -> None:
            self._movable_label_positions[label_key] = QPoint(pos)

        label._move_callback = remember
        if key in self._movable_label_positions:
            pos = self._movable_label_positions[key]
        else:
            pos = self._world_to_widget_pos(point, dx=dx, dy=dy)
            self._movable_label_positions[key] = QPoint(pos)
        label.move(max(0, min(pos.x(), max(self.width() - label.width(), 0))),
                   max(0, min(pos.y(), max(self.height() - label.height(), 0))))
        label.show()
        label.raise_()

    def _room_dimensions_from_last_config(self) -> Tuple[float, float, float]:
        cfg = self._last_cfg or DEFAULT_CONFIG
        geometry = cfg.get("geometry", {})
        Lx = max(_safe_float(geometry.get("Lx"), 5.0), 0.1)
        Ly = max(_safe_float(geometry.get("Ly"), 4.0), 0.1)
        Lz = max(_safe_float(geometry.get("Lz"), 2.7), 0.1)
        return Lx, Ly, Lz

    def set_camera_preset(self, preset: str) -> None:
        if self._plotter is None:
            return
        Lx, Ly, Lz = self._room_dimensions_from_last_config()
        center = (0.5 * Lx, 0.5 * Ly, 0.5 * Lz)
        distance = max(Lx, Ly, Lz, 1.0) * 2.35
        key = preset.strip().lower().replace("-", "_")
        if key in {"xy", "top", "dessus"}:
            position = (center[0], center[1], center[2] + distance)
            view_up = (0.0, 1.0, 0.0)
        elif key in {"xz", "front"}:
            position = (center[0], center[1] - distance, center[2])
            view_up = (0.0, 0.0, 1.0)
        elif key in {"zy", "yz", "side"}:
            position = (center[0] + distance, center[1], center[2])
            view_up = (0.0, 0.0, 1.0)
        elif key in {"iso", "isometric", "isotrope"}:
            position = (center[0] + 0.95 * distance, center[1] - 1.05 * distance, center[2] + 0.78 * distance)
            view_up = (0.0, 0.0, 1.0)
        else:
            position = (center[0] + 0.95 * distance, center[1] - 1.05 * distance, center[2] + 0.78 * distance)
            view_up = (0.0, 0.0, 1.0)
        try:
            self._plotter.camera_position = [position, center, view_up]
            if hasattr(self._plotter, "enable_parallel_projection"):
                self._plotter.enable_parallel_projection()
            else:
                self._plotter.camera.parallel_projection = True
            self._plotter.reset_camera()
            self._camera_position = self._plotter.camera_position
            self._plotter.render()
        except Exception:
            self._plotter.reset_camera()
            self._plotter.render()

    def reset_camera(self) -> None:
        self.set_camera_preset("iso")

    def update_scene(self, cfg: Dict[str, Any], estimated_source: Any = None, comparison_estimates: Any = None, reset_camera: bool = False) -> None:
        self._last_cfg = cfg
        self._last_estimated_source = None if estimated_source is None else np.asarray(estimated_source, dtype=float).reshape(-1)[:3]
        geometry = cfg.get("geometry", {})
        Lx = max(_safe_float(geometry.get("Lx"), 5.0), 0.1)
        Ly = max(_safe_float(geometry.get("Ly"), 4.0), 0.1)
        Lz = max(_safe_float(geometry.get("Lz"), 2.7), 0.1)
        source = parse_source(geometry.get("source"))
        mics = parse_mics_csv(geometry.get("mics_csv"))

        if self._fallback_label is not None:
            self._fallback_label.setText(
                "Vue 3D indisponible.\n\n"
                f"Chambre : {Lx:.2f} × {Ly:.2f} × {Lz:.2f} m\n"
                f"Source : {vector_to_text(source)}\n"
                f"Microphones : {len(mics)}\n\n"
                "Installe : pip install pyvista pyvistaqt vtk"
            )
            return

        if self._plotter is None or pv is None:
            return

        if not reset_camera:
            try:
                self._camera_position = self._plotter.camera_position
            except Exception:
                self._camera_position = None

        self._plotter.clear()
        self._active_movable_label_keys = set()
        scale = max(min(Lx, Ly, Lz), 0.1)
        mic_radius = max(0.035 * scale, 0.035)
        src_radius = max(0.055 * scale, 0.055)

        # Room wireframe.
        room = pv.Cube(center=(Lx / 2.0, Ly / 2.0, Lz / 2.0), x_length=Lx, y_length=Ly, z_length=Lz)
        self._plotter.add_mesh(room, style="wireframe", color="#475569", line_width=1.4, name="room")

        # Transparent floor and back wall provide COMSOL-like spatial cues.
        floor = pv.Plane(center=(Lx / 2.0, Ly / 2.0, 0.0), direction=(0.0, 0.0, 1.0), i_size=Lx, j_size=Ly)
        self._plotter.add_mesh(floor, color="#dbeafe", opacity=0.35, show_edges=True, edge_color="#b6c6d9", name="floor")
        back_wall = pv.Plane(center=(Lx / 2.0, Ly, Lz / 2.0), direction=(0.0, 1.0, 0.0), i_size=Lx, j_size=Lz)
        self._plotter.add_mesh(back_wall, color="#f1f5f9", opacity=0.18, show_edges=True, edge_color="#d4dde8", name="back_wall")

        # Heatmap d'interférence directe/réfléchie dans le plan horizontal du micro de référence.
        heatmap_cfg = cfg.get("interference_heatmap", {})
        if _safe_bool(heatmap_cfg.get("enabled", False)) and mics.size:
            try:
                ref_idx = max(1, int(_safe_float(heatmap_cfg.get("ref_mic_index"), 1.0))) - 1
                ref_idx = int(np.clip(ref_idx, 0, len(mics) - 1))
                ref_mic = np.asarray(mics[ref_idx], dtype=float).reshape(-1)[:3]
                plane_z = float(np.clip(ref_mic[2], 0.0, Lz))
                freq_hz = max(_safe_float(heatmap_cfg.get("freq_hz"), cfg.get("signal", {}).get("f_low", 1000.0)), 1e-6)
                beta = _reflection_beta_from_config(cfg, default=0.7)
                grid_n = _safe_int(heatmap_cfg.get("grid_n"), 90)
                opacity = float(np.clip(_safe_float(heatmap_cfg.get("opacity"), 0.65), 0.05, 0.95))
                c = max(_safe_float(cfg.get("propagation", {}).get("c"), 343.0), 1e-6)
                floor_z = _safe_float(cfg.get("propagation", {}).get("floor_z"), 0.0)

                Xh, Yh, eta = interference_ratio_grid(
                    source=source,
                    Lx=Lx,
                    Ly=Ly,
                    plane_z=plane_z,
                    floor_z=floor_z,
                    freq_hz=freq_hz,
                    beta=beta,
                    c=c,
                    grid_n=grid_n,
                )
                Zh = np.full_like(Xh, plane_z)
                heat_grid = pv.StructuredGrid(Xh, Yh, Zh)
                heat_grid["eta"] = eta.ravel(order="F")
                self._plotter.add_mesh(
                    heat_grid,
                    scalars="eta",
                    cmap="coolwarm",
                    clim=(-1.0, 1.0),
                    opacity=opacity,
                    show_edges=False,
                    name="interference_heatmap",
                    scalar_bar_args={
                        "title": "Interférence η",
                        "n_labels": 5,
                        "fmt": "%.2f",
                    },
                )
                self._plotter.add_mesh(
                    pv.Plane(center=(Lx / 2.0, Ly / 2.0, plane_z), direction=(0.0, 0.0, 1.0), i_size=Lx, j_size=Ly),
                    style="wireframe",
                    color="#334155",
                    opacity=0.25,
                    line_width=1.0,
                    name="interference_plane_frame",
                )
                self._add_movable_label(
                    "interference_heatmap",
                    f"Heatmap η — M{ref_idx + 1}, z={plane_z:.2f} m, f={freq_hz:.0f} Hz",
                    (0.08 * Lx, 0.08 * Ly, plane_z),
                    "#0f766e",
                    dx=12,
                    dy=-30,
                )
            except Exception as exc:
                # La scène reste utilisable même si la heatmap échoue.
                self._add_movable_label("interference_heatmap_error", f"Heatmap indisponible : {exc}", (0.05 * Lx, 0.05 * Ly, Lz), "#b45309")

        # Source.
        source = np.clip(source, [0.0, 0.0, 0.0], [Lx, Ly, Lz])
        source_mesh = pv.Sphere(radius=src_radius, center=tuple(source), theta_resolution=32, phi_resolution=16)
        self._plotter.add_mesh(source_mesh, color="#ef4444", smooth_shading=True, name="source")
        self._add_movable_label("source", "Source", source, "#ef4444", dx=12, dy=-30)

        # Microphones.
        if mics.size:
            clipped = np.column_stack([
                np.clip(mics[:, 0], 0.0, Lx),
                np.clip(mics[:, 1], 0.0, Ly),
                np.clip(mics[:, 2], 0.0, Lz),
            ])
            for i, point in enumerate(clipped):
                mic_mesh = pv.Sphere(radius=mic_radius, center=tuple(point), theta_resolution=24, phi_resolution=12)
                self._plotter.add_mesh(mic_mesh, color="#2563eb", smooth_shading=True, name=f"mic_{i}")
            labels = [f"M{i + 1}" for i in range(len(clipped))]
            self._plotter.add_point_labels(
                clipped, labels,
                font_size=10, point_color="#2563eb", text_color="#1e3a8a",
                shape_opacity=0.10, always_visible=False,
            )

        # Famille de microphones B affichée uniquement en mode comparaison.
        # La famille A correspond aux microphones standards de l’onglet "Pièce", déjà tracés en bleu ci-dessus.
        comparison = cfg.get("comparison", {})
        if _safe_bool(comparison.get("enabled", False)) and str(comparison.get("mode", "")) == "Deux familles de micros":
            mics_b = parse_mics_csv(comparison.get("mics_b_csv", ""))
            if mics_b.size:
                clipped_b = np.column_stack([
                    np.clip(mics_b[:, 0], 0.0, Lx),
                    np.clip(mics_b[:, 1], 0.0, Ly),
                    np.clip(mics_b[:, 2], 0.0, Lz),
                ])
                b_color = "#f59e0b"
                for i, point in enumerate(clipped_b):
                    mic_mesh = pv.Sphere(radius=mic_radius * 0.92, center=tuple(point), theta_resolution=24, phi_resolution=12)
                    self._plotter.add_mesh(mic_mesh, color=b_color, smooth_shading=True, name=f"comparison_mic_b_{i}")
                labels_b = [f"B{i + 1}" for i in range(len(clipped_b))]
                self._plotter.add_point_labels(
                    clipped_b, labels_b,
                    font_size=10, point_color=b_color, text_color="#7c2d12",
                    shape_opacity=0.12, always_visible=False,
                )

        # Estimated source after a simulation or a comparison.
        if comparison_estimates:
            for idx, item in enumerate(comparison_estimates):
                if isinstance(item, dict):
                    estimated = np.asarray(item.get("position"), dtype=float).reshape(-1)[:3]
                    label = str(item.get("label", f"Estimation {idx + 1}"))
                    color = str(item.get("color", "#22c55e"))
                else:
                    estimated = np.asarray(item, dtype=float).reshape(-1)[:3]
                    label = f"Estimation {idx + 1}"
                    color = "#22c55e"
                estimated = np.clip(estimated, [0.0, 0.0, 0.0], [Lx, Ly, Lz])
                est_mesh = pv.Sphere(radius=src_radius * 0.9, center=tuple(estimated), theta_resolution=32, phi_resolution=16)
                self._plotter.add_mesh(est_mesh, color=color, smooth_shading=True, name=f"comparison_estimated_source_{idx}")
                display_label = "Famille A" if label.lower().startswith("famille a") else label
                display_label = "Famille B" if display_label.lower().startswith("famille b") else display_label
                self._add_movable_label(f"comparison_estimated_{idx}", display_label, estimated, color, dx=12, dy=-30)
                line = pv.Line(tuple(source), tuple(estimated))
                self._plotter.add_mesh(line, color=color, line_width=3, name=f"comparison_error_line_{idx}")
        elif estimated_source is not None:
            estimated = np.asarray(estimated_source, dtype=float).reshape(-1)[:3]
            estimated = np.clip(estimated, [0.0, 0.0, 0.0], [Lx, Ly, Lz])
            est_mesh = pv.Sphere(radius=src_radius * 0.9, center=tuple(estimated), theta_resolution=32, phi_resolution=16)
            self._plotter.add_mesh(est_mesh, color="#22c55e", smooth_shading=True, name="estimated_source")
            self._add_movable_label("estimated", "Estimée", estimated, "#22c55e", dx=12, dy=-30)
            line = pv.Line(tuple(source), tuple(estimated))
            self._plotter.add_mesh(line, color="#f59e0b", line_width=3, name="error_line")

        try:
            self._plotter.show_bounds(
                bounds=(0, Lx, 0, Ly, 0, Lz),
                grid="back",
                location="outer",
                xtitle="x [m]",
                ytitle="y [m]",
                ztitle="z [m]",
                font_size=10,
                color="#334155",
            )
            self._plotter.add_axes(line_width=2, labels_off=False)
        except Exception:
            pass

        self._clear_movable_labels()

        if reset_camera or self._camera_position is None:
            self.set_camera_preset("iso")
        else:
            try:
                self._plotter.camera_position = self._camera_position
            except Exception:
                self.set_camera_preset("iso")
        self._plotter.render()


class DOEWorker(QObject):
    """Background worker used by the DOE tab.

    The worker owns the long-running DOE loop. It never touches widgets directly;
    every update is sent through Qt signals so the main UI remains responsive.
    """

    log_message = Signal(str)
    progress_changed = Signal(int, int, str)
    status_changed = Signal(str, str)
    finished_ok = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        *,
        workbook_path: str,
        results_path: str,
        sheet_name: str,
        start_index: int,
        limit: Optional[int],
        dry_run: bool,
        force: bool,
        stop_on_error: bool,
        max_failures: Optional[int],
        geometry_slots: Optional[Dict[str, str]],
        overwrite_results: bool = False,
        auto_workflow: bool = False,
        include_validation: bool = False,
    ) -> None:
        super().__init__()
        self.workbook_path = workbook_path
        self.results_path = results_path or None
        self.sheet_name = sheet_name
        self.start_index = start_index
        self.limit = limit
        self.dry_run = dry_run
        self.force = force
        self.stop_on_error = stop_on_error
        self.max_failures = max_failures
        self.geometry_slots = geometry_slots or {}
        self.overwrite_results = overwrite_results
        self.auto_workflow = auto_workflow
        self.include_validation = include_validation
        self._stop_requested = False

    def request_stop(self) -> None:
        self._stop_requested = True
        self.log_message.emit("Arrêt demandé : la campagne s'arrêtera après le run en cours.")

    def _should_stop(self) -> bool:
        return self._stop_requested

    def run(self) -> None:
        try:
            if DOERunner is None:
                raise RuntimeError(f"Module DOE indisponible : {DOE_IMPORT_ERROR}")
            if self.auto_workflow:
                if DOEAutomationRunner is None or DOEAutomationPlan is None:
                    raise RuntimeError(f"Automatisation DOE indisponible : {DOE_IMPORT_ERROR}")
                auto = DOEAutomationRunner(
                    self.workbook_path,
                    self.results_path or None,
                    geometry_slots=self.geometry_slots,
                )
                limits = {}
                if self.limit is not None:
                    for sheet in ("A_runs_screen", "A_runs_map", "B_PB12", "C1_PB12_common", "C2_alpha_refine", "D_validation"):
                        limits[sheet] = self.limit
                plan = DOEAutomationPlan.full(include_validation=self.include_validation, limits=limits)
                plan = DOEAutomationPlan(
                    run_stage_a_screen=True,
                    run_stage_a_map=True,
                    run_stage_b=True,
                    run_stage_c1=True,
                    run_stage_c2=True,
                    run_stage_d=self.include_validation,
                    resume=not self.force,
                    force=self.force,
                    stop_on_error=self.stop_on_error,
                    max_failures=self.max_failures,
                    limits=limits,
                )

                def on_auto_progress(progress: Any) -> None:
                    if progress.event == "workflow_started":
                        self.log_message.emit(progress.message)
                    elif progress.event == "step_started":
                        self.status_changed.emit(f"DOE auto : {progress.step}", "running")
                        self.log_message.emit(f"Étape {progress.step} — {progress.message}")
                    elif progress.event == "analysis":
                        self.log_message.emit(f"Analyse {progress.step} — {progress.message}")
                    elif progress.event == "run_progress" and progress.batch_progress is not None:
                        bp = progress.batch_progress
                        if bp.event in {"completed", "failed", "skipped", "stopped"}:
                            self.log_message.emit(f"[{progress.step}] [{bp.index}/{bp.total}] {bp.event} {bp.run_id or '-'} — {bp.message}")
                            done = int(bp.completed) + int(bp.skipped) + int(bp.failed)
                            self.progress_changed.emit(done, int(bp.total), bp.message or bp.event)
                    elif progress.event == "step_finished":
                        self.log_message.emit(f"Fin {progress.step} — {progress.message}")
                    elif progress.event == "workflow_finished":
                        self.log_message.emit(progress.message)

                summary = auto.run(
                    plan,
                    overwrite_results=self.overwrite_results,
                    progress=on_auto_progress,
                    should_stop=self._should_stop,
                )
                self.finished_ok.emit({"mode": "auto_workflow", "summary": summary})
                return

            runner = DOERunner(
                self.workbook_path,
                self.results_path or None,
                geometry_slots=self.geometry_slots,
            )
            results_path = runner.prepare_results(overwrite=self.overwrite_results)
            self.log_message.emit(f"Fichier DOE : {self.workbook_path}")
            self.log_message.emit(f"Fichier résultats : {results_path}")
            self.log_message.emit(f"Feuille : {self.sheet_name}")

            resume = not self.force
            if self.dry_run:
                plan = runner.plan_sheet(
                    self.sheet_name,
                    resume=resume,
                    force=self.force,
                    start_index=self.start_index,
                    limit=self.limit,
                )
                pending = [item for item in plan if item.status == "pending"]
                skipped = [item for item in plan if item.status == "skipped_success"]
                lines = [
                    "Plan DOE",
                    "=" * 52,
                    f"Feuille          : {self.sheet_name}",
                    f"Fichier résultats: {results_path}",
                    f"Runs considérés  : {len(plan)}",
                    f"À lancer         : {len(pending)}",
                    f"À ignorer        : {len(skipped)}",
                    "",
                    "Aperçu :",
                ]
                for item in plan[:30]:
                    reason = f" | {item.reason}" if item.reason else ""
                    lines.append(f"  row={item.row_index:04d} | {item.run_id:<14} | {item.status}{reason}")
                if len(plan) > 30:
                    lines.append(f"  ... {len(plan) - 30} lignes supplémentaires")
                text = "\n".join(lines)
                self.progress_changed.emit(0, len(plan), "Prévisualisation terminée")
                self.finished_ok.emit({"mode": "dry_run", "text": text, "results_path": str(results_path)})
                return

            def on_progress(progress: Any) -> None:
                if progress.event == "started":
                    self.status_changed.emit("Campagne DOE en cours", "running")
                    self.log_message.emit(progress.message)
                elif progress.event == "skipped":
                    self.log_message.emit(f"[{progress.index}/{progress.total}] Ignoré {progress.run_id} — {progress.message}")
                elif progress.event == "completed":
                    self.log_message.emit(f"[{progress.index}/{progress.total}] OK {progress.run_id} — {progress.message}")
                elif progress.event == "failed":
                    self.log_message.emit(f"[{progress.index}/{progress.total}] Échec {progress.run_id} — {progress.message}")
                elif progress.event == "stopped":
                    self.log_message.emit(f"[{progress.index}/{progress.total}] Stop {progress.run_id or '-'} — {progress.message}")
                elif progress.event == "finished":
                    self.log_message.emit(progress.message)

                done = int(progress.completed) + int(progress.skipped) + int(progress.failed)
                self.progress_changed.emit(done, int(progress.total), progress.message or progress.event)

            summary = runner.run_sheet(
                self.sheet_name,
                resume=resume,
                force=self.force,
                start_index=self.start_index,
                limit=self.limit,
                stop_on_error=self.stop_on_error,
                max_failures=self.max_failures,
                progress=on_progress,
                should_stop=self._should_stop,
                replace_existing=True,
            )
            self.finished_ok.emit({"mode": "run_sheet", "summary": summary})
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class ComsolChamberSimUI(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ChamberSim — COMSOL Style Workspace")
        self.resize(1540, 940)
        self.setMinimumSize(1180, 760)

        self.controls: Dict[Tuple[str, ...], Any] = {}
        self.estimated_source: Optional[np.ndarray] = None
        self.comparison_estimates: list[Dict[str, Any]] = []
        self._is_applying_config = False
        self._is_computing = False
        self._auto_compute_enabled = True
        self._auto_compute_delay_ms = 400
        self._doe_thread: Optional[QThread] = None
        self._doe_worker: Optional[DOEWorker] = None
        self._doe_running = False

        self._scene_refresh_timer = QTimer(self)
        self._scene_refresh_timer.setSingleShot(True)
        self._scene_refresh_timer.timeout.connect(self.refresh_scene)

        self._auto_compute_timer = QTimer(self)
        self._auto_compute_timer.setSingleShot(True)
        self._auto_compute_timer.timeout.connect(self._auto_compute)

        self._build_ui()
        self._build_menu()
        self._apply_config(DEFAULT_CONFIG)
        self._auto_compute_timer.stop()
        self.refresh_scene(reset_camera=True)
        self._append_log("Interface initialisée.")

    def closeEvent(self, event: Any) -> None:  # noqa: N802 - Qt naming convention
        if self._doe_running:
            reply = QMessageBox.question(
                self,
                "Campagne DOE en cours",
                "Une campagne DOE est en cours. Demander l'arrêt proprement ? Tu pourras quitter quand le run courant sera terminé.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                event.ignore()
                return
            self.stop_doe_sheet()
            event.ignore()
            return
        event.accept()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        outer.addWidget(self._build_ribbon())

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_model_builder())

        center_splitter = QSplitter(Qt.Vertical)
        center_splitter.setChildrenCollapsible(False)
        center_splitter.addWidget(self._build_graphics_panel())
        center_splitter.addWidget(self._build_bottom_panel())
        center_splitter.setSizes([640, 230])
        splitter.addWidget(center_splitter)

        splitter.addWidget(self._build_settings_panel())
        splitter.setSizes([260, 900, 410])
        outer.addWidget(splitter, 1)

        self.statusBar().showMessage("Prêt")

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("File")
        load_action = QAction("Load config", self)
        load_action.triggered.connect(self.load_config)
        save_action = QAction("Save config", self)
        save_action.triggered.connect(self.save_config)
        reset_action = QAction("Reset defaults", self)
        reset_action.triggered.connect(lambda: self._apply_config(DEFAULT_CONFIG))
        file_menu.addAction(load_action)
        file_menu.addAction(save_action)
        file_menu.addSeparator()
        file_menu.addAction(reset_action)

        study_menu = self.menuBar().addMenu("Study")
        run_action = QAction("Compute", self)
        run_action.triggered.connect(self.run_experiment)
        study_menu.addAction(run_action)

        doe_menu = self.menuBar().addMenu("DOE")
        doe_tab_action = QAction("Ouvrir l'onglet DOE", self)
        doe_tab_action.triggered.connect(lambda: self.settings_tabs.setCurrentIndex(6))
        doe_preview_action = QAction("Prévisualiser la feuille", self)
        doe_preview_action.triggered.connect(self.preview_doe_sheet)
        doe_run_action = QAction("Lancer la feuille", self)
        doe_run_action.triggered.connect(self.run_doe_sheet)
        doe_menu.addAction(doe_tab_action)
        doe_menu.addSeparator()
        doe_menu.addAction(doe_preview_action)
        doe_menu.addAction(doe_run_action)

        view_menu = self.menuBar().addMenu("View")
        refresh_action = QAction("Rebuild 3D view", self)
        refresh_action.triggered.connect(lambda: self.refresh_scene(reset_camera=False))
        reset_camera_action = QAction("Reset camera", self)
        reset_camera_action.triggered.connect(self.reset_camera)
        view_xy_action = QAction("Camera XY / Top", self)
        view_xy_action.triggered.connect(lambda: self.set_camera_preset("xy"))
        view_xz_action = QAction("Camera XZ", self)
        view_xz_action.triggered.connect(lambda: self.set_camera_preset("xz"))
        view_zy_action = QAction("Camera ZY", self)
        view_zy_action.triggered.connect(lambda: self.set_camera_preset("zy"))
        view_iso_action = QAction("Camera isotrope", self)
        view_iso_action.triggered.connect(lambda: self.set_camera_preset("iso"))
        view_menu.addAction(refresh_action)
        view_menu.addAction(reset_camera_action)
        view_menu.addSeparator()
        view_menu.addAction(view_xy_action)
        view_menu.addAction(view_xz_action)
        view_menu.addAction(view_zy_action)
        view_menu.addAction(view_iso_action)

    def _build_ribbon(self) -> QWidget:
        ribbon = QFrame()
        ribbon.setObjectName("Ribbon")
        ribbon.setFixedHeight(104)
        layout = QHBoxLayout(ribbon)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        layout.addWidget(self._ribbon_group("Fichier", [
            ("Charger", self.load_config, False),
            ("Enregistrer", self.save_config, False),
            ("Défauts", lambda: self._apply_config(DEFAULT_CONFIG), False),
        ]))
        layout.addWidget(self._ribbon_group("Étude", [
            ("Calculer", self.run_experiment, True),
            ("Auto", self._toggle_auto_compute, False, True),
            ("Effacer", self.clear_results, False),
        ]))
        layout.addWidget(self._ribbon_group("Vue 3D", [
            ("Rebuild", lambda: self.refresh_scene(reset_camera=False), False),
            ("Reset", self.reset_camera, False),
        ]))
        layout.addWidget(self._ribbon_group("Caméra", [
            ("XY", lambda: self.set_camera_preset("xy"), False),
            ("XZ", lambda: self.set_camera_preset("xz"), False),
            ("ZY", lambda: self.set_camera_preset("zy"), False),
            ("Iso", lambda: self.set_camera_preset("iso"), False),
        ]))
        layout.addWidget(self._ribbon_group("Microphones", [
            ("Carré 4", self.preset_square_4, False),
            ("Hexa 6", self.preset_hex_6, False),
            ("Aléatoire", self.preset_random, False),
        ]))
        layout.addWidget(self._ribbon_group("Navigation", [
            ("Pièce", lambda: self.settings_tabs.setCurrentIndex(0), False),
            ("Signal", lambda: self.settings_tabs.setCurrentIndex(1), False),
            ("Algo", lambda: self.settings_tabs.setCurrentIndex(3), False),
            ("Comp.", lambda: self.settings_tabs.setCurrentIndex(4), False),
            ("DOE", lambda: self.settings_tabs.setCurrentIndex(6), False),
        ]))
        layout.addStretch(1)
        return ribbon

    def _ribbon_group(self, title: str, buttons: Iterable[Tuple[Any, ...]]) -> QWidget:
        group = QFrame()
        group.setObjectName("RibbonGroup")
        group.setMinimumWidth(130)
        box = QVBoxLayout(group)
        box.setContentsMargins(6, 6, 6, 4)
        box.setSpacing(4)
        row = QHBoxLayout()
        row.setSpacing(4)
        for spec in buttons:
            text, callback, primary = spec[:3]
            checkable = bool(spec[3]) if len(spec) > 3 else False
            button = QToolButton()
            button.setText(text)
            button.setToolButtonStyle(Qt.ToolButtonTextOnly)
            button.setCheckable(checkable)
            if checkable:
                button.setChecked(self._auto_compute_enabled)
            if text == "Auto":
                self.auto_compute_button = button
                button.setToolTip(
                    "Recalcule automatiquement la simulation après une modification de paramètre.\n"
                    "Le lancement est temporisé pour éviter de recalculer à chaque frappe clavier."
                )
            if primary:
                button.setObjectName("PrimaryTool")
            button.clicked.connect(callback)
            row.addWidget(button)
        box.addLayout(row, 1)
        label = QLabel(title)
        label.setObjectName("RibbonGroupTitle")
        label.setAlignment(Qt.AlignCenter)
        box.addWidget(label)
        return group

    def _build_model_builder(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("LeftPanel")
        panel.setMinimumWidth(230)
        panel.setMaximumWidth(340)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        title = QLabel("Model Builder")
        title.setObjectName("PanelTitle")
        layout.addWidget(title)

        self.model_tree = QTreeWidget()
        self.model_tree.setHeaderHidden(True)
        self._populate_model_tree()
        self.model_tree.itemClicked.connect(self._on_tree_item_clicked)
        layout.addWidget(self.model_tree, 1)

        metrics = QGroupBox("Résumé")
        grid = QGridLayout(metrics)
        grid.setContentsMargins(8, 14, 8, 8)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)
        self.metric_mics = self._metric_label_pair(grid, 0, "MICROS", "—")
        self.metric_room = self._metric_label_pair(grid, 1, "CHAMBRE", "—")
        self.metric_algorithm = self._metric_label_pair(grid, 2, "ALGO", "—")
        self.metric_fs = self._metric_label_pair(grid, 3, "FS", "—")
        layout.addWidget(metrics)
        return panel

    def _populate_model_tree(self) -> None:
        self.model_tree.clear()
        root = QTreeWidgetItem(["heating_circuit.mph / ChamberSim"])
        global_defs = QTreeWidgetItem(["Global Definitions"])
        global_defs.addChildren([QTreeWidgetItem(["Parameters"]), QTreeWidgetItem(["Functions"]), QTreeWidgetItem(["Default Model Inputs"])])
        component = QTreeWidgetItem(["Component 1"])
        geometry = QTreeWidgetItem(["Geometry"])
        geometry.addChildren([QTreeWidgetItem(["Room"]), QTreeWidgetItem(["Source"]), QTreeWidgetItem(["Microphones"]), QTreeWidgetItem(["Safety Margin"])])
        physics = QTreeWidgetItem(["Physics"])
        physics.addChildren([QTreeWidgetItem(["Acoustic Propagation"]), QTreeWidgetItem(["Noise"]), QTreeWidgetItem(["Denoising"]), QTreeWidgetItem(["Localization Algorithms"]), QTreeWidgetItem(["Comparison"])])
        mesh = QTreeWidgetItem(["Mesh / Sampling"])
        study = QTreeWidgetItem(["Study 1"])
        study.addChildren([QTreeWidgetItem(["Compute"]), QTreeWidgetItem(["Solver Log"])])
        results = QTreeWidgetItem(["Results"])
        results.addChildren([QTreeWidgetItem(["3D Scene"]), QTreeWidgetItem(["Summary"]), QTreeWidgetItem(["Errors"]), QTreeWidgetItem(["Export"])] )
        component.addChildren([geometry, physics, mesh])
        root.addChildren([global_defs, component, study, results])
        self.model_tree.addTopLevelItem(root)
        self.model_tree.expandAll()

    def _metric_label_pair(self, grid: QGridLayout, row: int, label: str, value: str) -> QLabel:
        lab = QLabel(label)
        lab.setObjectName("MetricLabel")
        val = QLabel(value)
        val.setObjectName("MetricValue")
        val.setWordWrap(True)
        grid.addWidget(lab, row, 0)
        grid.addWidget(val, row, 1)
        return val

    def _build_graphics_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("GraphicsPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QFrame()
        header.setFixedHeight(38)
        header.setStyleSheet("background: #f8fafc; border-bottom: 1px solid #cbd5e1;")
        h = QHBoxLayout(header)
        h.setContentsMargins(10, 4, 10, 4)
        h.setSpacing(6)
        title = QLabel("Graphics — Chambre acoustique 3D")
        title.setObjectName("PanelTitle")
        h.addWidget(title)
        h.addStretch(1)
        self.scene_info_label = QLabel("—")
        self.scene_info_label.setObjectName("SubtleText")
        h.addWidget(self.scene_info_label)
        for cam_label, cam_preset, tip in (
            ("XY", "xy", "Vue de dessus : plan x-y, axe z sortant."),
            ("XZ", "xz", "Vue de face : plan x-z, regard selon y."),
            ("ZY", "zy", "Vue latérale : plan z-y, regard selon x."),
            ("Iso", "iso", "Vue isométrique 3D globale."),
        ):
            btn = QPushButton(cam_label)
            btn.setObjectName("SmallAction")
            btn.setToolTip(tip)
            btn.clicked.connect(lambda _checked=False, preset=cam_preset: self.set_camera_preset(preset))
            h.addWidget(btn)
        layout.addWidget(header)

        self.scene_view = SceneViewWidget()
        layout.addWidget(self.scene_view, 1)
        return panel

    def _build_settings_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("RightPanel")
        panel.setMinimumWidth(360)
        panel.setMaximumWidth(520)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        title = QLabel("Settings")
        title.setObjectName("PanelTitle")
        subtitle = QLabel("Configuration en temps réel : la vue 3D se met à jour dès que tu modifies la pièce, la source ou les micros.")
        subtitle.setObjectName("SubtleText")
        subtitle.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(subtitle)

        self.settings_tabs = QTabWidget()
        self.settings_tabs.addTab(self._scroll(self._tab_geometry()), "Pièce")
        self.settings_tabs.addTab(self._scroll(self._tab_signal()), "Signal")
        self.settings_tabs.addTab(self._scroll(self._tab_propagation()), "Propagation")
        self.settings_tabs.addTab(self._scroll(self._tab_algorithms()), "Algorithmes")
        self.settings_tabs.addTab(self._scroll(self._tab_comparison()), "Comparaison")
        self.settings_tabs.addTab(self._scroll(self._tab_outputs()), "Sorties")
        self.settings_tabs.addTab(self._scroll(self._tab_doe()), "DOE")
        layout.addWidget(self.settings_tabs, 1)
        return panel

    def _build_bottom_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("BottomPanel")
        panel.setMinimumHeight(180)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        top = QHBoxLayout()
        top.setSpacing(8)
        title = QLabel("Messages / Progress / Results")
        title.setObjectName("PanelTitle")
        top.addWidget(title)
        top.addStretch(1)
        self.status_pill = QLabel("En attente")
        self.status_pill.setObjectName("SubtleText")
        top.addWidget(self.status_pill)
        clear = QPushButton("Effacer")
        clear.setObjectName("SmallAction")
        clear.clicked.connect(self.clear_results)
        top.addWidget(clear)
        layout.addLayout(top)

        self.bottom_tabs = QTabWidget()
        summary = QWidget()
        summary_lay = QGridLayout(summary)
        summary_lay.setContentsMargins(6, 6, 6, 6)
        summary_lay.setHorizontalSpacing(8)
        summary_lay.setVerticalSpacing(8)
        self.result_algo_value = self._result_field(summary_lay, 0, 0, "Algorithme")
        self.result_error_value = self._result_field(summary_lay, 0, 1, "Erreur localisation")
        self.result_true_value = self._result_field(summary_lay, 1, 0, "Source vraie")
        self.result_estimated_value = self._result_field(summary_lay, 1, 1, "Source estimée")

        self.summary_console = QPlainTextEdit()
        self.summary_console.setReadOnly(True)
        self.summary_console.setMaximumBlockCount(120)
        summary_lay.addWidget(self.summary_console, 2, 0, 1, 2)

        self.log_console = QPlainTextEdit()
        self.log_console.setObjectName("Console")
        self.log_console.setReadOnly(True)
        self.log_console.setMaximumBlockCount(1000)

        self.error_console = QPlainTextEdit()
        self.error_console.setObjectName("ErrorConsole")
        self.error_console.setReadOnly(True)
        self.error_console.setMaximumBlockCount(400)

        self.comparison_console = QPlainTextEdit()
        self.comparison_console.setObjectName("Console")
        self.comparison_console.setReadOnly(True)
        self.comparison_console.setMaximumBlockCount(600)
        self.comparison_console.setPlainText("Aucune comparaison lancée.")

        self.bottom_tabs.addTab(summary, "Résumé")
        self.bottom_tabs.addTab(self.comparison_console, "Comparaison")
        self.bottom_tabs.addTab(self.log_console, "Messages")
        self.bottom_tabs.addTab(self.error_console, "Erreurs")
        layout.addWidget(self.bottom_tabs, 1)
        return panel

    def _result_field(self, grid: QGridLayout, row: int, col: int, title: str) -> QLabel:
        box = QFrame()
        box.setStyleSheet("background: #f8fafc; border: 1px solid #d4dde8; border-radius: 3px;")
        lay = QVBoxLayout(box)
        lay.setContentsMargins(8, 6, 8, 6)
        lab = QLabel(title.upper())
        lab.setObjectName("MetricLabel")
        val = QLabel("—")
        val.setObjectName("MetricValue")
        val.setWordWrap(True)
        lay.addWidget(lab)
        lay.addWidget(val)
        grid.addWidget(box, row, col)
        return val

    def _scroll(self, widget: QWidget) -> QScrollArea:
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setWidget(widget)
        return area

    # ------------------------------------------------------------------
    # Settings tabs
    # ------------------------------------------------------------------
    def _tab_geometry(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(10)
        lay.addWidget(self._group("Pièce", [
            self._line(("geometry", "Lx"), "Longueur Lx [m]"),
            self._line(("geometry", "Ly"), "Largeur Ly [m]"),
            self._line(("geometry", "Lz"), "Hauteur Lz [m]"),
            self._line(("geometry", "margin"), "Marge sécurité [m]"),
        ]))
        lay.addWidget(self._group("Source sonore", [
            self._line(("geometry", "source", "x"), "x [m]"),
            self._line(("geometry", "source", "y"), "y [m]"),
            self._line(("geometry", "source", "z"), "z [m]"),
        ]))

        mic_group = QGroupBox("Positions des microphones")
        mic_lay = QVBoxLayout(mic_group)
        help_label = QLabel("Une ligne par micro : x,y,z. Exemple : 1.20,0.80,1.50")
        help_label.setObjectName("FieldHelp")
        self.mics_text = QTextEdit()
        self.mics_text.setMinimumHeight(180)
        self.mics_text.setPlaceholderText("x,y,z\n1.000,1.000,1.200\n2.000,1.000,1.200")
        self.mics_text.textChanged.connect(self._on_config_changed)
        self.controls[("geometry", "mics_csv")] = self.mics_text
        mic_tip = _parameter_tooltip(("geometry", "mics_csv"), "Positions des microphones")
        self.mics_text.setProperty("config_path", "geometry.mics_csv")
        self.mics_text.setToolTip(mic_tip)
        help_label.setToolTip(mic_tip)
        mic_group.setToolTip(mic_tip)
        mic_lay.addWidget(help_label)
        mic_lay.addWidget(self.mics_text)
        lay.addWidget(mic_group)
        lay.addStretch(1)
        return page

    def _tab_signal(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(10)
        lay.addWidget(self._group("Signal source", [
            self._line(("signal", "fs"), "fs [Hz]"),
            self._line(("signal", "duration"), "Durée [s]"),
            self._line(("signal", "rms"), "RMS"),
            self._line(("signal", "f_low"), "f_low [Hz]"),
            self._line(("signal", "f_high"), "f_high [Hz]"),
            self._line(("signal", "seed"), "Graine"),
        ]))
        lay.addWidget(self._group("Modulation", [
            self._check(("signal", "use_mod"), "Activer la modulation"),
            self._line(("signal", "f_mod"), "f_mod [Hz]"),
            self._line(("signal", "mod_depth"), "Profondeur"),
        ]))
        lay.addWidget(self._group("Fade-in / Fade-out", [
            self._check(("signal", "use_fade"), "Activer le fade"),
            self._line(("signal", "fade_in"), "Fondu entrée [s]"),
            self._line(("signal", "fade_out"), "Fondu sortie [s]"),
        ]))
        lay.addStretch(1)
        return page

    def _tab_propagation(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(10)
        lay.addWidget(self._group("Propagation acoustique", [
            self._line(("propagation", "c"), "c [m/s]"),
            self._check(("propagation", "use_spreading"), "Atténuation géométrique 1/r"),
            self._line(("propagation", "gain_at_1m"), "Gain à 1 m"),
            self._check(("propagation", "use_floor_image"), "Image source du sol rigide"),
            self._line(("propagation", "floor_z"), "Hauteur sol z [m]"),
        ]))
        heatmap_help = QLabel(
            "Affiche η(x,y) dans le plan horizontal du micro choisi. "
            "η > 0 : direct + réfléchi constructif ; η < 0 : destructif."
        )
        heatmap_help.setObjectName("FieldHelp")
        heatmap_help.setWordWrap(True)
        lay.addWidget(self._group("Heatmap d'interférence sol", [
            self._check(("interference_heatmap", "enabled"), "Activer la heatmap d'interférence"),
            self._line(("interference_heatmap", "freq_hz"), "Fréquence [Hz]"),
            self._line(("interference_heatmap", "ref_mic_index"), "Micro référence (1..N)"),
            self._line(("interference_heatmap", "beta"), "β réflexion"),
            self._line(("interference_heatmap", "grid_n"), "Résolution grille"),
            self._line(("interference_heatmap", "opacity"), "Opacité"),
            ("", heatmap_help),
        ]))
        lay.addWidget(self._group("Bruit", [
            self._check(("propagation", "add_noise"), "Ajouter du bruit selon SNR"),
            self._line(("propagation", "snr_db"), "SNR [dB]"),
            self._check(("propagation", "noise_indep"), "Bruit indépendant par micro"),
            self._line(("propagation", "noise_seed"), "Graine bruit"),
        ]))
        lay.addWidget(self._group("Débruitage — Wiener STFT", [
            self._check(("propagation", "enable_denoise"), "Activer le débruitage"),
            self._line(("propagation", "denoise_nperseg"), "STFT nperseg"),
            self._line(("propagation", "denoise_noverlap"), "STFT noverlap"),
            self._line(("propagation", "denoise_noise_head"), "Tête bruit [s]"),
            self._line(("propagation", "denoise_noise_tail"), "Queue bruit [s]"),
            self._line(("propagation", "denoise_gain_floor"), "Plancher du gain"),
        ]))
        lay.addStretch(1)
        return page

    def _tab_algorithms(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(10)
        lay.addWidget(self._group("Sélection", [
            self._check(("algorithms", "enable_algorithms"), "Activer les algorithmes"),
            self._combo(("algorithms", "alg_choice"), "Algorithme", ALGORITHMS),
            self._check(("algorithms", "plot_estimated_source"), "Afficher la source estimée en 3D"),
        ]))
        lay.addWidget(self._group("TDOA — GCC-PHAT", [
            self._line(("algorithms", "tdoa_interp"), "Interpolation"),
            self._line(("algorithms", "tdoa_grid_dx"), "Pas grille dx [m]"),
            self._line(("algorithms", "tdoa_grid_dy"), "Pas grille dy [m]"),
            self._line(("algorithms", "tdoa_grid_dz"), "Pas grille dz [m]"),
            self._line(("algorithms", "tdoa_z_fixed"), "z fixé [m]"),
        ]))
        lay.addWidget(self._group("Energy methods — ER-LS / ER-NLS", [
            self._line(("algorithms", "er_ref_idx"), "ref_idx"),
            self._line(("algorithms", "er_window_s"), "Fenêtre [s]"),
            self._line(("algorithms", "er_hop_s"), "Pas temporel [s]"),
            self._line(("algorithms", "er_trim_frac"), "Fraction tronquée"),
            self._line(("algorithms", "er_kappa_eps"), "kappa_eps"),
            self._line(("algorithms", "ernls_max_iter"), "Itérations max"),
            self._line(("algorithms", "ernls_lam"), "Lambda"),
            self._line(("algorithms", "ernls_tol_step"), "Tolérance pas"),
            self._line(("algorithms", "ernls_tol_cost"), "Tolérance coût"),
        ]))
        lay.addWidget(self._group("MLE / EM ground", [
            self._combo(("algorithms", "mleem_model_type"), "Type de modèle", ["additive", "coherent"]),
            self._combo(("algorithms", "mleem_init_method"), "Initialisation", ["barycenter", "er_ls_ground"]),
            self._check(("algorithms", "mleem_estimate_alpha"), "Estimer alpha"),
            self._line(("algorithms", "mleem_alpha_init"), "alpha_init"),
            self._line(("algorithms", "mleem_alpha_min"), "alpha_min"),
            self._line(("algorithms", "mleem_alpha_max"), "alpha_max"),
            self._line(("algorithms", "mleem_alpha_grid_size"), "Grille alpha"),
            self._line(("algorithms", "mleem_max_iter"), "Itérations max"),
            self._line(("algorithms", "mleem_lam"), "Lambda"),
            self._line(("algorithms", "mleem_tol_step"), "Tolérance pas"),
            self._line(("algorithms", "mleem_tol_cost"), "Tolérance coût"),
            self._line(("algorithms", "mleem_fd_eps"), "fd_eps"),
            self._line(("algorithms", "mleem_barycenter_z_offset"), "Décalage z barycentre"),
        ]))
        lay.addStretch(1)
        return page


    def _tab_comparison(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(10)

        intro = QLabel(
            "Outil de comparaison : lance deux simulations avec la même source et affiche les deux estimations sur la même vue 3D. "
            "Pour comparer deux familles de micros, la famille A est celle définie dans l’onglet Pièce ; tu renseignes seulement la famille B ici."
        )
        intro.setObjectName("SubtleText")
        intro.setWordWrap(True)
        lay.addWidget(intro)

        lay.addWidget(self._group("Activation", [
            self._check(("comparison", "enabled"), "Activer l'outil de comparaison"),
            self._combo(("comparison", "mode"), "Mode", ["Deux algorithmes", "Deux familles de micros"]),
        ]))
        lay.addWidget(self._group("Mode : deux algorithmes", [
            self._combo(("comparison", "algorithm_a"), "Algorithme A", ALGORITHMS),
            self._combo(("comparison", "algorithm_b"), "Algorithme B", ALGORITHMS),
        ]))
        lay.addWidget(self._group("Mode : deux familles de micros", [
            self._combo(("comparison", "mic_algorithm"), "Algorithme commun", ALGORITHMS),
        ]))


        mic_b_group = QGroupBox("Famille de micros B")
        mic_b_lay = QVBoxLayout(mic_b_group)
        mic_b_help = QLabel("Une ligne par micro : x,y,z. La famille A reste celle de l’onglet Pièce ; la famille B sera affichée en orange sur la vue 3D.")
        mic_b_help.setObjectName("FieldHelp")
        self.comparison_mics_b_text = QTextEdit()
        self.comparison_mics_b_text.setMinimumHeight(120)
        self.comparison_mics_b_text.setPlaceholderText("x,y,z\n1.000,1.000,1.200\n2.500,3.300,1.200\n4.000,1.000,1.200")
        self.comparison_mics_b_text.textChanged.connect(self._on_config_changed)
        self._apply_parameter_tooltip(self.comparison_mics_b_text, ("comparison", "mics_b_csv"), "Famille de micros B")
        self.controls[("comparison", "mics_b_csv")] = self.comparison_mics_b_text
        mic_b_lay.addWidget(mic_b_help)
        mic_b_lay.addWidget(self.comparison_mics_b_text)
        lay.addWidget(mic_b_group)

        actions = QHBoxLayout()
        actions.setSpacing(8)
        self.comparison_run_button = QPushButton("Lancer la comparaison")
        self.comparison_run_button.setObjectName("PrimaryTool")
        self.comparison_run_button.clicked.connect(self.run_comparison)
        self.comparison_copy_b_button = QPushButton("Micros actuels → B")
        self.comparison_copy_b_button.setObjectName("SmallAction")
        self.comparison_copy_b_button.clicked.connect(lambda: self._copy_current_mics_to_comparison("B"))
        actions.addWidget(self.comparison_run_button)
        actions.addWidget(self.comparison_copy_b_button)
        actions.addStretch(1)
        lay.addLayout(actions)

        enabled_widget = self.controls.get(("comparison", "enabled"))
        mode_widget = self.controls.get(("comparison", "mode"))
        if isinstance(enabled_widget, QCheckBox):
            enabled_widget.stateChanged.connect(self._sync_comparison_controls)
        if isinstance(mode_widget, QComboBox):
            mode_widget.currentTextChanged.connect(self._sync_comparison_controls)
        QTimer.singleShot(0, self._sync_comparison_controls)

        lay.addStretch(1)
        return page


    def _tab_doe(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(10)

        intro = QLabel(
            "Pilotage DOE : charge un classeur Excel, choisis une feuille, puis lance les runs. "
            "Les résultats sont sauvegardés après chaque run pour permettre la reprise automatique."
        )
        intro.setObjectName("SubtleText")
        intro.setWordWrap(True)
        lay.addWidget(intro)

        if DOE_IMPORT_ERROR is not None:
            warning = QLabel(f"Module DOE indisponible : {DOE_IMPORT_ERROR}")
            warning.setObjectName("SubtleText")
            warning.setWordWrap(True)
            lay.addWidget(warning)

        files_group = QGroupBox("Fichiers")
        files_form = QFormLayout(files_group)
        files_form.setContentsMargins(10, 14, 10, 10)
        files_form.setHorizontalSpacing(8)
        files_form.setVerticalSpacing(8)

        workbook_row = QHBoxLayout()
        self.doe_workbook_line = QLineEdit()
        self.doe_workbook_line.setPlaceholderText("DOE_MAITRISE.xlsx")
        self.doe_workbook_line.setMinimumHeight(28)
        self.doe_workbook_line.textChanged.connect(self._on_doe_workbook_path_changed)
        browse_workbook = QPushButton("Parcourir")
        browse_workbook.setObjectName("SmallAction")
        browse_workbook.clicked.connect(self._browse_doe_workbook)
        workbook_row.addWidget(self.doe_workbook_line, 1)
        workbook_row.addWidget(browse_workbook)
        files_form.addRow(QLabel("Classeur DOE"), workbook_row)

        results_row = QHBoxLayout()
        self.doe_results_line = QLineEdit()
        self.doe_results_line.setPlaceholderText("Automatique : *_results.xlsx")
        self.doe_results_line.setMinimumHeight(28)
        browse_results = QPushButton("Choisir")
        browse_results.setObjectName("SmallAction")
        browse_results.clicked.connect(self._browse_doe_results)
        results_row.addWidget(self.doe_results_line, 1)
        results_row.addWidget(browse_results)
        files_form.addRow(QLabel("Résultats Excel"), results_row)
        lay.addWidget(files_group)

        run_group = QGroupBox("Exécution")
        run_form = QFormLayout(run_group)
        run_form.setContentsMargins(10, 14, 10, 10)
        run_form.setHorizontalSpacing(8)
        run_form.setVerticalSpacing(8)

        self.doe_sheet_combo = QComboBox()
        self.doe_sheet_combo.addItems(list(DOE_RUN_SHEETS) or ["A_runs_screen"])
        run_form.addRow(QLabel("Feuille"), self.doe_sheet_combo)

        self.doe_start_spin = QSpinBox()
        self.doe_start_spin.setRange(1, 1_000_000)
        self.doe_start_spin.setValue(1)
        run_form.addRow(QLabel("Premier run"), self.doe_start_spin)

        self.doe_limit_spin = QSpinBox()
        self.doe_limit_spin.setRange(0, 1_000_000)
        self.doe_limit_spin.setValue(5)
        self.doe_limit_spin.setSpecialValueText("Tous")
        run_form.addRow(QLabel("Limite"), self.doe_limit_spin)

        self.doe_max_failures_spin = QSpinBox()
        self.doe_max_failures_spin.setRange(0, 1_000_000)
        self.doe_max_failures_spin.setValue(0)
        self.doe_max_failures_spin.setSpecialValueText("Aucune")
        run_form.addRow(QLabel("Arrêt après échecs"), self.doe_max_failures_spin)

        self.doe_geometry_slots_line = QLineEdit()
        self.doe_geometry_slots_line.setPlaceholderText("Ex. BEST_A_1=A-G05; BEST_A_2=A-G08")
        self.doe_geometry_slots_line.setMinimumHeight(28)
        run_form.addRow(QLabel("Slots géométrie"), self.doe_geometry_slots_line)

        self.doe_force_check = QCheckBox("Recalculer les runs déjà réussis")
        self.doe_stop_error_check = QCheckBox("Arrêter dès le premier échec")
        self.doe_overwrite_results_check = QCheckBox("Recréer le fichier résultats avant lancement")
        self.doe_include_validation_check = QCheckBox("Inclure D_validation dans l'automatisation complète")
        run_form.addRow(self.doe_force_check)
        run_form.addRow(self.doe_stop_error_check)
        run_form.addRow(self.doe_overwrite_results_check)
        run_form.addRow(self.doe_include_validation_check)
        lay.addWidget(run_group)

        actions = QHBoxLayout()
        actions.setSpacing(8)
        self.doe_preview_button = QPushButton("Prévisualiser")
        self.doe_preview_button.setObjectName("SmallAction")
        self.doe_preview_button.clicked.connect(self.preview_doe_sheet)
        self.doe_run_button = QPushButton("Lancer")
        self.doe_run_button.setObjectName("PrimaryTool")
        self.doe_run_button.clicked.connect(self.run_doe_sheet)
        self.doe_auto_button = QPushButton("Automatisation complète")
        self.doe_auto_button.setObjectName("PrimaryTool")
        self.doe_auto_button.clicked.connect(self.run_doe_auto_workflow)
        self.doe_stop_button = QPushButton("Arrêter")
        self.doe_stop_button.setObjectName("SmallAction")
        self.doe_stop_button.setEnabled(False)
        self.doe_stop_button.clicked.connect(self.stop_doe_sheet)
        actions.addWidget(self.doe_preview_button)
        actions.addWidget(self.doe_run_button)
        actions.addWidget(self.doe_auto_button)
        actions.addWidget(self.doe_stop_button)
        actions.addStretch(1)
        lay.addLayout(actions)

        self.doe_progress = QProgressBar()
        self.doe_progress.setRange(0, 1)
        self.doe_progress.setValue(0)
        lay.addWidget(self.doe_progress)

        self.doe_status_label = QLabel("DOE en attente")
        self.doe_status_label.setObjectName("SubtleText")
        lay.addWidget(self.doe_status_label)

        self.doe_console = QPlainTextEdit()
        self.doe_console.setObjectName("Console")
        self.doe_console.setReadOnly(True)
        self.doe_console.setMaximumBlockCount(2000)
        self.doe_console.setMinimumHeight(210)
        lay.addWidget(self.doe_console, 1)
        return page

    def _tab_outputs(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(10)
        lay.addWidget(self._group("Graphiques et sorties", [
            self._check(("plots", "plot_scene"), "Afficher la scène 3D"),
            self._check(("plots", "plot_source_time"), "Afficher le signal source"),
            self._check(("plots", "plot_source_spectrum"), "Afficher le spectre source"),
            self._check(("plots", "plot_mics"), "Afficher les signaux micros"),
            self._check(("plots", "plot_mics_zoom"), "Afficher le zoom des micros"),
            self._check(("plots", "plot_source_spectrogram"), "Afficher le spectrogramme source"),
            self._check(("plots", "plot_mic_spectrogram"), "Afficher un spectrogramme micro"),
            self._line(("plots", "mic_spec_index"), "Index micro spectrogramme"),
            self._check(("plots", "plot_denoise_compare"), "Comparer avant / après débruitage"),
            self._line(("plots", "plot_denoise_mic"), "Index micro débruitage"),
            self._check(("plots", "print_delta_r"), "Afficher les écarts Δr"),
            self._line(("plots", "zoom_tmax"), "Zoom tmax [s]"),
        ]))
        lay.addStretch(1)
        return page

    def _group(self, title: str, rows: Iterable[Tuple[str, QWidget]]) -> QGroupBox:
        group = QGroupBox(title)
        form = QFormLayout(group)
        form.setContentsMargins(10, 14, 10, 10)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(8)
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        for label, widget in rows:
            if label:
                lab = QLabel(label)
                lab.setObjectName("FieldHelp")
                tip = widget.toolTip()
                if tip:
                    lab.setToolTip(tip)
                form.addRow(lab, widget)
            else:
                form.addRow(widget)
        return group

    def _apply_parameter_tooltip(self, widget: QWidget, path: Tuple[str, ...], display_name: str) -> None:
        widget.setProperty("config_path", ".".join(path))
        widget.setToolTip(_parameter_tooltip(path, display_name))

    def _line(self, path: Tuple[str, ...], label: str) -> Tuple[str, QWidget]:
        w = QLineEdit()
        w.setMinimumHeight(28)
        w.textChanged.connect(self._on_config_changed)
        self._apply_parameter_tooltip(w, path, label)
        self.controls[path] = w
        return label, w

    def _check(self, path: Tuple[str, ...], text: str) -> Tuple[str, QWidget]:
        w = QCheckBox(text)
        w.stateChanged.connect(self._on_config_changed)
        self._apply_parameter_tooltip(w, path, text)
        self.controls[path] = w
        return "", w

    def _combo(self, path: Tuple[str, ...], label: str, values: Iterable[str]) -> Tuple[str, QWidget]:
        w = QComboBox()
        w.addItems(list(values))
        w.currentTextChanged.connect(self._on_config_changed)
        self._apply_parameter_tooltip(w, path, label)
        self.controls[path] = w
        return label, w


    # ------------------------------------------------------------------
    # DOE tab actions
    # ------------------------------------------------------------------
    def _browse_doe_workbook(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choisir le classeur DOE",
            str(PROJECT_ROOT),
            "Excel files (*.xlsx);;All files (*)",
        )
        if not path:
            return
        self.doe_workbook_line.setText(path)

    def _browse_doe_results(self) -> None:
        default = self._default_doe_results_path()
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Choisir le fichier résultats DOE",
            str(default) if default is not None else str(PROJECT_ROOT / "DOE_MAITRISE_results.xlsx"),
            "Excel files (*.xlsx);;All files (*)",
        )
        if not path:
            return
        self.doe_results_line.setText(path)

    def _on_doe_workbook_path_changed(self) -> None:
        if not hasattr(self, "doe_results_line"):
            return
        workbook_text = self.doe_workbook_line.text().strip()
        results_text = self.doe_results_line.text().strip()
        if workbook_text and not results_text:
            default = self._default_doe_results_path()
            if default is not None:
                self.doe_results_line.setPlaceholderText(str(default))
        self._refresh_doe_sheet_list()

    def _refresh_doe_sheet_list(self) -> None:
        if not hasattr(self, "doe_sheet_combo") or DOEManager is None:
            return
        workbook_text = self.doe_workbook_line.text().strip()
        if not workbook_text or not Path(workbook_text).exists():
            return
        current = self.doe_sheet_combo.currentText()
        try:
            manager = DOEManager(workbook_text)
            sheets = manager.list_run_sheets()
            if not sheets:
                return
            self.doe_sheet_combo.blockSignals(True)
            self.doe_sheet_combo.clear()
            self.doe_sheet_combo.addItems(sheets)
            if current in sheets:
                self.doe_sheet_combo.setCurrentText(current)
            self.doe_sheet_combo.blockSignals(False)
            if hasattr(self, "doe_status_label"):
                self.doe_status_label.setText(f"{len(sheets)} feuilles DOE détectées.")
        except Exception as exc:
            if hasattr(self, "doe_status_label"):
                self.doe_status_label.setText(f"Impossible de lire le DOE : {exc}")

    def _default_doe_results_path(self) -> Optional[Path]:
        workbook_text = self.doe_workbook_line.text().strip() if hasattr(self, "doe_workbook_line") else ""
        if not workbook_text:
            return None
        workbook = Path(workbook_text).expanduser()
        return workbook.with_name(f"{workbook.stem}_results.xlsx")

    def _parse_doe_geometry_slots(self) -> Dict[str, str]:
        raw = self.doe_geometry_slots_line.text().strip() if hasattr(self, "doe_geometry_slots_line") else ""
        slots: Dict[str, str] = {}
        if not raw:
            return slots
        for chunk in raw.replace("\n", ";").replace(",", ";").split(";"):
            item = chunk.strip()
            if not item:
                continue
            if "=" not in item:
                raise ValueError(f"Mapping géométrique invalide : {item!r}. Format attendu : SLOT=A-G01")
            key, value = item.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key or not value:
                raise ValueError(f"Mapping géométrique incomplet : {item!r}")
            slots[key] = value
        return slots

    def _doe_limit_value(self) -> Optional[int]:
        value = int(self.doe_limit_spin.value())
        return None if value <= 0 else value

    def _doe_max_failures_value(self) -> Optional[int]:
        value = int(self.doe_max_failures_spin.value())
        return None if value <= 0 else value

    def _collect_doe_worker_args(self, dry_run: bool) -> Dict[str, Any]:
        workbook = self.doe_workbook_line.text().strip()
        if not workbook:
            raise ValueError("Choisis d'abord le fichier DOE_MAITRISE.xlsx.")
        if not Path(workbook).exists():
            raise FileNotFoundError(f"Fichier DOE introuvable : {workbook}")
        results = self.doe_results_line.text().strip()
        if not results:
            default = self._default_doe_results_path()
            results = str(default) if default is not None else ""
        if not results:
            raise ValueError("Impossible de déterminer le fichier résultats.")
        return {
            "workbook_path": workbook,
            "results_path": results,
            "sheet_name": self.doe_sheet_combo.currentText().strip(),
            "start_index": int(self.doe_start_spin.value()),
            "limit": self._doe_limit_value(),
            "dry_run": dry_run,
            "force": bool(self.doe_force_check.isChecked()),
            "stop_on_error": bool(self.doe_stop_error_check.isChecked()),
            "max_failures": self._doe_max_failures_value(),
            "geometry_slots": self._parse_doe_geometry_slots(),
            "overwrite_results": bool(self.doe_overwrite_results_check.isChecked()) and not dry_run,
            "auto_workflow": False,
            "include_validation": bool(getattr(self, "doe_include_validation_check", QCheckBox()).isChecked()),
        }

    def preview_doe_sheet(self) -> None:
        self._start_doe_worker(dry_run=True)

    def run_doe_sheet(self) -> None:
        self._start_doe_worker(dry_run=False)

    def run_doe_auto_workflow(self) -> None:
        self._start_doe_worker(dry_run=False, auto_workflow=True)

    def _start_doe_worker(self, *, dry_run: bool, auto_workflow: bool = False) -> None:
        if self._doe_running:
            QMessageBox.information(self, "DOE", "Une campagne DOE est déjà en cours.")
            return
        if DOE_IMPORT_ERROR is not None:
            QMessageBox.critical(self, "DOE indisponible", str(DOE_IMPORT_ERROR))
            return
        try:
            kwargs = self._collect_doe_worker_args(dry_run=dry_run)
            kwargs["auto_workflow"] = bool(auto_workflow)
        except Exception as exc:
            QMessageBox.warning(self, "Paramètres DOE invalides", str(exc))
            return

        self._auto_compute_timer.stop()
        if self._auto_compute_enabled:
            self._append_log("Auto compute suspendu pendant l'exécution DOE.")

        self._set_doe_controls_running(True)
        self._doe_running = True
        self.doe_console.clear()
        self.doe_progress.setRange(0, 1)
        self.doe_progress.setValue(0)
        self.doe_status_label.setText("Prévisualisation en cours" if dry_run else "Campagne DOE en cours")
        self._set_status("DOE en cours" if not dry_run else "Prévisualisation DOE", "running")
        self.bottom_tabs.setCurrentIndex(2)

        self._doe_thread = QThread(self)
        self._doe_worker = DOEWorker(**kwargs)
        self._doe_worker.moveToThread(self._doe_thread)
        self._doe_thread.started.connect(self._doe_worker.run)
        self._doe_worker.log_message.connect(self._append_doe_log)
        self._doe_worker.progress_changed.connect(self._on_doe_progress)
        self._doe_worker.status_changed.connect(self._set_status)
        self._doe_worker.finished_ok.connect(self._on_doe_finished)
        self._doe_worker.failed.connect(self._on_doe_failed)
        self._doe_worker.finished_ok.connect(lambda _payload: self._doe_thread.quit() if self._doe_thread is not None else None)
        self._doe_worker.failed.connect(lambda _message: self._doe_thread.quit() if self._doe_thread is not None else None)
        self._doe_thread.finished.connect(self._cleanup_doe_worker)
        self._doe_thread.start()

    def stop_doe_sheet(self) -> None:
        if self._doe_worker is not None:
            self._doe_worker.request_stop()
            self.doe_status_label.setText("Arrêt demandé : attente de la fin du run courant.")
            self.doe_stop_button.setEnabled(False)

    def _set_doe_controls_running(self, running: bool) -> None:
        for attr in (
            "doe_preview_button",
            "doe_run_button",
            "doe_auto_button",
            "doe_workbook_line",
            "doe_results_line",
            "doe_sheet_combo",
            "doe_start_spin",
            "doe_limit_spin",
            "doe_max_failures_spin",
            "doe_geometry_slots_line",
            "doe_force_check",
            "doe_stop_error_check",
            "doe_overwrite_results_check",
            "doe_include_validation_check",
        ):
            if hasattr(self, attr):
                getattr(self, attr).setEnabled(not running)
        if hasattr(self, "doe_stop_button"):
            self.doe_stop_button.setEnabled(running)

    def _append_doe_log(self, text: str) -> None:
        message = str(text)
        if hasattr(self, "doe_console"):
            self.doe_console.appendPlainText(message)
            self.doe_console.verticalScrollBar().setValue(self.doe_console.verticalScrollBar().maximum())
        self._append_log(f"[DOE] {message}")

    def _on_doe_progress(self, done: int, total: int, message: str) -> None:
        total = max(int(total), 1)
        done = max(0, min(int(done), total))
        self.doe_progress.setRange(0, total)
        self.doe_progress.setValue(done)
        self.doe_status_label.setText(f"{done}/{total} — {message}")
        self.statusBar().showMessage(f"DOE : {done}/{total}")

    def _on_doe_finished(self, payload: Any) -> None:
        try:
            mode = payload.get("mode") if isinstance(payload, dict) else None
            if mode == "dry_run":
                text = str(payload.get("text", "Prévisualisation terminée."))
                self.doe_console.setPlainText(text)
                self.doe_status_label.setText("Prévisualisation terminée")
                self._set_status("DOE prêt", "success")
                self.statusBar().showMessage("Prévisualisation DOE terminée")
                return

            summary = payload.get("summary") if isinstance(payload, dict) else None
            if summary is not None:
                lines = summary.summary_lines()
                self.doe_console.appendPlainText("\n" + "=" * 64 + "\nRésumé campagne DOE\n" + "=" * 64)
                self.doe_console.appendPlainText("\n".join(lines))
                if getattr(summary, "run_ids_failed", None):
                    self.doe_console.appendPlainText("\nRuns échoués :")
                    for run_id in summary.run_ids_failed[:30]:
                        self.doe_console.appendPlainText(f"- {run_id}")
                state = "success" if getattr(summary, "failed", 0) == 0 else "error"
                text = "DOE terminé" if state == "success" else "DOE terminé avec échecs"
                self._set_status(text, state)
                self.doe_status_label.setText(text)
                self.statusBar().showMessage(text)
        finally:
            pass

    def _on_doe_failed(self, message: str) -> None:
        self._append_doe_log(f"ERREUR DOE : {message}")
        self._set_status("Échec DOE", "error")
        self.doe_status_label.setText("Échec DOE")
        QMessageBox.critical(self, "Erreur DOE", message)

    def _cleanup_doe_worker(self) -> None:
        self._doe_running = False
        self._set_doe_controls_running(False)
        if self._doe_worker is not None:
            self._doe_worker.deleteLater()
            self._doe_worker = None
        if self._doe_thread is not None:
            self._doe_thread.deleteLater()
            self._doe_thread = None

    def _toggle_auto_compute(self, checked: bool = False) -> None:
        self._auto_compute_enabled = bool(checked)
        if hasattr(self, "auto_compute_button"):
            self.auto_compute_button.setChecked(self._auto_compute_enabled)
        if self._auto_compute_enabled:
            self.statusBar().showMessage("Auto compute activé")
            self._auto_compute_timer.start(250)
        else:
            self._auto_compute_timer.stop()
            self.statusBar().showMessage("Auto compute désactivé")

    def _auto_compute(self) -> None:
        if self._is_applying_config or not self._auto_compute_enabled or self._is_computing:
            return
        self.run_experiment(auto=True)

    # ------------------------------------------------------------------
    # Config mapping and realtime updates
    # ------------------------------------------------------------------
    def _on_tree_item_clicked(self, item: QTreeWidgetItem, _column: int) -> None:
        text = item.text(0).lower()
        if "room" in text or "source" in text or "micro" in text or "geometry" in text:
            self.settings_tabs.setCurrentIndex(0)
        elif "propagation" in text or "noise" in text or "denoising" in text:
            self.settings_tabs.setCurrentIndex(2)
        elif "algorithm" in text:
            self.settings_tabs.setCurrentIndex(3)
        elif "comparison" in text or "comparaison" in text:
            self.settings_tabs.setCurrentIndex(4)
        elif "sampling" in text or "mesh" in text:
            self.settings_tabs.setCurrentIndex(1)
        elif "result" in text or "3d" in text:
            self.bottom_tabs.setCurrentIndex(0)

    def _on_config_changed(self) -> None:
        if self._is_applying_config:
            return
        sender = self.sender() if hasattr(self, "sender") else None
        sender_path = str(sender.property("config_path")) if sender is not None and sender.property("config_path") else ""
        comparison_only = sender_path.startswith("comparison.")
        heatmap_only = sender_path.startswith("interference_heatmap.")
        self.estimated_source = None
        self.comparison_estimates = []
        self._update_metrics()
        self._scene_refresh_timer.start(120)
        if self._auto_compute_enabled and not comparison_only and not heatmap_only:
            self._auto_compute_timer.start(self._auto_compute_delay_ms)
            self.statusBar().showMessage(
                f"Paramètre modifié — auto compute dans {self._auto_compute_delay_ms / 1000:.1f} s"
            )
        elif comparison_only:
            self.statusBar().showMessage("Paramètre de comparaison modifié")
        elif heatmap_only:
            self.statusBar().showMessage("Paramètre de heatmap modifié")

    def _get_config(self) -> Dict[str, Any]:
        cfg = deep_copy_config(DEFAULT_CONFIG)
        for path, widget in self.controls.items():
            target = cfg
            for part in path[:-1]:
                target = target.setdefault(part, {})
            key = path[-1]
            if isinstance(widget, QLineEdit):
                target[key] = widget.text()
            elif isinstance(widget, QCheckBox):
                target[key] = widget.isChecked()
            elif isinstance(widget, QComboBox):
                target[key] = widget.currentText()
            elif isinstance(widget, QTextEdit):
                target[key] = widget.toPlainText().strip()
        return cfg

    def _apply_config(self, cfg_in: Dict[str, Any]) -> None:
        self._is_applying_config = True
        try:
            cfg = deep_copy_config(cfg_in)
            for path, widget in self.controls.items():
                value: Any = cfg
                for part in path:
                    value = value.get(part, {}) if isinstance(value, dict) else {}
                if isinstance(widget, QLineEdit):
                    widget.setText(str(value))
                elif isinstance(widget, QCheckBox):
                    widget.setChecked(bool(value))
                elif isinstance(widget, QComboBox):
                    idx = widget.findText(str(value))
                    widget.setCurrentIndex(idx if idx >= 0 else 0)
                elif isinstance(widget, QTextEdit):
                    widget.setPlainText(str(value))
        finally:
            self._is_applying_config = False
        self.estimated_source = None
        self.comparison_estimates = []
        self._update_metrics()
        self.refresh_scene(reset_camera=True)
        if self._auto_compute_enabled:
            self._auto_compute_timer.start(self._auto_compute_delay_ms)

    def refresh_scene(self, reset_camera: bool = False) -> None:
        if not hasattr(self, "scene_view"):
            return
        cfg = self._get_config() if self.controls else DEFAULT_CONFIG
        self.scene_view.update_scene(cfg, estimated_source=self.estimated_source, comparison_estimates=self.comparison_estimates, reset_camera=reset_camera)
        geometry = cfg.get("geometry", {})
        Lx = _safe_float(geometry.get("Lx"), 0.0)
        Ly = _safe_float(geometry.get("Ly"), 0.0)
        Lz = _safe_float(geometry.get("Lz"), 0.0)
        mics = parse_mics_csv(geometry.get("mics_csv"))
        self.scene_info_label.setText(f"{Lx:.2f} × {Ly:.2f} × {Lz:.2f} m | {len(mics)} micros")
        self.statusBar().showMessage("Vue 3D mise à jour")

    def set_camera_preset(self, preset: str) -> None:
        if hasattr(self, "scene_view"):
            labels = {"xy": "vue XY", "xz": "vue XZ", "zy": "vue ZY", "iso": "vue isotrope"}
            self.scene_view.set_camera_preset(preset)
            self.statusBar().showMessage(f"Caméra : {labels.get(preset, preset)}")

    def reset_camera(self) -> None:
        self.set_camera_preset("iso")

    def _update_metrics(self) -> None:
        if not hasattr(self, "metric_mics"):
            return
        try:
            cfg = self._get_config()
            geometry = cfg.get("geometry", {})
            signal = cfg.get("signal", {})
            algorithms = cfg.get("algorithms", {})
            Lx = _safe_float(geometry.get("Lx"), 0.0)
            Ly = _safe_float(geometry.get("Ly"), 0.0)
            Lz = _safe_float(geometry.get("Lz"), 0.0)
            mics = parse_mics_csv(geometry.get("mics_csv"))
            self.metric_mics.setText(str(len(mics)))
            self.metric_room.setText(f"{Lx:g} × {Ly:g} × {Lz:g} m")
            self.metric_algorithm.setText(str(algorithms.get("alg_choice", "—")))
            self.metric_fs.setText(f"{signal.get('fs', '—')} Hz")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def save_config(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save config", str(PROJECT_ROOT / "config.json"), "JSON files (*.json);;All files (*)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._get_config(), f, indent=2, ensure_ascii=False)
            self._append_log(f"Config saved: {path}")
            self.statusBar().showMessage(f"Config enregistrée : {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Save error", str(exc))

    def load_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load config", str(PROJECT_ROOT), "JSON files (*.json);;All files (*)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            self._apply_config(cfg)
            self._append_log(f"Config loaded: {path}")
            self.statusBar().showMessage(f"Config chargée : {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Load error", str(exc))

    def run_experiment(self, auto: bool = False) -> None:
        if self._is_computing:
            return
        self._auto_compute_timer.stop()
        self._is_computing = True
        self.clear_results(keep_log=True)
        cfg = self._get_config()
        self.estimated_source = None
        self.comparison_estimates = []
        self.refresh_scene(reset_camera=False)
        self._show_run_started(cfg)
        self._append_log("Starting experiment..." + (" [auto]" if auto else ""))
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        try:
            # show_plots=False keeps the workflow inside this COMSOL-like UI.
            result = run_experiment_from_config(cfg, log=self._append_log, show_plots=False)
            result_dict = result.as_dict() if hasattr(result, "as_dict") else {}
            self._append_log("Finished successfully.")
            if result_dict:
                self._append_log(json.dumps(result_dict, indent=2, ensure_ascii=False))
            self._show_run_success(result, cfg, result_dict)
        except Exception as exc:
            if not auto:
                QMessageBox.critical(self, "Experiment error", str(exc))
            self._show_run_error(exc)
            self._append_log(f"ERROR: {exc}")
        finally:
            QApplication.restoreOverrideCursor()
            self._is_computing = False
            self._update_metrics()

    def _show_run_started(self, cfg: Dict[str, Any]) -> None:
        self._set_status("Simulation en cours", "running")
        algorithms = cfg.get("algorithms", {})
        geometry = cfg.get("geometry", {})
        mics = parse_mics_csv(geometry.get("mics_csv"))
        self.result_algo_value.setText(str(algorithms.get("alg_choice", "—")))
        self.result_error_value.setText("Calcul en cours…")
        self.result_true_value.setText(vector_to_text(parse_source(geometry.get("source"))))
        self.result_estimated_value.setText("En attente")
        self.summary_console.setPlainText(
            "Simulation lancée.\n\n"
            f"Algorithme : {algorithms.get('alg_choice', '—')}\n"
            f"Microphones : {len(mics)}\n"
            f"Fréquence d'échantillonnage : {cfg.get('signal', {}).get('fs', '—')} Hz\n\n"
            "La source estimée sera affichée en vert dans la vue 3D si l'algorithme retourne une position."
        )
        self.bottom_tabs.setCurrentIndex(0)
        self.statusBar().showMessage("Simulation en cours")

    def _show_run_success(self, result: Any, cfg: Dict[str, Any], result_dict: Dict[str, Any]) -> None:
        self._set_status("Terminé", "success")
        algorithm = str(cfg.get("algorithms", {}).get("alg_choice", "—"))
        true_source = getattr(result, "true_source", None)
        estimated_source = getattr(result, "estimated_source", None)
        self.result_algo_value.setText(algorithm)
        self.result_true_value.setText(vector_to_text(true_source))
        self.result_estimated_value.setText(vector_to_text(estimated_source) if estimated_source is not None else "Non estimée")

        error_text = "Non disponible"
        interpretation = "Aucune source estimée n'a été retournée par l'algorithme sélectionné."
        if estimated_source is not None and true_source is not None:
            true_arr = np.asarray(true_source, dtype=float).reshape(-1)[:3]
            est_arr = np.asarray(estimated_source, dtype=float).reshape(-1)[:3]
            error = float(np.linalg.norm(est_arr - true_arr))
            error_text = f"{error:.4f} m"
            interpretation = "Plus cette valeur est proche de 0 m, plus la localisation est précise."
            self.estimated_source = est_arr
            self._append_log(f"Localization error norm: {error:.4f} m")
        else:
            self.estimated_source = None
        self.result_error_value.setText(error_text)
        self.refresh_scene(reset_camera=False)

        highlights = self._extract_result_highlights(result_dict)
        highlight_text = "\n".join(f"- {key}: {value}" for key, value in highlights) if highlights else "- Aucun indicateur numérique supplémentaire détecté."
        self.summary_console.setPlainText(
            "Simulation terminée avec succès.\n\n"
            f"Algorithme utilisé : {algorithm}\n"
            f"Source vraie : {vector_to_text(true_source)}\n"
            f"Source estimée : {vector_to_text(estimated_source) if estimated_source is not None else 'Non estimée'}\n"
            f"Erreur de localisation : {error_text}\n\n"
            f"Interprétation : {interpretation}\n\n"
            "Informations utiles extraites du résultat :\n"
            f"{highlight_text}\n"
        )
        self.bottom_tabs.setCurrentIndex(0)
        self.statusBar().showMessage("Simulation terminée")

    def _show_run_error(self, exc: Exception) -> None:
        self._set_status("Échec", "error")
        message = str(exc) or exc.__class__.__name__
        self.result_error_value.setText("Simulation interrompue")
        self.summary_console.setPlainText(
            "La simulation n'a pas pu se terminer.\n\n"
            f"Erreur détectée : {message}\n\n"
            "À vérifier en priorité : valeurs numériques, format des microphones x,y,z, dimensions de la pièce, choix d'algorithme."
        )
        self.error_console.appendPlainText(message)
        self.bottom_tabs.setCurrentIndex(3)
        self.statusBar().showMessage("Erreur de simulation")


    # ------------------------------------------------------------------
    # Comparison actions
    # ------------------------------------------------------------------
    def _sync_comparison_controls(self, *_args: Any) -> None:
        enabled_widget = self.controls.get(("comparison", "enabled"))
        mode_widget = self.controls.get(("comparison", "mode"))
        enabled = bool(enabled_widget.isChecked()) if isinstance(enabled_widget, QCheckBox) else False
        mode = mode_widget.currentText() if isinstance(mode_widget, QComboBox) else "Deux algorithmes"
        algorithm_mode = mode == "Deux algorithmes"
        mic_mode = mode == "Deux familles de micros"
        for path, active in (
            (("comparison", "algorithm_a"), algorithm_mode),
            (("comparison", "algorithm_b"), algorithm_mode),
            (("comparison", "mic_algorithm"), mic_mode),
            (("comparison", "mics_b_csv"), mic_mode),
        ):
            widget = self.controls.get(path)
            if widget is not None:
                widget.setEnabled(enabled and active)
        for attr in ("comparison_run_button", "comparison_copy_b_button"):
            if hasattr(self, attr):
                getattr(self, attr).setEnabled(enabled)

    def _copy_current_mics_to_comparison(self, target: str) -> None:
        try:
            cfg = self._get_config()
            text = str(cfg.get("geometry", {}).get("mics_csv", "")).strip()
            if target.upper() == "B" and hasattr(self, "comparison_mics_b_text"):
                self.comparison_mics_b_text.setPlainText(text)
        except Exception as exc:
            QMessageBox.warning(self, "Comparaison", str(exc))

    def _comparison_entry_from_result(self, label: str, result: Any, cfg: Dict[str, Any]) -> Dict[str, Any]:
        true_source = getattr(result, "true_source", None)
        estimated_source = getattr(result, "estimated_source", None)
        error = None
        if estimated_source is not None and true_source is not None:
            true_arr = np.asarray(true_source, dtype=float).reshape(-1)[:3]
            est_arr = np.asarray(estimated_source, dtype=float).reshape(-1)[:3]
            error = float(np.linalg.norm(est_arr - true_arr))
        return {
            "label": label,
            "algorithm": str(cfg.get("algorithms", {}).get("alg_choice", "—")),
            "n_mics": len(parse_mics_csv(cfg.get("geometry", {}).get("mics_csv", ""))),
            "true_source": true_source,
            "estimated_source": estimated_source,
            "error": error,
            "result": result,
        }

    def _format_comparison_entry(self, entry: Dict[str, Any]) -> str:
        error = entry.get("error")
        error_text = "Non disponible" if error is None else f"{float(error):.4f} m"
        return (
            f"{entry.get('label', '—')}\n"
            f"  Algorithme : {entry.get('algorithm', '—')}\n"
            f"  Micros     : {entry.get('n_mics', '—')}\n"
            f"  Estimation : {vector_to_text(entry.get('estimated_source'))}\n"
            f"  Erreur     : {error_text}"
        )

    def run_comparison(self) -> None:
        if self._is_computing:
            return
        cfg_base = self._get_config()
        comparison = cfg_base.get("comparison", {})
        if not _safe_bool(comparison.get("enabled", False)):
            QMessageBox.information(self, "Comparaison", "Active d'abord l'outil de comparaison.")
            return

        self._auto_compute_timer.stop()
        self._is_computing = True
        self.estimated_source = None
        self.comparison_estimates = []
        self._set_status("Comparaison en cours", "running")
        if hasattr(self, "comparison_console"):
            self.comparison_console.setPlainText("Comparaison lancée...\n")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()

        try:
            mode = str(comparison.get("mode", "Deux algorithmes"))
            configs: list[tuple[str, Dict[str, Any]]] = []
            if mode == "Deux familles de micros":
                algorithm = str(comparison.get("mic_algorithm", cfg_base.get("algorithms", {}).get("alg_choice", "ER_LS")))
                if algorithm == "None":
                    raise ValueError("Choisis un algorithme commun différent de 'None'.")
                mics_a = str(cfg_base.get("geometry", {}).get("mics_csv", "")).strip()
                mics_b = str(comparison.get("mics_b_csv", "")).strip()
                if len(parse_mics_csv(mics_a)) == 0:
                    raise ValueError("La famille A correspond aux microphones de l’onglet Pièce et doit contenir au moins un point x,y,z.")
                if len(parse_mics_csv(mics_b)) == 0:
                    raise ValueError("La famille de microphones B doit contenir au moins un point x,y,z.")
                cfg_a = deep_copy_config(cfg_base)
                cfg_b = deep_copy_config(cfg_base)
                cfg_a["geometry"]["mics_csv"] = mics_a
                cfg_b["geometry"]["mics_csv"] = mics_b
                cfg_a["geometry"].pop("algorithm_mics_csv", None)
                cfg_b["geometry"].pop("algorithm_mics_csv", None)
                cfg_a["geometry"].pop("est_mics_csv", None)
                cfg_b["geometry"].pop("est_mics_csv", None)
                cfg_a["algorithms"]["alg_choice"] = algorithm
                cfg_b["algorithms"]["alg_choice"] = algorithm
                configs = [("Famille A — onglet Pièce", cfg_a), ("Famille B", cfg_b)]
            else:
                algorithm_a = str(comparison.get("algorithm_a", "ER_LS"))
                algorithm_b = str(comparison.get("algorithm_b", "ER_NLS"))
                if algorithm_a == "None" or algorithm_b == "None":
                    raise ValueError("Choisis deux algorithmes de localisation différents de 'None'.")
                cfg_a = deep_copy_config(cfg_base)
                cfg_b = deep_copy_config(cfg_base)
                cfg_a["algorithms"]["alg_choice"] = algorithm_a
                cfg_b["algorithms"]["alg_choice"] = algorithm_b
                configs = [(f"Algorithme A — {algorithm_a}", cfg_a), (f"Algorithme B — {algorithm_b}", cfg_b)]

            entries: list[Dict[str, Any]] = []
            for label, cfg in configs:
                self._append_log(f"[Comparaison] Lancement : {label}")
                result = run_experiment_from_config(
                    cfg,
                    log=lambda msg, prefix=label: self._append_log(f"[{prefix}] {msg}"),
                    show_plots=False,
                )
                entries.append(self._comparison_entry_from_result(label, result, cfg))

            colors = ["#2563eb", "#f59e0b"] if mode == "Deux familles de micros" else ["#22c55e", "#a855f7"]
            self.comparison_estimates = []
            for idx, entry in enumerate(entries):
                est = entry.get("estimated_source")
                if est is not None:
                    self.comparison_estimates.append({
                        "label": str(entry.get("label", f"Estimation {idx + 1}")),
                        "position": np.asarray(est, dtype=float).reshape(-1)[:3],
                        "color": colors[idx % len(colors)],
                    })
            self.refresh_scene(reset_camera=False)

            lines = ["Comparaison terminée.", ""]
            lines.extend(self._format_comparison_entry(entry) for entry in entries)
            if len(entries) == 2 and entries[0].get("estimated_source") is not None and entries[1].get("estimated_source") is not None:
                p0 = np.asarray(entries[0]["estimated_source"], dtype=float).reshape(-1)[:3]
                p1 = np.asarray(entries[1]["estimated_source"], dtype=float).reshape(-1)[:3]
                delta = float(np.linalg.norm(p0 - p1))
                lines.extend(["", f"Écart entre les deux estimations : {delta:.4f} m"])
            text = "\n".join(lines)
            if hasattr(self, "comparison_console"):
                self.comparison_console.setPlainText(text)
            self.summary_console.setPlainText(text)
            self.result_algo_value.setText("Comparaison")
            self.result_true_value.setText(vector_to_text(entries[0].get("true_source") if entries else None))
            self.result_estimated_value.setText("\n".join(f"{e.get('label')}: {vector_to_text(e.get('estimated_source'))}" for e in entries))
            if entries:
                error_parts = []
                for e in entries:
                    err = e.get("error")
                    error_parts.append(f"{e.get('label')}: {'—' if err is None else f'{float(err):.4f} m'}")
                self.result_error_value.setText("\n".join(error_parts))
            self._set_status("Comparaison terminée", "success")
            self.bottom_tabs.setCurrentIndex(1)
            self.statusBar().showMessage("Comparaison terminée")
        except Exception as exc:
            QMessageBox.critical(self, "Erreur comparaison", str(exc))
            self._set_status("Échec comparaison", "error")
            if hasattr(self, "comparison_console"):
                self.comparison_console.setPlainText(f"La comparaison a échoué.\n\nErreur : {exc}")
            self._append_log(f"[Comparaison] ERROR: {exc}")
        finally:
            QApplication.restoreOverrideCursor()
            self._is_computing = False
            self._update_metrics()

    def _set_status(self, text: str, state: str) -> None:
        colors = {
            "idle": ("#f1f5f9", "#475569", "#cbd5e1"),
            "running": ("#dbeafe", "#1d4ed8", "#93c5fd"),
            "success": ("#dcfce7", "#166534", "#86efac"),
            "error": ("#fee2e2", "#991b1b", "#fecaca"),
        }
        background, color, border = colors.get(state, colors["idle"])
        self.status_pill.setText(text)
        self.status_pill.setStyleSheet(f"background: {background}; color: {color}; border: 1px solid {border}; border-radius: 10px; padding: 4px 8px;")

    def clear_results(self, keep_log: bool = False) -> None:
        if hasattr(self, "summary_console"):
            self.summary_console.clear()
        if hasattr(self, "error_console"):
            self.error_console.clear()
        if hasattr(self, "comparison_console"):
            self.comparison_console.setPlainText("Aucune comparaison lancée.")
        self.estimated_source = None
        self.comparison_estimates = []
        if hasattr(self, "scene_view"):
            self.refresh_scene(reset_camera=False)
        if hasattr(self, "log_console") and not keep_log:
            self.log_console.clear()
        for attr in ("result_algo_value", "result_error_value", "result_true_value", "result_estimated_value"):
            if hasattr(self, attr):
                getattr(self, attr).setText("—")
        self._set_status("En attente", "idle")
        self.statusBar().showMessage("Résultats effacés")

    def _append_log(self, text: str) -> None:
        message = str(text)
        if hasattr(self, "log_console"):
            self.log_console.appendPlainText(message)
            self.log_console.verticalScrollBar().setValue(self.log_console.verticalScrollBar().maximum())
        lower = message.lower()
        is_error = any(token in lower for token in ("error", "exception", "traceback", "failed", "échec", "erreur"))
        is_warning = any(token in lower for token in ("warning", "warn", "attention"))
        if hasattr(self, "error_console") and (is_error or is_warning):
            prefix = "ERREUR" if is_error else "AVERTISSEMENT"
            self.error_console.appendPlainText(f"[{prefix}] {message}")
            self.error_console.verticalScrollBar().setValue(self.error_console.verticalScrollBar().maximum())

    def _extract_result_highlights(self, data: Dict[str, Any], max_items: int = 6) -> list[tuple[str, str]]:
        highlights: list[tuple[str, str]] = []

        def walk(prefix: str, value: Any) -> None:
            if len(highlights) >= max_items:
                return
            if isinstance(value, dict):
                for key, sub_value in value.items():
                    walk(f"{prefix}.{key}" if prefix else str(key), sub_value)
                    if len(highlights) >= max_items:
                        return
            elif isinstance(value, (int, float, str, bool)) or value is None:
                if len(str(value)) <= 90:
                    highlights.append((prefix.replace("_", " "), str(value)))
            elif isinstance(value, (list, tuple)) and 0 < len(value) <= 4:
                if all(isinstance(item, (int, float, str, bool)) or item is None for item in value):
                    highlights.append((prefix.replace("_", " "), str(value)))

        for preferred_key in ("algorithm", "success", "cost", "n_iter", "iterations", "alpha", "snr_db"):
            if preferred_key in data:
                walk(preferred_key, data[preferred_key])
        if len(highlights) < max_items:
            walk("", data)

        unique: list[tuple[str, str]] = []
        seen = set()
        for key, value in highlights:
            if key and key not in seen:
                unique.append((key, value))
                seen.add(key)
            if len(unique) >= max_items:
                break
        return unique

    # ------------------------------------------------------------------
    # Microphone presets
    # ------------------------------------------------------------------
    def preset_square_4(self) -> None:
        try:
            cfg = self._get_config()
            Lx = _safe_float(cfg["geometry"]["Lx"], 5.0)
            Ly = _safe_float(cfg["geometry"]["Ly"], 4.0)
            Lz = _safe_float(cfg["geometry"]["Lz"], 2.7)
            margin = _safe_float(cfg["geometry"].get("margin"), 0.2)
            z = min(1.5, max(margin, 0.4 * Lz))
            cx, cy = 0.5 * Lx, 0.5 * Ly
            spacing = min(0.8, 0.28 * min(Lx, Ly))
            half = 0.5 * spacing
            mics = np.array([
                [cx - half, cy - half, z],
                [cx + half, cy - half, z],
                [cx - half, cy + half, z],
                [cx + half, cy + half, z],
            ], dtype=float)
            mics[:, 0] = np.clip(mics[:, 0], margin, Lx - margin)
            mics[:, 1] = np.clip(mics[:, 1], margin, Ly - margin)
            mics[:, 2] = np.clip(mics[:, 2], margin, Lz - margin)
            self._set_mics_from_array(mics)
        except Exception as exc:
            QMessageBox.critical(self, "Preset error", str(exc))

    def preset_hex_6(self) -> None:
        try:
            cfg = self._get_config()
            Lx = _safe_float(cfg["geometry"]["Lx"], 5.0)
            Ly = _safe_float(cfg["geometry"]["Ly"], 4.0)
            Lz = _safe_float(cfg["geometry"]["Lz"], 2.7)
            margin = _safe_float(cfg["geometry"].get("margin"), 0.2)
            z = min(1.5, max(margin, 0.4 * Lz))
            cx, cy = 0.5 * Lx, 0.5 * Ly
            radius = min(0.9, 0.32 * min(Lx, Ly))
            angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
            mics = np.stack([cx + radius * np.cos(angles), cy + radius * np.sin(angles), np.full(6, z)], axis=1)
            mics[:, 0] = np.clip(mics[:, 0], margin, Lx - margin)
            mics[:, 1] = np.clip(mics[:, 1], margin, Ly - margin)
            mics[:, 2] = np.clip(mics[:, 2], margin, Lz - margin)
            self._set_mics_from_array(mics)
        except Exception as exc:
            QMessageBox.critical(self, "Preset error", str(exc))

    def preset_random(self) -> None:
        try:
            n, ok = QInputDialog.getInt(self, "Placement aléatoire", "Nombre de microphones :", 6, 1, 256, 1)
            if not ok:
                return
            seed_txt, ok = QInputDialog.getText(self, "Placement aléatoire", "Graine (int ou None) :", text="42")
            if not ok:
                return
            seed = None if str(seed_txt).strip().lower() in ("", "none", "null") else _safe_int(seed_txt, 42)
            cfg = self._get_config()
            Lx = _safe_float(cfg["geometry"]["Lx"], 5.0)
            Ly = _safe_float(cfg["geometry"]["Ly"], 4.0)
            Lz = _safe_float(cfg["geometry"]["Lz"], 2.7)
            margin = _safe_float(cfg["geometry"].get("margin"), 0.2)
            rng = np.random.default_rng(seed)
            mics = np.column_stack([
                rng.uniform(margin, max(margin, Lx - margin), size=n),
                rng.uniform(margin, max(margin, Ly - margin), size=n),
                rng.uniform(margin, max(margin, Lz - margin), size=n),
            ])
            self._set_mics_from_array(mics)
        except Exception as exc:
            QMessageBox.critical(self, "Preset error", str(exc))

    def _set_mics_from_array(self, arr: np.ndarray) -> None:
        lines = [f"{x:.3f},{y:.3f},{z:.3f}" for x, y, z in arr]
        self.mics_text.setPlainText("\n".join(lines))
        self._on_config_changed()


def apply_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    app.setFont(QFont("Aptos", 10))
    app.setStyleSheet(COMSOL_QSS)


def main() -> int:
    if QT_IMPORT_ERROR is not None:
        print("PySide6 is not installed.")
        print("Install dependencies with: pip install PySide6 pyvista pyvistaqt vtk numpy")
        print(f"Original error: {QT_IMPORT_ERROR}")
        return 1

    app = QApplication(sys.argv)
    apply_theme(app)
    window = ComsolChamberSimUI()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

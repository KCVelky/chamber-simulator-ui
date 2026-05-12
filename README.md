# ChamberSim — version nettoyée

Cette version garde uniquement deux lanceurs d'interface au niveau racine :

- `ui_original.py` : interface Tkinter originale, conservée comme référence fonctionnelle.
- `ui_modern.py` : nouvelle interface PySide6, plus moderne et plus proche d'une vraie application desktop.

Le coeur métier reste dans `src/chamber_sim/`.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate       # Windows PowerShell
pip install -r requirements.txt
pip install -e .
```

## Lancement

Interface moderne :

```bash
python ui_modern.py
```

Interface originale :

```bash
python ui_original.py
```

## Nettoyage réalisé

Supprimé de la version nettoyée :

- scripts de test et de développement : `01_build_scene.py`, `02_test_source_signal.py`, `03_test_propagation.py`, `test.py`, `test_2.py`, `test.txt` ;
- doublon strict : `10_ui_experiment_2.py` ;
- anciennes variantes UI remplacées par `ui_original.py` et `ui_modern.py` ;
- tous les dossiers `__pycache__` et fichiers `.pyc` ;
- fichiers annexes non nécessaires au lancement de l'UI.

## Nouvelle organisation

```text
PYTHON-clean/
├── ui_original.py
├── ui_modern.py
├── requirements.txt
├── pyproject.toml
├── README.md
└── src/
    └── chamber_sim/
```

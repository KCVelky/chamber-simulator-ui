from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allows running this script directly from the project root without installing the package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chamber_sim.doe import DOEManager


def _default_workbook_path() -> Path:
    candidate = PROJECT_ROOT / "DOE_MAITRISE.xlsx"
    if candidate.exists():
        return candidate
    return Path("DOE_MAITRISE.xlsx")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prévisualise et valide le fichier DOE Excel.")
    parser.add_argument(
        "workbook",
        nargs="?",
        default=str(_default_workbook_path()),
        help="Chemin du fichier DOE .xlsx. Défaut : DOE_MAITRISE.xlsx dans le dossier projet.",
    )
    parser.add_argument(
        "--sheet",
        default=None,
        help="Nom d'une feuille DOE à prévisualiser, par exemple A_runs_screen.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=3,
        help="Nombre de lignes exemple à afficher.",
    )
    parser.add_argument(
        "--prepare-results",
        action="store_true",
        help="Crée ou met à jour le fichier DOE_MAITRISE_results.xlsx avec les feuilles de sortie.",
    )
    parser.add_argument(
        "--results",
        default=None,
        help="Chemin personnalisé pour le fichier résultats.",
    )
    parser.add_argument(
        "--overwrite-results",
        action="store_true",
        help="Écrase le fichier résultats existant. À utiliser avec prudence.",
    )
    args = parser.parse_args()

    manager = DOEManager(args.workbook, results_path=args.results)

    print("=" * 72)
    print("DOE preview")
    print("=" * 72)

    info = manager.discover()
    print(f"Fichier DOE : {info.path}")
    print(f"Nombre de feuilles : {len(info.sheets)}")
    print()

    print("Feuilles DOE détectées :")
    for row in manager.overview():
        status = "OK" if row.status == "ok" else "ABSENTE"
        print(f"- {row.sheet_name:<20} | {status:<7} | stage={row.stage:<2} | runs={row.n_runs:<5} | {row.description}")
    print()

    report = manager.validate()
    print("Validation :")
    for line in report.summary_lines():
        print(" ", line)
    print()

    if args.sheet:
        preview = manager.preview_sheet(args.sheet, sample_size=args.sample_size)
        print(f"Prévisualisation : {preview.sheet_name}")
        print(f"Stage : {preview.stage}")
        print(f"Nombre de runs : {preview.n_runs}")
        print("Colonnes :")
        print(" ", ", ".join(preview.headers))
        print("Exemples :")
        print(json.dumps(preview.sample_rows, ensure_ascii=False, indent=2, default=str))
        print()

    if args.prepare_results:
        results_path = manager.prepare_results(overwrite=args.overwrite_results)
        print(f"Fichier résultats prêt : {results_path}")
        print("Feuilles de sortie créées si nécessaire : runs_results, iterations_results, geometry_features, crb_map, stage_summary")
        print()

    if report.has_errors:
        print("Résultat : erreurs détectées. Corrige le fichier DOE avant exécution complète.")
        return 2

    print("Résultat : fichier DOE prêt pour l'étape suivante.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

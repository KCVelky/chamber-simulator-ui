from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chamber_sim.doe.runner import DOERunner


def _parse_geometry_slots(values: list[str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in values or []:
        if "=" not in item:
            raise ValueError(f"Mapping invalide : {item!r}. Format attendu : SLOT=A-G01")
        key, value = item.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one DOE entry and write the result to Excel.")
    parser.add_argument("workbook", help="Chemin vers DOE_MAITRISE.xlsx")
    parser.add_argument("--sheet", default="A_runs_screen", help="Feuille DOE à lire")
    parser.add_argument("--run-id", default=None, help="run_id à lancer")
    parser.add_argument("--row", type=int, default=None, help="Index de run dans la feuille, 1 = premier run")
    parser.add_argument("--results", default=None, help="Fichier Excel résultats. Défaut : *_results.xlsx")
    parser.add_argument(
        "--geometry-slot",
        action="append",
        default=[],
        help="Résolution d'un placeholder, ex. BEST_A_1=A-G01. Peut être répété.",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="N'efface pas les anciennes lignes du même run_id avant d'ajouter le résultat.",
    )
    parser.add_argument(
        "--overwrite-results",
        action="store_true",
        help="Recrée le fichier résultats à partir du template DOE avant d'écrire.",
    )
    parser.add_argument("--json", action="store_true", help="Affiche la ligne runs_results en JSON.")
    args = parser.parse_args()

    slots = _parse_geometry_slots(args.geometry_slot)
    runner = DOERunner(args.workbook, args.results, geometry_slots=slots)
    runner.prepare_results(overwrite=args.overwrite_results)

    result = runner.run_once(
        args.sheet,
        run_id=args.run_id,
        row_index=args.row,
        replace_existing=not args.keep_existing,
        write_results=True,
    )

    print("=" * 72)
    print("DOE run unique -> Excel")
    print("=" * 72)
    print(f"Feuille        : {result.sheet_name}")
    print(f"Run ID         : {result.run_id}")
    print(f"Succès         : {result.success}")
    print(f"Fichier Excel  : {result.results_path}")
    print(f"Itérations     : {result.iteration_count}")
    print(f"Erreur 3D [m]  : {result.run_row.get('err_3d')}")
    print(f"Erreur XY [m]  : {result.run_row.get('err_xy')}")
    print(f"Erreur Z [m]   : {result.run_row.get('err_z')}")
    print(f"Score          : {result.run_row.get('score')}")
    if not result.success:
        print(f"Erreur         : {result.run_row.get('error_message')}")

    if result.logs:
        print("\nDerniers logs simulation :")
        for line in result.logs[-12:]:
            print(f"  {line}")

    if args.json:
        print("\nLigne runs_results :")
        print(json.dumps(result.run_row, ensure_ascii=False, indent=2))

    print("\nRésultat : écriture Excel OK.")
    return 0 if result.success else 2


if __name__ == "__main__":
    raise SystemExit(main())

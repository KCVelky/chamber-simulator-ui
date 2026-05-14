from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chamber_sim.doe.runner import DOEBatchProgress, DOERunner


def _parse_geometry_slots(values: list[str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in values or []:
        if "=" not in item:
            raise ValueError(f"Mapping invalide : {item!r}. Format attendu : SLOT=A-G01")
        key, value = item.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def _print_progress(progress: DOEBatchProgress) -> None:
    if progress.event == "started":
        print(f"Démarrage : {progress.message}")
        return
    if progress.event == "finished":
        print(f"Terminé : {progress.message}")
        return

    prefix = f"[{progress.index}/{progress.total}]"
    run_id = progress.run_id or "-"
    if progress.event == "skipped":
        print(f"{prefix} SKIP  {run_id} | {progress.message}")
    elif progress.event == "completed":
        print(f"{prefix} OK    {run_id} | {progress.message}")
    elif progress.event == "failed":
        print(f"{prefix} FAIL  {run_id} | {progress.message}")
    elif progress.event == "stopped":
        print(f"{prefix} STOP  {run_id} | {progress.message}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a DOE sheet and save every result to Excel.")
    parser.add_argument("workbook", help="Chemin vers DOE_MAITRISE.xlsx")
    parser.add_argument("--sheet", default="A_runs_screen", help="Feuille DOE à lancer")
    parser.add_argument("--results", default=None, help="Fichier Excel résultats. Défaut : *_results.xlsx")
    parser.add_argument("--start-index", type=int, default=1, help="Index 1-based du premier run à considérer")
    parser.add_argument("--limit", type=int, default=None, help="Nombre maximum de runs à considérer")
    parser.add_argument(
        "--geometry-slot",
        action="append",
        default=[],
        help="Résolution d'un placeholder, ex. BEST_A_1=A-G01. Peut être répété.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recalcule aussi les runs déjà réussis dans runs_results.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Désactive la reprise automatique. Équivalent logique à --force pour le filtrage.",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="N'efface pas les anciennes lignes du même run_id avant d'ajouter le résultat.",
    )
    parser.add_argument(
        "--overwrite-results",
        action="store_true",
        help="Recrée le fichier résultats à partir du template DOE avant de lancer.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Arrête la campagne dès le premier échec.",
    )
    parser.add_argument(
        "--max-failures",
        type=int,
        default=None,
        help="Arrête la campagne après N échecs. Non défini = pas de limite.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Affiche le plan d'exécution sans lancer les simulations.",
    )
    args = parser.parse_args()

    slots = _parse_geometry_slots(args.geometry_slot)
    runner = DOERunner(args.workbook, args.results, geometry_slots=slots)
    runner.prepare_results(overwrite=args.overwrite_results)

    resume = not args.no_resume and not args.force
    if args.dry_run:
        plan = runner.plan_sheet(
            args.sheet,
            resume=resume,
            force=args.force,
            start_index=args.start_index,
            limit=args.limit,
        )
        pending = [item for item in plan if item.status == "pending"]
        skipped = [item for item in plan if item.status == "skipped_success"]
        print("=" * 72)
        print("Plan DOE")
        print("=" * 72)
        print(f"Feuille          : {args.sheet}")
        print(f"Fichier résultats: {runner.results_path}")
        print(f"Runs considérés  : {len(plan)}")
        print(f"À lancer         : {len(pending)}")
        print(f"À ignorer        : {len(skipped)}")
        print("\nAperçu :")
        for item in plan[:20]:
            print(f"  row={item.row_index:04d} | {item.run_id:<14} | {item.status} | {item.reason}")
        if len(plan) > 20:
            print(f"  ... {len(plan) - 20} lignes supplémentaires")
        return 0

    summary = runner.run_sheet(
        args.sheet,
        resume=resume,
        force=args.force,
        start_index=args.start_index,
        limit=args.limit,
        stop_on_error=args.stop_on_error,
        max_failures=args.max_failures,
        progress=_print_progress,
        replace_existing=not args.keep_existing,
    )

    print("\n" + "=" * 72)
    print("Résumé campagne DOE")
    print("=" * 72)
    for line in summary.summary_lines():
        print(line)
    if summary.run_ids_failed:
        print("\nRuns échoués :")
        for run_id in summary.run_ids_failed[:30]:
            print(f"  - {run_id}")
        if len(summary.run_ids_failed) > 30:
            print(f"  ... {len(summary.run_ids_failed) - 30} autres")

    return 0 if summary.failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

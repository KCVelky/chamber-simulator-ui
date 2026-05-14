from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chamber_sim.doe.orchestrator import DOEAutomationPlan, DOEAutomationProgress, DOEAutomationRunner
from chamber_sim.doe.analytics import DOEAnalyzer


def _parse_slots(values: list[str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in values or []:
        if "=" not in item:
            raise ValueError(f"Mapping invalide : {item!r}. Format attendu : SLOT=A-G01")
        key, value = item.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def _parse_limits(values: list[str] | None) -> dict[str, int | None]:
    out: dict[str, int | None] = {}
    for item in values or []:
        if "=" not in item:
            raise ValueError(f"Limite invalide : {item!r}. Format attendu : SHEET=10")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip().lower()
        out[key] = None if value in {"all", "tous", "none", ""} else int(value)
    return out


def _progress(progress: DOEAutomationProgress) -> None:
    if progress.event == "workflow_started":
        print(f"[AUTO] {progress.message}")
    elif progress.event == "step_started":
        print(f"\n[AUTO] Étape {progress.step} — {progress.message}")
    elif progress.event == "analysis":
        print(f"[AUTO] Analyse {progress.step} — {progress.message}")
    elif progress.event == "step_finished":
        print(f"[AUTO] Fin {progress.step} — {progress.message}")
    elif progress.event == "workflow_finished":
        print(f"\n[AUTO] {progress.message}")
    elif progress.event == "run_progress" and progress.batch_progress is not None:
        batch = progress.batch_progress
        if batch.event in {"completed", "failed", "skipped", "stopped"}:
            print(f"  [{batch.index}/{batch.total}] {batch.event.upper():<9} {batch.run_id or '-'} | {batch.message}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the intelligent DOE workflow A→B→C→D.")
    parser.add_argument("workbook", help="Chemin vers DOE_MAITRISE.xlsx")
    parser.add_argument("--results", default=None, help="Fichier Excel résultats. Défaut : *_results.xlsx")
    parser.add_argument("--overwrite-results", action="store_true", help="Recrée le fichier résultats avant lancement")
    parser.add_argument("--force", action="store_true", help="Recalcule les runs déjà réussis")
    parser.add_argument("--no-resume", action="store_true", help="Désactive la reprise automatique")
    parser.add_argument("--stop-on-error", action="store_true", help="Arrête au premier échec")
    parser.add_argument("--max-failures", type=int, default=None, help="Arrête après N échecs")
    parser.add_argument("--include-validation", action="store_true", help="Inclut D_validation dans le workflow complet")
    parser.add_argument("--skip-a-screen", action="store_true", help="Ne lance pas A_runs_screen, analyse seulement les résultats existants")
    parser.add_argument("--skip-a-map", action="store_true", help="Ne lance pas A_runs_map")
    parser.add_argument("--skip-b", action="store_true", help="Ne lance pas B_PB12")
    parser.add_argument("--skip-c1", action="store_true", help="Ne lance pas C1_PB12_common")
    parser.add_argument("--skip-c2", action="store_true", help="Ne lance pas C2_alpha_refine")
    parser.add_argument("--geometry-slot", action="append", default=[], help="Mapping initial, ex. BEST_A_1=A-G01")
    parser.add_argument("--limit", action="append", default=[], help="Limite par feuille, ex. A_runs_screen=10. Peut être répété.")
    parser.add_argument("--analyze-only", action="store_true", help="Ne lance aucun run, met seulement à jour les synthèses/décisions Excel")
    args = parser.parse_args()

    if args.analyze_only:
        runner = DOEAutomationRunner(args.workbook, args.results, geometry_slots=_parse_slots(args.geometry_slot))
        # Ensure output workbook exists through the orchestrator's runner.
        from chamber_sim.doe.runner import DOERunner
        base = DOERunner(args.workbook, args.results, geometry_slots=_parse_slots(args.geometry_slot))
        results = base.prepare_results(overwrite=args.overwrite_results)
        analyzer = DOEAnalyzer(args.workbook, results)
        report = analyzer.write_report()
        print("Analyse DOE mise à jour.")
        print(f"Fichier résultats : {results}")
        for key, value in sorted(report.selections.decisions.items()):
            print(f"  - {key}: {value}")
        return 0

    plan = DOEAutomationPlan(
        run_stage_a_screen=not args.skip_a_screen,
        run_stage_a_map=not args.skip_a_map,
        run_stage_b=not args.skip_b,
        run_stage_c1=not args.skip_c1,
        run_stage_c2=not args.skip_c2,
        run_stage_d=bool(args.include_validation),
        resume=not args.no_resume and not args.force,
        force=bool(args.force),
        stop_on_error=bool(args.stop_on_error),
        max_failures=args.max_failures,
        limits=_parse_limits(args.limit),
    )

    auto = DOEAutomationRunner(
        args.workbook,
        args.results,
        geometry_slots=_parse_slots(args.geometry_slot),
    )
    summary = auto.run(plan, overwrite_results=args.overwrite_results, progress=_progress)

    print("\n" + "=" * 72)
    print("Résumé automatisation DOE")
    print("=" * 72)
    for line in summary.summary_lines():
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

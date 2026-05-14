from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from .analytics import DOEAnalyzer, DOEAnalysisReport, ScoringPolicy
from .config_builder import DOEChamberMapping
from .runner import DOEBatchProgress, DOEBatchSummary, DOERunner


@dataclass(frozen=True)
class AutomationStep:
    name: str
    sheet_name: str | None
    status: str
    message: str = ""
    summary: DOEBatchSummary | None = None


@dataclass(frozen=True)
class DOEAutomationProgress:
    event: str  # workflow_started | step_started | run_progress | analysis | step_finished | workflow_finished | stopped
    step: str
    sheet_name: str | None
    message: str
    elapsed_s: float
    batch_progress: DOEBatchProgress | None = None


@dataclass(frozen=True)
class DOEAutomationSummary:
    results_path: Path
    steps: list[AutomationStep]
    geometry_slots: dict[str, str]
    algorithm_slots: dict[str, dict[str, Any]]
    decisions: dict[str, Any]
    elapsed_s: float
    stopped_early: bool = False

    def summary_lines(self) -> list[str]:
        lines = [
            f"Fichier résultats : {self.results_path}",
            f"Temps total       : {self.elapsed_s:.2f} s",
            f"Arrêt anticipé    : {'oui' if self.stopped_early else 'non'}",
            "",
            "Décisions automatiques :",
        ]
        if self.decisions:
            for key, value in sorted(self.decisions.items()):
                lines.append(f"  - {key}: {value}")
        else:
            lines.append("  - aucune décision disponible")
        lines.extend(["", "Étapes :"])
        for step in self.steps:
            suffix = f" | {step.message}" if step.message else ""
            lines.append(f"  - {step.name}: {step.status}{suffix}")
        return lines


AutomationProgressCallback = Callable[[DOEAutomationProgress], None]
StopPredicate = Callable[[], bool]


@dataclass(frozen=True)
class DOEAutomationPlan:
    run_stage_a_screen: bool = True
    run_stage_a_map: bool = True
    run_stage_b: bool = True
    run_stage_c1: bool = True
    run_stage_c2: bool = True
    run_stage_d: bool = False
    resume: bool = True
    force: bool = False
    stop_on_error: bool = False
    max_failures: int | None = None
    limits: dict[str, int | None] = field(default_factory=dict)

    @classmethod
    def full(cls, *, include_validation: bool = False, limits: Mapping[str, int | None] | None = None) -> "DOEAutomationPlan":
        return cls(run_stage_d=include_validation, limits=dict(limits or {}))


class DOEAutomationRunner:
    """Intelligent A→B→C→D orchestration for the DOE workbook.

    The orchestrator is deliberately conservative: it runs one sheet at a time,
    writes every simulation result immediately, then re-analyzes the Excel file to
    decide the next placeholders. Therefore the workflow is resumable and every
    decision remains visible in ``automation_decisions`` and ``best_candidates``.
    """

    def __init__(
        self,
        workbook_path: str | Path,
        results_path: str | Path | None = None,
        *,
        chamber: DOEChamberMapping | None = None,
        geometry_slots: Mapping[str, str] | None = None,
        algorithm_slots: Mapping[str, Mapping[str, Any]] | None = None,
        scoring_policy: ScoringPolicy | None = None,
    ) -> None:
        self.workbook_path = Path(workbook_path).expanduser().resolve()
        self.results_path = None if results_path is None else Path(results_path).expanduser().resolve()
        self.chamber = chamber
        self.geometry_slots = {str(k): str(v) for k, v in (geometry_slots or {}).items()}
        self.algorithm_slots = {str(k): dict(v) for k, v in (algorithm_slots or {}).items()}
        self.scoring_policy = scoring_policy or ScoringPolicy()

    def _new_runner(self) -> DOERunner:
        return DOERunner(
            self.workbook_path,
            self.results_path,
            chamber=self.chamber,
            geometry_slots=self.geometry_slots,
            algorithm_slots=self.algorithm_slots,
        )

    def _emit(
        self,
        progress: AutomationProgressCallback | None,
        event: str,
        step: str,
        sheet_name: str | None,
        message: str,
        t0: float,
        batch_progress: DOEBatchProgress | None = None,
    ) -> None:
        if progress is not None:
            progress(DOEAutomationProgress(event, step, sheet_name, message, time.perf_counter() - t0, batch_progress))

    def analyze_and_update_slots(self) -> DOEAnalysisReport:
        if self.results_path is None:
            raise RuntimeError("results_path non initialisé.")
        analyzer = DOEAnalyzer(self.workbook_path, self.results_path, policy=self.scoring_policy)
        report = analyzer.write_report()
        self.geometry_slots.update(report.selections.geometry_slots)
        self.algorithm_slots.update(report.selections.algorithm_slots)
        return report

    def run(
        self,
        plan: DOEAutomationPlan | None = None,
        *,
        overwrite_results: bool = False,
        progress: AutomationProgressCallback | None = None,
        should_stop: StopPredicate | None = None,
    ) -> DOEAutomationSummary:
        plan = plan or DOEAutomationPlan.full(include_validation=False)
        t0 = time.perf_counter()
        steps: list[AutomationStep] = []
        stopped = False

        runner = self._new_runner()
        self.results_path = runner.prepare_results(overwrite=overwrite_results)
        self._emit(progress, "workflow_started", "workflow", None, f"Résultats : {self.results_path}", t0)

        def stop_requested() -> bool:
            return should_stop() if should_stop is not None else False

        def run_sheet_step(step_name: str, sheet_name: str) -> DOEBatchSummary | None:
            nonlocal stopped
            if stop_requested():
                stopped = True
                self._emit(progress, "stopped", step_name, sheet_name, "Arrêt demandé avant cette étape.", t0)
                steps.append(AutomationStep(step_name, sheet_name, "stopped", "Arrêt demandé"))
                return None

            self._emit(progress, "step_started", step_name, sheet_name, f"Lancement {sheet_name}", t0)
            runner = self._new_runner()
            runner.results_path = self.results_path

            def on_batch(batch: DOEBatchProgress) -> None:
                self._emit(progress, "run_progress", step_name, sheet_name, batch.message or batch.event, t0, batch)

            summary = runner.run_sheet(
                sheet_name,
                resume=plan.resume,
                force=plan.force,
                limit=plan.limits.get(sheet_name),
                stop_on_error=plan.stop_on_error,
                max_failures=plan.max_failures,
                progress=on_batch,
                should_stop=stop_requested,
                replace_existing=True,
            )
            if summary.stopped_early:
                stopped = True
            status = "ok" if summary.failed == 0 and not summary.stopped_early else "partial"
            steps.append(AutomationStep(step_name, sheet_name, status, f"{summary.succeeded} OK, {summary.failed} échecs, {summary.skipped} ignorés", summary))
            self._emit(progress, "step_finished", step_name, sheet_name, steps[-1].message, t0)
            return summary

        def analyze_step(label: str) -> DOEAnalysisReport:
            report = self.analyze_and_update_slots()
            msg = "; ".join(f"{k}={v}" for k, v in sorted(report.selections.geometry_slots.items()))
            if report.selections.algorithm_slots:
                msg += " | algo prêt"
            self._emit(progress, "analysis", label, None, msg or "Analyse mise à jour.", t0)
            return report

        if plan.run_stage_a_screen:
            run_sheet_step("A_screen", "A_runs_screen")
            analyze_step("A_screen_analysis")
            if stopped and plan.stop_on_error:
                return self._finish(steps, t0, stopped, progress)
        else:
            analyze_step("A_screen_analysis")

        if plan.run_stage_a_map:
            # A_map needs BEST_A_1 and BEST_A_2 from the screening analysis.
            if not {"BEST_A_1", "BEST_A_2"}.issubset(self.geometry_slots):
                steps.append(AutomationStep("A_map", "A_runs_map", "skipped", "BEST_A_1/BEST_A_2 indisponibles"))
            else:
                run_sheet_step("A_map", "A_runs_map")
                analyze_step("A_map_analysis")
                if stopped and plan.stop_on_error:
                    return self._finish(steps, t0, stopped, progress)

        # If A_map has not been run yet, BEST_GLOBAL falls back to A_screen.
        analyze_step("global_selection")

        if plan.run_stage_b:
            if "BEST_GLOBAL" not in self.geometry_slots:
                steps.append(AutomationStep("B", "B_PB12", "skipped", "BEST_GLOBAL indisponible"))
            else:
                run_sheet_step("B", "B_PB12")
                analyze_step("B_analysis")

        if plan.run_stage_c1:
            if "BEST_GLOBAL" not in self.geometry_slots:
                steps.append(AutomationStep("C1", "C1_PB12_common", "skipped", "BEST_GLOBAL indisponible"))
            else:
                run_sheet_step("C1", "C1_PB12_common")
                analyze_step("C1_analysis")

        if plan.run_stage_c2:
            if "BEST_GLOBAL" not in self.geometry_slots:
                steps.append(AutomationStep("C2", "C2_alpha_refine", "skipped", "BEST_GLOBAL indisponible"))
            else:
                run_sheet_step("C2", "C2_alpha_refine")
                analyze_step("C2_analysis")

        if plan.run_stage_d:
            analyze_step("pre_validation_analysis")
            if not {"BEST_1", "BEST_2"}.issubset(self.geometry_slots):
                steps.append(AutomationStep("D", "D_validation", "skipped", "BEST_1/BEST_2 indisponibles"))
            else:
                run_sheet_step("D", "D_validation")
                analyze_step("D_analysis")

        return self._finish(steps, t0, stopped, progress)

    def _finish(
        self,
        steps: list[AutomationStep],
        t0: float,
        stopped: bool,
        progress: AutomationProgressCallback | None,
    ) -> DOEAutomationSummary:
        if self.results_path is None:
            raise RuntimeError("results_path non initialisé.")
        report = self.analyze_and_update_slots()
        summary = DOEAutomationSummary(
            results_path=self.results_path,
            steps=steps,
            geometry_slots=dict(self.geometry_slots),
            algorithm_slots=dict(self.algorithm_slots),
            decisions=dict(report.selections.decisions),
            elapsed_s=time.perf_counter() - t0,
            stopped_early=stopped,
        )
        self._emit(progress, "workflow_finished", "workflow", None, "Workflow DOE terminé.", t0)
        return summary

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from chamber_sim.experiment_runner import ExperimentResult, run_experiment_from_config

from .config_builder import DOEChamberMapping, DOEConfigBuildResult, DOEConfigBuilder
from .excel_io import get_run_statuses, prepare_results_workbook
from .result_writer import (
    build_failure_run_row,
    build_iteration_rows,
    build_success_run_row,
    utc_now_iso,
    write_execution_to_workbook,
)
from .schemas import DOE_RUN_SHEETS


@dataclass(frozen=True)
class DOERunOnceResult:
    sheet_name: str
    run_id: str | None
    results_path: Path
    success: bool
    run_row: dict[str, Any]
    iteration_count: int
    logs: list[str]
    build_result: DOEConfigBuildResult | None = None
    experiment_result: ExperimentResult | None = None


@dataclass(frozen=True)
class DOERunPlanItem:
    """One DOE row planned for a batch execution."""

    row_index: int
    run_id: str
    status: str  # pending | skipped_success | selected
    reason: str = ""


@dataclass(frozen=True)
class DOEBatchProgress:
    """Progress payload emitted during a DOE sheet execution."""

    event: str  # started | skipped | completed | failed | stopped | finished
    sheet_name: str
    run_id: str | None
    index: int
    total: int
    completed: int
    skipped: int
    failed: int
    elapsed_s: float
    message: str = ""
    result: DOERunOnceResult | None = None


@dataclass(frozen=True)
class DOEBatchSummary:
    """Summary returned after running a sheet."""

    sheet_name: str
    results_path: Path
    planned_total: int
    executed: int
    succeeded: int
    failed: int
    skipped: int
    stopped_early: bool
    elapsed_s: float
    run_ids_executed: list[str] = field(default_factory=list)
    run_ids_failed: list[str] = field(default_factory=list)
    run_ids_skipped: list[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return 0.0 if self.executed == 0 else self.succeeded / self.executed

    def summary_lines(self) -> list[str]:
        return [
            f"Feuille DOE      : {self.sheet_name}",
            f"Fichier résultats: {self.results_path}",
            f"Runs planifiés   : {self.planned_total}",
            f"Exécutés         : {self.executed}",
            f"Réussis          : {self.succeeded}",
            f"Échoués          : {self.failed}",
            f"Ignorés reprise  : {self.skipped}",
            f"Taux réussite    : {100.0 * self.success_rate:.1f}%",
            f"Temps total      : {self.elapsed_s:.2f} s",
            f"Arrêt anticipé   : {'oui' if self.stopped_early else 'non'}",
        ]


ProgressCallback = Callable[[DOEBatchProgress], None]
StopPredicate = Callable[[], bool]


class DOERunner:
    """Run DOE entries and persist results.

    Step 4 scope:
    - execute a full DOE sheet or a limited subset;
    - resume automatically by skipping runs already successful in ``runs_results``;
    - save after each run through ``run_once``;
    - expose progress callbacks for future UI integration.
    """

    def __init__(
        self,
        workbook_path: str | Path,
        results_path: str | Path | None = None,
        *,
        chamber: DOEChamberMapping | None = None,
        geometry_slots: Mapping[str, str] | None = None,
        algorithm_slots: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        self.workbook_path = Path(workbook_path).expanduser().resolve()
        self.results_path = None if results_path is None else Path(results_path).expanduser().resolve()
        self.builder = DOEConfigBuilder(
            self.workbook_path,
            chamber=chamber,
            geometry_slots=geometry_slots,
            algorithm_slots=algorithm_slots,
        )

    def prepare_results(self, overwrite: bool = False) -> Path:
        self.results_path = prepare_results_workbook(self.workbook_path, self.results_path, overwrite=overwrite)
        return self.results_path

    def run_once(
        self,
        sheet_name: str,
        *,
        run_id: str | None = None,
        row_index: int | None = None,
        replace_existing: bool = True,
        write_results: bool = True,
    ) -> DOERunOnceResult:
        if sheet_name not in DOE_RUN_SHEETS:
            raise KeyError(f"Feuille DOE inconnue : {sheet_name}")

        if self.results_path is None:
            self.prepare_results(overwrite=False)
        assert self.results_path is not None

        stage = DOE_RUN_SHEETS[sheet_name].stage
        logs: list[str] = []
        started = utc_now_iso()
        t0 = time.perf_counter()
        build: DOEConfigBuildResult | None = None
        effective_run_id = run_id

        try:
            if run_id is not None:
                build = self.builder.build_config_by_id(sheet_name, run_id)
            else:
                build = self.builder.build_config_by_row(sheet_name, row_index or 1)
            effective_run_id = build.run_id

            experiment = run_experiment_from_config(build.config, log=logs.append, show_plots=False)
            finished = utc_now_iso()
            elapsed = time.perf_counter() - t0
            run_row = build_success_run_row(
                build,
                experiment,
                started_at=started,
                finished_at=finished,
                time_s=elapsed,
            )
            iteration_rows = build_iteration_rows(build.run_id, experiment)

            if write_results:
                write_execution_to_workbook(
                    self.results_path,
                    run_row,
                    iteration_rows,
                    replace_existing=replace_existing,
                )

            return DOERunOnceResult(
                sheet_name=sheet_name,
                run_id=build.run_id,
                results_path=self.results_path,
                success=bool(run_row.get("success_flag")),
                run_row=dict(run_row),
                iteration_count=len(iteration_rows),
                logs=logs,
                build_result=build,
                experiment_result=experiment,
            )

        except Exception as exc:
            finished = utc_now_iso()
            elapsed = time.perf_counter() - t0
            run_row = build_failure_run_row(
                run_id=effective_run_id,
                sheet_name=sheet_name,
                stage=stage,
                started_at=started,
                finished_at=finished,
                time_s=elapsed,
                error_message=f"{type(exc).__name__}: {exc}",
                build=build,
            )
            if write_results:
                write_execution_to_workbook(
                    self.results_path,
                    run_row,
                    [],
                    replace_existing=replace_existing,
                )
            return DOERunOnceResult(
                sheet_name=sheet_name,
                run_id=effective_run_id,
                results_path=self.results_path,
                success=False,
                run_row=dict(run_row),
                iteration_count=0,
                logs=logs,
                build_result=build,
                experiment_result=None,
            )

    def plan_sheet(
        self,
        sheet_name: str,
        *,
        resume: bool = True,
        force: bool = False,
        start_index: int = 1,
        limit: int | None = None,
    ) -> list[DOERunPlanItem]:
        """Return the execution plan for a DOE sheet.

        ``start_index`` is 1-based within the DOE records, not the Excel row number.
        Successful runs found in the results workbook are marked as skipped when
        ``resume=True``. Failed or partial runs are retried by default.
        """

        if sheet_name not in DOE_RUN_SHEETS:
            raise KeyError(f"Feuille DOE inconnue : {sheet_name}")
        if start_index < 1:
            raise ValueError("start_index doit être >= 1.")

        if self.results_path is None:
            self.prepare_results(overwrite=False)
        assert self.results_path is not None

        records = self.builder.records_for_sheet(sheet_name)
        selected = list(enumerate(records, start=1))[start_index - 1 :]
        if limit is not None:
            if limit < 0:
                raise ValueError("limit doit être positif ou nul.")
            selected = selected[:limit]

        statuses = get_run_statuses(self.results_path) if resume and not force else {}
        plan: list[DOERunPlanItem] = []
        for idx, record in selected:
            run_id = self.builder.extract_run_id(sheet_name, record)
            previous = statuses.get(run_id)
            if previous and previous.success_flag is True:
                plan.append(
                    DOERunPlanItem(
                        row_index=idx,
                        run_id=run_id,
                        status="skipped_success",
                        reason=f"déjà réussi dans {self.results_path.name}",
                    )
                )
            else:
                plan.append(DOERunPlanItem(row_index=idx, run_id=run_id, status="pending"))
        return plan

    def run_sheet(
        self,
        sheet_name: str,
        *,
        resume: bool = True,
        force: bool = False,
        start_index: int = 1,
        limit: int | None = None,
        stop_on_error: bool = False,
        max_failures: int | None = None,
        progress: ProgressCallback | None = None,
        should_stop: StopPredicate | None = None,
        replace_existing: bool = True,
    ) -> DOEBatchSummary:
        """Execute a DOE sheet sequentially.

        The method is intentionally conservative: each run is written to Excel
        immediately by ``run_once``. This makes long campaigns resumable and safe
        against crashes or manual interruption.
        """

        if self.results_path is None:
            self.prepare_results(overwrite=False)
        assert self.results_path is not None

        plan = self.plan_sheet(
            sheet_name,
            resume=resume,
            force=force,
            start_index=start_index,
            limit=limit,
        )
        total = len(plan)
        skipped_ids = [item.run_id for item in plan if item.status == "skipped_success"]
        skipped_count = 0
        executed_ids: list[str] = []
        failed_ids: list[str] = []
        succeeded = 0
        failed = 0
        stopped = False
        t0 = time.perf_counter()

        def emit(event: str, item: DOERunPlanItem | None, message: str = "", result: DOERunOnceResult | None = None) -> None:
            if progress is None:
                return
            progress(
                DOEBatchProgress(
                    event=event,
                    sheet_name=sheet_name,
                    run_id=item.run_id if item is not None else None,
                    index=plan.index(item) + 1 if item is not None and item in plan else 0,
                    total=total,
                    completed=succeeded,
                    skipped=skipped_count,
                    failed=failed,
                    elapsed_s=time.perf_counter() - t0,
                    message=message,
                    result=result,
                )
            )

        emit("started", None, f"{total} runs planifiés.")

        for position, item in enumerate(plan, start=1):
            if item.status == "skipped_success":
                skipped_count += 1
                emit("skipped", item, item.reason)
                continue

            if should_stop is not None and should_stop():
                stopped = True
                emit("stopped", item, "Arrêt demandé avant le lancement du run.")
                break

            result = self.run_once(
                sheet_name,
                run_id=item.run_id,
                replace_existing=replace_existing,
                write_results=True,
            )
            executed_ids.append(item.run_id)

            if result.success:
                succeeded += 1
                err = result.run_row.get("err_3d")
                msg = "run terminé"
                if isinstance(err, (float, int)):
                    msg += f" | err_3d={float(err):.4f} m"
                emit("completed", item, msg, result)
            else:
                failed += 1
                failed_ids.append(item.run_id)
                message = str(result.run_row.get("error_message") or "échec sans message")
                emit("failed", item, message, result)
                if stop_on_error:
                    stopped = True
                    emit("stopped", item, "Arrêt après le premier échec.", result)
                    break
                if max_failures is not None and failed >= max_failures:
                    stopped = True
                    emit("stopped", item, f"Arrêt après {failed} échecs.", result)
                    break

        elapsed = time.perf_counter() - t0
        summary = DOEBatchSummary(
            sheet_name=sheet_name,
            results_path=self.results_path,
            planned_total=total,
            executed=len(executed_ids),
            succeeded=succeeded,
            failed=failed,
            skipped=skipped_count,
            stopped_early=stopped,
            elapsed_s=elapsed,
            run_ids_executed=executed_ids,
            run_ids_failed=failed_ids,
            run_ids_skipped=skipped_ids[:skipped_count],
        )
        if progress is not None:
            progress(
                DOEBatchProgress(
                    event="finished",
                    sheet_name=sheet_name,
                    run_id=None,
                    index=total,
                    total=total,
                    completed=succeeded,
                    skipped=skipped_count,
                    failed=failed,
                    elapsed_s=elapsed,
                    message="Campagne terminée.",
                    result=None,
                )
            )
        return summary

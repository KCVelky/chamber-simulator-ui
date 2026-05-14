from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from chamber_sim.experiment_runner import ExperimentResult

from .config_builder import DOEConfigBuildResult
from .excel_io import append_output_rows


def utc_now_iso() -> str:
    """Return a compact ISO timestamp suitable for Excel cells and logs."""

    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _to_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _xyz(value: Any) -> tuple[float | None, float | None, float | None]:
    if value is None:
        return None, None, None
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        return None, None, None
    if arr.size < 3:
        return None, None, None
    return _to_float_or_none(arr[0]), _to_float_or_none(arr[1]), _to_float_or_none(arr[2])


def _debug_n_iter(debug: Any) -> int | None:
    if not isinstance(debug, Mapping):
        return None

    solver_debug = debug.get("solver_debug")
    if isinstance(solver_debug, Mapping):
        for key in ("history_cost", "history_x", "history_step", "iterations"):
            values = solver_debug.get(key)
            if values is not None:
                try:
                    return int(len(values))
                except Exception:
                    pass

    for key in ("history", "iterations", "cost_history"):
        values = debug.get(key)
        if values is not None:
            try:
                return int(len(values))
            except Exception:
                pass

    return None


def compute_error_metrics(true_source: Any, estimated_source: Any) -> dict[str, float | None]:
    """Compute 3D, horizontal and vertical localization errors."""

    try:
        true = np.asarray(true_source, dtype=float).reshape(3)
        est = np.asarray(estimated_source, dtype=float).reshape(3)
    except Exception:
        return {"err_3d": None, "err_xy": None, "err_z": None}

    delta = est - true
    return {
        "err_3d": float(np.linalg.norm(delta)),
        "err_xy": float(np.linalg.norm(delta[:2])),
        "err_z": float(abs(delta[2])),
    }


def compute_default_score(err_3d: float | None, success: bool, time_s: float | None = None) -> float:
    """Simple first DOE score.

    The full scoring policy will be refined in later steps. For now, the score is mostly
    the 3D error, with a tiny time penalty so impossible/failed runs are sorted last.
    """

    if not success or err_3d is None:
        return 1e9
    return float(err_3d) + 0.001 * float(time_s or 0.0)


def build_success_run_row(
    build: DOEConfigBuildResult,
    experiment: ExperimentResult,
    *,
    started_at: str,
    finished_at: str,
    time_s: float,
) -> dict[str, Any]:
    """Convert one successful simulator result into a runs_results row."""

    x_true, y_true, z_true = _xyz(experiment.true_source)
    x_hat, y_hat, z_hat = _xyz(experiment.estimated_source)
    errors = compute_error_metrics(experiment.true_source, experiment.estimated_source)
    success = experiment.estimated_source is not None
    n_iter = _debug_n_iter(experiment.debug)
    raw = build.config.get("doe", {}).get("raw_record", {})

    return {
        "run_id": build.run_id,
        "stage": build.summary.get("stage"),
        "input_sheet": build.sheet_name,
        "status": "completed" if success else "no_estimate",
        "success_flag": bool(success),
        "started_at": started_at,
        "finished_at": finished_at,
        "time_s": float(time_s),
        "algorithm": experiment.algorithm,
        "geometry_slot": raw.get("geometry_slot"),
        "geom_run_id": build.geometry_id,
        "source_id": build.source_id,
        "x_true": x_true,
        "y_true": y_true,
        "z_true": z_true,
        "x_hat": x_hat,
        "y_hat": y_hat,
        "z_hat": z_hat,
        "err_3d": errors["err_3d"],
        "err_xy": errors["err_xy"],
        "err_z": errors["err_z"],
        "residual": _to_float_or_none(experiment.residual),
        "n_iter": n_iter,
        "score": compute_default_score(errors["err_3d"], success, time_s),
        "error_message": None,
    }


def build_failure_run_row(
    *,
    run_id: str | None,
    sheet_name: str,
    stage: str | None = None,
    started_at: str,
    finished_at: str,
    time_s: float,
    error_message: str,
    build: DOEConfigBuildResult | None = None,
) -> dict[str, Any]:
    """Convert a failed DOE execution into a runs_results row."""

    raw = build.config.get("doe", {}).get("raw_record", {}) if build is not None else {}
    x_true, y_true, z_true = _xyz(build.source_position if build is not None else None)

    return {
        "run_id": run_id or (build.run_id if build is not None else None),
        "stage": stage or (build.summary.get("stage") if build is not None else None),
        "input_sheet": sheet_name,
        "status": "failed",
        "success_flag": False,
        "started_at": started_at,
        "finished_at": finished_at,
        "time_s": float(time_s),
        "algorithm": build.summary.get("algorithm") if build is not None else None,
        "geometry_slot": raw.get("geometry_slot"),
        "geom_run_id": build.geometry_id if build is not None else None,
        "source_id": build.source_id if build is not None else None,
        "x_true": x_true,
        "y_true": y_true,
        "z_true": z_true,
        "x_hat": None,
        "y_hat": None,
        "z_hat": None,
        "err_3d": None,
        "err_xy": None,
        "err_z": None,
        "residual": None,
        "n_iter": None,
        "score": 1e9,
        "error_message": error_message[:32000],
    }


def _array_or_none(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        return np.asarray(value)
    except Exception:
        return None


def build_iteration_rows(run_id: str, experiment: ExperimentResult) -> list[dict[str, Any]]:
    """Extract solver iteration history for iterations_results.

    For MLE/EM this uses debug['solver_debug']['history_*']. The function is tolerant:
    if another algorithm does not provide history, it simply returns an empty list.
    """

    debug = experiment.debug
    if not isinstance(debug, Mapping):
        return []
    solver_debug = debug.get("solver_debug")
    if not isinstance(solver_debug, Mapping):
        return []

    history_x = _array_or_none(solver_debug.get("history_x"))
    history_cost = _array_or_none(solver_debug.get("history_cost"))
    history_step = _array_or_none(solver_debug.get("history_step"))
    history_alpha = _array_or_none(solver_debug.get("history_alpha"))

    lengths = [len(arr) for arr in (history_x, history_cost, history_step, history_alpha) if arr is not None]
    if not lengths:
        return []
    n = max(lengths)

    rows: list[dict[str, Any]] = []
    for i in range(n):
        x_hat = y_hat = z_hat = None
        if history_x is not None and i < len(history_x):
            x_hat, y_hat, z_hat = _xyz(history_x[i])
        rows.append(
            {
                "run_id": run_id,
                "iteration": i,
                "cost": _to_float_or_none(history_cost[i]) if history_cost is not None and i < len(history_cost) else None,
                "step_norm": _to_float_or_none(history_step[i]) if history_step is not None and i < len(history_step) else None,
                "x_hat": x_hat,
                "y_hat": y_hat,
                "z_hat": z_hat,
                "alpha": _to_float_or_none(history_alpha[i]) if history_alpha is not None and i < len(history_alpha) else None,
                "notes": None,
            }
        )
    return rows


def write_execution_to_workbook(
    results_path: str | Path,
    run_row: Mapping[str, Any],
    iteration_rows: list[Mapping[str, Any]] | None = None,
    *,
    replace_existing: bool = True,
) -> None:
    """Write one DOE run result and optional iteration rows into Excel."""

    rows_by_sheet: dict[str, list[Mapping[str, Any]]] = {"runs_results": [run_row]}
    if iteration_rows:
        rows_by_sheet["iterations_results"] = iteration_rows

    run_id = str(run_row.get("run_id") or "").strip() or None
    append_output_rows(
        results_path,
        rows_by_sheet,
        replace_run_id=run_id if replace_existing else None,
    )

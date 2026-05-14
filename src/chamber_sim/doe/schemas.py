from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

IssueLevel = Literal["info", "warning", "error"]


@dataclass(frozen=True)
class ColumnSchema:
    """Column expected in a DOE input sheet."""

    name: str
    required: bool = True
    description: str = ""


@dataclass(frozen=True)
class SheetSchema:
    """Description of a DOE input sheet.

    The uploaded workbook uses three descriptive rows, then the header on row 4.
    """

    name: str
    stage: str
    header_row: int = 4
    run_id_column: str = "run_id"
    columns: tuple[ColumnSchema, ...] = field(default_factory=tuple)
    description: str = ""

    @property
    def required_columns(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.columns if c.required)


@dataclass(frozen=True)
class OutputSheetSchema:
    """Description of a DOE output sheet created in the results workbook."""

    name: str
    headers: tuple[str, ...]
    description: str = ""


@dataclass
class ValidationIssue:
    level: IssueLevel
    sheet: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(issue.level == "error" for issue in self.issues)

    @property
    def has_warnings(self) -> bool:
        return any(issue.level == "warning" for issue in self.issues)

    def add(self, level: IssueLevel, sheet: str, message: str, **details: Any) -> None:
        self.issues.append(ValidationIssue(level=level, sheet=sheet, message=message, details=details))

    def extend(self, other: "ValidationReport") -> None:
        self.issues.extend(other.issues)

    def summary_lines(self) -> list[str]:
        if not self.issues:
            return ["Validation OK: aucune erreur détectée."]

        lines: list[str] = []
        for issue in self.issues:
            prefix = {"info": "INFO", "warning": "WARNING", "error": "ERROR"}[issue.level]
            lines.append(f"[{prefix}] {issue.sheet}: {issue.message}")
        return lines


DOE_RUN_SHEETS: dict[str, SheetSchema] = {
    "A_runs_screen": SheetSchema(
        name="A_runs_screen",
        stage="A",
        description="Stage A - screening géométrique.",
        columns=(
            ColumnSchema("run_id"),
            ColumnSchema("stage"),
            ColumnSchema("geom_run_id"),
            ColumnSchema("source_screen_id"),
            ColumnSchema("n_mics"),
            ColumnSchema("n_heights"),
            ColumnSchema("opening_h"),
            ColumnSchema("balance"),
            ColumnSchema("x_norm"),
            ColumnSchema("y_norm"),
            ColumnSchema("range_class"),
            ColumnSchema("src_z_m"),
        ),
    ),
    "A_runs_map": SheetSchema(
        name="A_runs_map",
        stage="A",
        description="Stage A - cartographie complète après screening.",
        columns=(
            ColumnSchema("run_id"),
            ColumnSchema("stage"),
            ColumnSchema("geometry_slot"),
            ColumnSchema("source_id"),
            ColumnSchema("x_norm"),
            ColumnSchema("y_norm"),
            ColumnSchema("range_class"),
            ColumnSchema("src_z_m"),
            ColumnSchema("model_type"),
            ColumnSchema("init_method"),
        ),
    ),
    "B_PB12": SheetSchema(
        name="B_PB12",
        stage="B",
        description="Stage B - robustesse bruit, durée, erreurs de calibration.",
        columns=(
            ColumnSchema("pb_run"),
            ColumnSchema("geometry_slot"),
            ColumnSchema("B1_snr"),
            ColumnSchema("B2_duration_s"),
            ColumnSchema("B3_window_s"),
            ColumnSchema("B4_hop_ratio"),
            ColumnSchema("hop_s", required=False),
            ColumnSchema("B5_trim_frac"),
            ColumnSchema("B6_gain_err_db"),
            ColumnSchema("B7_mic_pos_err_mm"),
            ColumnSchema("B8_c_err_pct"),
            ColumnSchema("stress_anchor"),
            ColumnSchema("commentaire", required=False),
            ColumnSchema("run_id"),
        ),
    ),
    "C1_PB12_common": SheetSchema(
        name="C1_PB12_common",
        stage="C1",
        description="Stage C1 - criblage algorithmique commun.",
        columns=(
            ColumnSchema("pb_run"),
            ColumnSchema("model_type"),
            ColumnSchema("init_method"),
            ColumnSchema("estimate_alpha"),
            ColumnSchema("lam"),
            ColumnSchema("max_iter"),
            ColumnSchema("fd_eps"),
            ColumnSchema("geometry_slot"),
            ColumnSchema("positions_set"),
            ColumnSchema("commentaire", required=False),
            ColumnSchema("run_id"),
        ),
    ),
    "C2_alpha_refine": SheetSchema(
        name="C2_alpha_refine",
        stage="C2",
        description="Stage C2 - raffinement alpha.",
        columns=(
            ColumnSchema("alpha_case"),
            ColumnSchema("model_type"),
            ColumnSchema("init_method"),
            ColumnSchema("estimate_alpha"),
            ColumnSchema("alpha_init"),
            ColumnSchema("alpha_grid_size"),
            ColumnSchema("lam"),
            ColumnSchema("max_iter"),
            ColumnSchema("fd_eps"),
            ColumnSchema("geometry_slot"),
            ColumnSchema("positions_set"),
            ColumnSchema("commentaire", required=False),
            ColumnSchema("run_id"),
        ),
    ),
    "D_validation": SheetSchema(
        name="D_validation",
        stage="D",
        description="Stage D - validation finale chambre.",
        columns=(
            ColumnSchema("run_id"),
            ColumnSchema("geometry_slot"),
            ColumnSchema("point_id"),
            ColumnSchema("x_norm"),
            ColumnSchema("y_norm"),
            ColumnSchema("src_z_m"),
            ColumnSchema("repetition"),
            ColumnSchema("placement_mode"),
            ColumnSchema("calibration_state"),
            ColumnSchema("algo_setting"),
        ),
    ),
}


OUTPUT_SHEETS: dict[str, OutputSheetSchema] = {
    "runs_results": OutputSheetSchema(
        name="runs_results",
        description="Une ligne par run exécuté.",
        headers=(
            "run_id",
            "stage",
            "input_sheet",
            "status",
            "success_flag",
            "started_at",
            "finished_at",
            "time_s",
            "algorithm",
            "geometry_slot",
            "geom_run_id",
            "source_id",
            "x_true",
            "y_true",
            "z_true",
            "x_hat",
            "y_hat",
            "z_hat",
            "err_3d",
            "err_xy",
            "err_z",
            "residual",
            "n_iter",
            "score",
            "error_message",
        ),
    ),
    "iterations_results": OutputSheetSchema(
        name="iterations_results",
        description="Historique d'itérations des solveurs, utile pour la convergence.",
        headers=(
            "run_id",
            "iteration",
            "cost",
            "step_norm",
            "x_hat",
            "y_hat",
            "z_hat",
            "alpha",
            "notes",
        ),
    ),
    "geometry_features": OutputSheetSchema(
        name="geometry_features",
        description="Indicateurs calculés pour chaque géométrie microphonique.",
        headers=(
            "geometry_id",
            "geometry_slot",
            "n_mics",
            "aperture_x",
            "aperture_y",
            "aperture_z",
            "mean_pair_distance",
            "min_pair_distance",
            "condition_metric",
            "notes",
        ),
    ),
    "crb_map": OutputSheetSchema(
        name="crb_map",
        description="Carte CRB par géométrie et position source.",
        headers=(
            "geometry_id",
            "source_id",
            "x_norm",
            "y_norm",
            "src_z_m",
            "crb_x",
            "crb_y",
            "crb_z",
            "crb_trace",
            "fim_cond",
            "notes",
        ),
    ),
    "stage_summary": OutputSheetSchema(
        name="stage_summary",
        description="Synthèse automatique par étape.",
        headers=(
            "stage",
            "input_sheet",
            "n_runs",
            "n_success",
            "success_rate",
            "rmse_3d",
            "median_err_3d",
            "p90_err_3d",
            "mean_time_s",
            "best_run_id",
            "best_score",
            "selected_geometry_or_algo",
            "notes",
        ),
    ),
    "best_candidates": OutputSheetSchema(
        name="best_candidates",
        description="Classement automatique des meilleures géométries et réglages algorithmiques.",
        headers=(
            "stage",
            "input_sheet",
            "candidate_type",
            "candidate_id",
            "rank",
            "n_runs",
            "n_success",
            "success_rate",
            "rmse_3d",
            "median_err_3d",
            "p90_err_3d",
            "mean_time_s",
            "score",
            "notes",
        ),
    ),
    "automation_decisions": OutputSheetSchema(
        name="automation_decisions",
        description="Décisions prises automatiquement pour chaîner les étapes DOE.",
        headers=(
            "decision_key",
            "selected_value",
            "source_stage",
            "score",
            "n_runs",
            "timestamp",
            "notes",
        ),
    ),
    "factor_effects": OutputSheetSchema(
        name="factor_effects",
        description="Effets principaux estimés à partir du plan de robustesse B.",
        headers=(
            "stage",
            "factor",
            "best_level",
            "worst_level",
            "effect_on_score",
            "effect_on_rmse_3d",
            "best_score",
            "worst_score",
            "notes",
        ),
    ),

}

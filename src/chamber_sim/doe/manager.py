from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .excel_io import (
    DOEWorkbookInfo,
    discover_workbook,
    get_existing_run_ids,
    prepare_results_workbook,
    read_header,
    read_sheet_records,
)
from .schemas import DOE_RUN_SHEETS, SheetSchema, ValidationReport
from .config_builder import DOEChamberMapping, DOEConfigBuilder, DOEConfigBuildResult


@dataclass(frozen=True)
class DOESheetPreview:
    sheet_name: str
    stage: str
    n_runs: int
    headers: list[str]
    sample_rows: list[dict[str, Any]]


@dataclass(frozen=True)
class DOEOverviewRow:
    sheet_name: str
    stage: str
    status: str
    n_runs: int
    description: str


class DOEManager:
    """High-level object used by scripts and, later, by the UI.

    Step 1 responsibilities:
    - discover workbook sheets;
    - validate expected DOE run sheets;
    - preview runs;
    - prepare a separate result workbook.
    """

    def __init__(self, workbook_path: str | Path, results_path: str | Path | None = None) -> None:
        self.workbook_path = Path(workbook_path).expanduser().resolve()
        self.results_path = None if results_path is None else Path(results_path).expanduser().resolve()

    def discover(self) -> DOEWorkbookInfo:
        return discover_workbook(self.workbook_path)

    def list_run_sheets(self) -> list[str]:
        info = self.discover()
        available = set(info.sheet_names)
        return [name for name in DOE_RUN_SHEETS if name in available]

    def get_schema(self, sheet_name: str) -> SheetSchema:
        try:
            return DOE_RUN_SHEETS[sheet_name]
        except KeyError as exc:
            known = ", ".join(DOE_RUN_SHEETS)
            raise KeyError(f"Feuille DOE inconnue : {sheet_name}. Feuilles connues : {known}") from exc

    def preview_sheet(self, sheet_name: str, sample_size: int = 5) -> DOESheetPreview:
        schema = self.get_schema(sheet_name)
        headers = read_header(self.workbook_path, schema)
        records = read_sheet_records(self.workbook_path, schema)
        return DOESheetPreview(
            sheet_name=schema.name,
            stage=schema.stage,
            n_runs=len(records),
            headers=[h for h in headers if h],
            sample_rows=records[:sample_size],
        )

    def overview(self) -> list[DOEOverviewRow]:
        info = self.discover()
        available = set(info.sheet_names)
        rows: list[DOEOverviewRow] = []
        for sheet_name, schema in DOE_RUN_SHEETS.items():
            if sheet_name not in available:
                rows.append(
                    DOEOverviewRow(
                        sheet_name=sheet_name,
                        stage=schema.stage,
                        status="missing",
                        n_runs=0,
                        description=schema.description,
                    )
                )
                continue
            n_runs = len(read_sheet_records(self.workbook_path, schema))
            rows.append(
                DOEOverviewRow(
                    sheet_name=sheet_name,
                    stage=schema.stage,
                    status="ok",
                    n_runs=n_runs,
                    description=schema.description,
                )
            )
        return rows

    def validate(self) -> ValidationReport:
        report = ValidationReport()
        info = self.discover()
        available = set(info.sheet_names)
        all_run_ids: dict[str, str] = {}

        for sheet_name, schema in DOE_RUN_SHEETS.items():
            if sheet_name not in available:
                report.add("error", sheet_name, "Feuille DOE attendue absente.")
                continue

            headers = [h for h in read_header(self.workbook_path, schema) if h]
            missing = [col for col in schema.required_columns if col not in headers]
            if missing:
                report.add("error", sheet_name, "Colonnes obligatoires manquantes.", missing=missing)

            records = read_sheet_records(self.workbook_path, schema)
            if not records:
                report.add("warning", sheet_name, "Aucun run détecté dans cette feuille.")
                continue

            if schema.run_id_column not in headers:
                report.add(
                    "error",
                    sheet_name,
                    f"Colonne run_id introuvable : '{schema.run_id_column}'.",
                )
                continue

            local_ids: set[str] = set()
            duplicates_local: set[str] = set()
            duplicates_global: set[str] = set()
            missing_run_id_rows: list[int] = []

            for idx, record in enumerate(records, start=schema.header_row + 1):
                value = record.get(schema.run_id_column)
                if value is None or str(value).strip() == "":
                    missing_run_id_rows.append(idx)
                    continue
                run_id = str(value).strip()
                if run_id in local_ids:
                    duplicates_local.add(run_id)
                local_ids.add(run_id)

                if run_id in all_run_ids and all_run_ids[run_id] != sheet_name:
                    duplicates_global.add(run_id)
                else:
                    all_run_ids[run_id] = sheet_name

            if missing_run_id_rows:
                report.add(
                    "error",
                    sheet_name,
                    "Certaines lignes de runs n'ont pas de run_id.",
                    rows=missing_run_id_rows[:20],
                )
            if duplicates_local:
                report.add(
                    "error",
                    sheet_name,
                    "run_id dupliqués dans la feuille.",
                    run_ids=sorted(duplicates_local),
                )
            if duplicates_global:
                report.add(
                    "warning",
                    sheet_name,
                    "run_id déjà utilisé dans une autre feuille DOE.",
                    run_ids=sorted(duplicates_global),
                )

            report.add("info", sheet_name, f"{len(records)} runs détectés.")

        return report


    def config_builder(
        self,
        chamber: DOEChamberMapping | None = None,
        geometry_slots: dict[str, str] | None = None,
    ) -> DOEConfigBuilder:
        """Return a builder able to convert DOE rows into simulator configs."""

        return DOEConfigBuilder(
            self.workbook_path,
            chamber=chamber,
            geometry_slots=geometry_slots,
        )

    def build_config_by_id(
        self,
        sheet_name: str,
        run_id: str,
        chamber: DOEChamberMapping | None = None,
        geometry_slots: dict[str, str] | None = None,
    ) -> DOEConfigBuildResult:
        """Convert one DOE run into a full simulator configuration."""

        return self.config_builder(chamber=chamber, geometry_slots=geometry_slots).build_config_by_id(sheet_name, run_id)

    def prepare_results(self, overwrite: bool = False) -> Path:
        results = prepare_results_workbook(self.workbook_path, self.results_path, overwrite=overwrite)
        self.results_path = results
        return results

    def existing_result_run_ids(self) -> set[str]:
        if self.results_path is None:
            return set()
        return get_existing_run_ids(self.results_path)

    def runner(
        self,
        chamber: DOEChamberMapping | None = None,
        geometry_slots: dict[str, str] | None = None,
    ):
        """Return a DOE runner connected to this workbook and result file."""

        from .runner import DOERunner

        return DOERunner(
            self.workbook_path,
            self.results_path,
            chamber=chamber,
            geometry_slots=geometry_slots,
        )

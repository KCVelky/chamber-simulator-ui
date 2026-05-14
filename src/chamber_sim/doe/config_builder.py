from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from chamber_sim.experiment_runner import deep_copy_config

from .excel_io import read_sheet_records
from .schemas import ColumnSchema, DOE_RUN_SHEETS, SheetSchema


@dataclass(frozen=True)
class DOEChamberMapping:
    """Physical convention used to convert DOE coded values into simulator inputs.

    The workbook stores microphone coordinates relative to the array center and source
    positions as normalized coordinates. This object centralizes the convention so it
    can be changed later from the UI without rewriting the DOE code.
    """

    Lx: float = 8.0
    Ly: float = 6.0
    Lz: float = 4.0
    margin: float = 0.05
    array_center_x: float = 4.0
    array_center_y: float = 3.0
    source_center_x: float = 4.0
    source_center_y: float = 3.0
    near_span_x: float = 1.25
    near_span_y: float = 0.95
    mid_span_x: float = 2.25
    mid_span_y: float = 1.70
    far_span_x: float = 3.35
    far_span_y: float = 2.45
    baseline_algorithm: str = "MLE_EM_GROUND"
    baseline_model_type: str = "coherent"
    baseline_init_method: str = "er_ls_ground"
    baseline_snr_db: float = 20.0
    baseline_duration_s: float = 6.0
    baseline_window_s: float = 0.50
    baseline_hop_s: float = 0.25
    baseline_trim_frac: float = 0.10
    baseline_source_z_m: float = 0.80

    def span_for_range(self, range_class: Any) -> tuple[float, float]:
        key = str(range_class or "mid").strip().lower()
        if key == "near":
            return self.near_span_x, self.near_span_y
        if key == "far":
            return self.far_span_x, self.far_span_y
        return self.mid_span_x, self.mid_span_y


@dataclass(frozen=True)
class GeometryRealization:
    geometry_id: str
    n_mics: int
    n_heights: str
    opening_h: str
    balance: str
    preset_rule: str
    relative_positions: np.ndarray
    absolute_positions: np.ndarray

    def to_mics_csv(self) -> str:
        return positions_to_csv(self.absolute_positions)


@dataclass(frozen=True)
class DOEConfigBuildResult:
    sheet_name: str
    run_id: str
    geometry_id: str
    source_id: str | None
    config: dict[str, Any]
    source_position: tuple[float, float, float]
    mic_positions: list[tuple[float, float, float]]
    summary: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "sheet_name": self.sheet_name,
            "run_id": self.run_id,
            "geometry_id": self.geometry_id,
            "source_id": self.source_id,
            "source_position": list(self.source_position),
            "mic_positions": [list(row) for row in self.mic_positions],
            "summary": self.summary,
            "config": self.config,
        }


_GEOM_SHEET = SheetSchema(
    name="A_geom_realizations",
    stage="A",
    columns=(
        ColumnSchema("geom_run_id"),
        ColumnSchema("n_mics"),
        ColumnSchema("n_heights"),
        ColumnSchema("opening_h"),
        ColumnSchema("balance"),
        ColumnSchema("preset_rule"),
        ColumnSchema("mic_coordinates_xyz_m"),
    ),
)


def _as_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value).strip()


def _as_float(value: Any, default: float | None = None) -> float:
    if value is None or str(value).strip() == "":
        if default is None:
            raise ValueError("Valeur numérique manquante.")
        return float(default)
    if isinstance(value, str) and value.strip().lower() in {"inf", "infty", "∞", "infinite"}:
        return math.inf
    return float(value)


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None or str(value).strip() == "":
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "oui", "on"}


def _stable_seed(text: str, modulo: int = 2_147_483_647) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % modulo


def parse_mic_coordinates(text: str) -> np.ndarray:
    """Parse '(x,y,z); (x,y,z)' coordinates from the DOE workbook."""

    raw = _as_str(text)
    if not raw:
        raise ValueError("Coordonnées micro absentes dans A_geom_realizations.")

    matches = re.findall(r"\(([^)]*)\)", raw)
    if not matches:
        # Fallback for a simpler 'x,y,z; x,y,z' syntax.
        matches = [part.strip() for part in raw.split(";") if part.strip()]

    rows: list[list[float]] = []
    for item in matches:
        parts = [p.strip() for p in item.split(",")]
        if len(parts) != 3:
            raise ValueError(f"Coordonnée micro invalide : {item!r}")
        rows.append([float(parts[0]), float(parts[1]), float(parts[2])])

    arr = np.asarray(rows, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("Les coordonnées micro doivent former une matrice Mx3.")
    return arr


def positions_to_csv(positions: np.ndarray) -> str:
    arr = np.asarray(positions, dtype=float)
    return "\n".join(f"{x:.6f},{y:.6f},{z:.6f}" for x, y, z in arr)


def _geometry_features(positions: np.ndarray) -> dict[str, float]:
    arr = np.asarray(positions, dtype=float)
    aperture = np.ptp(arr, axis=0)
    if arr.shape[0] <= 1:
        mean_pair = min_pair = 0.0
    else:
        dists = []
        for i in range(arr.shape[0]):
            for j in range(i + 1, arr.shape[0]):
                dists.append(float(np.linalg.norm(arr[i] - arr[j])))
        mean_pair = float(np.mean(dists)) if dists else 0.0
        min_pair = float(np.min(dists)) if dists else 0.0
    return {
        "aperture_x": float(aperture[0]),
        "aperture_y": float(aperture[1]),
        "aperture_z": float(aperture[2]),
        "mean_pair_distance": mean_pair,
        "min_pair_distance": min_pair,
    }


class DOEConfigBuilder:
    """Build simulator configurations from DOE workbook rows.

    Step 2 scope:
    - turn one DOE row into a full ``run_experiment_from_config`` configuration;
    - resolve Stage A explicit geometries;
    - allow placeholder geometry slots such as BEST_A_1 through a mapping.
    """

    def __init__(
        self,
        workbook_path: str | Path,
        chamber: DOEChamberMapping | None = None,
        geometry_slots: Mapping[str, str] | None = None,
        algorithm_slots: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        self.workbook_path = Path(workbook_path).expanduser().resolve()
        self.chamber = chamber or DOEChamberMapping()
        self.geometry_slots = {str(k): str(v) for k, v in (geometry_slots or {}).items()}
        self.algorithm_slots = {str(k): dict(v) for k, v in (algorithm_slots or {}).items()}
        self._geometry_cache: dict[str, GeometryRealization] | None = None

    def geometry_table(self) -> dict[str, GeometryRealization]:
        if self._geometry_cache is not None:
            return self._geometry_cache

        records = read_sheet_records(self.workbook_path, _GEOM_SHEET)
        out: dict[str, GeometryRealization] = {}
        for rec in records:
            geometry_id = _as_str(rec.get("geom_run_id"))
            if not geometry_id:
                continue
            rel = parse_mic_coordinates(_as_str(rec.get("mic_coordinates_xyz_m")))
            abs_pos = rel.copy()
            abs_pos[:, 0] += self.chamber.array_center_x
            abs_pos[:, 1] += self.chamber.array_center_y
            out[geometry_id] = GeometryRealization(
                geometry_id=geometry_id,
                n_mics=int(_as_float(rec.get("n_mics"), rel.shape[0])),
                n_heights=_as_str(rec.get("n_heights")),
                opening_h=_as_str(rec.get("opening_h")),
                balance=_as_str(rec.get("balance")),
                preset_rule=_as_str(rec.get("preset_rule")),
                relative_positions=rel,
                absolute_positions=abs_pos,
            )

        if not out:
            raise ValueError("Aucune géométrie trouvée dans A_geom_realizations.")
        self._geometry_cache = out
        return out

    def records_for_sheet(self, sheet_name: str) -> list[dict[str, Any]]:
        if sheet_name not in DOE_RUN_SHEETS:
            raise KeyError(f"Feuille DOE inconnue : {sheet_name}")
        return read_sheet_records(self.workbook_path, DOE_RUN_SHEETS[sheet_name])

    def get_record(self, sheet_name: str, run_id: str | None = None, row_index: int | None = None) -> dict[str, Any]:
        records = self.records_for_sheet(sheet_name)
        if not records:
            raise ValueError(f"Aucun run trouvé dans {sheet_name}.")
        if run_id is None and row_index is None:
            return records[0]
        if row_index is not None:
            if row_index < 1 or row_index > len(records):
                raise IndexError(f"row_index doit être entre 1 et {len(records)}.")
            return records[row_index - 1]

        target = _as_str(run_id)
        for rec in records:
            if self.extract_run_id(sheet_name, rec) == target:
                return rec
        raise KeyError(f"run_id introuvable dans {sheet_name} : {target}")

    def extract_run_id(self, sheet_name: str, record: Mapping[str, Any]) -> str:
        schema = DOE_RUN_SHEETS[sheet_name]
        value = record.get(schema.run_id_column)
        if value is None:
            # Fallback for sheets where the practical identifier has another name.
            for key in ("run_id", "pb_run", "alpha_case"):
                if record.get(key) is not None:
                    value = record.get(key)
                    break
        run_id = _as_str(value)
        if not run_id:
            raise ValueError(f"Impossible de déterminer le run_id pour {sheet_name}.")
        return run_id

    def resolve_geometry_id(self, record: Mapping[str, Any]) -> str:
        direct = _as_str(record.get("geom_run_id"))
        if direct:
            return direct

        slot = _as_str(record.get("geometry_slot"))
        if slot:
            if slot in self.geometry_slots:
                return self.geometry_slots[slot]
            if slot.startswith("A-G"):
                return slot
            known = ", ".join(f"{k}={v}" for k, v in sorted(self.geometry_slots.items())) or "aucun"
            raise ValueError(
                f"Le placeholder de géométrie '{slot}' n'est pas encore résolu. "
                f"Passe un mapping, par exemple --geometry-slot {slot}=A-G01. "
                f"Mappings actuels : {known}."
            )

        raise ValueError("Aucune géométrie DOE trouvée dans la ligne (geom_run_id ou geometry_slot).")

    def resolve_geometry(self, record: Mapping[str, Any]) -> GeometryRealization:
        geometry_id = self.resolve_geometry_id(record)
        table = self.geometry_table()
        if geometry_id not in table:
            raise KeyError(f"Géométrie introuvable dans A_geom_realizations : {geometry_id}")
        return table[geometry_id]

    def source_position_from_record(self, record: Mapping[str, Any]) -> tuple[float, float, float]:
        x_norm = _as_float(record.get("x_norm"), 0.0)
        y_norm = _as_float(record.get("y_norm"), 0.0)
        range_class = record.get("range_class", "mid")
        span_x, span_y = self.chamber.span_for_range(range_class)

        x = self.chamber.source_center_x + x_norm * span_x
        y = self.chamber.source_center_y + y_norm * span_y
        z = _as_float(record.get("src_z_m"), self.chamber.baseline_source_z_m)

        # Keep the generated source safely inside the simulated room.
        m = self.chamber.margin
        x = min(max(x, m), self.chamber.Lx - m)
        y = min(max(y, m), self.chamber.Ly - m)
        z = min(max(z, m), self.chamber.Lz - m)
        return float(x), float(y), float(z)

    def build_config(self, sheet_name: str, record: Mapping[str, Any]) -> DOEConfigBuildResult:
        run_id = self.extract_run_id(sheet_name, record)
        geometry = self.resolve_geometry(record)
        source_xyz = self.source_position_from_record(record)
        source_id = (
            _as_str(record.get("source_screen_id"))
            or _as_str(record.get("source_id"))
            or _as_str(record.get("point_id"))
            or None
        )

        nominal_mics = np.asarray(geometry.absolute_positions, dtype=float)
        true_mics, algorithm_mics, perturbation_mm = self._build_true_and_algorithm_mics(
            nominal_mics, sheet_name, record, run_id
        )

        cfg = deep_copy_config()
        cfg["geometry"].update(
            {
                "Lx": str(self.chamber.Lx),
                "Ly": str(self.chamber.Ly),
                "Lz": str(self.chamber.Lz),
                "margin": str(self.chamber.margin),
                "source": {"x": f"{source_xyz[0]:.6f}", "y": f"{source_xyz[1]:.6f}", "z": f"{source_xyz[2]:.6f}"},
                "mics_csv": positions_to_csv(true_mics),
                "algorithm_mics_csv": positions_to_csv(algorithm_mics),
            }
        )

        self._apply_stage_defaults(cfg, sheet_name, record, run_id)
        self._apply_algorithm_slot(cfg, record)
        self._apply_gain_errors(cfg, record, run_id, true_mics.shape[0])
        self._disable_plots_for_doe(cfg)

        features = _geometry_features(algorithm_mics)
        cfg["doe"] = {
            "sheet_name": sheet_name,
            "run_id": run_id,
            "stage": DOE_RUN_SHEETS[sheet_name].stage,
            "geometry_id": geometry.geometry_id,
            "source_id": source_id,
            "source_position_m": list(source_xyz),
            "geometry_features": features,
            "mic_position_perturbation_mm": perturbation_mm,
            "raw_record": dict(record),
        }

        summary = {
            "stage": DOE_RUN_SHEETS[sheet_name].stage,
            "algorithm": cfg["algorithms"].get("alg_choice"),
            "model_type": cfg["algorithms"].get("mleem_model_type"),
            "init_method": cfg["algorithms"].get("mleem_init_method"),
            "n_mics": geometry.n_mics,
            "n_heights": geometry.n_heights,
            "opening_h": geometry.opening_h,
            "balance": geometry.balance,
            "snr_db": cfg["propagation"].get("snr_db"),
            "add_noise": cfg["propagation"].get("add_noise"),
            "duration_s": cfg["signal"].get("duration"),
            "window_s": cfg["algorithms"].get("er_window_s"),
            "hop_s": cfg["algorithms"].get("er_hop_s"),
            "mic_position_perturbation_mm": perturbation_mm,
            "mic_gain_errors_db": cfg["propagation"].get("mic_gain_errors_db"),
            **features,
        }

        return DOEConfigBuildResult(
            sheet_name=sheet_name,
            run_id=run_id,
            geometry_id=geometry.geometry_id,
            source_id=source_id,
            config=cfg,
            source_position=source_xyz,
            mic_positions=[tuple(map(float, row)) for row in geometry.absolute_positions.tolist()],
            summary=summary,
        )

    def _build_true_and_algorithm_mics(
        self,
        nominal_mics: np.ndarray,
        sheet_name: str,
        record: Mapping[str, Any],
        run_id: str,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Return physical and algorithm microphone coordinates for DOE perturbations.

        The simulator propagates sound to ``mics_csv``. Algorithms localize with
        ``algorithm_mics_csv``. This lets Stage B/D model calibration errors: the
        physical array can be slightly perturbed while the estimator still uses the
        nominal geometry, unless the validation row says calibration is ``after``.
        """

        stage = DOE_RUN_SHEETS[sheet_name].stage
        perturbation_mm = 0.0
        calibration_state = _as_str(record.get("calibration_state"), "before").lower()

        if stage == "B":
            perturbation_mm = _as_float(record.get("B7_mic_pos_err_mm"), 0.0)
        elif stage == "D" and _as_str(record.get("placement_mode"), "nominal").lower() == "perturbed":
            # Validation uses a deterministic representative placement perturbation.
            perturbation_mm = 5.0

        true_mics = np.asarray(nominal_mics, dtype=float).copy()
        if perturbation_mm > 0.0:
            rng = np.random.default_rng(_stable_seed(run_id + ":mic_positions"))
            true_mics = true_mics + rng.normal(0.0, perturbation_mm / 1000.0, size=true_mics.shape)

        if stage == "D" and calibration_state == "after":
            algorithm_mics = true_mics.copy()
        else:
            algorithm_mics = np.asarray(nominal_mics, dtype=float).copy()
        return true_mics, algorithm_mics, float(perturbation_mm)

    def _apply_gain_errors(self, cfg: dict[str, Any], record: Mapping[str, Any], run_id: str, n_mics: int) -> None:
        gain_err_db = _as_float(record.get("B6_gain_err_db"), 0.0)
        if gain_err_db <= 0.0:
            cfg["propagation"].pop("mic_gain_errors_db", None)
            return
        rng = np.random.default_rng(_stable_seed(run_id + ":mic_gains"))
        errors = rng.normal(0.0, gain_err_db, size=int(n_mics))
        cfg["propagation"]["mic_gain_errors_db"] = ",".join(f"{v:.6f}" for v in errors)

    def _apply_algorithm_slot(self, cfg: dict[str, Any], record: Mapping[str, Any]) -> None:
        slot = _as_str(record.get("algo_setting"))
        if not slot:
            return
        settings = self.algorithm_slots.get(slot)
        if not settings:
            return
        for key, value in settings.items():
            cfg["algorithms"][str(key)] = value

    def _apply_stage_defaults(self, cfg: dict[str, Any], sheet_name: str, record: Mapping[str, Any], run_id: str) -> None:
        stage = DOE_RUN_SHEETS[sheet_name].stage

        # Deterministic seeds make DOE runs reproducible while still varying run to run.
        cfg["signal"]["seed"] = str(_stable_seed(run_id + ":signal"))
        cfg["propagation"]["noise_seed"] = str(_stable_seed(run_id + ":noise"))

        cfg["signal"]["duration"] = str(self.chamber.baseline_duration_s)
        cfg["propagation"].update({"add_noise": True, "snr_db": str(self.chamber.baseline_snr_db)})
        cfg["algorithms"].update(
            {
                "enable_algorithms": True,
                "alg_choice": self.chamber.baseline_algorithm,
                "er_window_s": str(self.chamber.baseline_window_s),
                "er_hop_s": str(self.chamber.baseline_hop_s),
                "er_trim_frac": str(self.chamber.baseline_trim_frac),
                "mleem_model_type": self.chamber.baseline_model_type,
                "mleem_init_method": self.chamber.baseline_init_method,
            }
        )

        # Stage A map already carries algorithm choices. Stage C does too.
        model_type = record.get("model_type")
        init_method = record.get("init_method")
        if model_type is not None:
            cfg["algorithms"]["mleem_model_type"] = _as_str(model_type)
        if init_method is not None:
            cfg["algorithms"]["mleem_init_method"] = _as_str(init_method)

        if stage == "B":
            snr = _as_float(record.get("B1_snr"), self.chamber.baseline_snr_db)
            if math.isinf(snr):
                cfg["propagation"]["add_noise"] = False
            else:
                cfg["propagation"].update({"add_noise": True, "snr_db": str(snr)})
            duration = _as_float(record.get("B2_duration_s"), self.chamber.baseline_duration_s)
            window_s = _as_float(record.get("B3_window_s"), self.chamber.baseline_window_s)
            hop_s = record.get("hop_s")
            if hop_s is None:
                hop_ratio = _as_float(record.get("B4_hop_ratio"), 0.5)
                hop_s = window_s * hop_ratio
            trim_frac = _as_float(record.get("B5_trim_frac"), self.chamber.baseline_trim_frac)
            c_err_pct = _as_float(record.get("B8_c_err_pct"), 0.0)

            cfg["signal"]["duration"] = str(duration)
            cfg["algorithms"]["er_window_s"] = str(window_s)
            cfg["algorithms"]["er_hop_s"] = str(_as_float(hop_s, self.chamber.baseline_hop_s))
            cfg["algorithms"]["er_trim_frac"] = str(trim_frac)
            cfg["propagation"]["c"] = str(343.0 * (1.0 + c_err_pct / 100.0))

        if stage in {"C1", "C2"}:
            cfg["algorithms"].update(
                {
                    "mleem_model_type": _as_str(record.get("model_type"), self.chamber.baseline_model_type),
                    "mleem_init_method": _as_str(record.get("init_method"), self.chamber.baseline_init_method),
                    "mleem_estimate_alpha": _as_bool(record.get("estimate_alpha"), False),
                    "mleem_lam": str(_as_float(record.get("lam"), 0.01)),
                    "mleem_max_iter": str(int(_as_float(record.get("max_iter"), 40))),
                    "mleem_fd_eps": str(_as_float(record.get("fd_eps"), 1e-5)),
                }
            )
            if record.get("alpha_init") is not None:
                cfg["algorithms"]["mleem_alpha_init"] = str(_as_float(record.get("alpha_init"), 1.0))
            if record.get("alpha_grid_size") is not None:
                cfg["algorithms"]["mleem_alpha_grid_size"] = str(int(_as_float(record.get("alpha_grid_size"), 31)))

        if stage == "D":
            # The final validation may later resolve BEST_GLOBAL algorithm settings.
            # For now, use the current baseline unless the row is expanded in future steps.
            cfg["signal"]["seed"] = str(_stable_seed(run_id + ":signal:" + _as_str(record.get("repetition"))))
            cfg["propagation"]["noise_seed"] = str(_stable_seed(run_id + ":noise:" + _as_str(record.get("calibration_state"))))

    @staticmethod
    def _disable_plots_for_doe(cfg: dict[str, Any]) -> None:
        cfg["plots"].update(
            {
                "plot_scene": False,
                "plot_source_time": False,
                "plot_source_spectrum": False,
                "plot_mic_spectrogram": False,
                "plot_mics": False,
                "plot_mics_zoom": False,
                "plot_source_spectrogram": False,
                "plot_denoise_compare": False,
                "print_delta_r": False,
            }
        )
        cfg["algorithms"]["plot_estimated_source"] = False

    def build_config_by_id(self, sheet_name: str, run_id: str) -> DOEConfigBuildResult:
        return self.build_config(sheet_name, self.get_record(sheet_name, run_id=run_id))

    def build_config_by_row(self, sheet_name: str, row_index: int = 1) -> DOEConfigBuildResult:
        return self.build_config(sheet_name, self.get_record(sheet_name, row_index=row_index))

    @staticmethod
    def save_config_json(result: DOEConfigBuildResult, output_path: str | Path) -> Path:
        path = Path(output_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(result.to_json_dict(), f, ensure_ascii=False, indent=2)
        return path

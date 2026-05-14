from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chamber_sim.doe.config_builder import DOEConfigBuilder
from chamber_sim.experiment_runner import run_experiment_from_config


def _parse_geometry_slots(values: list[str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in values or []:
        if "=" not in item:
            raise ValueError(f"Mapping invalide : {item!r}. Format attendu : SLOT=A-G01")
        key, value = item.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a simulator config from one DOE row.")
    parser.add_argument("workbook", help="Chemin vers DOE_MAITRISE.xlsx")
    parser.add_argument("--sheet", default="A_runs_screen", help="Feuille DOE à lire")
    parser.add_argument("--run-id", default=None, help="run_id à convertir")
    parser.add_argument("--row", type=int, default=None, help="Index de run dans la feuille, 1 = premier run")
    parser.add_argument(
        "--geometry-slot",
        action="append",
        default=[],
        help="Résolution d'un placeholder, ex. BEST_A_1=A-G01. Peut être répété.",
    )
    parser.add_argument("--output", default=None, help="Chemin JSON de sortie")
    parser.add_argument("--run-once", action="store_true", help="Optionnel : lance ce run unique pour vérifier l'adaptateur")
    args = parser.parse_args()

    slots = _parse_geometry_slots(args.geometry_slot)
    builder = DOEConfigBuilder(args.workbook, geometry_slots=slots)

    if args.run_id:
        result = builder.build_config_by_id(args.sheet, args.run_id)
    else:
        result = builder.build_config_by_row(args.sheet, args.row or 1)

    print("=" * 72)
    print("DOE -> configuration simulateur")
    print("=" * 72)
    print(f"Feuille       : {result.sheet_name}")
    print(f"Run ID        : {result.run_id}")
    print(f"Géométrie     : {result.geometry_id}")
    print(f"Source ID     : {result.source_id}")
    print(f"Source [m]    : {tuple(round(v, 4) for v in result.source_position)}")
    print(f"Micros        : {len(result.mic_positions)}")
    print(f"Algorithme    : {result.summary.get('algorithm')}")
    print(f"Modèle MLE/EM : {result.summary.get('model_type')}")
    print(f"Init MLE/EM   : {result.summary.get('init_method')}")
    print(f"Bruit         : add_noise={result.summary.get('add_noise')} | snr_db={result.summary.get('snr_db')}")
    print(f"Durée         : {result.summary.get('duration_s')} s")
    print("Aperçu micros :")
    for idx, pos in enumerate(result.mic_positions[:8], start=1):
        print(f"  M{idx:02d}: ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")
    if len(result.mic_positions) > 8:
        print(f"  ... {len(result.mic_positions) - 8} micros supplémentaires")

    output = args.output
    if output is None:
        output = str(Path(args.workbook).with_name(f"{result.run_id}_config.json"))
    output_path = builder.save_config_json(result, output)
    print(f"\nConfig JSON écrite : {output_path}")

    if args.run_once:
        print("\nLancement d'un run unique de test...")
        logs: list[str] = []
        exp = run_experiment_from_config(result.config, log=logs.append, show_plots=False)
        print("Run terminé.")
        print(json.dumps(exp.as_dict(), ensure_ascii=False, indent=2))
        print("\nLogs simulation :")
        for line in logs[-12:]:
            print(f"  {line}")

    print("\nRésultat : conversion DOE -> config OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

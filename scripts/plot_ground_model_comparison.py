#!/usr/bin/env python3
"""
Visualisation comparative : modèle champ libre vs modèle avec sol.

Objectif
--------
Générer une scène où le sol existe physiquement dans les mesures simulées,
puis comparer deux localisations réalisées avec la même famille de microphones :

1) ER-LS classique : inversion avec modèle champ libre, donc sans prise en compte du sol.
2) MLE-EM sol : inversion avec modèle direct + source image, donc cohérent avec la scène.

Le script produit une figure PNG avec :
- microphones,
- source réelle,
- source image,
- estimation ER-LS champ libre,
- estimation MLE-EM avec sol,
- plan du sol,
- segments d'erreur entre la vraie source et les estimations.

Lancement depuis la racine du projet :
    python scripts/plot_ground_model_comparison.py

Exemples :
    python scripts/plot_ground_model_comparison.py --mic-family one_sided --output outputs/ground_comparison.png
    python scripts/plot_ground_model_comparison.py --mic-family sphere --source 4 6 1.2 --snr-db 35
    python scripts/plot_ground_model_comparison.py --mic-family three_arrays --estimate-alpha
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Permet de lancer le script sans installer le package en editable.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chamber_sim.scene import (  # noqa: E402
    Scene,
    Source,
    MicArray,
    as_vec3,
    make_semi_anechoic_room,
)
from chamber_sim.signals.source_signal import (  # noqa: E402
    BandpassSpec,
    FadeSpec,
    SourceSignalConfig,
    generate_source_signal,
)
from chamber_sim.propagation.free_field import (  # noqa: E402
    PropagationConfig,
    simulate_mic_signals_free_field,
)
from chamber_sim.noise.add_noise import NoiseConfig, add_awgn_snr  # noqa: E402
from chamber_sim.er_ls.config import EnergyConfig, ERLSConfig  # noqa: E402
from chamber_sim.er_ls.er_ls import er_ls  # noqa: E402
from chamber_sim.mle_em import (  # noqa: E402
    GroundGeometryConfig,
    MLEEMGroundConfig,
    MLEEMInitConfig,
    MLEEMDebugFlags,
    image_source,
    mle_em_ground,
)


# -----------------------------------------------------------------------------
# Géométries micro : familles simples pour démonstration pédagogique
# -----------------------------------------------------------------------------

def fibonacci_sphere(center: np.ndarray, radius: float, n_points: int) -> np.ndarray:
    """Points quasi-uniformes sur une sphère."""
    center = np.asarray(center, dtype=float).reshape(3)
    idx = np.arange(n_points, dtype=float)
    phi = np.pi * (3.0 - np.sqrt(5.0))
    z = 1.0 - 2.0 * (idx + 0.5) / n_points
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    theta = phi * idx
    pts = np.column_stack([np.cos(theta) * r, np.sin(theta) * r, z])
    return center[None, :] + radius * pts


def build_mic_family(name: str, source: np.ndarray) -> np.ndarray:
    """
    Construit une famille de micros.

    Les géométries sont volontairement simples et lisibles :
    - sphere      : micros autour de la source, quasi-uniformes ;
    - one_sided   : micros d'un seul côté, sur plusieurs hauteurs ;
    - three_arrays: trois petites antennes sphériques de 10 micros.
    """
    src = np.asarray(source, dtype=float).reshape(3)

    if name == "sphere":
        # Même famille de microphones autour de la source.
        # On force les points trop bas à rester au-dessus du sol pour garder une scène physique.
        mics = fibonacci_sphere(center=src + np.array([0.0, 0.0, 0.15]), radius=2.1, n_points=24)
        mics[:, 2] = np.maximum(mics[:, 2], 0.25)
        return mics

    if name == "one_sided":
        # Géométrie très utile pour faire apparaître le biais vertical/horizontal :
        # tous les micros voient le sol, mais avec des trajets réfléchis différents.
        offsets = np.array([
            [-2.4, -1.6, -0.85],
            [-2.1, -0.6, -0.45],
            [-2.3,  0.6,  0.10],
            [-2.0,  1.6,  0.55],
            [-3.2, -1.2,  0.35],
            [-3.0,  0.0,  0.85],
            [-3.3,  1.2,  1.30],
            [-1.5, -1.9,  0.65],
            [-1.3, -0.3,  1.15],
            [-1.6,  1.7,  1.65],
            [-2.7, -2.2,  1.55],
            [-2.9,  2.1,  0.95],
        ], dtype=float)
        mics = src[None, :] + offsets
        mics[:, 2] = np.maximum(mics[:, 2], 0.25)
        return mics

    if name == "three_arrays":
        centers = np.array([
            src + np.array([-2.8, -1.7, 0.15]),
            src + np.array([-2.9,  1.7, 0.35]),
            src + np.array([-1.6,  0.0, 1.20]),
        ])
        clouds = []
        for k, c in enumerate(centers):
            cloud = fibonacci_sphere(center=c, radius=0.55, n_points=10)
            cloud[:, 2] = np.maximum(cloud[:, 2], 0.25)
            clouds.append(cloud)
        return np.vstack(clouds)

    raise ValueError(f"Famille de microphones inconnue : {name}")


# -----------------------------------------------------------------------------
# Simulation + estimation
# -----------------------------------------------------------------------------

def compute_errors(true_source: np.ndarray, estimates: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    x_true = np.asarray(true_source, dtype=float).reshape(3)
    for name, x_hat in estimates.items():
        x_hat = np.asarray(x_hat, dtype=float).reshape(3)
        diff = x_hat - x_true
        out[name] = {
            "error_3d_m": float(np.linalg.norm(diff)),
            "error_xy_m": float(np.linalg.norm(diff[:2])),
            "error_z_m": float(abs(diff[2])),
        }
    return out


def run_comparison(args: argparse.Namespace) -> Tuple[dict, dict]:
    source_true = np.asarray(args.source, dtype=float).reshape(3)
    room = make_semi_anechoic_room(args.room[0], args.room[1], args.room[2], floor_rigid=True)
    mic_positions = build_mic_family(args.mic_family, source_true)

    scene = Scene(
        room=room,
        mic_array=MicArray(mic_positions, name=args.mic_family),
        sources=[Source(position=source_true, name="Source réelle")],
    )

    # Source large bande basse fréquence : le modèle MLE-EM cohérent utilise cette bande.
    sig_cfg = SourceSignalConfig(
        fs=float(args.fs),
        duration=float(args.duration),
        band=BandpassSpec(float(args.f_low), float(args.f_high)),
        rms=1.0,
        fade=FadeSpec(fade_in=0.03, fade_out=0.03),
        seed=int(args.seed),
    )
    t, s = generate_source_signal(sig_cfg)

    # Important : ici le SOL EXISTE dans les mesures simulées.
    # Le trajet image est donc réellement injecté dans les signaux micro.
    prop_cfg = PropagationConfig(
        c=float(args.sound_speed),
        include_spherical_spreading=True,
        gain_at_1m=1.0,
        include_rigid_floor_image=True,
        floor_z=float(args.floor_z),
    )
    y, direct_distances = simulate_mic_signals_free_field(scene, t, s, prop_cfg)

    if args.snr_db is not None:
        y, noise_rms = add_awgn_snr(
            y,
            NoiseConfig(snr_db=float(args.snr_db), per_mic_independent=True, seed=int(args.seed) + 1),
        )
    else:
        noise_rms = 0.0

    energy_cfg = EnergyConfig(remove_mean=True, window_s=None, trim_frac=0.10)

    # Méthode A : ER-LS classique, volontairement incohérente avec la scène.
    # Elle suppose que les énergies suivent seulement 1/r^2.
    x_erls, erls_residual_rms, dbg_erls = er_ls(
        mic_positions=mic_positions,
        y=y,
        fs=float(args.fs),
        cfg_E=energy_cfg,
        cfg=ERLSConfig(ref_idx=int(args.ref_idx), kappa_eps=1e-3, min_pairs=3),
    )

    # Méthode B : MLE-EM avec sol, cohérente avec les mesures.
    gcfg = GroundGeometryConfig(
        floor_z=float(args.floor_z),
        enforce_z_positive=True,
        z_margin=1e-5,
    )
    mle_cfg = MLEEMGroundConfig(
        model_type=args.ground_model,
        max_iter=int(args.max_iter),
        estimate_alpha=bool(args.estimate_alpha),
        alpha_init=float(args.alpha),
        alpha_min=0.0,
        alpha_max=1.5,
        alpha_grid_size=31,
        f_low_hz=float(args.f_low),
        f_high_hz=float(args.f_high),
        sound_speed=float(args.sound_speed),
        estimate_noise_floor=False,
    )
    init_cfg = MLEEMInitConfig(method=args.init_method, barycenter_z_offset=0.50)

    # On démarre depuis ER-LS pour montrer que le MLE peut corriger le modèle physique.
    # Si l'ER-LS tombe sous le sol, le clamp interne impose z > floor_z.
    x0_for_mle = x_erls.copy()
    x0_for_mle[2] = max(x0_for_mle[2], float(args.floor_z) + 0.15)

    x_mle, mle_residual_rms, dbg_mle = mle_em_ground(
        mic_positions=mic_positions,
        y=y,
        fs=float(args.fs),
        x0=x0_for_mle if args.use_erls_init else None,
        alpha=float(args.alpha),
        cfg_E=energy_cfg,
        cfg_init=init_cfg,
        cfg=mle_cfg,
        gcfg=gcfg,
        dbg_flags=MLEEMDebugFlags(keep_alpha_grid_details=False, keep_iteration_details=True),
    )

    src_img = image_source(source_true, gcfg)
    estimates = {
        "ER-LS champ libre": x_erls,
        "MLE-EM avec sol": x_mle,
    }
    errors = compute_errors(source_true, estimates)

    results = {
        "source_true": source_true,
        "source_image": src_img,
        "mic_positions": mic_positions,
        "direct_distances": direct_distances,
        "x_erls_free_field": x_erls,
        "x_mle_ground": x_mle,
        "errors": errors,
        "residual_rms": {
            "ER-LS champ libre": float(erls_residual_rms),
            "MLE-EM avec sol": float(mle_residual_rms),
        },
        "alpha_hat_mle": float(dbg_mle.get("alpha_hat", args.alpha)),
        "source_energy_hat_mle": float(dbg_mle.get("K_hat", np.nan)),
        "noise_rms": float(noise_rms),
        "energy_measured": np.asarray(dbg_mle.get("E", dbg_erls.get("E", [])), dtype=float),
    }

    context = {
        "scene": scene,
        "args": args,
        "debug_erls": dbg_erls,
        "debug_mle": dbg_mle,
    }
    return results, context


# -----------------------------------------------------------------------------
# Visualisation
# -----------------------------------------------------------------------------

def set_axes_equal(ax) -> None:
    """Ratio égal sur les axes 3D."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)
    radius = 0.5 * max(x_range, y_range, z_range)
    ax.set_xlim3d([x_middle - radius, x_middle + radius])
    ax.set_ylim3d([y_middle - radius, y_middle + radius])
    ax.set_zlim3d([z_middle - radius, z_middle + radius])


def add_ground_plane(ax, xlim, ylim, floor_z: float) -> None:
    verts = [[
        (xlim[0], ylim[0], floor_z),
        (xlim[1], ylim[0], floor_z),
        (xlim[1], ylim[1], floor_z),
        (xlim[0], ylim[1], floor_z),
    ]]
    plane = Poly3DCollection(verts, alpha=0.18)
    ax.add_collection3d(plane)


def plot_results(results: dict, args: argparse.Namespace, output: Path) -> None:
    src = np.asarray(results["source_true"], dtype=float)
    src_img = np.asarray(results["source_image"], dtype=float)
    mics = np.asarray(results["mic_positions"], dtype=float)
    x_erls = np.asarray(results["x_erls_free_field"], dtype=float)
    x_mle = np.asarray(results["x_mle_ground"], dtype=float)

    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.45, 1.0], height_ratios=[1.0, 0.72])

    # Vue 3D principale
    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax3d.set_title("Effet du sol : même mesure, deux modèles d'inversion")

    # Limites utiles autour de la scène
    all_pts = np.vstack([mics, src[None, :], src_img[None, :], x_erls[None, :], x_mle[None, :]])
    pad = 0.8
    xlim = [float(np.min(all_pts[:, 0]) - pad), float(np.max(all_pts[:, 0]) + pad)]
    ylim = [float(np.min(all_pts[:, 1]) - pad), float(np.max(all_pts[:, 1]) + pad)]
    zlim = [min(float(args.floor_z) - 0.35, float(np.min(all_pts[:, 2]) - pad)), float(np.max(all_pts[:, 2]) + pad)]

    add_ground_plane(ax3d, xlim, ylim, floor_z=float(args.floor_z))

    ax3d.scatter(mics[:, 0], mics[:, 1], mics[:, 2], s=45, label="Micros")
    ax3d.scatter(src[0], src[1], src[2], marker="*", s=220, label="Source réelle")
    ax3d.scatter(src_img[0], src_img[1], src_img[2], marker="*", s=120, alpha=0.35, label="Source image")
    ax3d.scatter(x_erls[0], x_erls[1], x_erls[2], marker="X", s=150, label="ER-LS champ libre")
    ax3d.scatter(x_mle[0], x_mle[1], x_mle[2], marker="D", s=130, label="MLE-EM avec sol")

    # Segments d'erreur
    ax3d.plot([src[0], x_erls[0]], [src[1], x_erls[1]], [src[2], x_erls[2]], linestyle="--", linewidth=2)
    ax3d.plot([src[0], x_mle[0]], [src[1], x_mle[1]], [src[2], x_mle[2]], linestyle="--", linewidth=2)

    # Quelques rayons directs et réfléchis pour montrer la physique
    for mic in mics[:: max(1, len(mics) // 8)]:
        ax3d.plot([src[0], mic[0]], [src[1], mic[1]], [src[2], mic[2]], linewidth=0.8, alpha=0.35)
        ax3d.plot([src_img[0], mic[0]], [src_img[1], mic[1]], [src_img[2], mic[2]], linewidth=0.8, alpha=0.18)

    ax3d.text(src[0], src[1], src[2], "  vraie source")
    ax3d.text(x_erls[0], x_erls[1], x_erls[2], "  ER-LS")
    ax3d.text(x_mle[0], x_mle[1], x_mle[2], "  MLE-EM")
    ax3d.text(src_img[0], src_img[1], src_img[2], "  image", alpha=0.65)

    ax3d.set_xlabel("x [m]")
    ax3d.set_ylabel("y [m]")
    ax3d.set_zlabel("z [m]")
    ax3d.set_xlim(xlim)
    ax3d.set_ylim(ylim)
    ax3d.set_zlim(zlim)
    ax3d.view_init(elev=22, azim=-55)
    set_axes_equal(ax3d)
    ax3d.legend(loc="upper left")

    # Erreurs
    ax_err = fig.add_subplot(gs[0, 1])
    methods = ["ER-LS champ libre", "MLE-EM avec sol"]
    err3d = [results["errors"][m]["error_3d_m"] for m in methods]
    errxy = [results["errors"][m]["error_xy_m"] for m in methods]
    errz = [results["errors"][m]["error_z_m"] for m in methods]
    x = np.arange(len(methods))
    width = 0.24
    ax_err.bar(x - width, err3d, width, label="erreur 3D")
    ax_err.bar(x, errxy, width, label="erreur xy")
    ax_err.bar(x + width, errz, width, label="erreur z")
    ax_err.set_xticks(x)
    ax_err.set_xticklabels(["ER-LS\nchamp libre", "MLE-EM\navec sol"])
    ax_err.set_ylabel("Erreur [m]")
    ax_err.set_title("Erreur de localisation")
    ax_err.grid(True, axis="y", alpha=0.3)
    ax_err.legend()

    # Bloc texte / tableau minimal
    ax_txt = fig.add_subplot(gs[1, 1])
    ax_txt.axis("off")
    erls = results["errors"]["ER-LS champ libre"]
    mle = results["errors"]["MLE-EM avec sol"]
    txt = (
        "Lecture de la figure\n"
        "---------------------\n"
        "Le sol est présent dans la simulation des mesures.\n"
        "ER-LS utilise un modèle 1/r² champ libre : il explique\n"
        "une partie de l'effet du sol comme une fausse distance.\n"
        "MLE-EM utilise direct + source image : le modèle inverse\n"
        "est cohérent avec la physique simulée.\n\n"
        f"Famille micro : {args.mic_family}\n"
        f"Modèle sol MLE : {args.ground_model}, alpha_hat = {results['alpha_hat_mle']:.3f}\n"
        f"Source réelle : [{src[0]:.2f}, {src[1]:.2f}, {src[2]:.2f}] m\n"
        f"ER-LS : [{x_erls[0]:.2f}, {x_erls[1]:.2f}, {x_erls[2]:.2f}] m | erreur 3D = {erls['error_3d_m']:.3f} m\n"
        f"MLE-EM : [{x_mle[0]:.2f}, {x_mle[1]:.2f}, {x_mle[2]:.2f}] m | erreur 3D = {mle['error_3d_m']:.3f} m\n"
    )
    ax_txt.text(0.0, 1.0, txt, va="top", ha="left", family="monospace", fontsize=10)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=int(args.dpi), bbox_inches="tight")
    plt.close(fig)


def save_json_summary(results: dict, output_png: Path) -> Path:
    summary_path = output_png.with_suffix(".json")
    serializable = {}
    for k, v in results.items():
        if isinstance(v, np.ndarray):
            serializable[k] = v.tolist()
        elif isinstance(v, dict):
            serializable[k] = {
                kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv)
                for kk, vv in v.items()
            }
        else:
            serializable[k] = v
    summary_path.write_text(json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary_path


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare ER-LS champ libre vs MLE-EM avec sol sur les mêmes mesures.",
    )
    parser.add_argument("--mic-family", choices=["sphere", "one_sided", "three_arrays"], default="one_sided")
    parser.add_argument("--source", nargs=3, type=float, default=[4.0, 6.0, 1.25], metavar=("X", "Y", "Z"))
    parser.add_argument("--room", nargs=3, type=float, default=[10.0, 12.0, 5.0], metavar=("LX", "LY", "LZ"))
    parser.add_argument("--floor-z", type=float, default=0.0)
    parser.add_argument("--fs", type=float, default=4000.0)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--f-low", type=float, default=80.0)
    parser.add_argument("--f-high", type=float, default=260.0)
    parser.add_argument("--sound-speed", type=float, default=343.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--snr-db", type=float, default=None, help="Ajoute un bruit blanc au SNR demandé. Par défaut : pas de bruit.")
    parser.add_argument("--ref-idx", type=int, default=0, help="Micro de référence pour ER-LS.")
    parser.add_argument("--ground-model", choices=["coherent", "additive"], default="coherent")
    parser.add_argument("--alpha", type=float, default=1.0, help="Coefficient de réflexion effectif utilisé par MLE-EM.")
    parser.add_argument("--estimate-alpha", action="store_true", help="Laisse MLE-EM estimer alpha.")
    parser.add_argument("--init-method", choices=["barycenter", "er_ls_ground"], default="barycenter")
    parser.add_argument("--use-erls-init", action="store_true", default=True, help="Initialise MLE-EM avec ER-LS clampé au-dessus du sol.")
    parser.add_argument("--no-erls-init", dest="use_erls_init", action="store_false")
    parser.add_argument("--max-iter", type=int, default=45)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "ground_model_comparison.png")
    parser.add_argument("--dpi", type=int, default=170)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results, _context = run_comparison(args)
    output = Path(args.output)
    plot_results(results, args, output)
    summary = save_json_summary(results, output)

    print("\nComparaison terminée")
    print("====================")
    print(f"Figure : {output}")
    print(f"Résumé : {summary}")
    print("\nErreurs :")
    for method, vals in results["errors"].items():
        print(
            f"  - {method:<20s} | "
            f"3D={vals['error_3d_m']:.4f} m | "
            f"xy={vals['error_xy_m']:.4f} m | "
            f"z={vals['error_z_m']:.4f} m"
        )


if __name__ == "__main__":
    main()

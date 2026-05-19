#!/usr/bin/env python3
"""
Ground model comparison rendered with the same visual logic as the UI real-time
3D scene.

The script simulates one real scene where the floor exists, then displays:
- Microphones
- Source
- Estimate 1: ground-aware model
- Estimate 2: free-field model

By default the script tries to use the PyVista rendering backend used by the UI
real-time scene. If PyVista/VTK is not installed, it automatically falls back to
a Matplotlib renderer that keeps the same colors and layout philosophy.

Run from the project root:
    python scripts/plot_ground_model_comparison.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

# Robust project-root detection.
HERE = Path(__file__).resolve()
CANDIDATE_ROOTS = [HERE.parents[1], Path.cwd(), Path("/mnt/data/chamber-simulator-ui-main-ground-viz-pyvista"), Path("/mnt/data/chamber-simulator-ui-main")]
PROJECT_ROOT = next((p for p in CANDIDATE_ROOTS if (p / "src" / "chamber_sim").exists()), HERE.parents[1])
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chamber_sim.scene import Scene, Source, MicArray, make_semi_anechoic_room  # noqa: E402
from chamber_sim.signals.source_signal import BandpassSpec, FadeSpec, SourceSignalConfig, generate_source_signal  # noqa: E402
from chamber_sim.propagation.free_field import PropagationConfig, simulate_mic_signals_free_field  # noqa: E402
from chamber_sim.noise.add_noise import NoiseConfig, add_awgn_snr  # noqa: E402
from chamber_sim.er_ls.config import EnergyConfig, ERLSConfig  # noqa: E402
from chamber_sim.er_ls.er_ls import er_ls  # noqa: E402
from chamber_sim.mle_em import GroundGeometryConfig, MLEEMGroundConfig, MLEEMInitConfig, MLEEMDebugFlags, mle_em_ground  # noqa: E402


# -----------------------------------------------------------------------------
# Fixed microphone family requested by the user
# -----------------------------------------------------------------------------

MIC_POSITIONS = np.array(
    [
        [2.0, 2.0, 1.3],
        [5.0, 1.0, 1.2],
        [7.0, 3.0, 0.9],
        [4.0, 7.0, 1.1],
        [7.0, 5.0, 1.8],
        [1.0, 5.0, 3.0],
    ],
    dtype=float,
)


# -----------------------------------------------------------------------------
# Colors copied from the UI real-time 3D scene
# -----------------------------------------------------------------------------

UI_COLORS = {
    "background": "#ffffff",
    "room": "#475569",
    "floor": "#dbeafe",
    "floor_edge": "#b6c6d9",
    "back_wall": "#f1f5f9",
    "back_wall_edge": "#d4dde8",
    "source": "#ef4444",
    "microphone": "#2563eb",
    "estimate_1": "#22c55e",  # UI estimate green
    "estimate_2": "#f59e0b",  # requested orange for second estimate
    "axis_text": "#334155",
    "label_text": "#0f172a",
}


# -----------------------------------------------------------------------------
# Simulation and estimation
# -----------------------------------------------------------------------------

def compute_errors(true_source: np.ndarray, estimates: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    true_source = np.asarray(true_source, dtype=float).reshape(3)
    out: Dict[str, Dict[str, float]] = {}
    for name, estimate in estimates.items():
        estimate = np.asarray(estimate, dtype=float).reshape(3)
        diff = estimate - true_source
        out[name] = {
            "error_3d_m": float(np.linalg.norm(diff)),
            "error_xy_m": float(np.linalg.norm(diff[:2])),
            "error_z_m": float(abs(diff[2])),
        }
    return out


def run_comparison(args: argparse.Namespace) -> dict:
    source_true = np.asarray(args.source, dtype=float).reshape(3)
    mic_positions = MIC_POSITIONS.copy()

    room = make_semi_anechoic_room(args.room[0], args.room[1], args.room[2], floor_rigid=True)
    scene = Scene(
        room=room,
        mic_array=MicArray(mic_positions, name="custom_family"),
        sources=[Source(position=source_true, name="Source")],
    )

    sig_cfg = SourceSignalConfig(
        fs=float(args.fs),
        duration=float(args.duration),
        band=BandpassSpec(float(args.f_low), float(args.f_high)),
        rms=1.0,
        fade=FadeSpec(fade_in=0.03, fade_out=0.03),
        seed=int(args.seed),
    )
    t, signal = generate_source_signal(sig_cfg)

    # The measurement model includes the floor in both cases.
    prop_cfg = PropagationConfig(
        c=float(args.sound_speed),
        include_spherical_spreading=True,
        gain_at_1m=1.0,
        include_rigid_floor_image=True,
        floor_z=float(args.floor_z),
    )
    y, direct_distances = simulate_mic_signals_free_field(scene, t, signal, prop_cfg)

    if args.snr_db is not None:
        y, noise_rms = add_awgn_snr(
            y,
            NoiseConfig(snr_db=float(args.snr_db), per_mic_independent=True, seed=int(args.seed) + 1),
        )
    else:
        noise_rms = 0.0

    energy_cfg = EnergyConfig(remove_mean=True, window_s=None, trim_frac=0.10)

    # Estimate 2: ER-LS with a free-field model. This intentionally ignores the floor.
    x_without_ground, erls_residual_rms, dbg_erls = er_ls(
        mic_positions=mic_positions,
        y=y,
        fs=float(args.fs),
        cfg_E=energy_cfg,
        cfg=ERLSConfig(ref_idx=int(args.ref_idx), kappa_eps=1e-3, min_pairs=3),
    )

    # Estimate 1: MLE-EM with a ground-aware propagation model.
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

    x0_for_mle = x_without_ground.copy()
    x0_for_mle[2] = max(x0_for_mle[2], float(args.floor_z) + 0.15)

    x_with_ground, mle_residual_rms, dbg_mle = mle_em_ground(
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

    estimates = {
        "Estimate 1 - with ground": x_with_ground,
        "Estimate 2 - without ground": x_without_ground,
    }

    return {
        "source_true": source_true,
        "mic_positions": mic_positions,
        "direct_distances": direct_distances,
        "x_estimate_1_with_ground": x_with_ground,
        "x_estimate_2_without_ground": x_without_ground,
        "errors": compute_errors(source_true, estimates),
        "residual_rms": {
            "Estimate 1 - with ground": float(mle_residual_rms),
            "Estimate 2 - without ground": float(erls_residual_rms),
        },
        "alpha_hat_mle": float(dbg_mle.get("alpha_hat", args.alpha)),
        "source_energy_hat_mle": float(dbg_mle.get("K_hat", np.nan)),
        "noise_rms": float(noise_rms),
        "energy_measured": np.asarray(dbg_mle.get("E", dbg_erls.get("E", [])), dtype=float),
    }


# -----------------------------------------------------------------------------
# PyVista rendering: same graphical backend/style as the UI real-time scene
# -----------------------------------------------------------------------------

def _add_point_labels(plotter, points: np.ndarray, labels: list[str], *, point_color: str, text_color: str, font_size: int = 13) -> None:
    """Add labels with PyVista, accepting minor API differences between versions."""
    kwargs = dict(
        font_size=font_size,
        point_color=point_color,
        text_color=text_color,
        shape_opacity=0.12,
        always_visible=True,
    )
    try:
        plotter.add_point_labels(points, labels, bold=True, **kwargs)
    except TypeError:
        plotter.add_point_labels(points, labels, **kwargs)


def render_with_ui_pyvista(results: dict, args: argparse.Namespace, output: Path) -> bool:
    """
    Render with the same PyVista logic used in SceneViewWidget.update_scene().

    Returns False if PyVista/VTK is unavailable.
    """
    try:
        import pyvista as pv
    except Exception:
        return False

    pv.OFF_SCREEN = True

    Lx, Ly, Lz = (float(v) for v in args.room)
    source = np.asarray(results["source_true"], dtype=float).reshape(3)
    mics = np.asarray(results["mic_positions"], dtype=float)
    est1 = np.asarray(results["x_estimate_1_with_ground"], dtype=float).reshape(3)
    est2 = np.asarray(results["x_estimate_2_without_ground"], dtype=float).reshape(3)

    output.parent.mkdir(parents=True, exist_ok=True)

    plotter = pv.Plotter(off_screen=True, window_size=(1400, 1100))
    plotter.set_background(UI_COLORS["background"])

    scale = max(min(Lx, Ly, Lz), 0.1)
    mic_radius = float(args.mic_radius) if args.mic_radius is not None else max(0.035 * scale, 0.035)
    estimate_radius = float(args.estimate_radius)
    source_radius = float(args.source_radius)

    # Room wireframe: copied from UI.
    room = pv.Cube(center=(Lx / 2.0, Ly / 2.0, Lz / 2.0), x_length=Lx, y_length=Ly, z_length=Lz)
    plotter.add_mesh(room, style="wireframe", color=UI_COLORS["room"], line_width=1.4, name="room")

    # Transparent floor and back wall: copied from UI.
    floor = pv.Plane(center=(Lx / 2.0, Ly / 2.0, float(args.floor_z)), direction=(0.0, 0.0, 1.0), i_size=Lx, j_size=Ly)
    plotter.add_mesh(
        floor,
        color=UI_COLORS["floor"],
        opacity=0.35,
        show_edges=True,
        edge_color=UI_COLORS["floor_edge"],
        name="floor",
    )

    back_wall = pv.Plane(center=(Lx / 2.0, Ly, Lz / 2.0), direction=(0.0, 1.0, 0.0), i_size=Lx, j_size=Lz)
    plotter.add_mesh(
        back_wall,
        color=UI_COLORS["back_wall"],
        opacity=0.18,
        show_edges=True,
        edge_color=UI_COLORS["back_wall_edge"],
        name="back_wall",
    )

    # Microphones: every microphone has exactly the same color and opacity.
    clipped_mics = np.column_stack(
        [
            np.clip(mics[:, 0], 0.0, Lx),
            np.clip(mics[:, 1], 0.0, Ly),
            np.clip(mics[:, 2], 0.0, Lz),
        ]
    )
    for i, point in enumerate(clipped_mics):
        mic_mesh = pv.Sphere(radius=mic_radius, center=tuple(point), theta_resolution=32, phi_resolution=16)
        plotter.add_mesh(mic_mesh, color=UI_COLORS["microphone"], opacity=1.0, smooth_shading=True, name=f"mic_{i + 1}")

    # Estimate 2 first: orange, requested.
    est2_mesh = pv.Sphere(radius=estimate_radius, center=tuple(np.clip(est2, [0, 0, 0], [Lx, Ly, Lz])), theta_resolution=40, phi_resolution=20)
    plotter.add_mesh(est2_mesh, color=UI_COLORS["estimate_2"], opacity=0.96, smooth_shading=True, name="estimate_2_without_ground")

    # Estimate 1 second: UI green.
    est1_mesh = pv.Sphere(radius=estimate_radius, center=tuple(np.clip(est1, [0, 0, 0], [Lx, Ly, Lz])), theta_resolution=40, phi_resolution=20)
    plotter.add_mesh(est1_mesh, color=UI_COLORS["estimate_1"], opacity=0.96, smooth_shading=True, name="estimate_1_with_ground")

    # Source last and larger: visually dominant / first layer.
    source_mesh = pv.Sphere(radius=source_radius, center=tuple(np.clip(source, [0, 0, 0], [Lx, Ly, Lz])), theta_resolution=48, phi_resolution=24)
    plotter.add_mesh(source_mesh, color=UI_COLORS["source"], opacity=1.0, smooth_shading=True, name="source")

    # Labels: bold if supported by the local PyVista version.
    _add_point_labels(plotter, np.asarray([est2]), ["Estimate 2"], point_color=UI_COLORS["estimate_2"], text_color=UI_COLORS["estimate_2"], font_size=13)
    _add_point_labels(plotter, np.asarray([est1]), ["Estimate 1"], point_color=UI_COLORS["estimate_1"], text_color=UI_COLORS["estimate_1"], font_size=13)
    _add_point_labels(plotter, np.asarray([source]), ["Source"], point_color=UI_COLORS["source"], text_color=UI_COLORS["label_text"], font_size=14)

    # Same camera preset as the UI isometric view.
    center = (0.5 * Lx, 0.5 * Ly, 0.5 * Lz)
    distance = max(Lx, Ly, Lz, 1.0) * 2.35
    position = (center[0] + 0.95 * distance, center[1] - 1.05 * distance, center[2] + 0.78 * distance)
    view_up = (0.0, 0.0, 1.0)
    plotter.camera_position = [position, center, view_up]
    try:
        plotter.enable_parallel_projection()
    except Exception:
        plotter.camera.parallel_projection = True

    plotter.show_bounds(
        bounds=(0, Lx, 0, Ly, 0, Lz),
        grid="back",
        location="outer",
        xtitle="x [m]",
        ytitle="y [m]",
        ztitle="z [m]",
        font_size=10,
        color=UI_COLORS["axis_text"],
    )
    plotter.add_axes(line_width=2, labels_off=False)
    plotter.screenshot(str(output), return_img=False)
    plotter.close()
    return True


# -----------------------------------------------------------------------------
# Matplotlib fallback, used only when PyVista is not available in the environment
# -----------------------------------------------------------------------------

def render_matplotlib_fallback(results: dict, args: argparse.Namespace, output: Path) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

    def draw_sphere(ax, center, radius, color, *, alpha=1.0, resolution=32, zorder=10):
        center = np.asarray(center, dtype=float).reshape(3)
        u = np.linspace(0, 2 * np.pi, resolution)
        v = np.linspace(0, np.pi, max(14, resolution // 2))
        x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
        y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
        z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
        ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0.0, antialiased=True, shade=True, zorder=zorder)

    def add_label(ax, xyz, text, color, dx=0.06, dy=0.06, dz=0.06, size=10):
        xyz = np.asarray(xyz, dtype=float).reshape(3)
        label = ax.text(
            xyz[0] + dx,
            xyz[1] + dy,
            xyz[2] + dz,
            text,
            fontsize=size,
            fontweight="bold",
            color=color,
            zorder=100,
        )
        label.set_path_effects([pe.withStroke(linewidth=3.5, foreground="white")])

    def add_room(ax, room_dims):
        Lx, Ly, Lz = room_dims
        corners = np.array([[0,0,0],[Lx,0,0],[Lx,Ly,0],[0,Ly,0],[0,0,Lz],[Lx,0,Lz],[Lx,Ly,Lz],[0,Ly,Lz]], dtype=float)
        edges = []
        for i in range(8):
            for j in range(i + 1, 8):
                if np.sum(np.abs(corners[i] - corners[j]) > 1e-12) == 1:
                    edges.append([corners[i], corners[j]])
        ax.add_collection3d(Line3DCollection(edges, colors=UI_COLORS["room"], linewidths=1.05, alpha=0.95))

    def add_floor(ax, room_dims):
        Lx, Ly, _ = room_dims
        verts = [[(0,0,args.floor_z),(Lx,0,args.floor_z),(Lx,Ly,args.floor_z),(0,Ly,args.floor_z)]]
        ax.add_collection3d(Poly3DCollection(verts, facecolors=UI_COLORS["floor"], edgecolors=UI_COLORS["floor_edge"], linewidths=0.8, alpha=0.55))

    Lx, Ly, Lz = (float(v) for v in args.room)
    room_dims = (Lx, Ly, Lz)
    src = np.asarray(results["source_true"], dtype=float)
    mics = np.asarray(results["mic_positions"], dtype=float)
    est1 = np.asarray(results["x_estimate_1_with_ground"], dtype=float)
    est2 = np.asarray(results["x_estimate_2_without_ground"], dtype=float)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8.8, 7.8), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    ax.computed_zorder = False

    add_floor(ax, room_dims)
    add_room(ax, room_dims)

    # Microphones: same blue and same opacity.
    for mic in mics:
        draw_sphere(ax, mic, args.mic_radius, UI_COLORS["microphone"], alpha=1.0, resolution=22, zorder=20)

    # Estimate 2 orange, Estimate 1 green, Source red and drawn last.
    draw_sphere(ax, est2, args.estimate_radius, UI_COLORS["estimate_2"], alpha=0.96, resolution=30, zorder=40)
    draw_sphere(ax, est1, args.estimate_radius, UI_COLORS["estimate_1"], alpha=0.96, resolution=30, zorder=45)
    draw_sphere(ax, src, args.source_radius, UI_COLORS["source"], alpha=1.0, resolution=36, zorder=90)

    add_label(ax, est2, "Estimate 2", UI_COLORS["estimate_2"], dx=0.12, dy=0.08, dz=0.14, size=10)
    add_label(ax, est1, "Estimate 1", UI_COLORS["estimate_1"], dx=0.18, dy=0.18, dz=0.18, size=10)
    add_label(ax, src, "Source", UI_COLORS["label_text"], dx=0.02, dy=-0.15, dz=0.20, size=11)

    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    ax.set_zlim(0, Lz)
    ax.set_box_aspect(room_dims)
    ax.view_init(elev=24, azim=-44)
    ax.set_xlabel("x [m]", labelpad=10, fontweight="bold")
    ax.set_ylabel("y [m]", labelpad=10, fontweight="bold")
    ax.set_zlabel("z [m]", labelpad=10, fontweight="bold")
    ticks = np.arange(0, max(room_dims) + 0.01, 2.0)
    ax.set_xticks(ticks[ticks <= Lx])
    ax.set_yticks(ticks[ticks <= Ly])
    ax.set_zticks(ticks[ticks <= Lz])
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"]["color"] = "#e8eef7"
        axis._axinfo["grid"]["linewidth"] = 0.55
        axis._axinfo["grid"]["linestyle"] = "-"
        axis._axinfo["axisline"]["color"] = UI_COLORS["room"]
    ax.xaxis.set_pane_color((1,1,1,0.0))
    ax.yaxis.set_pane_color((1,1,1,0.0))
    ax.zaxis.set_pane_color((1,1,1,0.0))
    ax.tick_params(colors=UI_COLORS["room"], labelsize=9)

    fig.tight_layout(pad=0.4)
    fig.savefig(output, dpi=int(args.dpi), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


# -----------------------------------------------------------------------------
# IO / CLI
# -----------------------------------------------------------------------------

def save_json_summary(results: dict, output_png: Path) -> Path:
    def to_jsonable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        return obj

    path = output_png.with_suffix(".json")
    path.write_text(json.dumps(to_jsonable(results), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the ground-model comparison using the UI 3D scene style.")
    parser.add_argument("--backend", choices=["auto", "pyvista", "matplotlib"], default="auto",
                        help="auto uses the UI PyVista backend when available, then Matplotlib fallback.")
    parser.add_argument("--source", nargs=3, type=float, default=[4.0, 6.0, 4.0], metavar=("X", "Y", "Z"))
    parser.add_argument("--room", nargs=3, type=float, default=[8.0, 8.0, 8.0], metavar=("LX", "LY", "LZ"))
    parser.add_argument("--floor-z", type=float, default=0.0)
    parser.add_argument("--fs", type=float, default=4000.0)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--f-low", type=float, default=80.0)
    parser.add_argument("--f-high", type=float, default=260.0)
    parser.add_argument("--sound-speed", type=float, default=343.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--snr-db", type=float, default=None)
    parser.add_argument("--ref-idx", type=int, default=0)
    parser.add_argument("--ground-model", choices=["coherent", "additive"], default="coherent")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--estimate-alpha", action="store_true")
    parser.add_argument("--init-method", choices=["barycenter", "er_ls_ground"], default="barycenter")
    parser.add_argument("--use-erls-init", action="store_true", default=True)
    parser.add_argument("--no-erls-init", dest="use_erls_init", action="store_false")
    parser.add_argument("--max-iter", type=int, default=45)

    # Same radius logic as the UI, but exposed for clean figure tuning.
    parser.add_argument("--mic-radius", type=float, default=0.16)
    parser.add_argument("--source-radius", type=float, default=0.36)
    parser.add_argument("--estimate-radius", type=float, default=0.30)

    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "ground_model_comparison_ui_native.png")
    parser.add_argument("--dpi", type=int, default=170)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_comparison(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    backend_used = None
    if args.backend in {"auto", "pyvista"}:
        ok = render_with_ui_pyvista(results, args, output)
        if ok:
            backend_used = "pyvista-ui"
        elif args.backend == "pyvista":
            raise RuntimeError("PyVista/VTK is not installed. Install with: pip install pyvista pyvistaqt vtk")

    if backend_used is None:
        render_matplotlib_fallback(results, args, output)
        backend_used = "matplotlib-fallback"

    summary = save_json_summary(results, output)

    print("\nGround model comparison completed")
    print("=================================")
    print(f"Backend : {backend_used}")
    print(f"Figure  : {output}")
    print(f"Summary : {summary}")
    print("\nErrors:")
    for method, vals in results["errors"].items():
        print(
            f"  - {method:<28s} | "
            f"3D={vals['error_3d_m']:.4f} m | "
            f"xy={vals['error_xy_m']:.4f} m | "
            f"z={vals['error_z_m']:.4f} m"
        )


if __name__ == "__main__":
    main()

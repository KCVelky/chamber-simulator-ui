from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class VizOptions:
    show_rays: bool = True
    show_room: bool = True
    show_axes: bool = True
    equal_aspect: bool = True
    mic_label: bool = True
    source_label: bool = True
    title: str = "3D Scene"
    elev: float = 18.0
    azim: float = -55.0


def _set_axes_equal(ax) -> None:
    """Make 3D plot have equal aspect ratio (so room looks like a box, not stretched)."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max(x_range, y_range, z_range)

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def _draw_room_wireframe(ax, Lx: float, Ly: float, Lz: float) -> None:
    # 8 corners of a rectangular box
    corners = np.array([
        [0,  0,  0],
        [Lx, 0,  0],
        [Lx, Ly, 0],
        [0,  Ly, 0],
        [0,  0,  Lz],
        [Lx, 0,  Lz],
        [Lx, Ly, Lz],
        [0,  Ly, Lz],
    ], dtype=float)

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom
        (4, 5), (5, 6), (6, 7), (7, 4),  # top
        (0, 4), (1, 5), (2, 6), (3, 7),  # verticals
    ]

    for i, j in edges:
        xs = [corners[i, 0], corners[j, 0]]
        ys = [corners[i, 1], corners[j, 1]]
        zs = [corners[i, 2], corners[j, 2]]
        ax.plot(xs, ys, zs)


def plot_scene_3d(scene, options: Optional[VizOptions] = None):
    """
    Plot Room + MicArray + Sources.

    Expects 'scene' to have:
      - scene.room.size -> (3,)
      - scene.mic_array.positions -> (M,3)
      - scene.sources -> list of objects with .position and .name
    """
    if options is None:
        options = VizOptions()

    Lx, Ly, Lz = map(float, scene.room.size)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=options.elev, azim=options.azim)

    # Room
    if options.show_room:
        _draw_room_wireframe(ax, Lx, Ly, Lz)

    # Micros
    mics = np.asarray(scene.mic_array.positions, dtype=float)
    ax.scatter(mics[:, 0], mics[:, 1], mics[:, 2], marker="o", s=40, label="Micros")
    if options.mic_label:
        for k, (x, y, z) in enumerate(mics):
            ax.text(x, y, z, f"M{k}", fontsize=8)


    # Sources
    for s in scene.sources:
        p = np.asarray(s.position, dtype=float).reshape(3,)
        ax.scatter([p[0]], [p[1]], [p[2]], marker="^", s=140, label="Source")
        if options.source_label:
            ax.text(p[0], p[1], p[2], s.name, fontsize=9)

    # Rays: source -> each mic (optional)
    if getattr(options, "show_rays", False) and len(scene.sources) > 0:
        src0 = np.asarray(scene.sources[0].position, dtype=float).reshape(3,)
        for (x, y, z) in mics:
            ax.plot([src0[0], x], [src0[1], y], [src0[2], z], linewidth=0.8)


    ax.legend(loc="upper left")

    plt.tight_layout()
    return fig, ax
    


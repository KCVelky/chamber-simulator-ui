from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

if TYPE_CHECKING:
    from chamber_sim.er_ls.geometry import ApolloniusSphere
else:
    ApolloniusSphere = object


@dataclass(frozen=True)
class VizSphereOptions:
    max_spheres: int = 4                 # combien de sphères à afficher
    sphere_mesh: Tuple[int, int] = (28, 14)  # (n_theta, n_phi) résolution surface
    base_alpha: float = 0.35             # opacité de base
    alpha_decay: float = 0.70            # chaque sphère suivante est plus transparente
    alpha_slider_init: float = 1.0       # multiplicateur initial (0..1.5 typiquement)
    alpha_slider_max: float = 1.5
    show_wireframe: bool = False         # True => wireframe, False => surface
    equal_aspect: bool = True
    title: str = "Apollonius spheres (ER)"


def _set_axes_equal(ax):
    """Force aspect equal in 3D (matplotlib ne le fait pas bien par défaut)."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def _sphere_mesh(c: np.ndarray, r: float, n_theta: int, n_phi: int):
    """Retourne X,Y,Z pour une sphère centrée en c de rayon r."""
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta)
    phi = np.linspace(0.0, np.pi, n_phi)
    tt, pp = np.meshgrid(theta, phi)

    x = c[0] + r * np.cos(tt) * np.sin(pp)
    y = c[1] + r * np.sin(tt) * np.sin(pp)
    z = c[2] + r * np.cos(pp)
    return x, y, z


def _select_spheres(
    spheres: List[ApolloniusSphere],
    max_spheres: int,
    mode: str = "radius",
    choose_indices: Optional[Sequence[int]] = None,
) -> List[ApolloniusSphere]:
    """
    Sélectionne quelques sphères à afficher.
    - choose_indices: si fourni, on prend exactement ces indices (dans la liste spheres)
    - mode:
        "radius" => prend les plus grandes sphères (visuellement parlant)
        "closest" => prend les plus petites (souvent plus informatives localement)
    """
    if choose_indices is not None:
        out = []
        for k in choose_indices:
            if 0 <= k < len(spheres):
                out.append(spheres[k])
        return out[:max_spheres]

    if len(spheres) <= max_spheres:
        return spheres

    if mode == "closest":
        order = np.argsort([sp.rho for sp in spheres])  # petites d'abord
    else:  # "radius"
        order = np.argsort([-sp.rho for sp in spheres])  # grandes d'abord

    return [spheres[i] for i in order[:max_spheres]]


def plot_apollonius_spheres_3d(
    mic_positions: np.ndarray,
    spheres: List[ApolloniusSphere],
    x_true: Optional[np.ndarray] = None,
    x_hat: Optional[np.ndarray] = None,
    room_size: Optional[np.ndarray] = None,
    choose_indices: Optional[Sequence[int]] = None,
    select_mode: str = "radius",
    opts: VizSphereOptions = VizSphereOptions(),
):
    """
    Plot interactif : micros + (source vraie/estimée) + quelques sphères d’Apollonius
    avec curseur pour opacité.

    Paramètres
    ----------
    mic_positions : (M,3)
    spheres : list[ApolloniusSphere]
    x_true : (3,) optionnel
    x_hat : (3,) optionnel
    room_size : (3,) optionnel => limite axes à [0,Lx] etc.
    choose_indices : liste d'indices (dans spheres) à forcer
    select_mode : "radius" ou "closest"
    opts : VizSphereOptions
    """
    mics = np.asarray(mic_positions, float)
    if mics.ndim != 2 or mics.shape[1] != 3:
        raise ValueError("mic_positions must be (M,3)")
    if len(spheres) == 0:
        raise ValueError("No spheres to plot.")

    # choisir quelques sphères
    sph_sel = _select_spheres(
        spheres=spheres,
        max_spheres=int(opts.max_spheres),
        mode=select_mode,
        choose_indices=choose_indices,
    )

    # figure
    fig = plt.figure(figsize=(9.5, 7.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(opts.title)

    # plot mics
    ax.scatter(mics[:, 0], mics[:, 1], mics[:, 2], s=45, marker="o", label="Mics")

    # sources
    if x_true is not None:
        xt = np.asarray(x_true, float).ravel()
        ax.scatter([xt[0]], [xt[1]], [xt[2]], s=120, marker="*", label="True")

    if x_hat is not None:
        xh = np.asarray(x_hat, float).ravel()
        ax.scatter([xh[0]], [xh[1]], [xh[2]], s=120, marker="X", label="Estimated")

    # limites axes
    if room_size is not None:
        L = np.asarray(room_size, float).ravel()
        ax.set_xlim(0, L[0])
        ax.set_ylim(0, L[1])
        ax.set_zlim(0, L[2])

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")

    # sphères : alphas décroissants + slider multiplicateur
    n_theta, n_phi = opts.sphere_mesh
    base_alphas = [opts.base_alpha * (opts.alpha_decay ** k) for k in range(len(sph_sel))]

    sphere_artists = []
    for k, sp in enumerate(sph_sel):
        c = np.asarray(sp.c, float).ravel()
        r = float(sp.rho)
        X, Y, Z = _sphere_mesh(c, r, n_theta, n_phi)

        a = np.clip(base_alphas[k], 0.01, 1.0)

        if opts.show_wireframe:
            art = ax.plot_wireframe(X, Y, Z, rstride=2, cstride=2, linewidth=0.6, alpha=a)
        else:
            art = ax.plot_surface(X, Y, Z, linewidth=0.0, antialiased=True, alpha=a)

        sphere_artists.append((art, base_alphas[k], sp))

    ax.legend(loc="upper left")

    if opts.equal_aspect:
        _set_axes_equal(ax)

    # ---- Slider opacité
    plt.subplots_adjust(bottom=0.16)
    ax_alpha = fig.add_axes([0.18, 0.06, 0.64, 0.03])
    slider = Slider(
        ax=ax_alpha,
        label="Opacity x",
        valmin=0.0,
        valmax=float(opts.alpha_slider_max),
        valinit=float(opts.alpha_slider_init),
    )

    def _update(val):
        mult = float(slider.val)
        for (art, a0, _sp) in sphere_artists:
            a = float(np.clip(a0 * mult, 0.0, 1.0))
            # plot_surface renvoie un Poly3DCollection
            try:
                art.set_alpha(a)
            except Exception:
                # wireframe retourne une Line3DCollection (ou liste), set_alpha marche aussi
                try:
                    art.set_alpha(a)
                except Exception:
                    pass
        fig.canvas.draw_idle()

    slider.on_changed(_update)

    # petit texte info (quelles sphères)
    info = "Spheres shown:\n" + "\n".join(
        [f"mic={sp.mic_idx} ref={sp.ref_idx}  rho={sp.rho:.2f}  kappa={sp.kappa:.3f}" for sp in sph_sel]
    )
    fig.text(0.02, 0.02, info, fontsize=9, family="monospace")

    return fig, ax

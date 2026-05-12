from __future__ import annotations

import json
from tkinter import filedialog

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))



import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import matplotlib.pyplot as plt

from chamber_sim.scene import Scene, MicArray, Source, SourceType, as_vec3, make_semi_anechoic_room
from chamber_sim.viz3d import plot_scene_3d, VizOptions
from chamber_sim.noise import NoiseConfig, add_awgn_snr
from chamber_sim.tdoa import GCCPHATConfig, gcc_phat_tdoa, GridSearchConfig, localize_tdoa_grid
from chamber_sim.tdoa.localization import MultiRefTDOAConfig, estimate_source_multiref_tdoa
from chamber_sim.tdoa.denoise import DenoiseConfig
from chamber_sim.er_ls.config import EnergyConfig, ERLSConfig
from chamber_sim.er_ls import er_ls
from chamber_sim.er_nls import er_nls
from chamber_sim.er_nls.config import ERNLSConfig
from chamber_sim.er_ls.er_ls_ground import er_ls_ground, ERLSGroundConfig
from chamber_sim.er_ls.ground_model import GroundConfig
from chamber_sim.er_nls.er_nls_ground import er_nls_ground, ERNLSGroundConfig
from chamber_sim.mle_em import (
    MLEEMInitConfig,
    MLEEMGroundConfig,
    GroundGeometryConfig as MLEEMGeometryConfig,
    MLEEMDebugFlags,
    mle_em_ground,
)

from chamber_sim.signals import (
    BandpassSpec,
    ModulationSpec,
    FadeSpec,
    SourceSignalConfig,
    generate_source_signal,
)

from chamber_sim.propagation import PropagationConfig, simulate_mic_signals_free_field


# -----------------------
# Helpers
# -----------------------

def _parse_floats_csv(text: str) -> np.ndarray:
    """
    Parse mic positions from text like:
      3.0,3.0,1.5
      3.5,3.0,1.5
    Returns (M,3)
    """
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    rows = []
    for ln in lines:
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) != 3:
            raise ValueError(f"Each mic line must have 3 comma-separated values. Got: {ln}")
        rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
    arr = np.array(rows, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("Mic positions must be an (M,3) array.")
    return arr


def _safe_float(var: tk.StringVar, name: str) -> float:
    try:
        return float(var.get())
    except Exception as e:
        raise ValueError(f"Invalid float for '{name}': {var.get()}") from e


def _safe_int_or_none(var: tk.StringVar, name: str) -> int | None:
    v = var.get().strip()
    if v.lower() in ("", "none", "null"):
        return None
    try:
        return int(v)
    except Exception as e:
        raise ValueError(f"Invalid int/None for '{name}': {var.get()}") from e


# -----------------------
# UI App
# -----------------------

class ExperimentUI(ttk.Frame):
    def __init__(self, master: tk.Tk):
        super().__init__(master)
        self.master = master
        self.master.title("ChamberSim – Experiment UI")
        self.grid(sticky="nsew")

        self._build_vars()
        self._build_layout()

    def _build_vars(self):
        # Room
        self.Lx = tk.StringVar(value="8.0")
        self.Ly = tk.StringVar(value="6.0")
        self.Lz = tk.StringVar(value="4.0")
        self.margin = tk.StringVar(value="0.05")

        # Source position
        self.src_x = tk.StringVar(value="5.5")
        self.src_y = tk.StringVar(value="2.0")
        self.src_z = tk.StringVar(value="1.2")

        # Mic positions text
        self.mic_text = tk.StringVar(value=
            "3.0,3.0,1.5\n"
            "3.5,3.0,1.5\n"
            "3.0,3.5,1.5\n"
            "3.5,3.5,1.5\n"
        )

        # Signal
        self.fs = tk.StringVar(value="8000")
        self.duration = tk.StringVar(value="6.0")
        self.f_low = tk.StringVar(value="80.0")
        self.f_high = tk.StringVar(value="260.0")
        self.rms = tk.StringVar(value="1.0")

        # Modulation
        self.use_mod = tk.BooleanVar(value=True)
        self.f_mod = tk.StringVar(value="0.25")
        self.mod_depth = tk.StringVar(value="0.2")

        # Fade
        self.use_fade = tk.BooleanVar(value=True)
        self.fade_in = tk.StringVar(value="0.2")
        self.fade_out = tk.StringVar(value="0.2")

        # Seed
        self.seed = tk.StringVar(value="42")  # can be "None"

        # Propagation
        self.c = tk.StringVar(value="343.0")
        self.use_spreading = tk.BooleanVar(value=True)
        self.gain_at_1m = tk.StringVar(value="1.0")
        self.use_floor_image = tk.BooleanVar(value=True)
        self.floor_z = tk.StringVar(value="0.0")
        # Noise
        self.add_noise = tk.BooleanVar(value=False)
        self.snr_db = tk.StringVar(value="20.0")
        self.noise_indep = tk.BooleanVar(value=True)
        self.noise_seed = tk.StringVar(value="123")  # can be None

        # Denoise (Wiener STFT)
        self.enable_denoise = tk.BooleanVar(value=True)

        self.denoise_nperseg = tk.StringVar(value="1024")
        self.denoise_noverlap = tk.StringVar(value="768")
        self.denoise_noise_head = tk.StringVar(value="0.20")
        self.denoise_noise_tail = tk.StringVar(value="0.20")
        self.denoise_gain_floor = tk.StringVar(value="0.05")

        # Debug denoise plots
        self.plot_denoise_compare = tk.BooleanVar(value=False)
        self.plot_denoise_mic = tk.StringVar(value="0")

        # What to plot
        self.plot_scene = tk.BooleanVar(value=True)
        self.plot_source_time = tk.BooleanVar(value=True)
        self.plot_source_spectrum = tk.BooleanVar(value=True)
        self.plot_mics = tk.BooleanVar(value=True)
        self.plot_mics_zoom = tk.BooleanVar(value=True)
        self.print_delta_r = tk.BooleanVar(value=True)
        self.plot_source_spectrogram = tk.BooleanVar(value=False)
        self.plot_mic_spectrogram = tk.BooleanVar(value=False)
        self.mic_spec_index = tk.StringVar(value="0")  # micro affiché en spectrogramme

        # Zoom config
        self.zoom_tmax = tk.StringVar(value="0.2")

        # Algorithms
        self.enable_algorithms = tk.BooleanVar(value=True)

        self.alg_choice = tk.StringVar(value="None")  # "None" or "TDOA"

        # TDOA parameters
        self.tdoa_interp = tk.StringVar(value="8")
        self.tdoa_grid_dx = tk.StringVar(value="0.10")
        self.tdoa_grid_dy = tk.StringVar(value="0.10")
        self.tdoa_grid_dz = tk.StringVar(value="0.10")
        self.tdoa_z_fixed = tk.StringVar(value="")  # empty => None

        self.plot_estimated_source = tk.BooleanVar(value=True)

        # Energy (ER-LS / ER-NLS) parameters
        self.er_ref_idx = tk.StringVar(value="0")          # micro de référence
        self.er_window_s = tk.StringVar(value="0.50")      # fenêtre énergie [s]
        self.er_hop_s = tk.StringVar(value="0.25")         # hop [s]
        self.er_trim_frac = tk.StringVar(value="0.10")     # trimmed mean
        self.er_kappa_eps = tk.StringVar(value="0.001")    # stabilité sphères

        # ER-NLS parameters
        self.ernls_max_iter = tk.StringVar(value="50")
        self.ernls_lam = tk.StringVar(value="0.01")
        self.ernls_tol_step = tk.StringVar(value="1e-6")
        self.ernls_tol_cost = tk.StringVar(value="1e-9")

        # MLE/EM ground parameters
        self.mleem_model_type = tk.StringVar(value="coherent")
        self.mleem_init_method = tk.StringVar(value="er_ls_ground")
        self.mleem_max_iter = tk.StringVar(value="40")
        self.mleem_lam = tk.StringVar(value="0.01")
        self.mleem_tol_step = tk.StringVar(value="1e-6")
        self.mleem_tol_cost = tk.StringVar(value="1e-9")
        self.mleem_estimate_alpha = tk.BooleanVar(value=False)
        self.mleem_alpha_init = tk.StringVar(value="1.0")
        self.mleem_alpha_min = tk.StringVar(value="0.0")
        self.mleem_alpha_max = tk.StringVar(value="1.5")
        self.mleem_alpha_grid_size = tk.StringVar(value="31")
        self.mleem_fd_eps = tk.StringVar(value="1e-5")
        self.mleem_barycenter_z_offset = tk.StringVar(value="0.50")


    def _build_layout(self):
        self.master.columnconfigure(0, weight=1)
        self.master.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        nb = ttk.Notebook(self)
        nb.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        self.rowconfigure(0, weight=1)

        tab_geom = ttk.Frame(nb)
        tab_signal = ttk.Frame(nb)
        tab_prop = ttk.Frame(nb)
        tab_plots = ttk.Frame(nb)
        tab_algo = ttk.Frame(nb)


        nb.add(tab_geom, text="Geometry")
        nb.add(tab_signal, text="Signal")
        nb.add(tab_prop, text="Propagation")
        nb.add(tab_plots, text="Plots")
        nb.add(tab_algo, text="Algorithms")


        # --- Geometry tab
        self._section_room(tab_geom).grid(row=0, column=0, sticky="ew", padx=8, pady=6)
        self._section_source(tab_geom).grid(row=1, column=0, sticky="ew", padx=8, pady=6)
        self._section_mics(tab_geom).grid(row=2, column=0, sticky="nsew", padx=8, pady=6)
        tab_geom.columnconfigure(0, weight=1)
        tab_geom.rowconfigure(2, weight=1)

        # --- Signal tab
        self._section_signal(tab_signal).grid(row=0, column=0, sticky="ew", padx=8, pady=6)
        self._section_modulation(tab_signal).grid(row=1, column=0, sticky="ew", padx=8, pady=6)
        self._section_fade(tab_signal).grid(row=2, column=0, sticky="ew", padx=8, pady=6)
        self._section_seed(tab_signal).grid(row=3, column=0, sticky="ew", padx=8, pady=6)
        tab_signal.columnconfigure(0, weight=1)

        # --- Prop tab
        self._section_prop(tab_prop).grid(row=0, column=0, sticky="ew", padx=8, pady=6)
        tab_prop.columnconfigure(0, weight=1)

        # --- Algo tab
        self._section_algorithms(tab_algo).grid(row=0, column=0, sticky="ew", padx=8, pady=6)
        tab_algo.columnconfigure(0, weight=1)


        # --- Plots tab
        self._section_plots(tab_plots).grid(row=0, column=0, sticky="ew", padx=8, pady=6)
        self._section_zoom(tab_plots).grid(row=1, column=0, sticky="ew", padx=8, pady=6)
        tab_plots.columnconfigure(0, weight=1)

        # Bottom buttons
        btns = ttk.Frame(self)
        btns.grid(row=1, column=0, sticky="ew", padx=8, pady=(0,8))
        btns.columnconfigure(0, weight=1)

        save_btn = ttk.Button(btns, text="Save config", command=self.save_config)
        load_btn = ttk.Button(btns, text="Load config", command=self.load_config)
        run_btn  = ttk.Button(btns, text="Run experiment", command=self.run_experiment)

        save_btn.grid(row=0, column=0, sticky="w")
        load_btn.grid(row=0, column=1, sticky="w", padx=(8,0))
        run_btn.grid(row=0, column=2, sticky="e")
        btns.columnconfigure(2, weight=1)


    def _section_room(self, parent):
        f = ttk.LabelFrame(parent, text="Room")
        for j in range(6):
            f.columnconfigure(j, weight=1)

        ttk.Label(f, text="Lx [m]").grid(row=0, column=0, sticky="w")
        ttk.Entry(f, textvariable=self.Lx, width=10).grid(row=0, column=1, sticky="w")

        ttk.Label(f, text="Ly [m]").grid(row=0, column=2, sticky="w")
        ttk.Entry(f, textvariable=self.Ly, width=10).grid(row=0, column=3, sticky="w")

        ttk.Label(f, text="Lz [m]").grid(row=0, column=4, sticky="w")
        ttk.Entry(f, textvariable=self.Lz, width=10).grid(row=0, column=5, sticky="w")

        ttk.Label(f, text="Margin [m]").grid(row=1, column=0, sticky="w")
        ttk.Entry(f, textvariable=self.margin, width=10).grid(row=1, column=1, sticky="w")

        return f

    def _section_source(self, parent):
        f = ttk.LabelFrame(parent, text="Source")
        for j in range(6):
            f.columnconfigure(j, weight=1)

        ttk.Label(f, text="x [m]").grid(row=0, column=0, sticky="w")
        ttk.Entry(f, textvariable=self.src_x, width=10).grid(row=0, column=1, sticky="w")

        ttk.Label(f, text="y [m]").grid(row=0, column=2, sticky="w")
        ttk.Entry(f, textvariable=self.src_y, width=10).grid(row=0, column=3, sticky="w")

        ttk.Label(f, text="z [m]").grid(row=0, column=4, sticky="w")
        ttk.Entry(f, textvariable=self.src_z, width=10).grid(row=0, column=5, sticky="w")

        return f

    def _section_mics(self, parent):
        f = ttk.LabelFrame(parent, text="Mic positions (one mic per line: x,y,z)")
        f.columnconfigure(0, weight=1)
        f.rowconfigure(0, weight=1)

        self.mic_box = tk.Text(f, height=8, wrap="none")
        self.mic_box.grid(row=0, column=0, sticky="nsew", padx=6, pady=(6,4))
        self.mic_box.insert("1.0", self.mic_text.get())

        btn_row = ttk.Frame(f)
        btn_row.grid(row=1, column=0, sticky="ew", padx=6, pady=(0,6))
        btn_row.columnconfigure(0, weight=1)

        ttk.Button(btn_row, text="Preset: square 4", command=self.preset_square_4).grid(row=0, column=0, sticky="w")
        ttk.Button(btn_row, text="Preset: hex 6", command=self.preset_hex_6).grid(row=0, column=1, sticky="w", padx=(8,0))
        ttk.Button(btn_row, text="Preset: random", command=self.preset_random).grid(row=0, column=2, sticky="w", padx=(8,0))

        return f


    def _section_signal(self, parent):
        f = ttk.LabelFrame(parent, text="Signal")
        for j in range(6):
            f.columnconfigure(j, weight=1)

        ttk.Label(f, text="fs [Hz]").grid(row=0, column=0, sticky="w")
        ttk.Entry(f, textvariable=self.fs, width=10).grid(row=0, column=1, sticky="w")

        ttk.Label(f, text="Duration [s]").grid(row=0, column=2, sticky="w")
        ttk.Entry(f, textvariable=self.duration, width=10).grid(row=0, column=3, sticky="w")

        ttk.Label(f, text="RMS").grid(row=0, column=4, sticky="w")
        ttk.Entry(f, textvariable=self.rms, width=10).grid(row=0, column=5, sticky="w")

        ttk.Label(f, text="f_low [Hz]").grid(row=1, column=0, sticky="w")
        ttk.Entry(f, textvariable=self.f_low, width=10).grid(row=1, column=1, sticky="w")

        ttk.Label(f, text="f_high [Hz]").grid(row=1, column=2, sticky="w")
        ttk.Entry(f, textvariable=self.f_high, width=10).grid(row=1, column=3, sticky="w")

        return f

    def _section_modulation(self, parent):
        f = ttk.LabelFrame(parent, text="Modulation (optional)")
        for j in range(6):
            f.columnconfigure(j, weight=1)

        ttk.Checkbutton(f, text="Use modulation", variable=self.use_mod).grid(row=0, column=0, sticky="w")

        ttk.Label(f, text="f_mod [Hz]").grid(row=0, column=2, sticky="w")
        ttk.Entry(f, textvariable=self.f_mod, width=10).grid(row=0, column=3, sticky="w")

        ttk.Label(f, text="depth").grid(row=0, column=4, sticky="w")
        ttk.Entry(f, textvariable=self.mod_depth, width=10).grid(row=0, column=5, sticky="w")
        return f

    def _section_fade(self, parent):
        f = ttk.LabelFrame(parent, text="Fade-in/out (optional)")
        for j in range(6):
            f.columnconfigure(j, weight=1)

        ttk.Checkbutton(f, text="Use fade", variable=self.use_fade).grid(row=0, column=0, sticky="w")

        ttk.Label(f, text="fade_in [s]").grid(row=0, column=2, sticky="w")
        ttk.Entry(f, textvariable=self.fade_in, width=10).grid(row=0, column=3, sticky="w")

        ttk.Label(f, text="fade_out [s]").grid(row=0, column=4, sticky="w")
        ttk.Entry(f, textvariable=self.fade_out, width=10).grid(row=0, column=5, sticky="w")
        return f

    def _section_seed(self, parent):
        f = ttk.LabelFrame(parent, text="Random seed (optional)")
        f.columnconfigure(1, weight=1)
        ttk.Label(f, text="seed (int or None)").grid(row=0, column=0, sticky="w")
        ttk.Entry(f, textvariable=self.seed, width=12).grid(row=0, column=1, sticky="w")
        ttk.Label(f, text="Tip: put None for random each run").grid(row=0, column=2, sticky="w")
        return f

    def _section_prop(self, parent):
        f = ttk.LabelFrame(parent, text="Propagation")
        for j in range(6):
            f.columnconfigure(j, weight=1)

        ttk.Label(f, text="c [m/s]").grid(row=0, column=0, sticky="w")
        ttk.Entry(f, textvariable=self.c, width=10).grid(row=0, column=1, sticky="w")

        ttk.Checkbutton(f, text="Include 1/r spreading", variable=self.use_spreading).grid(row=0, column=2, sticky="w")

        ttk.Label(f, text="gain@1m").grid(row=0, column=4, sticky="w")
        ttk.Entry(f, textvariable=self.gain_at_1m, width=10).grid(row=0, column=5, sticky="w")

        ttk.Checkbutton(f, text="Rigid floor image source", variable=self.use_floor_image).grid(row=1, column=0, sticky="w")

        ttk.Label(f, text="floor z").grid(row=1, column=2, sticky="w")
        ttk.Entry(f, textvariable=self.floor_z, width=10).grid(row=1, column=3, sticky="w")
        # --- Noise controls ---
        ttk.Separator(f).grid(row=2, column=0, columnspan=6, sticky="ew", pady=(8, 6))

        ttk.Checkbutton(f, text="Add noise (SNR)", variable=self.add_noise).grid(row=3, column=0, sticky="w")

        ttk.Label(f, text="SNR [dB]").grid(row=3, column=2, sticky="w")
        ttk.Entry(f, textvariable=self.snr_db, width=10).grid(row=3, column=3, sticky="w")

        ttk.Checkbutton(f, text="Independent per mic", variable=self.noise_indep).grid(row=3, column=4, sticky="w")

        ttk.Label(f, text="Noise seed").grid(row=4, column=0, sticky="w")
        ttk.Entry(f, textvariable=self.noise_seed, width=12).grid(row=4, column=1, sticky="w")
        ttk.Label(f, text="(int or None)").grid(row=4, column=2, sticky="w")

        # --- Denoise controls ---
        ttk.Separator(f).grid(row=5, column=0, columnspan=6, sticky="ew", pady=(8, 6))

        ttk.Checkbutton(f, text="Enable denoise (Wiener STFT)", variable=self.enable_denoise).grid(row=6, column=0, sticky="w")

        ttk.Label(f, text="STFT nperseg").grid(row=6, column=2, sticky="w")
        ttk.Entry(f, textvariable=self.denoise_nperseg, width=10).grid(row=6, column=3, sticky="w")

        ttk.Label(f, text="STFT noverlap").grid(row=6, column=4, sticky="w")
        ttk.Entry(f, textvariable=self.denoise_noverlap, width=10).grid(row=6, column=5, sticky="w")

        ttk.Label(f, text="Noise head [s]").grid(row=7, column=0, sticky="w")
        ttk.Entry(f, textvariable=self.denoise_noise_head, width=10).grid(row=7, column=1, sticky="w")

        ttk.Label(f, text="Noise tail [s]").grid(row=7, column=2, sticky="w")
        ttk.Entry(f, textvariable=self.denoise_noise_tail, width=10).grid(row=7, column=3, sticky="w")

        ttk.Label(f, text="Gain floor").grid(row=7, column=4, sticky="w")
        ttk.Entry(f, textvariable=self.denoise_gain_floor, width=10).grid(row=7, column=5, sticky="w")



        return f
    
    def _section_algorithms(self, parent):
        f = ttk.LabelFrame(parent, text="Algorithms (source localization)")
        for j in range(6):
            f.columnconfigure(j, weight=1)

        ttk.Checkbutton(f, text="Enable algorithms", variable=self.enable_algorithms).grid(row=0, column=0, sticky="w")

        ttk.Label(f, text="Algorithm:").grid(row=1, column=0, sticky="w")
        algo_menu = ttk.Combobox(
            f,
            textvariable=self.alg_choice,
            values=["None", "TDOA", "ER_LS", "ER_NLS", "MLE_EM_GROUND"],
            state="readonly",
            width=12,
        )
        algo_menu.grid(row=1, column=1, sticky="w")

        ttk.Separator(f).grid(row=2, column=0, columnspan=6, sticky="ew", pady=(8, 6))

        # --- TDOA params ---
        ttk.Label(f, text="TDOA (GCC-PHAT) settings").grid(row=3, column=0, sticky="w")

        ttk.Label(f, text="interp").grid(row=4, column=0, sticky="w")
        ttk.Entry(f, textvariable=self.tdoa_interp, width=8).grid(row=4, column=1, sticky="w")

        ttk.Label(f, text="Grid dx [m]").grid(row=4, column=2, sticky="w")
        ttk.Entry(f, textvariable=self.tdoa_grid_dx, width=8).grid(row=4, column=3, sticky="w")

        ttk.Label(f, text="Grid dy [m]").grid(row=4, column=4, sticky="w")
        ttk.Entry(f, textvariable=self.tdoa_grid_dy, width=8).grid(row=4, column=5, sticky="w")

        ttk.Label(f, text="Grid dz [m]").grid(row=5, column=0, sticky="w")
        ttk.Entry(f, textvariable=self.tdoa_grid_dz, width=8).grid(row=5, column=1, sticky="w")

        ttk.Label(f, text="z_fixed [m] (optional)").grid(row=5, column=2, sticky="w")
        ttk.Entry(f, textvariable=self.tdoa_z_fixed, width=10).grid(row=5, column=3, sticky="w")

        ttk.Checkbutton(
            f,
            text="Plot estimated source in 3D",
            variable=self.plot_estimated_source
        ).grid(row=6, column=0, sticky="w", pady=(6, 0))

        ttk.Label(f, text="Note: TDOA is a baseline (fast). Energy methods come next.").grid(
            row=7, column=0, columnspan=6, sticky="w", pady=(6, 0)
        )

        ttk.Separator(f).grid(row=8, column=0, columnspan=6, sticky="ew", pady=(10, 6))

        ttk.Label(f, text="Energy methods (ER-LS / ER-NLS) settings").grid(row=9, column=0, sticky="w")

        ttk.Label(f, text="ref_idx").grid(row=10, column=0, sticky="w")
        ttk.Entry(f, textvariable=self.er_ref_idx, width=8).grid(row=10, column=1, sticky="w")

        ttk.Label(f, text="window_s [s]").grid(row=10, column=2, sticky="w")
        ttk.Entry(f, textvariable=self.er_window_s, width=8).grid(row=10, column=3, sticky="w")

        ttk.Label(f, text="hop_s [s]").grid(row=10, column=4, sticky="w")
        ttk.Entry(f, textvariable=self.er_hop_s, width=8).grid(row=10, column=5, sticky="w")

        ttk.Label(f, text="trim_frac").grid(row=11, column=0, sticky="w")
        ttk.Entry(f, textvariable=self.er_trim_frac, width=8).grid(row=11, column=1, sticky="w")

        ttk.Label(f, text="kappa_eps").grid(row=11, column=2, sticky="w")
        ttk.Entry(f, textvariable=self.er_kappa_eps, width=8).grid(row=11, column=3, sticky="w")

        ttk.Label(f, text="ER-NLS max_iter").grid(row=12, column=0, sticky="w")
        ttk.Entry(f, textvariable=self.ernls_max_iter, width=8).grid(row=12, column=1, sticky="w")

        ttk.Label(f, text="lam").grid(row=12, column=2, sticky="w")
        ttk.Entry(f, textvariable=self.ernls_lam, width=8).grid(row=12, column=3, sticky="w")

        ttk.Label(f, text="tol_step").grid(row=12, column=4, sticky="w")
        ttk.Entry(f, textvariable=self.ernls_tol_step, width=10).grid(row=12, column=5, sticky="w")

        ttk.Label(f, text="tol_cost").grid(row=13, column=0, sticky="w")
        ttk.Entry(f, textvariable=self.ernls_tol_cost, width=10).grid(row=13, column=1, sticky="w")

        ttk.Separator(f).grid(row=14, column=0, columnspan=6, sticky="ew", pady=(10, 6))

        ttk.Label(f, text="MLE/EM ground settings").grid(row=15, column=0, sticky="w")

        ttk.Label(f, text="model_type").grid(row=16, column=0, sticky="w")
        ttk.Combobox(
            f,
            textvariable=self.mleem_model_type,
            values=["additive", "coherent"],
            state="readonly",
            width=12,
        ).grid(row=16, column=1, sticky="w")

        ttk.Label(f, text="init_method").grid(row=16, column=2, sticky="w")
        ttk.Combobox(
            f,
            textvariable=self.mleem_init_method,
            values=["barycenter", "er_ls_ground"],
            state="readonly",
            width=12,
        ).grid(row=16, column=3, sticky="w")

        ttk.Checkbutton(f, text="Estimate alpha", variable=self.mleem_estimate_alpha).grid(row=16, column=4, sticky="w")

        ttk.Label(f, text="alpha_init").grid(row=17, column=0, sticky="w")
        ttk.Entry(f, textvariable=self.mleem_alpha_init, width=8).grid(row=17, column=1, sticky="w")

        ttk.Label(f, text="alpha_min").grid(row=17, column=2, sticky="w")
        ttk.Entry(f, textvariable=self.mleem_alpha_min, width=8).grid(row=17, column=3, sticky="w")

        ttk.Label(f, text="alpha_max").grid(row=17, column=4, sticky="w")
        ttk.Entry(f, textvariable=self.mleem_alpha_max, width=8).grid(row=17, column=5, sticky="w")

        ttk.Label(f, text="alpha_grid").grid(row=18, column=0, sticky="w")
        ttk.Entry(f, textvariable=self.mleem_alpha_grid_size, width=8).grid(row=18, column=1, sticky="w")

        ttk.Label(f, text="max_iter").grid(row=18, column=2, sticky="w")
        ttk.Entry(f, textvariable=self.mleem_max_iter, width=8).grid(row=18, column=3, sticky="w")

        ttk.Label(f, text="lam").grid(row=18, column=4, sticky="w")
        ttk.Entry(f, textvariable=self.mleem_lam, width=8).grid(row=18, column=5, sticky="w")

        ttk.Label(f, text="tol_step").grid(row=19, column=0, sticky="w")
        ttk.Entry(f, textvariable=self.mleem_tol_step, width=10).grid(row=19, column=1, sticky="w")

        ttk.Label(f, text="tol_cost").grid(row=19, column=2, sticky="w")
        ttk.Entry(f, textvariable=self.mleem_tol_cost, width=10).grid(row=19, column=3, sticky="w")

        ttk.Label(f, text="fd_eps").grid(row=19, column=4, sticky="w")
        ttk.Entry(f, textvariable=self.mleem_fd_eps, width=10).grid(row=19, column=5, sticky="w")

        ttk.Label(f, text="barycenter z offset").grid(row=20, column=0, sticky="w")
        ttk.Entry(f, textvariable=self.mleem_barycenter_z_offset, width=10).grid(row=20, column=1, sticky="w")

        return f


    def _section_plots(self, parent):
        f = ttk.LabelFrame(parent, text="Select plots / outputs")
        f.columnconfigure(0, weight=1)

        ttk.Checkbutton(f, text="Plot 3D scene (room + mics + source)", variable=self.plot_scene).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(f, text="Plot source signal (time)", variable=self.plot_source_time).grid(row=1, column=0, sticky="w")
        ttk.Checkbutton(f, text="Plot source spectrum (FFT)", variable=self.plot_source_spectrum).grid(row=2, column=0, sticky="w")
        ttk.Checkbutton(f, text="Plot mic spectrogram", variable=self.plot_mic_spectrogram).grid(row=4, column=0, sticky="w")

        row_mic = ttk.Frame(f)
        row_mic.grid(row=5, column=0, sticky="w", pady=(2, 0))
        ttk.Label(row_mic, text="Mic index for spectrogram:").grid(row=0, column=0, sticky="w")
        ttk.Entry(row_mic, textvariable=self.mic_spec_index, width=6).grid(row=0, column=1, sticky="w", padx=(6, 0))

        ttk.Checkbutton(f, text="Plot microphone signals", variable=self.plot_mics).grid(row=5, column=0, sticky="w")
        ttk.Checkbutton(f, text="Plot microphone signals (zoom)", variable=self.plot_mics_zoom).grid(row=6, column=0, sticky="w")
        ttk.Checkbutton(f,text="Plot source spectrogram",variable=self.plot_source_spectrogram).grid(row=7, column=0, sticky="w")
        ttk.Checkbutton(f, text="Print Δr (distances differences)", variable=self.print_delta_r).grid(row=8, column=0, sticky="w")

        ttk.Separator(f).grid(row=9, column=0, sticky="ew", pady=(8, 6))

        ttk.Checkbutton(f,text="Plot denoise (before / after Wiener)",variable=self.plot_denoise_compare).grid(row=10, column=0, sticky="w")

        row_dn = ttk.Frame(f)
        row_dn.grid(row=11, column=0, sticky="w", pady=(2, 0))
        ttk.Label(row_dn, text="Mic index:").grid(row=0, column=0, sticky="w")
        ttk.Entry(row_dn, textvariable=self.plot_denoise_mic, width=6).grid(row=0, column=1, sticky="w", padx=(6, 0))

        return f

    def _section_zoom(self, parent):
        f = ttk.LabelFrame(parent, text="Zoom")
        f.columnconfigure(1, weight=1)
        ttk.Label(f, text="tmax [s]").grid(row=0, column=0, sticky="w")
        ttk.Entry(f, textvariable=self.zoom_tmax, width=10).grid(row=0, column=1, sticky="w")
        return f

    def run_experiment(self):
        try:
            # --- Read geometry
            Lx = _safe_float(self.Lx, "Lx")
            Ly = _safe_float(self.Ly, "Ly")
            Lz = _safe_float(self.Lz, "Lz")
            margin = _safe_float(self.margin, "margin")

            src = np.array([
                _safe_float(self.src_x, "src_x"),
                _safe_float(self.src_y, "src_y"),
                _safe_float(self.src_z, "src_z"),
            ], dtype=float)

            mic_positions = _parse_floats_csv(self.mic_box.get("1.0", "end"))

            room = make_semi_anechoic_room(Lx=Lx, Ly=Ly, Lz=Lz, floor_rigid=True)
            mics = MicArray(positions=mic_positions, name="ui_mics")
            source = Source(position=as_vec3(float(src[0]), float(src[1]), float(src[2])),
                            kind=SourceType.MONOPOLE, name="S1")

            scene = Scene(room=room, mic_array=mics, sources=[source])
            scene.validate(margin=margin)

            # --- Signal cfg
            fs = _safe_float(self.fs, "fs")
            duration = _safe_float(self.duration, "duration")
            f_low = _safe_float(self.f_low, "f_low")
            f_high = _safe_float(self.f_high, "f_high")
            rms = _safe_float(self.rms, "rms")

            seed = _safe_int_or_none(self.seed, "seed")

            mod = None
            if self.use_mod.get():
                mod = ModulationSpec(
                    f_mod=_safe_float(self.f_mod, "f_mod"),
                    depth=_safe_float(self.mod_depth, "mod_depth"),
                )

            fade = None
            if self.use_fade.get():
                fade = FadeSpec(
                    fade_in=_safe_float(self.fade_in, "fade_in"),
                    fade_out=_safe_float(self.fade_out, "fade_out"),
                )

            sig_cfg = SourceSignalConfig(
                fs=fs,
                duration=duration,
                band=BandpassSpec(f_low=f_low, f_high=f_high),
                rms=rms,
                modulation=mod,
                fade=fade,
                seed=seed,
            )
            t, s = generate_source_signal(sig_cfg)
            fs = 1.0 / float(t[1] - t[0])


            # --- Propagation cfg
            prop_cfg = PropagationConfig(
                c=_safe_float(self.c, "c"),
                include_spherical_spreading=self.use_spreading.get(),
                gain_at_1m=_safe_float(self.gain_at_1m, "gain_at_1m"),
                include_rigid_floor_image=self.use_floor_image.get(),
                floor_z=_safe_float(self.floor_z, "floor_z"),
            )

            y, d = simulate_mic_signals_free_field(scene, t, s, prop_cfg)

            # --- Optional noise ---
            if self.add_noise.get():
                noise_cfg = NoiseConfig(
                    snr_db=_safe_float(self.snr_db, "snr_db"),
                    per_mic_independent=bool(self.noise_indep.get()),
                    seed=_safe_int_or_none(self.noise_seed, "noise_seed"),
                )
                y, noise_rms = add_awgn_snr(y, noise_cfg)
                print(f"Added noise: target SNR={noise_cfg.snr_db} dB, noise_rms={noise_rms:.6g}")

            x_hat = None  # estimated source position (optional)
            err = None
            dbg = None

            if self.enable_algorithms.get() and self.alg_choice.get() == "TDOA":
                fs = 1.0 / float(t[1] - t[0])

                # --- Grid search config (déjà dans ton UI) ---
                z_fixed_txt = self.tdoa_z_fixed.get().strip()
                z_fixed = None if z_fixed_txt == "" else float(z_fixed_txt)

                grid_cfg = GridSearchConfig(
                    dx=float(self.tdoa_grid_dx.get()),
                    dy=float(self.tdoa_grid_dy.get()),
                    dz=float(self.tdoa_grid_dz.get()),
                    z_fixed=z_fixed,
                )

                # --- Denoise config (pour l'instant: valeurs fixes, tu pourras en faire des champs UI après) ---
                den_cfg = DenoiseConfig(
                    nperseg=int(float(self.denoise_nperseg.get())),
                    noverlap=int(float(self.denoise_noverlap.get())),
                    noise_head_s=float(self.denoise_noise_head.get()),
                    noise_tail_s=float(self.denoise_noise_tail.get()),
                    gain_floor=float(self.denoise_gain_floor.get()),
                    f_low=float(self.f_low.get()),
                    f_high=float(self.f_high.get()),
                )

                algo_cfg = MultiRefTDOAConfig(
                    interp=int(float(self.tdoa_interp.get())),
                    enable_denoise=bool(self.enable_denoise.get()),
                    denoise=den_cfg,
                    fusion="median",
                )

                algo_cfg = MultiRefTDOAConfig(
                    interp=int(float(self.tdoa_interp.get())),
                    enable_denoise=True,
                    denoise=den_cfg,
                    fusion="median",
                )

                x_hat, dbg = estimate_source_multiref_tdoa(
                    mic_positions=np.asarray(scene.mic_array.positions, float),
                    y=y,                 # IMPORTANT: y doit déjà contenir le bruit si add_noise
                    fs=fs,
                    c=prop_cfg.c,
                    room_size=np.asarray(scene.room.size, float),
                    grid_cfg=grid_cfg,
                    cfg=algo_cfg,
                )

            elif self.enable_algorithms.get() and self.alg_choice.get() == "ER_LS":
                
                fs_sig = 1.0 / float(t[1] - t[0])

                cfg_E = EnergyConfig(
                    remove_mean=True,
                    window_s=float(self.er_window_s.get()),
                    hop_s=float(self.er_hop_s.get()),
                    trim_frac=float(self.er_trim_frac.get()),
                )

                if not self.use_floor_image.get():
                    # --- champ libre (ancienne version) ---
                    cfg_ls = ERLSConfig(
                        ref_idx=int(float(self.er_ref_idx.get())),
                        kappa_eps=float(self.er_kappa_eps.get()),
                        min_pairs=3,
                    )

                    x_hat, err, dbg = er_ls(
                        mic_positions=np.asarray(scene.mic_array.positions, float),
                        y=y,
                        fs=fs_sig,
                        cfg_E=cfg_E,
                        cfg=cfg_ls,
                        mic_gains=None,
                    )

                    # (option) plot sphères seulement en champ libre
                    from chamber_sim.er_ls.viz import plot_apollonius_spheres_3d
                    plot_apollonius_spheres_3d(
                        mic_positions=np.asarray(scene.mic_array.positions, float),
                        spheres=dbg["spheres"],
                        x_true=np.asarray(scene.sources[0].position, float),
                        x_hat=np.asarray(x_hat, float),
                        room_size=np.asarray(scene.room.size, float),
                        select_mode="radius",
                    )

                else:
                    # --- sol rigide (nouvelle version) ---
                    gcfg = GroundConfig(
                        floor_z=_safe_float(self.floor_z, "floor_z"),
                        enforce_z_positive=True,
                    )

                    cfg_g = ERLSGroundConfig(
                        # tu peux mettre des valeurs fixes ou plus tard les ajouter à l’UI
                        max_iter=30,
                        lam=1e-2,
                        tol_step=1e-6,
                        tol_cost=1e-9,
                        beta_min=0.0,
                        beta_max=2.0,
                        beta_grid=41,
                    )

                    # init : tu peux reprendre la source “true” pour tester, mais en réel ce sera None
                    x0 = None

                    x_hat, err, dbg = er_ls_ground(
                        mic_positions=np.asarray(scene.mic_array.positions, float),
                        y=y,
                        fs=fs_sig,
                        x0=x0,
                        cfg_E=cfg_E,
                        cfg=cfg_g,
                        gcfg=gcfg,
                    )

                    print("beta_hat:", dbg["beta_hat"], "K_hat:", dbg["K_hat"])


                print("----- ER-LS RESULTS -----")
                print("Estimated source [m]:", np.round(x_hat, 3))
                print("ER-LS residual RMS [m]:", err)
                print("True source [m]:", np.round(scene.sources[0].position, 3))

            elif self.enable_algorithms.get() and self.alg_choice.get() == "ER_NLS":
                fs_sig = 1.0 / float(t[1] - t[0])

                cfg_E = EnergyConfig(
                    remove_mean=True,
                    window_s=float(self.er_window_s.get()),
                    hop_s=float(self.er_hop_s.get()),
                    trim_frac=float(self.er_trim_frac.get()),
                )

                if not self.use_floor_image.get():
                    cfg_ls = ERLSConfig(
                        ref_idx=int(float(self.er_ref_idx.get())),
                        kappa_eps=float(self.er_kappa_eps.get()),
                        min_pairs=3,
                    )
                    cfg_nls = ERNLSConfig(
                        max_iter=int(float(self.ernls_max_iter.get())),
                        lam=float(self.ernls_lam.get()),
                        tol_step=float(self.ernls_tol_step.get()),
                        tol_cost=float(self.ernls_tol_cost.get()),
                        kappa_eps=float(self.er_kappa_eps.get()),
                    )

                    x_hat, err, dbg = er_nls(
                        mic_positions=np.asarray(scene.mic_array.positions, float),
                        y=y,
                        fs=fs_sig,
                        x0=None,
                        cfg_E=cfg_E,
                        cfg_ls=cfg_ls,
                        cfg=cfg_nls,
                        mic_gains=None,
                    )

                else:
                    gcfg = GroundConfig(
                        floor_z=_safe_float(self.floor_z, "floor_z"),
                        enforce_z_positive=True,
                    )

                    cfg_ls_g = ERLSGroundConfig(
                        max_iter=30,
                        lam=1e-2,
                        tol_step=1e-6,
                        tol_cost=1e-9,
                        beta_min=0.0,
                        beta_max=2.0,
                        beta_grid=41,
                    )

                    cfg_nls_g = ERNLSGroundConfig(
                        max_iter=int(float(self.ernls_max_iter.get())),
                        lam=float(self.ernls_lam.get()),
                        tol_step=float(self.ernls_tol_step.get()),
                        tol_cost=float(self.ernls_tol_cost.get()),
                    )

                    x_hat, err, dbg = er_nls_ground(
                        mic_positions=np.asarray(scene.mic_array.positions, float),
                        y=y,
                        fs=fs_sig,
                        x0=None,
                        beta=None,          # None => prend beta_hat de ER-LS-ground
                        cfg_E=cfg_E,
                        cfg_ls_g=cfg_ls_g,
                        cfg=cfg_nls_g,
                        gcfg=gcfg,
                    )

                    print("beta used:", dbg["beta"], "K_hat:", dbg["K_hat"])

            elif self.enable_algorithms.get() and self.alg_choice.get() == "MLE_EM_GROUND":
                if not self.use_floor_image.get():
                    raise ValueError("MLE_EM_GROUND requires 'Rigid floor image source' to be enabled.")

                fs_sig = 1.0 / float(t[1] - t[0])

                cfg_E = EnergyConfig(
                    remove_mean=True,
                    window_s=float(self.er_window_s.get()),
                    hop_s=float(self.er_hop_s.get()),
                    trim_frac=float(self.er_trim_frac.get()),
                )

                cfg_init = MLEEMInitConfig(
                    method=self.mleem_init_method.get(),
                    barycenter_z_offset=float(self.mleem_barycenter_z_offset.get()),
                )

                cfg_mle = MLEEMGroundConfig(
                    model_type=self.mleem_model_type.get(),
                    max_iter=int(float(self.mleem_max_iter.get())),
                    lam=float(self.mleem_lam.get()),
                    tol_step=float(self.mleem_tol_step.get()),
                    tol_cost=float(self.mleem_tol_cost.get()),
                    estimate_alpha=bool(self.mleem_estimate_alpha.get()),
                    alpha_init=float(self.mleem_alpha_init.get()),
                    alpha_min=float(self.mleem_alpha_min.get()),
                    alpha_max=float(self.mleem_alpha_max.get()),
                    alpha_grid_size=int(float(self.mleem_alpha_grid_size.get())),
                    fd_eps=float(self.mleem_fd_eps.get()),
                    f_low_hz=float(self.f_low.get()),
                    f_high_hz=float(self.f_high.get()),
                    sound_speed=float(self.c.get()),
                    estimate_noise_floor=False,
                    store_history=True,
                )

                gcfg_mle = MLEEMGeometryConfig(
                    floor_z=_safe_float(self.floor_z, "floor_z"),
                    enforce_z_positive=True,
                    z_margin=1e-6,
                )

                dbg_flags = MLEEMDebugFlags(
                    keep_alpha_grid_details=True,
                    keep_iteration_details=True,
                )

                x_hat, err, dbg = mle_em_ground(
                    mic_positions=np.asarray(scene.mic_array.positions, float),
                    y=y,
                    fs=fs_sig,
                    x0=None,
                    alpha=None,
                    cfg_E=cfg_E,
                    cfg_init=cfg_init,
                    cfg=cfg_mle,
                    gcfg=gcfg_mle,
                    dbg_flags=dbg_flags,
                )

                print("----- MLE/EM GROUND RESULTS -----")
                print("Estimated source [m]:", np.round(x_hat, 3))
                print("Residual RMS:", err)
                print("True source [m]:", np.round(scene.sources[0].position, 3))
                print("alpha_hat:", dbg.get("alpha_hat", None), "K_hat:", dbg.get("K_hat", None))

            # --- Plot denoise before / after ---
            if self.plot_denoise_compare.get() and dbg is not None:
                mic_idx = int(self.plot_denoise_mic.get())
                if mic_idx < 0 or mic_idx >= y.shape[0]:
                    raise ValueError(f"plot_denoise_mic out of range [0,{y.shape[0]-1}]")

                y_before = y[mic_idx]
                y_used = dbg.get("y_used", None) if isinstance(dbg, dict) else None
                y_after = (y_used[mic_idx] if y_used is not None else y[mic_idx])


                plt.figure(figsize=(10, 6))

                plt.subplot(2, 1, 1)
                plt.plot(t, y_before, linewidth=0.8)
                plt.title(f"Mic {mic_idx} — before Wiener")
                plt.xlabel("t [s]")
                plt.ylabel("Amplitude")

                plt.subplot(2, 1, 2)
                plt.plot(t, y_after, linewidth=0.8)
                plt.title(f"Mic {mic_idx} — after Wiener")
                plt.xlabel("t [s]")
                plt.ylabel("Amplitude")

                plt.tight_layout()

                plt.figure(figsize=(10, 5))

                Yb = np.fft.rfft(y_before)
                Ya = np.fft.rfft(y_after)
                freqs = np.fft.rfftfreq(y_before.size, d=1.0/fs)

                plt.plot(freqs, 20*np.log10(np.abs(Yb) + 1e-12), label="Before", alpha=0.7)
                plt.plot(freqs, 20*np.log10(np.abs(Ya) + 1e-12), label="After", alpha=0.7)
                plt.xlim(0, 500)
                plt.xlabel("f [Hz]")
                plt.ylabel("Magnitude [dB]")
                plt.title(f"Mic {mic_idx} spectrum — before / after Wiener")
                plt.legend()
                plt.tight_layout()



                print("----- MULTI-REF TDOA (GCC-PHAT) -----")
                print("Estimated source [m]:", np.round(x_hat, 3))
                print("True source [m]:", np.round(scene.sources[0].position, 3))
                print("Costs per ref:", np.round(dbg["costs_per_ref"], 6))




            if self.plot_mic_spectrogram.get():
                mic_idx = int(self.mic_spec_index.get())
                if mic_idx < 0 or mic_idx >= y.shape[0]:
                    raise ValueError(f"mic_spec_index out of range. Must be in [0, {y.shape[0]-1}]")

                plt.figure()
                plt.specgram(y[mic_idx], NFFT=2048, Fs=fs, noverlap=1024)
                plt.title(f"Mic M{mic_idx} spectrogram")
                plt.xlabel("t [s]")
                plt.ylabel("f [Hz]")
                plt.ylim(0, min(2000, 0.5 * fs))
                plt.tight_layout()


            # --- Outputs / plots
            if self.print_delta_r.get():
                d_ref = d[0]
                delta_r = d - d_ref
                print("Distances r_i [m]      :", np.round(d, 4))
                print("Delta r_i = r_i - r_0 :", np.round(delta_r, 4))

            if self.plot_scene.get():
                fig, ax = plot_scene_3d(scene, VizOptions(title="Chambre 3D – micros & source"))
                if x_hat is not None and self.plot_estimated_source.get():
                    ax.scatter([x_hat[0]], [x_hat[1]], [x_hat[2]], marker="X", s=160, label="Estimated")
                    ax.legend(loc="upper left")

            if self.plot_source_time.get():
                plt.figure()
                plt.plot(t, s)
                plt.title("Source signal s(t)")
                plt.xlabel("t [s]")
                plt.ylabel("Amplitude")
                plt.tight_layout()

            if self.plot_source_spectrum.get():
                S = np.fft.rfft(s)
                freqs = np.fft.rfftfreq(s.size, d=1.0 / fs)
                plt.figure()
                plt.plot(freqs, 20*np.log10(np.maximum(np.abs(S), 1e-12)))
                plt.title("Source spectrum |S(f)|")
                plt.xlabel("f [Hz]")
                plt.ylabel("Magnitude [dB]")
                plt.xlim(0, 0.5 * fs)
                plt.tight_layout()

            if self.plot_mics.get():
                plt.figure()
                ax = plt.gca()

                lines = []
                for i in range(y.shape[0]):
                    (ln,) = ax.plot(t, y[i], label=f"M{i}", linewidth=1.0)
                    lines.append(ln)

                ax.set_title("Microphone signals (click legend to show/hide)")
                ax.set_xlabel("t [s]")
                ax.set_ylabel("Amplitude")

                leg = ax.legend(loc="upper right", ncol=2, fontsize=9)
                leg.set_draggable(True)

                # Make legend entries clickable
                for legline in leg.get_lines():
                    legline.set_picker(5)
                for legtext in leg.get_texts():
                    legtext.set_picker(True)

                legend_map = {}
                for legline, legtext, origline in zip(leg.get_lines(), leg.get_texts(), lines):
                    legend_map[legline] = origline
                    legend_map[legtext] = origline

                def on_pick(event):
                    artist = event.artist
                    if artist not in legend_map:
                        return
                    orig = legend_map[artist]
                    visible = not orig.get_visible()
                    orig.set_visible(visible)

                    alpha = 1.0 if visible else 0.2
                    label = orig.get_label()

                    for txt in leg.get_texts():
                        if txt.get_text() == label:
                            txt.set_alpha(alpha)

                    event.canvas.draw()

                plt.gcf().canvas.mpl_connect("pick_event", on_pick)

            if self.plot_source_spectrogram.get():
                plt.figure()
                plt.specgram(s, NFFT=2048, Fs=fs, noverlap=1024)
                plt.title("Source spectrogram")
                plt.xlabel("t [s]")
                plt.ylabel("f [Hz]")
                plt.ylim(0, min(2000, 0.5 * fs))  # ajuste si tu veux
                plt.tight_layout()

            if self.plot_mics_zoom.get():
                tmax = _safe_float(self.zoom_tmax, "zoom_tmax")
                mask = t <= tmax

                plt.figure()
                ax = plt.gca()

                lines = []
                for i in range(y.shape[0]):
                    (ln,) = ax.plot(t[mask], y[i][mask], label=f"M{i}", linewidth=1.0)
                    lines.append(ln)

                ax.set_title(f"Microphone signals (zoom to {tmax}s) — click legend to show/hide")
                ax.set_xlabel("t [s]")
                ax.set_ylabel("Amplitude")

                leg = ax.legend(loc="upper right", ncol=2, fontsize=9)
                leg.set_draggable(True)

                for legline in leg.get_lines():
                    legline.set_picker(5)
                for legtext in leg.get_texts():
                    legtext.set_picker(True)

                legend_map = {}
                for legline, legtext, origline in zip(leg.get_lines(), leg.get_texts(), lines):
                    legend_map[legline] = origline
                    legend_map[legtext] = origline

                def on_pick(event):
                    artist = event.artist
                    if artist not in legend_map:
                        return
                    orig = legend_map[artist]
                    visible = not orig.get_visible()
                    orig.set_visible(visible)

                    alpha = 1.0 if visible else 0.2
                    label = orig.get_label()
                    for txt in leg.get_texts():
                        if txt.get_text() == label:
                            txt.set_alpha(alpha)

                    event.canvas.draw()

                plt.gcf().canvas.mpl_connect("pick_event", on_pick)


            plt.show()

        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def _get_config_dict(self) -> dict:
        """Read all UI fields into a JSON-serializable dict."""
        cfg = {
            "geometry": {
                "Lx": self.Lx.get(),
                "Ly": self.Ly.get(),
                "Lz": self.Lz.get(),
                "margin": self.margin.get(),
                "source": {"x": self.src_x.get(), "y": self.src_y.get(), "z": self.src_z.get()},
                "mics_csv": self.mic_box.get("1.0", "end").strip(),
            },
            "signal": {
                "fs": self.fs.get(),
                "duration": self.duration.get(),
                "f_low": self.f_low.get(),
                "f_high": self.f_high.get(),
                "rms": self.rms.get(),
                "seed": self.seed.get(),
                "use_mod": bool(self.use_mod.get()),
                "f_mod": self.f_mod.get(),
                "mod_depth": self.mod_depth.get(),
                "use_fade": bool(self.use_fade.get()),
                "fade_in": self.fade_in.get(),
                "fade_out": self.fade_out.get(),
            },
            "propagation": {
                "c": self.c.get(),
                "use_spreading": bool(self.use_spreading.get()),
                "gain_at_1m": self.gain_at_1m.get(),
                "use_floor_image": bool(self.use_floor_image.get()),
                "floor_z": self.floor_z.get(),
                "add_noise": bool(self.add_noise.get()),
                "snr_db": self.snr_db.get(),
                "noise_indep": bool(self.noise_indep.get()),
                "noise_seed": self.noise_seed.get(),
                "enable_denoise": bool(self.enable_denoise.get()),
                "denoise_nperseg": self.denoise_nperseg.get(),
                "denoise_noverlap": self.denoise_noverlap.get(),
                "denoise_noise_head": self.denoise_noise_head.get(),
                "denoise_noise_tail": self.denoise_noise_tail.get(),
                "denoise_gain_floor": self.denoise_gain_floor.get(),


            },
            "algorithms": {
                "enable_algorithms": bool(self.enable_algorithms.get()),
                "alg_choice": self.alg_choice.get(),

                "tdoa_interp": self.tdoa_interp.get(),
                "tdoa_grid_dx": self.tdoa_grid_dx.get(),
                "tdoa_grid_dy": self.tdoa_grid_dy.get(),
                "tdoa_grid_dz": self.tdoa_grid_dz.get(),
                "tdoa_z_fixed": self.tdoa_z_fixed.get(),

                "plot_estimated_source": bool(self.plot_estimated_source.get()),

                "er_ref_idx": self.er_ref_idx.get(),
                "er_window_s": self.er_window_s.get(),
                "er_hop_s": self.er_hop_s.get(),
                "er_trim_frac": self.er_trim_frac.get(),
                "er_kappa_eps": self.er_kappa_eps.get(),
                "ernls_max_iter": self.ernls_max_iter.get(),
                "ernls_lam": self.ernls_lam.get(),
                "ernls_tol_step": self.ernls_tol_step.get(),
                "ernls_tol_cost": self.ernls_tol_cost.get(),

                "mleem_model_type": self.mleem_model_type.get(),
                "mleem_init_method": self.mleem_init_method.get(),
                "mleem_max_iter": self.mleem_max_iter.get(),
                "mleem_lam": self.mleem_lam.get(),
                "mleem_tol_step": self.mleem_tol_step.get(),
                "mleem_tol_cost": self.mleem_tol_cost.get(),
                "mleem_estimate_alpha": bool(self.mleem_estimate_alpha.get()),
                "mleem_alpha_init": self.mleem_alpha_init.get(),
                "mleem_alpha_min": self.mleem_alpha_min.get(),
                "mleem_alpha_max": self.mleem_alpha_max.get(),
                "mleem_alpha_grid_size": self.mleem_alpha_grid_size.get(),
                "mleem_fd_eps": self.mleem_fd_eps.get(),
                "mleem_barycenter_z_offset": self.mleem_barycenter_z_offset.get(),

            },

            "plots": {
                "plot_source_spectrogram": bool(self.plot_source_spectrogram.get()),
                "plot_scene": bool(self.plot_scene.get()),
                "plot_source_time": bool(self.plot_source_time.get()),
                "plot_source_spectrum": bool(self.plot_source_spectrum.get()),
                "plot_mic_spectrogram": bool(self.plot_mic_spectrogram.get()),
                "mic_spec_index": self.mic_spec_index.get(),
                "plot_mics": bool(self.plot_mics.get()),
                "plot_mics_zoom": bool(self.plot_mics_zoom.get()),
                "print_delta_r": bool(self.print_delta_r.get()),
                "zoom_tmax": self.zoom_tmax.get(),
            },
        }
        return cfg

    def _apply_config_dict(self, cfg: dict) -> None:
        """Apply a config dict to the UI fields."""
        g = cfg.get("geometry", {})
        self.Lx.set(g.get("Lx", self.Lx.get()))
        self.Ly.set(g.get("Ly", self.Ly.get()))
        self.Lz.set(g.get("Lz", self.Lz.get()))
        self.margin.set(g.get("margin", self.margin.get()))

        src = g.get("source", {})
        self.src_x.set(src.get("x", self.src_x.get()))
        self.src_y.set(src.get("y", self.src_y.get()))
        self.src_z.set(src.get("z", self.src_z.get()))

        mics_csv = g.get("mics_csv", None)
        if mics_csv is not None:
            self.mic_box.delete("1.0", "end")
            self.mic_box.insert("1.0", mics_csv)

        s = cfg.get("signal", {})
        self.fs.set(s.get("fs", self.fs.get()))
        self.duration.set(s.get("duration", self.duration.get()))
        self.f_low.set(s.get("f_low", self.f_low.get()))
        self.f_high.set(s.get("f_high", self.f_high.get()))
        self.rms.set(s.get("rms", self.rms.get()))
        self.seed.set(s.get("seed", self.seed.get()))

        self.use_mod.set(bool(s.get("use_mod", self.use_mod.get())))
        self.f_mod.set(s.get("f_mod", self.f_mod.get()))
        self.mod_depth.set(s.get("mod_depth", self.mod_depth.get()))

        self.use_fade.set(bool(s.get("use_fade", self.use_fade.get())))
        self.fade_in.set(s.get("fade_in", self.fade_in.get()))
        self.fade_out.set(s.get("fade_out", self.fade_out.get()))

        a = cfg.get("algorithms", {})
        self.enable_algorithms.set(bool(a.get("enable_algorithms", self.enable_algorithms.get())))
        self.alg_choice.set(a.get("alg_choice", self.alg_choice.get()))

        self.tdoa_interp.set(a.get("tdoa_interp", self.tdoa_interp.get()))
        self.tdoa_grid_dx.set(a.get("tdoa_grid_dx", self.tdoa_grid_dx.get()))
        self.tdoa_grid_dy.set(a.get("tdoa_grid_dy", self.tdoa_grid_dy.get()))
        self.tdoa_grid_dz.set(a.get("tdoa_grid_dz", self.tdoa_grid_dz.get()))
        self.tdoa_z_fixed.set(a.get("tdoa_z_fixed", self.tdoa_z_fixed.get()))

        self.plot_estimated_source.set(bool(a.get("plot_estimated_source", self.plot_estimated_source.get())))

        self.er_ref_idx.set(a.get("er_ref_idx", self.er_ref_idx.get()))
        self.er_window_s.set(a.get("er_window_s", self.er_window_s.get()))
        self.er_hop_s.set(a.get("er_hop_s", self.er_hop_s.get()))
        self.er_trim_frac.set(a.get("er_trim_frac", self.er_trim_frac.get()))
        self.er_kappa_eps.set(a.get("er_kappa_eps", self.er_kappa_eps.get()))
        self.ernls_max_iter.set(a.get("ernls_max_iter", self.ernls_max_iter.get()))
        self.ernls_lam.set(a.get("ernls_lam", self.ernls_lam.get()))
        self.ernls_tol_step.set(a.get("ernls_tol_step", self.ernls_tol_step.get()))
        self.ernls_tol_cost.set(a.get("ernls_tol_cost", self.ernls_tol_cost.get()))

        self.mleem_model_type.set(a.get("mleem_model_type", self.mleem_model_type.get()))
        self.mleem_init_method.set(a.get("mleem_init_method", self.mleem_init_method.get()))
        self.mleem_max_iter.set(a.get("mleem_max_iter", self.mleem_max_iter.get()))
        self.mleem_lam.set(a.get("mleem_lam", self.mleem_lam.get()))
        self.mleem_tol_step.set(a.get("mleem_tol_step", self.mleem_tol_step.get()))
        self.mleem_tol_cost.set(a.get("mleem_tol_cost", self.mleem_tol_cost.get()))
        self.mleem_estimate_alpha.set(bool(a.get("mleem_estimate_alpha", self.mleem_estimate_alpha.get())))
        self.mleem_alpha_init.set(a.get("mleem_alpha_init", self.mleem_alpha_init.get()))
        self.mleem_alpha_min.set(a.get("mleem_alpha_min", self.mleem_alpha_min.get()))
        self.mleem_alpha_max.set(a.get("mleem_alpha_max", self.mleem_alpha_max.get()))
        self.mleem_alpha_grid_size.set(a.get("mleem_alpha_grid_size", self.mleem_alpha_grid_size.get()))
        self.mleem_fd_eps.set(a.get("mleem_fd_eps", self.mleem_fd_eps.get()))
        self.mleem_barycenter_z_offset.set(a.get("mleem_barycenter_z_offset", self.mleem_barycenter_z_offset.get()))

        p = cfg.get("propagation", {})
        self.c.set(p.get("c", self.c.get()))
        self.use_spreading.set(bool(p.get("use_spreading", self.use_spreading.get())))
        self.gain_at_1m.set(p.get("gain_at_1m", self.gain_at_1m.get()))
        self.use_floor_image.set(bool(p.get("use_floor_image", self.use_floor_image.get())))
        self.floor_z.set(p.get("floor_z", self.floor_z.get()))
        self.add_noise.set(bool(p.get("add_noise", self.add_noise.get())))
        self.snr_db.set(p.get("snr_db", self.snr_db.get()))
        self.noise_indep.set(bool(p.get("noise_indep", self.noise_indep.get())))
        self.noise_seed.set(p.get("noise_seed", self.noise_seed.get()))
        self.enable_denoise.set(bool(p.get("enable_denoise", self.enable_denoise.get())))
        self.denoise_nperseg.set(p.get("denoise_nperseg", self.denoise_nperseg.get()))
        self.denoise_noverlap.set(p.get("denoise_noverlap", self.denoise_noverlap.get()))
        self.denoise_noise_head.set(p.get("denoise_noise_head", self.denoise_noise_head.get()))
        self.denoise_noise_tail.set(p.get("denoise_noise_tail", self.denoise_noise_tail.get()))
        self.denoise_gain_floor.set(p.get("denoise_gain_floor", self.denoise_gain_floor.get()))



        pl = cfg.get("plots", {})
        self.plot_scene.set(bool(pl.get("plot_scene", self.plot_scene.get())))
        self.plot_source_time.set(bool(pl.get("plot_source_time", self.plot_source_time.get())))
        self.plot_source_spectrum.set(bool(pl.get("plot_source_spectrum", self.plot_source_spectrum.get())))
        self.plot_mic_spectrogram.set(bool(pl.get("plot_mic_spectrogram", self.plot_mic_spectrogram.get())))
        self.mic_spec_index.set(pl.get("mic_spec_index", self.mic_spec_index.get()))
        self.plot_mics.set(bool(pl.get("plot_mics", self.plot_mics.get())))
        self.plot_mics_zoom.set(bool(pl.get("plot_mics_zoom", self.plot_mics_zoom.get())))
        self.print_delta_r.set(bool(pl.get("print_delta_r", self.print_delta_r.get())))
        self.zoom_tmax.set(pl.get("zoom_tmax", self.zoom_tmax.get()))
        self.plot_source_spectrogram.set(bool(pl.get("plot_source_spectrogram", self.plot_source_spectrogram.get())))


    def save_config(self) -> None:
        cfg = self._get_config_dict()
        path = filedialog.asksaveasfilename(
            title="Save config",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
            messagebox.showinfo("Saved", f"Config saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save config:\n{e}")

    def load_config(self) -> None:
        path = filedialog.askopenfilename(
            title="Load config",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            self._apply_config_dict(cfg)
            messagebox.showinfo("Loaded", f"Config loaded from:\n{path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load config:\n{e}")

    def _set_mics_from_array(self, arr: np.ndarray) -> None:
        """Write mic positions (M,3) into the text box as CSV lines."""
        lines = [f"{x:.3f},{y:.3f},{z:.3f}" for x, y, z in arr]
        self.mic_box.delete("1.0", "end")
        self.mic_box.insert("1.0", "\n".join(lines))

    def preset_square_4(self) -> None:
        """4 mics in a square, centered in the room, at a given height."""
        try:
            Lx = float(self.Lx.get())
            Ly = float(self.Ly.get())
            Lz = float(self.Lz.get())
            margin = float(self.margin.get())

            z = min(1.5, max(margin, 0.4 * Lz))  # default-ish height
            cx, cy = 0.5 * Lx, 0.5 * Ly
            spacing = min(0.5, 0.25 * min(Lx, Ly))  # meters

            half = 0.5 * spacing
            mics = np.array([
                [cx - half, cy - half, z],
                [cx + half, cy - half, z],
                [cx - half, cy + half, z],
                [cx + half, cy + half, z],
            ], dtype=float)

            # keep inside margins
            mics[:, 0] = np.clip(mics[:, 0], margin, Lx - margin)
            mics[:, 1] = np.clip(mics[:, 1], margin, Ly - margin)
            mics[:, 2] = np.clip(mics[:, 2], margin, Lz - margin)

            self._set_mics_from_array(mics)
        except Exception as e:
            messagebox.showerror("Preset error", str(e))

    def preset_hex_6(self) -> None:
        """6 mics on a hexagon around the room center, in the horizontal plane."""
        try:
            Lx = float(self.Lx.get())
            Ly = float(self.Ly.get())
            Lz = float(self.Lz.get())
            margin = float(self.margin.get())

            z = min(1.5, max(margin, 0.4 * Lz))
            cx, cy = 0.5 * Lx, 0.5 * Ly
            radius = min(0.6, 0.30 * min(Lx, Ly))  # meters

            angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
            mics = np.stack([
                cx + radius * np.cos(angles),
                cy + radius * np.sin(angles),
                np.full(6, z)
            ], axis=1)

            mics[:, 0] = np.clip(mics[:, 0], margin, Lx - margin)
            mics[:, 1] = np.clip(mics[:, 1], margin, Ly - margin)
            mics[:, 2] = np.clip(mics[:, 2], margin, Lz - margin)

            self._set_mics_from_array(mics)
        except Exception as e:
            messagebox.showerror("Preset error", str(e))

    def preset_random(self) -> None:
        """Random mic positions inside the room (respecting margin)."""
        try:
            Lx = float(self.Lx.get())
            Ly = float(self.Ly.get())
            Lz = float(self.Lz.get())
            margin = float(self.margin.get())

            # ask for number of mics in a tiny popup
            win = tk.Toplevel(self.master)
            win.title("Random array")
            ttk.Label(win, text="Number of mics:").grid(row=0, column=0, padx=8, pady=8, sticky="w")
            n_var = tk.StringVar(value="6")
            ttk.Entry(win, textvariable=n_var, width=8).grid(row=0, column=1, padx=8, pady=8, sticky="w")

            ttk.Label(win, text="Seed (int or None):").grid(row=1, column=0, padx=8, pady=8, sticky="w")
            seed_var = tk.StringVar(value=self.seed.get())
            ttk.Entry(win, textvariable=seed_var, width=12).grid(row=1, column=1, padx=8, pady=8, sticky="w")

            def apply():
                try:
                    n = int(n_var.get())
                    if n <= 0:
                        raise ValueError("n must be > 0")

                    seed_txt = seed_var.get().strip()
                    if seed_txt.lower() in ("", "none", "null"):
                        seed = None
                    else:
                        seed = int(seed_txt)

                    rng = np.random.default_rng(seed)

                    mics = np.column_stack([
                        rng.uniform(margin, Lx - margin, size=n),
                        rng.uniform(margin, Ly - margin, size=n),
                        rng.uniform(margin, Lz - margin, size=n),
                    ])

                    self._set_mics_from_array(mics)
                    win.destroy()
                except Exception as e:
                    messagebox.showerror("Random preset error", str(e))

            ttk.Button(win, text="Apply", command=apply).grid(row=2, column=0, columnspan=2, padx=8, pady=10, sticky="e")

        except Exception as e:
            messagebox.showerror("Preset error", str(e))


def main():
    root = tk.Tk()
    root.geometry("820x650")
    root.minsize(820, 650)
    app = ExperimentUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()



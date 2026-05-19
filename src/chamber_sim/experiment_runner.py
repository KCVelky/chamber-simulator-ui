from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import dataclass
from io import StringIO
from typing import Any, Callable, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt

from chamber_sim.scene import Scene, MicArray, Source, SourceType, as_vec3, make_semi_anechoic_room
from chamber_sim.viz3d import plot_scene_3d, VizOptions
from chamber_sim.noise import NoiseConfig, add_awgn_snr
from chamber_sim.tdoa import GridSearchConfig
from chamber_sim.tdoa.localization import MultiRefTDOAConfig, estimate_source_multiref_tdoa
from chamber_sim.tdoa.denoise import DenoiseConfig
from chamber_sim.er_ls.config import EnergyConfig, ERLSConfig
from chamber_sim.er_ls import er_ls, er_ls_free_field
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

LogFn = Optional[Callable[[str], None]]


DEFAULT_CONFIG: Dict[str, Any] = {
    "geometry": {
        "Lx": "8.0",
        "Ly": "6.0",
        "Lz": "4.0",
        "margin": "0.05",
        "source": {"x": "5.5", "y": "2.0", "z": "1.2"},
        "mics_csv": "3.0,3.0,1.5\n3.5,3.0,1.5\n3.0,3.5,1.5\n3.5,3.5,1.5",
    },
    "signal": {
        "fs": "8000",
        "duration": "6.0",
        "f_low": "80.0",
        "f_high": "260.0",
        "rms": "1.0",
        "seed": "42",
        "use_mod": True,
        "f_mod": "0.25",
        "mod_depth": "0.2",
        "use_fade": True,
        "fade_in": "0.2",
        "fade_out": "0.2",
    },
    "propagation": {
        "c": "343.0",
        "use_spreading": True,
        "gain_at_1m": "1.0",
        "use_floor_image": True,
        "floor_z": "0.0",
        "add_noise": False,
        "snr_db": "20.0",
        "noise_indep": True,
        "noise_seed": "123",
        "enable_denoise": True,
        "denoise_nperseg": "1024",
        "denoise_noverlap": "768",
        "denoise_noise_head": "0.20",
        "denoise_noise_tail": "0.20",
        "denoise_gain_floor": "0.05",
    },
    "algorithms": {
        "enable_algorithms": True,
        "alg_choice": "None",
        "tdoa_interp": "8",
        "tdoa_grid_dx": "0.10",
        "tdoa_grid_dy": "0.10",
        "tdoa_grid_dz": "0.10",
        "tdoa_z_fixed": "",
        "plot_estimated_source": True,
        "er_ref_idx": "0",
        "er_window_s": "0.50",
        "er_hop_s": "0.25",
        "er_trim_frac": "0.10",
        "er_kappa_eps": "0.001",
        "ernls_max_iter": "50",
        "ernls_lam": "0.01",
        "ernls_tol_step": "1e-6",
        "ernls_tol_cost": "1e-9",
        "mleem_model_type": "coherent",
        "mleem_init_method": "er_ls_ground",
        "mleem_max_iter": "40",
        "mleem_lam": "0.01",
        "mleem_tol_step": "1e-6",
        "mleem_tol_cost": "1e-9",
        "mleem_estimate_alpha": False,
        "mleem_alpha_init": "1.0",
        "mleem_alpha_min": "0.0",
        "mleem_alpha_max": "1.5",
        "mleem_alpha_grid_size": "31",
        "mleem_fd_eps": "1e-5",
        "mleem_barycenter_z_offset": "0.50",
    },
    "plots": {
        "plot_scene": True,
        "plot_source_time": True,
        "plot_source_spectrum": True,
        "plot_mic_spectrogram": False,
        "mic_spec_index": "0",
        "plot_mics": True,
        "plot_mics_zoom": True,
        "plot_source_spectrogram": False,
        "print_delta_r": True,
        "zoom_tmax": "0.2",
        "plot_denoise_compare": False,
        "plot_denoise_mic": "0",
    },
    "comparison": {
        "enabled": False,
        "mode": "Deux algorithmes",
        "algorithm_a": "ER_LS",
        "algorithm_b": "ER_NLS",
        "mic_algorithm": "ER_LS",
        "mics_a_csv": "3.0,3.0,1.5\n3.5,3.0,1.5\n3.0,3.5,1.5\n3.5,3.5,1.5",
        "mics_b_csv": "2.8,3.0,1.5\n3.7,3.0,1.5\n3.25,3.45,1.5\n3.25,2.55,1.5\n3.25,3.0,1.95\n3.25,3.0,1.05",
    },
}


def deep_copy_config(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    import copy
    base = copy.deepcopy(DEFAULT_CONFIG)
    if cfg:
        for section, values in cfg.items():
            if isinstance(values, dict) and isinstance(base.get(section), dict):
                base[section].update(copy.deepcopy(values))
            else:
                base[section] = copy.deepcopy(values)
    return base


def _log(log: LogFn, text: str) -> None:
    if log is not None:
        log(str(text))
    else:
        print(text)


def _parse_floats_csv(text: str) -> np.ndarray:
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


def _float(value: Any, name: str) -> float:
    try:
        return float(value)
    except Exception as e:
        raise ValueError(f"Invalid float for '{name}': {value}") from e


def _int_or_none(value: Any, name: str) -> int | None:
    v = str(value).strip()
    if v.lower() in ("", "none", "null"):
        return None
    try:
        return int(v)
    except Exception as e:
        raise ValueError(f"Invalid int/None for '{name}': {value}") from e


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


@dataclass
class ExperimentResult:
    true_source: np.ndarray
    estimated_source: Optional[np.ndarray]
    residual: Optional[float]
    fs: float
    n_mics: int
    algorithm: str
    noise_added: bool
    debug: Optional[Dict[str, Any]] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "true_source": self.true_source.tolist(),
            "estimated_source": None if self.estimated_source is None else self.estimated_source.tolist(),
            "residual": self.residual,
            "fs": self.fs,
            "n_mics": self.n_mics,
            "algorithm": self.algorithm,
            "noise_added": self.noise_added,
        }


def run_experiment_from_config(cfg_in: Dict[str, Any], log: LogFn = None, show_plots: bool = True) -> ExperimentResult:
    cfg = deep_copy_config(cfg_in)
    g = cfg["geometry"]
    s_cfg_raw = cfg["signal"]
    p = cfg["propagation"]
    a = cfg["algorithms"]
    pl = cfg["plots"]

    # --- Geometry
    Lx = _float(g["Lx"], "Lx")
    Ly = _float(g["Ly"], "Ly")
    Lz = _float(g["Lz"], "Lz")
    margin = _float(g["margin"], "margin")
    src_raw = g.get("source", {})
    src = np.array([
        _float(src_raw.get("x", "0"), "src_x"),
        _float(src_raw.get("y", "0"), "src_y"),
        _float(src_raw.get("z", "0"), "src_z"),
    ], dtype=float)
    mic_positions = _parse_floats_csv(g.get("mics_csv", ""))
    algorithm_mics_csv = g.get("algorithm_mics_csv") or g.get("est_mics_csv") or g.get("mics_csv", "")
    algorithm_mic_positions = _parse_floats_csv(algorithm_mics_csv)

    room = make_semi_anechoic_room(Lx=Lx, Ly=Ly, Lz=Lz, floor_rigid=True)
    mics = MicArray(positions=mic_positions, name="ui_mics")
    source = Source(position=as_vec3(float(src[0]), float(src[1]), float(src[2])), kind=SourceType.MONOPOLE, name="S1")
    scene = Scene(room=room, mic_array=mics, sources=[source])
    scene.validate(margin=margin)

    # --- Signal
    fs_input = _float(s_cfg_raw["fs"], "fs")
    duration = _float(s_cfg_raw["duration"], "duration")
    f_low = _float(s_cfg_raw["f_low"], "f_low")
    f_high = _float(s_cfg_raw["f_high"], "f_high")
    rms = _float(s_cfg_raw["rms"], "rms")
    seed = _int_or_none(s_cfg_raw.get("seed", "None"), "seed")

    mod = None
    if _bool(s_cfg_raw.get("use_mod", False)):
        mod = ModulationSpec(f_mod=_float(s_cfg_raw["f_mod"], "f_mod"), depth=_float(s_cfg_raw["mod_depth"], "mod_depth"))

    fade = None
    if _bool(s_cfg_raw.get("use_fade", False)):
        fade = FadeSpec(fade_in=_float(s_cfg_raw["fade_in"], "fade_in"), fade_out=_float(s_cfg_raw["fade_out"], "fade_out"))

    sig_cfg = SourceSignalConfig(
        fs=fs_input,
        duration=duration,
        band=BandpassSpec(f_low=f_low, f_high=f_high),
        rms=rms,
        modulation=mod,
        fade=fade,
        seed=seed,
    )
    t, s = generate_source_signal(sig_cfg)
    fs = 1.0 / float(t[1] - t[0])
    _log(log, f"Signal generated: fs={fs:.1f} Hz, duration={duration:.3f} s, samples={s.size}")

    # --- Propagation
    prop_cfg = PropagationConfig(
        c=_float(p["c"], "c"),
        include_spherical_spreading=_bool(p.get("use_spreading", True)),
        gain_at_1m=_float(p["gain_at_1m"], "gain_at_1m"),
        include_rigid_floor_image=_bool(p.get("use_floor_image", True)),
        floor_z=_float(p["floor_z"], "floor_z"),
    )
    y, d = simulate_mic_signals_free_field(scene, t, s, prop_cfg)
    mic_gain_errors = str(p.get("mic_gain_errors_db", "") or "").strip()
    if mic_gain_errors:
        gains_db = np.array([float(item.strip()) for item in mic_gain_errors.split(",") if item.strip()], dtype=float)
        if gains_db.size != y.shape[0]:
            raise ValueError(f"mic_gain_errors_db must contain {y.shape[0]} values, got {gains_db.size}.")
        y = y * (10.0 ** (gains_db[:, None] / 20.0))
        _log(log, f"Applied per-microphone gain errors [dB]: {np.round(gains_db, 3)}")
    _log(log, f"Propagation simulated for {y.shape[0]} microphones.")

    noise_added = False
    if _bool(p.get("add_noise", False)):
        noise_cfg = NoiseConfig(
            snr_db=_float(p["snr_db"], "snr_db"),
            per_mic_independent=_bool(p.get("noise_indep", True)),
            seed=_int_or_none(p.get("noise_seed", "None"), "noise_seed"),
        )
        y, noise_rms = add_awgn_snr(y, noise_cfg)
        noise_added = True
        _log(log, f"Added noise: target SNR={noise_cfg.snr_db} dB, noise_rms={noise_rms:.6g}")

    x_hat = None
    err = None
    dbg = None
    alg_choice = str(a.get("alg_choice", "None"))

    if _bool(a.get("enable_algorithms", True)) and alg_choice == "TDOA":
        z_fixed_txt = str(a.get("tdoa_z_fixed", "")).strip()
        z_fixed = None if z_fixed_txt == "" else float(z_fixed_txt)
        grid_cfg = GridSearchConfig(
            dx=float(a["tdoa_grid_dx"]),
            dy=float(a["tdoa_grid_dy"]),
            dz=float(a["tdoa_grid_dz"]),
            z_fixed=z_fixed,
        )
        den_cfg = DenoiseConfig(
            nperseg=int(float(p["denoise_nperseg"])),
            noverlap=int(float(p["denoise_noverlap"])),
            noise_head_s=float(p["denoise_noise_head"]),
            noise_tail_s=float(p["denoise_noise_tail"]),
            gain_floor=float(p["denoise_gain_floor"]),
            f_low=float(s_cfg_raw["f_low"]),
            f_high=float(s_cfg_raw["f_high"]),
        )
        algo_cfg = MultiRefTDOAConfig(
            interp=int(float(a["tdoa_interp"])),
            enable_denoise=_bool(p.get("enable_denoise", True)),
            denoise=den_cfg,
            fusion="median",
        )
        x_hat, dbg = estimate_source_multiref_tdoa(
            mic_positions=np.asarray(algorithm_mic_positions, float),
            y=y,
            fs=fs,
            c=prop_cfg.c,
            room_size=np.asarray(scene.room.size, float),
            grid_cfg=grid_cfg,
            cfg=algo_cfg,
        )
        _log(log, "----- MULTI-REF TDOA (GCC-PHAT) -----")
        _log(log, f"Estimated source [m]: {np.round(x_hat, 3)}")
        _log(log, f"True source [m]: {np.round(scene.sources[0].position, 3)}")
        if isinstance(dbg, dict) and "costs_per_ref" in dbg:
            _log(log, f"Costs per ref: {np.round(dbg['costs_per_ref'], 6)}")


    elif _bool(a.get("enable_algorithms", True)) and alg_choice == "ER_LS_FREE_FIELD":
        cfg_E = EnergyConfig(remove_mean=True, window_s=float(a["er_window_s"]), hop_s=float(a["er_hop_s"]), trim_frac=float(a["er_trim_frac"]))
        cfg_ls = ERLSConfig(ref_idx=int(float(a["er_ref_idx"])), kappa_eps=float(a["er_kappa_eps"]), min_pairs=3)
        x_hat, err, dbg = er_ls_free_field(
            mic_positions=np.asarray(algorithm_mic_positions, float),
            y=y,
            fs=fs,
            cfg_E=cfg_E,
            cfg=cfg_ls,
            mic_gains=None,
        )
        _log(log, "----- ER-LS FREE FIELD RESULTS -----")
        _log(log, "Model used by estimator: classical inverse-square energy model only, E_i ≈ K / r_i²")
        if _bool(p.get("use_floor_image", True)):
            _log(log, "Note: floor/image reflection is present in simulated signals, but this algorithm intentionally ignores it.")
        _log(log, f"Estimated source [m]: {np.round(x_hat, 3)}")
        _log(log, f"Residual RMS [m]: {err}")
        _log(log, f"True source [m]: {np.round(scene.sources[0].position, 3)}")
        if show_plots:
            from chamber_sim.er_ls.viz import plot_apollonius_spheres_3d
            plot_apollonius_spheres_3d(
                mic_positions=np.asarray(algorithm_mic_positions, float),
                spheres=dbg["spheres"],
                x_true=np.asarray(scene.sources[0].position, float),
                x_hat=np.asarray(x_hat, float),
                room_size=np.asarray(scene.room.size, float),
                select_mode="radius",
            )

    elif _bool(a.get("enable_algorithms", True)) and alg_choice == "ER_LS":
        cfg_E = EnergyConfig(remove_mean=True, window_s=float(a["er_window_s"]), hop_s=float(a["er_hop_s"]), trim_frac=float(a["er_trim_frac"]))
        if not _bool(p.get("use_floor_image", True)):
            cfg_ls = ERLSConfig(ref_idx=int(float(a["er_ref_idx"])), kappa_eps=float(a["er_kappa_eps"]), min_pairs=3)
            x_hat, err, dbg = er_ls(np.asarray(algorithm_mic_positions, float), y, fs, cfg_E, cfg_ls, mic_gains=None)
            if show_plots:
                from chamber_sim.er_ls.viz import plot_apollonius_spheres_3d
                plot_apollonius_spheres_3d(
                    mic_positions=np.asarray(algorithm_mic_positions, float),
                    spheres=dbg["spheres"],
                    x_true=np.asarray(scene.sources[0].position, float),
                    x_hat=np.asarray(x_hat, float),
                    room_size=np.asarray(scene.room.size, float),
                    select_mode="radius",
                )
        else:
            gcfg = GroundConfig(floor_z=_float(p["floor_z"], "floor_z"), enforce_z_positive=True)
            cfg_g = ERLSGroundConfig(max_iter=30, lam=1e-2, tol_step=1e-6, tol_cost=1e-9, beta_min=0.0, beta_max=2.0, beta_grid=41)
            x_hat, err, dbg = er_ls_ground(
                mic_positions=np.asarray(algorithm_mic_positions, float),
                y=y,
                fs=fs,
                x0=None,
                cfg_E=cfg_E,
                cfg=cfg_g,
                gcfg=gcfg,
            )
            if isinstance(dbg, dict):
                _log(log, f"beta_hat: {dbg.get('beta_hat')} K_hat: {dbg.get('K_hat')}")
        _log(log, "----- ER-LS RESULTS -----")
        _log(log, f"Estimated source [m]: {np.round(x_hat, 3)}")
        _log(log, f"Residual RMS [m]: {err}")
        _log(log, f"True source [m]: {np.round(scene.sources[0].position, 3)}")

    elif _bool(a.get("enable_algorithms", True)) and alg_choice == "ER_NLS":
        cfg_E = EnergyConfig(remove_mean=True, window_s=float(a["er_window_s"]), hop_s=float(a["er_hop_s"]), trim_frac=float(a["er_trim_frac"]))
        if not _bool(p.get("use_floor_image", True)):
            cfg_ls = ERLSConfig(ref_idx=int(float(a["er_ref_idx"])), kappa_eps=float(a["er_kappa_eps"]), min_pairs=3)
            cfg_nls = ERNLSConfig(max_iter=int(float(a["ernls_max_iter"])), lam=float(a["ernls_lam"]), tol_step=float(a["ernls_tol_step"]), tol_cost=float(a["ernls_tol_cost"]), kappa_eps=float(a["er_kappa_eps"]))
            x_hat, err, dbg = er_nls(np.asarray(algorithm_mic_positions, float), y, fs, None, cfg_E, cfg_ls, cfg_nls, mic_gains=None)
        else:
            gcfg = GroundConfig(floor_z=_float(p["floor_z"], "floor_z"), enforce_z_positive=True)
            cfg_ls_g = ERLSGroundConfig(max_iter=30, lam=1e-2, tol_step=1e-6, tol_cost=1e-9, beta_min=0.0, beta_max=2.0, beta_grid=41)
            cfg_nls_g = ERNLSGroundConfig(max_iter=int(float(a["ernls_max_iter"])), lam=float(a["ernls_lam"]), tol_step=float(a["ernls_tol_step"]), tol_cost=float(a["ernls_tol_cost"]))
            x_hat, err, dbg = er_nls_ground(np.asarray(algorithm_mic_positions, float), y, fs, None, None, cfg_E, cfg_ls_g, cfg_nls_g, gcfg)
            if isinstance(dbg, dict):
                _log(log, f"beta used: {dbg.get('beta')} K_hat: {dbg.get('K_hat')}")
        _log(log, "----- ER-NLS RESULTS -----")
        _log(log, f"Estimated source [m]: {np.round(x_hat, 3)}")
        _log(log, f"Residual RMS [m]: {err}")
        _log(log, f"True source [m]: {np.round(scene.sources[0].position, 3)}")

    elif _bool(a.get("enable_algorithms", True)) and alg_choice == "MLE_EM_GROUND":
        if not _bool(p.get("use_floor_image", True)):
            raise ValueError("MLE_EM_GROUND requires 'Rigid floor image source' to be enabled.")
        cfg_E = EnergyConfig(remove_mean=True, window_s=float(a["er_window_s"]), hop_s=float(a["er_hop_s"]), trim_frac=float(a["er_trim_frac"]))
        cfg_init = MLEEMInitConfig(method=str(a["mleem_init_method"]), barycenter_z_offset=float(a["mleem_barycenter_z_offset"]))
        cfg_mle = MLEEMGroundConfig(
            model_type=str(a["mleem_model_type"]),
            max_iter=int(float(a["mleem_max_iter"])),
            lam=float(a["mleem_lam"]),
            tol_step=float(a["mleem_tol_step"]),
            tol_cost=float(a["mleem_tol_cost"]),
            estimate_alpha=_bool(a.get("mleem_estimate_alpha", False)),
            alpha_init=float(a["mleem_alpha_init"]),
            alpha_min=float(a["mleem_alpha_min"]),
            alpha_max=float(a["mleem_alpha_max"]),
            alpha_grid_size=int(float(a["mleem_alpha_grid_size"])),
            fd_eps=float(a["mleem_fd_eps"]),
            f_low_hz=float(s_cfg_raw["f_low"]),
            f_high_hz=float(s_cfg_raw["f_high"]),
            sound_speed=float(p["c"]),
            estimate_noise_floor=False,
            store_history=True,
        )
        gcfg_mle = MLEEMGeometryConfig(floor_z=_float(p["floor_z"], "floor_z"), enforce_z_positive=True, z_margin=1e-6)
        dbg_flags = MLEEMDebugFlags(keep_alpha_grid_details=True, keep_iteration_details=True)
        x_hat, err, dbg = mle_em_ground(np.asarray(algorithm_mic_positions, float), y, fs, None, None, cfg_E, cfg_init, cfg_mle, gcfg_mle, dbg_flags)
        _log(log, "----- MLE/EM GROUND RESULTS -----")
        _log(log, f"Estimated source [m]: {np.round(x_hat, 3)}")
        _log(log, f"Residual RMS: {err}")
        _log(log, f"True source [m]: {np.round(scene.sources[0].position, 3)}")
        if isinstance(dbg, dict):
            _log(log, f"alpha_hat: {dbg.get('alpha_hat', None)} K_hat: {dbg.get('K_hat', None)}")

    if _bool(pl.get("print_delta_r", True)):
        d_ref = d[0]
        delta_r = d - d_ref
        _log(log, f"Distances r_i [m]      : {np.round(d, 4)}")
        _log(log, f"Delta r_i = r_i - r_0 : {np.round(delta_r, 4)}")

    if show_plots:
        if _bool(pl.get("plot_denoise_compare", False)) and dbg is not None:
            mic_idx = int(pl.get("plot_denoise_mic", "0"))
            if mic_idx < 0 or mic_idx >= y.shape[0]:
                raise ValueError(f"plot_denoise_mic out of range [0,{y.shape[0]-1}]")
            y_before = y[mic_idx]
            y_used = dbg.get("y_used", None) if isinstance(dbg, dict) else None
            y_after = y_used[mic_idx] if y_used is not None else y[mic_idx]
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

        if _bool(pl.get("plot_mic_spectrogram", False)):
            mic_idx = int(pl.get("mic_spec_index", "0"))
            if mic_idx < 0 or mic_idx >= y.shape[0]:
                raise ValueError(f"mic_spec_index out of range. Must be in [0, {y.shape[0]-1}]")
            plt.figure()
            plt.specgram(y[mic_idx], NFFT=2048, Fs=fs, noverlap=1024)
            plt.title(f"Mic M{mic_idx} spectrogram")
            plt.xlabel("t [s]")
            plt.ylabel("f [Hz]")
            plt.ylim(0, min(2000, 0.5 * fs))
            plt.tight_layout()

        if _bool(pl.get("plot_scene", True)):
            fig, ax = plot_scene_3d(scene, VizOptions(title="Chambre 3D – micros & source"))
            if x_hat is not None and _bool(a.get("plot_estimated_source", True)):
                ax.scatter([x_hat[0]], [x_hat[1]], [x_hat[2]], marker="X", s=160, label="Estimated")
                ax.legend(loc="upper left")

        if _bool(pl.get("plot_source_time", True)):
            plt.figure()
            plt.plot(t, s)
            plt.title("Source signal s(t)")
            plt.xlabel("t [s]")
            plt.ylabel("Amplitude")
            plt.tight_layout()

        if _bool(pl.get("plot_source_spectrum", True)):
            S = np.fft.rfft(s)
            freqs = np.fft.rfftfreq(s.size, d=1.0 / fs)
            plt.figure()
            plt.plot(freqs, 20 * np.log10(np.maximum(np.abs(S), 1e-12)))
            plt.title("Source spectrum |S(f)|")
            plt.xlabel("f [Hz]")
            plt.ylabel("Magnitude [dB]")
            plt.xlim(0, 0.5 * fs)
            plt.tight_layout()

        if _bool(pl.get("plot_mics", True)):
            plt.figure()
            ax = plt.gca()
            lines = []
            for i in range(y.shape[0]):
                (ln,) = ax.plot(t, y[i], label=f"M{i}", linewidth=1.0)
                lines.append(ln)
            ax.set_title("Microphone signals")
            ax.set_xlabel("t [s]")
            ax.set_ylabel("Amplitude")
            ax.legend(loc="upper right", ncol=2, fontsize=9)

        if _bool(pl.get("plot_source_spectrogram", False)):
            plt.figure()
            plt.specgram(s, NFFT=2048, Fs=fs, noverlap=1024)
            plt.title("Source spectrogram")
            plt.xlabel("t [s]")
            plt.ylabel("f [Hz]")
            plt.ylim(0, min(2000, 0.5 * fs))
            plt.tight_layout()

        if _bool(pl.get("plot_mics_zoom", True)):
            tmax = _float(pl.get("zoom_tmax", "0.2"), "zoom_tmax")
            mask = t <= tmax
            plt.figure()
            ax = plt.gca()
            for i in range(y.shape[0]):
                ax.plot(t[mask], y[i][mask], label=f"M{i}", linewidth=1.0)
            ax.set_title(f"Microphone signals zoom to {tmax}s")
            ax.set_xlabel("t [s]")
            ax.set_ylabel("Amplitude")
            ax.legend(loc="upper right", ncol=2, fontsize=9)

        plt.show()

    return ExperimentResult(
        true_source=np.asarray(scene.sources[0].position, float),
        estimated_source=None if x_hat is None else np.asarray(x_hat, float),
        residual=None if err is None else float(err),
        fs=float(fs),
        n_mics=int(y.shape[0]),
        algorithm=alg_choice,
        noise_added=noise_added,
        debug=dbg,
    )

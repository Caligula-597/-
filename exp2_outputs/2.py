from __future__ import annotations

import math
import time
import json
from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import lax


# =========================================================
# Config
# =========================================================

@dataclass
class SimConfig:
    N: int = 256
    k: int = 7
    rho: float = 1.0
    v0: float = 0.05
    noise: float = 0.05      # strictly interpreted as D
    dt: float = 1.0
    include_self_in_alignment: bool = False
    dtype: Any = jnp.float32

    @property
    def L(self) -> float:
        return float((self.N / self.rho) ** (1.0 / 3.0))


@dataclass
class Exp2Config:
    burnin_steps: int = 2000
    measure_steps: int = 4000
    max_lag: int = 512
    dense_until: int = 64
    log_points: int = 24
    n_origins: int = 8
    nn_block: int = 1024
    eps: float = 1e-12


# =========================================================
# Basic vector helpers
# =========================================================

def normalize(x: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    n = jnp.linalg.norm(x, axis=-1, keepdims=True)
    n = jnp.maximum(n, eps)
    return x / n


def minimal_image(diff: jnp.ndarray, L: float) -> jnp.ndarray:
    return diff - L * jnp.round(diff / L)


def wrap_positions(r_unwrapped: jnp.ndarray, L: float) -> jnp.ndarray:
    return jnp.mod(r_unwrapped, L)


def trapz_compat(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x=x))
    return float(np.trapz(y, x=x))


# =========================================================
# Topological kNN under PBC
# =========================================================

def knn_topological_pbc(r_wrapped: jnp.ndarray, k: int, L: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Return:
        nbr_idx: [N, k]
        nbr_d2 : [N, k]
    """
    diff = r_wrapped[:, None, :] - r_wrapped[None, :, :]
    diff = minimal_image(diff, L)
    d2 = jnp.sum(diff * diff, axis=-1)

    N = r_wrapped.shape[0]
    big = jnp.array(1e30, dtype=d2.dtype)
    d2 = d2 + jnp.eye(N, dtype=d2.dtype) * big

    vals_neg, idx = lax.top_k(-d2, k)
    nbr_d2 = -vals_neg
    return idx.astype(jnp.int32), nbr_d2


# =========================================================
# State init / one-step dynamics
# =========================================================

def init_state(key: jax.Array, sim_cfg: SimConfig) -> Dict[str, jnp.ndarray]:
    key_r, key_u = jr.split(key)

    r_unwrapped = jr.uniform(
        key_r,
        shape=(sim_cfg.N, 3),
        minval=0.0,
        maxval=sim_cfg.L,
        dtype=sim_cfg.dtype,
    )
    r = wrap_positions(r_unwrapped, sim_cfg.L)

    u = jr.normal(key_u, shape=(sim_cfg.N, 3), dtype=sim_cfg.dtype)
    u = normalize(u)

    nbr_idx, nbr_d2 = knn_topological_pbc(r, sim_cfg.k, sim_cfg.L)

    return {
        "r": r,
        "r_unwrapped": r_unwrapped,
        "u": u,
        "nbr_idx": nbr_idx,
        "nbr_d2": nbr_d2,
    }


def step_state(state: Dict[str, jnp.ndarray], key: jax.Array, sim_cfg: SimConfig) -> Dict[str, jnp.ndarray]:
    """
    Strictly aligned to Experiment 1:
      1) topological neighbor averaging
      2) additive Gaussian white noise
      3) renormalization
      4) self-propelled motion
      5) recompute topological neighbors
    """
    r_unwrapped = state["r_unwrapped"]
    u = state["u"]
    nbr_idx = state["nbr_idx"]

    nbr_u = u[nbr_idx]   # [N, k, 3]

    if sim_cfg.include_self_in_alignment:
        A = (u + jnp.sum(nbr_u, axis=1)) / (sim_cfg.k + 1.0)
    else:
        A = jnp.mean(nbr_u, axis=1)

    eta = jr.normal(key, shape=u.shape, dtype=u.dtype)
    u_new = normalize(A + jnp.sqrt(2.0 * sim_cfg.noise * sim_cfg.dt) * eta)

    r_unwrapped_new = r_unwrapped + sim_cfg.v0 * sim_cfg.dt * u_new
    r_new = wrap_positions(r_unwrapped_new, sim_cfg.L)

    nbr_idx_new, nbr_d2_new = knn_topological_pbc(r_new, sim_cfg.k, sim_cfg.L)

    return {
        "r": r_new,
        "r_unwrapped": r_unwrapped_new,
        "u": u_new,
        "nbr_idx": nbr_idx_new,
        "nbr_d2": nbr_d2_new,
    }


def make_advance_n(sim_cfg: SimConfig):
    @jax.jit
    def advance_n(state: Dict[str, jnp.ndarray], key: jax.Array, n_steps: int):
        loop_key, out_key = jr.split(key)

        def body_fun(i, st):
            subkey = jr.fold_in(loop_key, i)
            return step_state(st, subkey, sim_cfg)

        state = lax.fori_loop(0, n_steps, body_fun, state)
        return state, out_key

    return advance_n


# =========================================================
# Lag / origin / save schedule
# =========================================================

def build_mixed_lags(max_lag: int, dense_until: int = 128, log_points: int = 48) -> np.ndarray:
    dense_until = min(dense_until, max_lag)
    dense = np.arange(0, dense_until + 1, dtype=np.int32)

    if max_lag <= dense_until:
        return dense

    start = max(dense_until + 1, 1)
    tail = np.unique(np.round(np.geomspace(start, max_lag, num=log_points)).astype(np.int32))
    lags = np.unique(np.concatenate([dense, tail]))

    if lags[0] != 0:
        lags = np.concatenate([[0], lags])

    return lags


def build_origin_times(measure_steps: int, max_lag: int, n_origins: int) -> np.ndarray:
    last_start = measure_steps - max_lag
    if last_start < 0:
        raise ValueError("measure_steps must be >= max_lag")

    if n_origins <= 1:
        return np.array([0], dtype=np.int32)

    return np.unique(np.linspace(0, last_start, n_origins, dtype=np.int32))


def build_save_schedule(origins: np.ndarray, lags: np.ndarray, measure_steps: int) -> np.ndarray:
    t = (origins[:, None] + lags[None, :]).ravel()
    t = np.unique(t)
    return t[t <= measure_steps]


# =========================================================
# Co-rotating frame (CPU side)
# =========================================================

def _normalize_np(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return (v / n).astype(np.float32)


def make_corotating_basis_np(p_vec: np.ndarray, prev_e1: Optional[np.ndarray], eps: float = 1e-12):
    """
    Right-handed basis [e1,e2,e3], where e3 || polarization.
    prev_e1 suppresses frame flip / twisting.
    """
    e3 = _normalize_np(p_vec, eps=eps)

    if prev_e1 is None:
        aux = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if abs(np.dot(aux, e3)) > 0.9:
            aux = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        e1 = aux - np.dot(aux, e3) * e3
        e1 = _normalize_np(e1, eps=eps)
    else:
        cand = prev_e1 - np.dot(prev_e1, e3) * e3
        if np.linalg.norm(cand) < eps:
            aux = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            if abs(np.dot(aux, e3)) > 0.9:
                aux = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            cand = aux - np.dot(aux, e3) * e3
        e1 = _normalize_np(cand, eps=eps)

    e2 = np.cross(e3, e1).astype(np.float32)
    e2 = _normalize_np(e2, eps=eps)
    e1 = np.cross(e2, e3).astype(np.float32)
    e1 = _normalize_np(e1, eps=eps)

    basis = np.stack([e1, e2, e3], axis=0).astype(np.float32)
    return basis, e1


def extract_frame_from_state_np(state_np: Dict[str, np.ndarray], prev_e1: Optional[np.ndarray], exp_cfg: Exp2Config):
    """
    Compact frame used in Experiment 2 analysis.
    """
    r = np.asarray(state_np["r_unwrapped"], dtype=np.float32)
    u = np.asarray(state_np["u"], dtype=np.float32)
    nbr_idx = np.asarray(state_np["nbr_idx"], dtype=np.int32)
    nbr_d2 = np.asarray(state_np["nbr_d2"], dtype=np.float32)

    ubar = u.mean(axis=0)
    phi = float(np.linalg.norm(ubar))

    basis, prev_e1 = make_corotating_basis_np(ubar, prev_e1, eps=exp_cfg.eps)

    r_cm = r.mean(axis=0)
    r_rel = r - r_cm

    delta_u = u - ubar

    nn2_mean = float(np.mean(np.min(nbr_d2, axis=1)))

    frame = {
        "nbr_idx": nbr_idx,
        "r_rel_lab": r_rel.astype(np.float32),
        "delta_u_lab": delta_u.astype(np.float32),
        "basis": basis.astype(np.float32),
        "phi": phi,
        "nn2_mean": nn2_mean,
    }
    return frame, prev_e1


# =========================================================
# CPU fallback nearest-neighbor metric scale
# =========================================================

def mean_nn2_chunked_np(x: np.ndarray, block: int = 1024) -> float:
    n = x.shape[0]
    mins = np.full(n, np.inf, dtype=np.float32)

    for a in range(0, n, block):
        xa = x[a:a + block]
        best = np.full(xa.shape[0], np.inf, dtype=np.float32)

        for b in range(0, n, block):
            xb = x[b:b + block]
            diff = xa[:, None, :] - xb[None, :, :]
            d2 = np.einsum("ijk,ijk->ij", diff, diff)

            if a == b:
                i = np.arange(xa.shape[0])
                d2[i, i] = np.inf

            best = np.minimum(best, d2.min(axis=1))

        mins[a:a + xa.shape[0]] = best

    return float(np.mean(mins))


# =========================================================
# Postprocess helpers
# =========================================================

def neighbor_overlap_fraction(nbr0: np.ndarray, nbr1: np.ndarray) -> float:
    """
    Directed overlap:
      q_i = |N_i(t) ∩ N_i(t+tau)| / k
      Q_n = mean_i q_i
    """
    same = (nbr0[:, :, None] == nbr1[:, None, :])   # [N, k, k]
    overlap = same.any(axis=2).sum(axis=1).astype(np.float32)
    return float(np.mean(overlap / nbr0.shape[1]))


def first_crossing_leq(x: np.ndarray, y: np.ndarray, threshold: float) -> float:
    idx = np.where(y <= threshold)[0]
    if len(idx) == 0:
        return np.nan
    j = idx[0]
    if j == 0:
        return float(x[0])
    x0, x1 = x[j - 1], x[j]
    y0, y1 = y[j - 1], y[j]
    if abs(y1 - y0) < 1e-12:
        return float(x1)
    w = (threshold - y0) / (y1 - y0)
    return float(x0 + w * (x1 - x0))


def first_crossing_geq(x: np.ndarray, y: np.ndarray, threshold: float) -> float:
    idx = np.where(y >= threshold)[0]
    if len(idx) == 0:
        return np.nan
    j = idx[0]
    if j == 0:
        return float(x[0])
    x0, x1 = x[j - 1], x[j]
    y0, y1 = y[j - 1], y[j]
    if abs(y1 - y0) < 1e-12:
        return float(x1)
    w = (threshold - y0) / (y1 - y0)
    return float(x0 + w * (x1 - x0))


def tau_area_until_nonpositive(lags: np.ndarray, curve: np.ndarray) -> float:
    idx = np.where(curve <= 0.0)[0]
    end = int(idx[0]) if len(idx) > 0 else len(curve) - 1

    if end <= 0:
        return 0.0

    x = np.asarray(lags[:end + 1], dtype=np.float64)
    y = np.clip(np.asarray(curve[:end + 1], dtype=np.float64), 0.0, None)
    return trapz_compat(y, x)


# =========================================================
# Single-seed postprocess
# =========================================================

def postprocess_frames(frames: Dict[int, Dict[str, Any]], origins: np.ndarray, lags: np.ndarray, sim_cfg: SimConfig, exp_cfg: Exp2Config):
    N = sim_cfg.N
    k = sim_cfg.k

    qn_vals = {int(l): [] for l in lags}
    cdelta_vals = {int(l): [] for l in lags}
    msd_vals = {int(l): [] for l in lags}
    phi_vals = []

    for t0 in origins:
        f0 = frames[int(t0)]
        phi_vals.append(f0["phi"])

        nbr0 = f0["nbr_idx"]
        B0 = f0["basis"]

        # express origin quantities in the origin basis
        x0 = f0["r_rel_lab"] @ B0.T
        du0 = f0["delta_u_lab"] @ B0.T

        den0 = float(np.mean(np.sum(du0 * du0, axis=1)))
        den0 = max(den0, exp_cfg.eps)

        for lag in lags:
            ft = frames[int(t0 + lag)]

            qn = neighbor_overlap_fraction(nbr0, ft["nbr_idx"])
            qn_vals[int(lag)].append(qn)

            # express t0+tau quantities in the SAME origin basis
            x_t = ft["r_rel_lab"] @ B0.T
            du_t = ft["delta_u_lab"] @ B0.T

            num = float(np.mean(np.sum(du0 * du_t, axis=1)))
            cdelta = num / den0
            cdelta_vals[int(lag)].append(cdelta)

            dr = x_t - x0
            msd = float(np.mean(np.sum(dr * dr, axis=1)))
            msd_vals[int(lag)].append(msd)

    lag_arr = np.asarray(lags, dtype=np.float64)

    def mean_sem(d):
        arr = np.asarray(d, dtype=np.float64)
        mean = float(arr.mean())
        sem = float(arr.std(ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0
        return mean, sem

    qn_mean = np.array([mean_sem(qn_vals[int(l)])[0] for l in lags], dtype=np.float64)
    qn_sem = np.array([mean_sem(qn_vals[int(l)])[1] for l in lags], dtype=np.float64)

    cdelta_mean = np.array([mean_sem(cdelta_vals[int(l)])[0] for l in lags], dtype=np.float64)
    cdelta_sem = np.array([mean_sem(cdelta_vals[int(l)])[1] for l in lags], dtype=np.float64)

    msd_mean = np.array([mean_sem(msd_vals[int(l)])[0] for l in lags], dtype=np.float64)
    msd_sem = np.array([mean_sem(msd_vals[int(l)])[1] for l in lags], dtype=np.float64)

    q_inf = k / (N - 1.0)
    qn_tilde_mean = (qn_mean - q_inf) / (1.0 - q_inf)
    qn_tilde_sem = qn_sem / (1.0 - q_inf)
    qn_tilde_mean[0] = 1.0

    nn2_vals = [frames[int(t0)]["nn2_mean"] for t0 in origins]
    if any(v is None for v in nn2_vals):
        ell_nn2 = float(
            np.mean([mean_nn2_chunked_np(frames[int(t0)]["r_rel_lab"], block=exp_cfg.nn_block) for t0 in origins])
        )
    else:
        ell_nn2 = float(np.mean(nn2_vals))

    tau_rw = tau_area_until_nonpositive(lag_arr, qn_tilde_mean)
    tau_rw_1e = first_crossing_leq(lag_arr, qn_tilde_mean, math.e ** (-1.0))

    tau_rel = tau_area_until_nonpositive(lag_arr, cdelta_mean)
    tau_rel_1e = first_crossing_leq(lag_arr, cdelta_mean, math.e ** (-1.0))

    tau_cage = first_crossing_geq(lag_arr, msd_mean, ell_nn2)

    return {
        "curves": {
            "lag": lag_arr,
            "Qn_mean": qn_mean,
            "Qn_sem": qn_sem,
            "Qn_tilde_mean": qn_tilde_mean,
            "Qn_tilde_sem": qn_tilde_sem,
            "Cdelta_mean": cdelta_mean,
            "Cdelta_sem": cdelta_sem,
            "MSDrel_mean": msd_mean,
            "MSDrel_sem": msd_sem,
        },
        "scalars": {
            "N": N,
            "k": k,
            "q_inf": q_inf,
            "ell_nn2": ell_nn2,
            "phi_mean": float(np.mean(phi_vals)),

            "tau_rw": float(tau_rw),
            "tau_rw_1e": float(tau_rw_1e),
            "tau_rw_window_limited": bool(np.isnan(tau_rw_1e) and qn_tilde_mean[-1] > np.exp(-1)),

            "tau_rel": float(tau_rel),
            "tau_rel_1e": float(tau_rel_1e),
            "tau_rel_window_limited": bool(np.isnan(tau_rel_1e) and cdelta_mean[-1] > np.exp(-1)),

            "tau_cage": float(tau_cage),
            "tau_cage_window_limited": bool(np.isnan(tau_cage) and msd_mean[-1] < ell_nn2),
        }
    }


# =========================================================
# Single seed runner
# =========================================================

def run_single_seed(seed: int, sim_cfg: SimConfig, exp_cfg: Exp2Config, advance_n=None, verbose: bool = True):
    key = jr.PRNGKey(seed)

    if verbose:
        print(f"[seed {seed}] init")
    state = init_state(key, sim_cfg)

    key = jr.PRNGKey(seed + 1234567)

    if advance_n is None:
        advance_n = make_advance_n(sim_cfg)

    if verbose:
        print(f"[seed {seed}] burn-in {exp_cfg.burnin_steps}")
    t0 = time.time()
    state, key = advance_n(state, key, exp_cfg.burnin_steps)
    if verbose:
        print(f"[seed {seed}] burn-in done in {time.time() - t0:.2f}s")

    lags = build_mixed_lags(exp_cfg.max_lag, exp_cfg.dense_until, exp_cfg.log_points)
    origins = build_origin_times(exp_cfg.measure_steps, exp_cfg.max_lag, exp_cfg.n_origins)
    save_times = build_save_schedule(origins, lags, exp_cfg.measure_steps)

    if verbose:
        print(f"[seed {seed}] lags={len(lags)}, origins={len(origins)}, saved_frames={len(save_times)}")

    frames = {}
    prev_e1 = None
    current_t = 0

    t1 = time.time()
    for i, t in enumerate(save_times):
        gap = int(t - current_t)
        if gap > 0:
            state, key = advance_n(state, key, gap)
            current_t = t

        state_np = {
            "r_unwrapped": np.asarray(jax.device_get(state["r_unwrapped"]), dtype=np.float32),
            "u": np.asarray(jax.device_get(state["u"]), dtype=np.float32),
            "nbr_idx": np.asarray(jax.device_get(state["nbr_idx"]), dtype=np.int32),
            "nbr_d2": np.asarray(jax.device_get(state["nbr_d2"]), dtype=np.float32),
        }

        frame, prev_e1 = extract_frame_from_state_np(state_np, prev_e1, exp_cfg)
        frames[int(t)] = frame

        if verbose and ((i + 1) % max(1, len(save_times) // 10) == 0):
            print(f"[seed {seed}] saved {i + 1}/{len(save_times)} frames")

    if verbose:
        print(f"[seed {seed}] sparse measurement done in {time.time() - t1:.2f}s")

    result = postprocess_frames(frames, origins, lags, sim_cfg, exp_cfg)
    result["seed"] = seed
    result["schedule"] = {
        "lags": lags,
        "origins": origins,
        "save_times": save_times,
    }
    return result


# =========================================================
# Multi-seed aggregation
# =========================================================

def aggregate_seed_results(results: List[Dict[str, Any]], n_boot: int = 1000, rng_seed: int = 0):
    rng = np.random.default_rng(rng_seed)

    lag = results[0]["curves"]["lag"]

    def stack_curve(name):
        return np.stack([r["curves"][name] for r in results], axis=0)

    def stack_scalar(name):
        return np.array([r["scalars"][name] for r in results], dtype=np.float64)

    agg_curves = {}
    for name in ["Qn_tilde_mean", "Cdelta_mean", "MSDrel_mean"]:
        arr = stack_curve(name)
        agg_curves[name] = {
            "mean": arr.mean(axis=0),
            "se": arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0]) if arr.shape[0] > 1 else np.zeros_like(arr[0]),
        }

    agg_scalars = {}
    for name in [
        "phi_mean", "ell_nn2",
        "tau_rw", "tau_rw_1e", "tau_rw_window_limited",
        "tau_rel", "tau_rel_1e", "tau_rel_window_limited",
        "tau_cage", "tau_cage_window_limited"
    ]:
        arr = stack_scalar(name)
        mean = float(arr.mean())
        se = float(arr.std(ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0

        if arr.size > 1 and arr.dtype != np.bool_:
            boot = []
            for _ in range(n_boot):
                sample = rng.choice(arr, size=arr.size, replace=True)
                boot.append(np.mean(sample))
            lo, hi = np.quantile(boot, [0.025, 0.975])
        else:
            lo, hi = mean, mean

        agg_scalars[name] = {
            "mean": mean,
            "se": se,
            "ci_lo": float(lo),
            "ci_hi": float(hi),
            "all": arr,
        }

    return {
        "lag": lag,
        "curves": agg_curves,
        "scalars": agg_scalars,
        "n_seeds": len(results),
    }


# =========================================================
# D-point / sweep runner
# =========================================================

def run_D_point(noise: float, seeds: List[int], sim_cfg: SimConfig, exp_cfg: Exp2Config, verbose: bool = True):
    sim_cfg_local = SimConfig(**asdict(sim_cfg))
    sim_cfg_local.noise = noise

    advance_n = make_advance_n(sim_cfg_local)

    all_results = []
    for s in seeds:
        res = run_single_seed(s, sim_cfg_local, exp_cfg, advance_n=advance_n, verbose=verbose)
        all_results.append(res)

    agg = aggregate_seed_results(all_results, n_boot=1000, rng_seed=123)
    agg["noise"] = noise
    agg["sim_cfg"] = {
    **asdict(sim_cfg_local),
    "L": sim_cfg_local.L,
    "dtype": str(sim_cfg_local.dtype),
    }
    agg["exp_cfg"] = asdict(exp_cfg)

    return {
        "seed_results": all_results,
        "agg": agg,
    }


def run_noise_sweep(noise_list: List[float], seeds: List[int], sim_cfg: SimConfig, exp_cfg: Exp2Config, verbose: bool = True):
    out = {}
    summary_rows = []

    for noise in noise_list:
        print(f"\n========== noise = {noise:.6f} ==========")
        block = run_D_point(noise, seeds, sim_cfg, exp_cfg, verbose=verbose)
        out[float(noise)] = block

        sc = block["agg"]["scalars"]
        summary_rows.append({
            "noise": noise,
            "n_seeds": block["agg"]["n_seeds"],

            "phi_mean": sc["phi_mean"]["mean"],
            "phi_se": sc["phi_mean"]["se"],

            "tau_rw": sc["tau_rw"]["mean"],
            "tau_rw_se": sc["tau_rw"]["se"],
            "tau_rw_ci_lo": sc["tau_rw"]["ci_lo"],
            "tau_rw_ci_hi": sc["tau_rw"]["ci_hi"],
            "tau_rw_window_limited_frac": sc["tau_rw_window_limited"]["mean"],

            "tau_rel": sc["tau_rel"]["mean"],
            "tau_rel_se": sc["tau_rel"]["se"],
            "tau_rel_ci_lo": sc["tau_rel"]["ci_lo"],
            "tau_rel_ci_hi": sc["tau_rel"]["ci_hi"],
            "tau_rel_window_limited_frac": sc["tau_rel_window_limited"]["mean"],

            "tau_cage": sc["tau_cage"]["mean"],
            "tau_cage_se": sc["tau_cage"]["se"],
            "tau_cage_ci_lo": sc["tau_cage"]["ci_lo"],
            "tau_cage_ci_hi": sc["tau_cage"]["ci_hi"],
            "tau_cage_window_limited_frac": sc["tau_cage_window_limited"]["mean"],

            "ell_nn2": sc["ell_nn2"]["mean"],
        })

    summary_df = pd.DataFrame(summary_rows).sort_values("noise").reset_index(drop=True)
    return out, summary_df


# =========================================================
# Plot helpers
# =========================================================

def plot_curves_for_D(block: Dict[str, Any], title_prefix: str = ""):
    agg = block["agg"]
    lag = agg["lag"]

    q = agg["curves"]["Qn_tilde_mean"]["mean"]
    q_se = agg["curves"]["Qn_tilde_mean"]["se"]

    c = agg["curves"]["Cdelta_mean"]["mean"]
    c_se = agg["curves"]["Cdelta_mean"]["se"]

    m = agg["curves"]["MSDrel_mean"]["mean"]
    m_se = agg["curves"]["MSDrel_mean"]["se"]

    noise = agg["noise"]
    sc = agg["scalars"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    ax = axes[0]
    ax.plot(lag, q, lw=2)
    ax.fill_between(lag, q - q_se, q + q_se, alpha=0.25)
    ax.set_xscale("log")
    ax.set_title(rf"$\tilde Q_n(\tau)$   D={noise:.4f}")
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$\tilde Q_n(\tau)$")
    ax.axhline(np.exp(-1), ls="--", lw=1)

    ax = axes[1]
    ax.plot(lag, c, lw=2)
    ax.fill_between(lag, c - c_se, c + c_se, alpha=0.25)
    ax.set_xscale("log")
    ax.set_title(r"$C_\delta(\tau)$")
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$C_\delta(\tau)$")
    ax.axhline(np.exp(-1), ls="--", lw=1)
    ax.axhline(0.0, ls="--", lw=1)

    ax = axes[2]
    ax.plot(lag, m, lw=2)
    ax.fill_between(lag, m - m_se, m + m_se, alpha=0.25)
    ax.set_xscale("log")
    ax.set_title(r"$\Delta_{\rm rel}^2(\tau)$")
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$\Delta_{\rm rel}^2(\tau)$")
    ax.axhline(sc["ell_nn2"]["mean"], ls="--", lw=1, label=r"$\ell_{nn}^2$")
    ax.legend()

    if title_prefix:
        fig.suptitle(title_prefix, y=1.03, fontsize=14)

    plt.tight_layout()
    plt.show()


def plot_timescales_vs_noise(summary_df: pd.DataFrame, title: str = ""):
    x = summary_df["noise"].values

    fig, axes = plt.subplots(1, 4, figsize=(19, 4.5))

    ax = axes[0]
    ax.errorbar(x, summary_df["phi_mean"], yerr=summary_df["phi_se"], marker="o", lw=1.8, capsize=3)
    ax.set_title(r"$\Phi$")
    ax.set_xlabel("D")
    ax.set_ylabel(r"$\Phi$")

    ax = axes[1]
    ax.errorbar(x, summary_df["tau_rw"], yerr=summary_df["tau_rw_se"], marker="o", lw=1.8, capsize=3)
    ax.set_title(r"$\tau_{rw}$")
    ax.set_xlabel("D")
    ax.set_ylabel(r"$\tau_{rw}$")
    ax.set_yscale("log")

    ax = axes[2]
    ax.errorbar(x, summary_df["tau_rel"], yerr=summary_df["tau_rel_se"], marker="o", lw=1.8, capsize=3)
    ax.set_title(r"$\tau_{rel}$")
    ax.set_xlabel("D")
    ax.set_ylabel(r"$\tau_{rel}$")
    ax.set_yscale("log")

    ax = axes[3]
    ax.errorbar(x, summary_df["tau_cage"], yerr=summary_df["tau_cage_se"], marker="o", lw=1.8, capsize=3)
    ax.set_title(r"$\tau_{cage}$")
    ax.set_xlabel("D")
    ax.set_ylabel(r"$\tau_{cage}$")
    ax.set_yscale("log")

    if title:
        fig.suptitle(title, y=1.03, fontsize=14)

    plt.tight_layout()
    plt.show()


# =========================================================
# Save helpers
# =========================================================

def save_summary_df(summary_df: pd.DataFrame, path_csv: str):
    summary_df.to_csv(path_csv, index=False)
    print(f"saved: {path_csv}")


def to_jsonable(obj):
    """
    Recursively convert common NumPy / JAX / dataclass-like objects
    into plain Python JSON-serializable objects.
    """
    # plain python
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    # numpy scalars
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)

    # numpy arrays
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # JAX arrays
    if hasattr(obj, "tolist") and "jax" in str(type(obj)).lower():
        return obj.tolist()

    # dict
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}

    # list / tuple
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]

    # dtype / scalar meta / type objects
    if isinstance(obj, type):
        return str(obj)

    # numpy / jax dtype-like objects
    if "dtype" in str(type(obj)).lower() or "scalarmeta" in str(type(obj)).lower():
        return str(obj)

    # fallback
    return str(obj)


def save_agg_block_jsonable(block: Dict[str, Any], path_json: str, save_curves: bool = True):
    agg = block["agg"]

    payload = {
        "noise": float(agg["noise"]),
        "n_seeds": int(agg["n_seeds"]),
        "sim_cfg": to_jsonable(agg["sim_cfg"]),
        "exp_cfg": to_jsonable(agg["exp_cfg"]),
        "scalars": to_jsonable(agg["scalars"]),
    }

    if save_curves and "lag" in agg and "curves" in agg:
        payload["lag"] = to_jsonable(agg["lag"])
        payload["curves"] = to_jsonable(agg["curves"])

    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"saved: {path_json}")


def export_curves_to_csv(block: Dict[str, Any], path_csv: str):
    """
    Export curve data to CSV format for plotting.
    Columns: lag, Qn_tilde_mean, Qn_tilde_se, Cdelta_mean, Cdelta_se, MSDrel_mean, MSDrel_se, ell_nn2
    """
    agg = block["agg"]
    lag = agg["lag"]
    curves = agg["curves"]
    ell_nn2 = agg["scalars"]["ell_nn2"]["mean"]

    df_data = {
        "lag": lag,
        "Qn_tilde_mean": curves["Qn_tilde_mean"]["mean"],
        "Qn_tilde_se": curves["Qn_tilde_mean"]["se"],
        "Cdelta_mean": curves["Cdelta_mean"]["mean"],
        "Cdelta_se": curves["Cdelta_mean"]["se"],
        "MSDrel_mean": curves["MSDrel_mean"]["mean"],
        "MSDrel_se": curves["MSDrel_mean"]["se"],
        "ell_nn2": [ell_nn2] * len(lag),
    }

    df = pd.DataFrame(df_data)
    df.to_csv(path_csv, index=False)
    print(f"saved curves to CSV: {path_csv}")


# =========================================================
# Main
# =========================================================

def main():
    # -------------------------
    # 1) smoke test
    # -------------------------
    print("\n===== SMOKE TEST =====")

    sim_cfg_smoke = SimConfig(
        N=256,
        k=7,
        rho=1.0,
        v0=0.05,
        noise=0.05,
        dt=1.0,
        include_self_in_alignment=False,
    )

    exp_cfg_smoke = Exp2Config(
        burnin_steps=2000,
        measure_steps=4000,
        max_lag=512,
        dense_until=64,
        log_points=24,
        n_origins=8,
    )

    smoke = run_single_seed(
        seed=0,
        sim_cfg=sim_cfg_smoke,
        exp_cfg=exp_cfg_smoke,
        verbose=True,
    )

    print("\nSMOKE SCALARS:")
    print(json.dumps(smoke["scalars"], indent=2, ensure_ascii=False))

    smoke_block = {
        "agg": {
            "noise": sim_cfg_smoke.noise,
            "n_seeds": 1,
            "lag": smoke["curves"]["lag"],
            "curves": {
                "Qn_tilde_mean": {
                    "mean": smoke["curves"]["Qn_tilde_mean"],
                    "se": np.zeros_like(smoke["curves"]["Qn_tilde_mean"]),
                },
                "Cdelta_mean": {
                    "mean": smoke["curves"]["Cdelta_mean"],
                    "se": np.zeros_like(smoke["curves"]["Cdelta_mean"]),
                },
                "MSDrel_mean": {
                    "mean": smoke["curves"]["MSDrel_mean"],
                    "se": np.zeros_like(smoke["curves"]["MSDrel_mean"]),
                },
            },
            "scalars": {
                "ell_nn2": {
                    "mean": smoke["scalars"]["ell_nn2"],
                }
            }
        }
    }
    plot_curves_for_D(smoke_block, title_prefix="smoke test")

    # -------------------------
    # 2) first pass production
    # -------------------------
    print("\n===== FIRST PASS SWEEP =====")

    sim_cfg = SimConfig(
        N=1024,
        k=7,
        rho=1.0,
        v0=0.05,
        noise=0.05,   # overwritten in sweep
        dt=1.0,
        include_self_in_alignment=False,
    )

    exp_cfg = Exp2Config(
        burnin_steps=20_000,
        measure_steps=50_000,
        max_lag=8192,
        dense_until=128,
        log_points=48,
        n_origins=24,
    )

    noise_list = [0.005, 0.02, 0.05, 0.0738, 0.10, 0.15]
    seeds = list(range(5))

    all_blocks, summary_df = run_noise_sweep(
        noise_list=noise_list,
        seeds=seeds,
        sim_cfg=sim_cfg,
        exp_cfg=exp_cfg,
        verbose=False,
    )

    print("\nSUMMARY:")
    print(summary_df)

    plot_timescales_vs_noise(summary_df, title="Experiment 2 first pass")
    for noise in noise_list:
        plot_curves_for_D(all_blocks[float(noise)], title_prefix=f"D={noise:.4f}")

    save_summary_df(summary_df, "exp2_summary_firstpass.csv")
    for noise in noise_list:
        save_agg_block_jsonable(all_blocks[float(noise)], f"exp2_noise_{noise:.4f}.json")


if __name__ == "__main__":
    main()
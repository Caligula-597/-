
from __future__ import annotations

import os
import math
import json
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

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
    N: int = 1024
    k: int = 7
    rho: float = 1.0
    v0: float = 0.05
    noise: float = 0.05   # interpreted as D
    dt: float = 1.0
    include_self_in_alignment: bool = False
    dtype: Any = jnp.float32

    @property
    def L(self) -> float:
        return float((self.N / self.rho) ** (1.0 / 3.0))


@dataclass
class Exp4Config:
    burnin_steps: int = 20_000
    measure_steps: int = 12_000
    eps: float = 1e-12

    # Strictly inherited from the Experiment 1 / 3 thermodynamic standard:
    #   sdot_proxy = < ||u_{t+1} - A_t||^2 / (2 D dt) >
    # and
    #   J = sdot_proxy / chi
    #
    # This script intentionally does NOT default to turning-cost proxies.
    use_strict_entropy_proxy: bool = True

    # bootstrap over seeds
    n_boot: int = 1000


@dataclass
class TwoStageGridConfig:
    coarse_D: List[float] = None
    refine_half_width_decades: float = 0.18
    refine_points_each_target: int = 7
    include_midpoint_refine: bool = True
    D_min: float = 0.003
    D_max: float = 0.20

    # when coarse optimum/critical point lands on a boundary,
    # densify the first/last coarse interval as a guardrail
    boundary_guard_points: int = 6

    def __post_init__(self):
        if self.coarse_D is None:
            self.coarse_D = [0.005, 0.02, 0.035, 0.05, 0.0738, 0.10, 0.15]


@dataclass
class DriftConfig:
    enabled: bool = True
    parameter_sets: Optional[List[Dict[str, Any]]] = None

    def __post_init__(self):
        if self.parameter_sets is None:
            self.parameter_sets = [
                {"label": "base", "rho": 1.0, "v0": 0.05},
                {"label": "rho_low", "rho": 0.8, "v0": 0.05},
                {"label": "rho_high", "rho": 1.2, "v0": 0.05},
                {"label": "v0_low", "rho": 1.0, "v0": 0.04},
                {"label": "v0_high", "rho": 1.0, "v0": 0.06},
            ]


# =========================================================
# Helpers
# =========================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def trapz_compat(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x=x))
    return float(np.trapz(y, x=x))


def normalize(x: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    n = jnp.linalg.norm(x, axis=-1, keepdims=True)
    n = jnp.maximum(n, eps)
    return x / n


def minimal_image(diff: jnp.ndarray, L: float) -> jnp.ndarray:
    return diff - L * jnp.round(diff / L)


def wrap_positions(r_unwrapped: jnp.ndarray, L: float) -> jnp.ndarray:
    return jnp.mod(r_unwrapped, L)


def to_jsonable(obj):
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "tolist") and "jax" in str(type(obj)).lower():
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, type):
        return str(obj)
    tname = str(type(obj)).lower()
    if "dtype" in tname or "scalarmeta" in tname:
        return str(obj)
    return str(obj)


def mean_se_ci(arr: np.ndarray, n_boot: int = 1000, rng: Optional[np.random.Generator] = None):
    arr = np.asarray(arr, dtype=float)
    mean = float(arr.mean())
    se = float(arr.std(ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0
    if arr.size <= 1:
        return mean, se, mean, mean
    if rng is None:
        rng = np.random.default_rng(0)
    boot = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=arr.size, replace=True)
        boot.append(np.mean(sample))
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return mean, se, float(lo), float(hi)


def nearest_sorted_unique_float(values: List[float], tol: float = 1e-12) -> List[float]:
    out = []
    for v in sorted(float(x) for x in values):
        if not out or abs(v - out[-1]) > tol:
            out.append(v)
    return out


def build_local_log_grid(target: float, D_min: float, D_max: float, half_width_decades: float, n_points: int) -> List[float]:
    target = float(np.clip(target, D_min, D_max))
    lo = max(D_min, target * (10.0 ** (-half_width_decades)))
    hi = min(D_max, target * (10.0 ** (half_width_decades)))
    if lo <= 0:
        lo = D_min
    if n_points <= 1 or abs(hi - lo) < 1e-14:
        return [target]
    return list(np.geomspace(lo, hi, n_points))


def build_interval_log_grid(lo: float, hi: float, n_points: int) -> List[float]:
    lo = float(lo)
    hi = float(hi)
    if n_points <= 1 or hi <= lo:
        return [lo]
    return list(np.geomspace(lo, hi, n_points))


# =========================================================
# Dynamics
# =========================================================

def knn_topological_pbc(r_wrapped: jnp.ndarray, k: int, L: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    diff = r_wrapped[:, None, :] - r_wrapped[None, :, :]
    diff = minimal_image(diff, L)
    d2 = jnp.sum(diff * diff, axis=-1)

    N = r_wrapped.shape[0]
    big = jnp.array(1e30, dtype=d2.dtype)
    d2 = d2 + jnp.eye(N, dtype=d2.dtype) * big

    vals_neg, idx = lax.top_k(-d2, k)
    nbr_d2 = -vals_neg
    return idx.astype(jnp.int32), nbr_d2


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


def step_state_with_D(state: Dict[str, jnp.ndarray], key: jax.Array, D: float, sim_cfg: SimConfig) -> Dict[str, jnp.ndarray]:
    r_unwrapped = state["r_unwrapped"]
    u = state["u"]
    nbr_idx = state["nbr_idx"]

    nbr_u = u[nbr_idx]
    if sim_cfg.include_self_in_alignment:
        A = (u + jnp.sum(nbr_u, axis=1)) / (sim_cfg.k + 1.0)
    else:
        A = jnp.mean(nbr_u, axis=1)

    eta = jr.normal(key, shape=u.shape, dtype=u.dtype)
    u_new = normalize(A + jnp.sqrt(2.0 * D * sim_cfg.dt) * eta)

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


def make_point_runner(sim_cfg: SimConfig, exp_cfg: Exp4Config):
    eps = exp_cfg.eps

    @jax.jit
    def run_point(seed: int, D: float, burnin_steps: int, measure_steps: int):
        key = jr.PRNGKey(seed)
        state = init_state(key, sim_cfg)
        key = jr.PRNGKey(seed + 1234567)

        burn_key, key = jr.split(key)

        def burn_body(i, st):
            subkey = jr.fold_in(burn_key, i)
            return step_state_with_D(st, subkey, D, sim_cfg)

        state = lax.fori_loop(0, burnin_steps, burn_body, state)

        meas_key, out_key = jr.split(key)

        def meas_body(i, carry):
            st, sum_phi, sum_phi2, sum_sdot, sum_align_cost, sum_turn_cost, sum_nnd = carry
            subkey = jr.fold_in(meas_key, i)

            u_old = st["u"]
            nbr_idx = st["nbr_idx"]
            nbr_u = u_old[nbr_idx]
            if sim_cfg.include_self_in_alignment:
                A = (u_old + jnp.sum(nbr_u, axis=1)) / (sim_cfg.k + 1.0)
            else:
                A = jnp.mean(nbr_u, axis=1)
            A_hat = normalize(A, eps)

            st_new = step_state_with_D(st, subkey, D, sim_cfg)
            u_new = st_new["u"]

            phi = jnp.linalg.norm(jnp.mean(u_new, axis=0))
            turn_cost = jnp.mean(1.0 - jnp.sum(u_old * u_new, axis=1))
            align_cost = jnp.mean(1.0 - jnp.sum(u_old * A_hat, axis=1))

            # strict thermodynamic proxy inherited from previous standards:
            #   \dot S proxy ~ ||u_{t+1} - A_t||^2 / (2 D dt)
            sdot_proxy = jnp.mean(jnp.sum((u_new - A) * (u_new - A), axis=1)) / jnp.maximum(2.0 * D * sim_cfg.dt, eps)

            nnd = jnp.mean(jnp.sqrt(jnp.maximum(st_new["nbr_d2"][:, 0], 0.0)))

            return (
                st_new,
                sum_phi + phi,
                sum_phi2 + phi * phi,
                sum_sdot + sdot_proxy,
                sum_align_cost + align_cost,
                sum_turn_cost + turn_cost,
                sum_nnd + nnd,
            )

        init = (state, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        state, sum_phi, sum_phi2, sum_sdot, sum_align_cost, sum_turn_cost, sum_nnd = lax.fori_loop(
            0,
            measure_steps,
            meas_body,
            init,
        )

        T = jnp.maximum(measure_steps, 1)
        phi_mean = sum_phi / T
        phi2_mean = sum_phi2 / T
        phi_var = jnp.maximum(phi2_mean - phi_mean * phi_mean, 0.0)
        chi = sim_cfg.N * phi_var
        sdot_proxy_mean = sum_sdot / T
        align_cost_mean = sum_align_cost / T
        turn_cost_mean = sum_turn_cost / T
        mean_nn_distance = sum_nnd / T
        J = sdot_proxy_mean / jnp.maximum(chi, eps)

        return {
            "phi_mean": phi_mean,
            "phi_var": phi_var,
            "chi": chi,
            "sdot_proxy_mean": sdot_proxy_mean,
            "work_mean": sdot_proxy_mean,  # keep name for downstream compatibility
            "align_cost_mean": align_cost_mean,
            "turn_cost_mean": turn_cost_mean,
            "mean_nn_distance": mean_nn_distance,
            "J": J,
            "D": jnp.asarray(D),
            "N": jnp.asarray(sim_cfg.N),
            "k": jnp.asarray(sim_cfg.k),
        }, out_key

    return run_point


# =========================================================
# Point-level aggregation
# =========================================================

def run_D_point(
    noise: float,
    seeds: List[int],
    sim_cfg: SimConfig,
    exp_cfg: Exp4Config,
    verbose: bool = True,
):
    sim_cfg_local = SimConfig(**asdict(sim_cfg))
    sim_cfg_local.noise = float(noise)
    point_runner = make_point_runner(sim_cfg_local, exp_cfg)

    seed_results = []
    for s in seeds:
        t0 = time.time()
        res, _ = point_runner(int(s), float(noise), int(exp_cfg.burnin_steps), int(exp_cfg.measure_steps))
        res = {k: float(v) for k, v in jax.device_get(res).items()}
        res["seed"] = int(s)
        seed_results.append(res)
        if verbose:
            print(f"  N={sim_cfg_local.N} k={sim_cfg_local.k} D={noise:.6f} seed={s} done in {time.time() - t0:.2f}s")

    rng = np.random.default_rng(123)
    scalar_names = [
        "phi_mean",
        "phi_var",
        "chi",
        "sdot_proxy_mean",
        "work_mean",
        "align_cost_mean",
        "turn_cost_mean",
        "mean_nn_distance",
        "J",
    ]

    agg_scalars = {}
    for name in scalar_names:
        arr = np.array([r[name] for r in seed_results], dtype=float)
        mean, se, lo, hi = mean_se_ci(arr, n_boot=exp_cfg.n_boot, rng=rng)
        agg_scalars[name] = {
            "mean": mean,
            "se": se,
            "ci_lo": lo,
            "ci_hi": hi,
            "all": arr,
        }

    block = {
        "noise": float(noise),
        "n_seeds": len(seed_results),
        "seed_results": seed_results,
        "agg": {
            "noise": float(noise),
            "n_seeds": len(seed_results),
            "sim_cfg": {**asdict(sim_cfg_local), "dtype": str(sim_cfg_local.dtype), "L": sim_cfg_local.L},
            "exp_cfg": asdict(exp_cfg),
            "scalars": agg_scalars,
        },
    }
    return block


def save_block_json(block: Dict[str, Any], path_json: str):
    payload = {
        "noise": float(block["noise"]),
        "n_seeds": int(block["n_seeds"]),
        "agg": to_jsonable(block["agg"]),
    }
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"saved: {path_json}")


# =========================================================
# k-sweep, bootstrap metrics, and two-stage refinement
# =========================================================

def compute_valley_depth(D: np.ndarray, J: np.ndarray, idx_opt: int, eps: float = 1e-12) -> float:
    left_idx = idx_opt - 1 if idx_opt - 1 >= 0 else None
    right_idx = idx_opt + 1 if idx_opt + 1 < len(D) else None

    shoulders = []
    if left_idx is not None:
        shoulders.append(float(J[left_idx]))
    if right_idx is not None:
        shoulders.append(float(J[right_idx]))
    if not shoulders:
        return np.nan

    shoulder = min(shoulders)
    jmin = float(J[idx_opt])
    return float((shoulder - jmin) / max(abs(jmin), eps))


def curve_metrics_from_mean_curves(D: np.ndarray, J: np.ndarray, chi: np.ndarray, eps: float = 1e-12) -> Dict[str, float]:
    D = np.asarray(D, dtype=float)
    J = np.asarray(J, dtype=float)
    chi = np.asarray(chi, dtype=float)

    idx_opt = int(np.argmin(J))
    idx_c = int(np.argmax(chi))

    D_opt = float(D[idx_opt])
    D_c = float(D[idx_c])
    J_min = float(J[idx_opt])
    delta = float(abs(D_c - D_opt))
    valley_depth = compute_valley_depth(D, J, idx_opt, eps=eps)

    return {
        "D_opt": D_opt,
        "D_c": D_c,
        "J_min": J_min,
        "Delta": delta,
        "valley_depth": valley_depth,
        "idx_opt": idx_opt,
        "idx_c": idx_c,
        "opt_on_left_boundary": bool(idx_opt == 0),
        "opt_on_right_boundary": bool(idx_opt == len(D) - 1),
        "opt_on_boundary": bool(idx_opt == 0 or idx_opt == len(D) - 1),
        "critical_on_left_boundary": bool(idx_c == 0),
        "critical_on_right_boundary": bool(idx_c == len(D) - 1),
        "critical_on_boundary": bool(idx_c == 0 or idx_c == len(D) - 1),
    }


def build_refine_grid_from_coarse(D_c: float, D_opt: float, coarse_D: List[float], coarse_base: Dict[str, Any], grid_cfg: TwoStageGridConfig) -> List[float]:
    pts = []
    pts.extend(build_local_log_grid(D_c, grid_cfg.D_min, grid_cfg.D_max, grid_cfg.refine_half_width_decades, grid_cfg.refine_points_each_target))
    pts.extend(build_local_log_grid(D_opt, grid_cfg.D_min, grid_cfg.D_max, grid_cfg.refine_half_width_decades, grid_cfg.refine_points_each_target))

    if grid_cfg.include_midpoint_refine:
        mid = math.sqrt(max(D_c, grid_cfg.D_min) * max(D_opt, grid_cfg.D_min))
        pts.extend(build_local_log_grid(mid, grid_cfg.D_min, grid_cfg.D_max, 0.12, max(5, grid_cfg.refine_points_each_target - 2)))

    coarse_D = nearest_sorted_unique_float(coarse_D)
    idx_opt = int(coarse_base["idx_opt"])
    idx_c = int(coarse_base["idx_c"])

    # boundary guardrails: if coarse optimum or Dc lands on a boundary, densify the edge interval
    if idx_opt == 0 or idx_c == 0:
        if len(coarse_D) >= 2:
            pts.extend(build_interval_log_grid(coarse_D[0], coarse_D[1], grid_cfg.boundary_guard_points))
    if idx_opt == len(coarse_D) - 1 or idx_c == len(coarse_D) - 1:
        if len(coarse_D) >= 2:
            pts.extend(build_interval_log_grid(coarse_D[-2], coarse_D[-1], grid_cfg.boundary_guard_points))

    return nearest_sorted_unique_float(pts)


def build_summary_df_from_blocks(blocks: Dict[float, Dict[str, Any]], N: int, k: int) -> pd.DataFrame:
    rows = []
    for D in sorted(blocks.keys()):
        sc = blocks[D]["agg"]["scalars"]
        rows.append({
            "N": N,
            "k": k,
            "noise": float(D),
            "n_seeds": blocks[D]["agg"]["n_seeds"],
            "phi_mean": sc["phi_mean"]["mean"],
            "phi_se": sc["phi_mean"]["se"],
            "chi_mean": sc["chi"]["mean"],
            "chi_se": sc["chi"]["se"],
            "work_mean": sc["work_mean"]["mean"],
            "work_se": sc["work_mean"]["se"],
            "sdot_proxy_mean": sc["sdot_proxy_mean"]["mean"],
            "sdot_proxy_se": sc["sdot_proxy_mean"]["se"],
            "align_cost_mean": sc["align_cost_mean"]["mean"],
            "turn_cost_mean": sc["turn_cost_mean"]["mean"],
            "J_mean": sc["J"]["mean"],
            "J_se": sc["J"]["se"],
            "mean_nn_distance": sc["mean_nn_distance"]["mean"],
        })
    return pd.DataFrame(rows).sort_values("noise").reset_index(drop=True)


def build_seed_table_from_blocks(blocks: Dict[float, Dict[str, Any]], N: int, k: int) -> pd.DataFrame:
    rows = []
    for D in sorted(blocks.keys()):
        for r in blocks[D]["seed_results"]:
            rows.append({
                "N": N,
                "k": k,
                "noise": float(D),
                "seed": int(r["seed"]),
                "phi_mean": float(r["phi_mean"]),
                "phi_var": float(r["phi_var"]),
                "chi": float(r["chi"]),
                "sdot_proxy_mean": float(r["sdot_proxy_mean"]),
                "work_mean": float(r["work_mean"]),
                "align_cost_mean": float(r["align_cost_mean"]),
                "turn_cost_mean": float(r["turn_cost_mean"]),
                "mean_nn_distance": float(r["mean_nn_distance"]),
                "J": float(r["J"]),
            })
    return pd.DataFrame(rows).sort_values(["noise", "seed"]).reset_index(drop=True)


def bootstrap_k_metrics(seed_df: pd.DataFrame, n_boot: int = 1000, rng_seed: int = 0) -> Dict[str, Any]:
    rng = np.random.default_rng(rng_seed)

    D_list = np.array(sorted(seed_df["noise"].unique()), dtype=float)
    seeds = np.array(sorted(seed_df["seed"].unique()), dtype=int)
    n_seeds = len(seeds)

    pivot_J = seed_df.pivot(index="seed", columns="noise", values="J").reindex(index=seeds, columns=D_list)
    pivot_chi = seed_df.pivot(index="seed", columns="noise", values="chi").reindex(index=seeds, columns=D_list)
    pivot_phi = seed_df.pivot(index="seed", columns="noise", values="phi_mean").reindex(index=seeds, columns=D_list)
    pivot_work = seed_df.pivot(index="seed", columns="noise", values="work_mean").reindex(index=seeds, columns=D_list)
    pivot_sdot = seed_df.pivot(index="seed", columns="noise", values="sdot_proxy_mean").reindex(index=seeds, columns=D_list)

    J_mat = pivot_J.values.astype(float)
    chi_mat = pivot_chi.values.astype(float)
    phi_mat = pivot_phi.values.astype(float)
    work_mat = pivot_work.values.astype(float)
    sdot_mat = pivot_sdot.values.astype(float)

    J_mean = J_mat.mean(axis=0)
    chi_mean = chi_mat.mean(axis=0)
    phi_mean = phi_mat.mean(axis=0)
    work_mean = work_mat.mean(axis=0)
    sdot_mean = sdot_mat.mean(axis=0)

    base = curve_metrics_from_mean_curves(D_list, J_mean, chi_mean)

    boot_records = []
    for _ in range(n_boot):
        idx = rng.choice(np.arange(n_seeds), size=n_seeds, replace=True)
        J_b = J_mat[idx].mean(axis=0)
        chi_b = chi_mat[idx].mean(axis=0)
        m = curve_metrics_from_mean_curves(D_list, J_b, chi_b)
        boot_records.append(m)

    boot_df = pd.DataFrame(boot_records)

    out = {
        "D_list": D_list,
        "J_curve_mean": J_mean,
        "chi_curve_mean": chi_mean,
        "phi_curve_mean": phi_mean,
        "work_curve_mean": work_mean,
        "sdot_curve_mean": sdot_mean,
        "base": base,
        "bootstrap_summary": {},
        "boot_df": boot_df,
    }

    for name in [
        "D_opt",
        "D_c",
        "J_min",
        "Delta",
        "valley_depth",
        "opt_on_boundary",
        "critical_on_boundary",
    ]:
        arr = boot_df[name].values
        if arr.dtype == bool:
            arr = arr.astype(float)
        arr = np.asarray(arr, dtype=float)
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0:
            fallback = float(base[name]) if name in base and not isinstance(base[name], bool) else float(bool(base.get(name, False)))
            out["bootstrap_summary"][name] = {
                "mean": fallback,
                "se": np.nan,
                "ci_lo": np.nan,
                "ci_hi": np.nan,
                "boot_sd": np.nan,
            }
        else:
            out["bootstrap_summary"][name] = {
                "mean": float(np.mean(arr)),
                "se": float(np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0,
                "ci_lo": float(np.quantile(arr, 0.025)),
                "ci_hi": float(np.quantile(arr, 0.975)),
                "boot_sd": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            }
    return out


def run_k_sweep_two_stage(
    N: int,
    k: int,
    seeds: List[int],
    sim_base: SimConfig,
    exp_cfg: Exp4Config,
    grid_cfg: TwoStageGridConfig,
    out_dir: str,
    verbose: bool = True,
):
    ensure_dir(out_dir)

    sim_cfg = SimConfig(**asdict(sim_base))
    sim_cfg.N = int(N)
    sim_cfg.k = int(k)

    blocks: Dict[float, Dict[str, Any]] = {}

    # stage A: coarse scan
    coarse_grid = nearest_sorted_unique_float(grid_cfg.coarse_D)
    print(f"\n===== stage A coarse: N={N}, k={k} =====")
    for D in coarse_grid:
        blocks[float(D)] = run_D_point(D, seeds, sim_cfg, exp_cfg, verbose=verbose)

    coarse_summary_df = build_summary_df_from_blocks(blocks, N, k)
    coarse_seed_df = build_seed_table_from_blocks(blocks, N, k)
    coarse_metrics = bootstrap_k_metrics(coarse_seed_df, n_boot=exp_cfg.n_boot, rng_seed=123 + 17 * k + 1000 * N)

    D_c_coarse = coarse_metrics["base"]["D_c"]
    D_opt_coarse = coarse_metrics["base"]["D_opt"]

    # stage B: refine near Dc and Dopt + boundary guard
    refine_grid = build_refine_grid_from_coarse(
        D_c=D_c_coarse,
        D_opt=D_opt_coarse,
        coarse_D=coarse_grid,
        coarse_base=coarse_metrics["base"],
        grid_cfg=grid_cfg,
    )
    refine_grid = [d for d in refine_grid if not any(abs(d - c) < 1e-12 for c in coarse_grid)]

    print(f"===== stage B refine: N={N}, k={k}, coarse Dc={D_c_coarse:.6f}, coarse Dopt={D_opt_coarse:.6f} =====")
    print("refine grid:", [round(x, 6) for x in refine_grid])

    for D in refine_grid:
        blocks[float(D)] = run_D_point(D, seeds, sim_cfg, exp_cfg, verbose=verbose)

    summary_df = build_summary_df_from_blocks(blocks, N, k)
    seed_df = build_seed_table_from_blocks(blocks, N, k)
    metrics = bootstrap_k_metrics(seed_df, n_boot=exp_cfg.n_boot, rng_seed=321 + 19 * k + 1000 * N)

    for D in sorted(blocks.keys()):
        save_block_json(blocks[D], os.path.join(out_dir, f"exp4_N{N}_k{k}_noise_{D:.6f}.json"))

    summary_path = os.path.join(out_dir, f"exp4_summary_N{N}_k{k}.csv")
    seed_path = os.path.join(out_dir, f"exp4_seed_table_N{N}_k{k}.csv")
    summary_df.to_csv(summary_path, index=False)
    seed_df.to_csv(seed_path, index=False)
    print(f"saved: {summary_path}")
    print(f"saved: {seed_path}")

    metrics_payload = {
        "N": N,
        "k": k,
        "rho": sim_cfg.rho,
        "v0": sim_cfg.v0,
        "base": to_jsonable(metrics["base"]),
        "bootstrap_summary": to_jsonable(metrics["bootstrap_summary"]),
        "D_list": to_jsonable(metrics["D_list"]),
        "J_curve_mean": to_jsonable(metrics["J_curve_mean"]),
        "chi_curve_mean": to_jsonable(metrics["chi_curve_mean"]),
        "phi_curve_mean": to_jsonable(metrics["phi_curve_mean"]),
        "work_curve_mean": to_jsonable(metrics["work_curve_mean"]),
        "sdot_curve_mean": to_jsonable(metrics["sdot_curve_mean"]),
    }
    with open(os.path.join(out_dir, f"exp4_k_metrics_N{N}_k{k}.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2, ensure_ascii=False)

    plot_point_curves(summary_df, metrics, title=f"Exp4 point curves, N={N}, k={k}", out_path=os.path.join(out_dir, f"exp4_point_curves_N{N}_k{k}.png"))

    return {
        "blocks": blocks,
        "summary_df": summary_df,
        "seed_df": seed_df,
        "metrics": metrics,
        "sim_cfg": sim_cfg,
        "exp_cfg": exp_cfg,
    }


# =========================================================
# Across-k summary, window definition, plots
# =========================================================

def compute_window_from_k_metrics(k_metrics_df: pd.DataFrame) -> Tuple[List[int], int]:
    df = k_metrics_df.sort_values("k").reset_index(drop=True)
    idx_star = int(df["J_min_mean"].idxmin())
    k_star = int(df.loc[idx_star, "k"])
    J_star = float(df.loc[idx_star, "J_min_mean"])
    sigma_star = float(df.loc[idx_star, "J_min_boot_sd"])

    in_window = []
    for _, row in df.iterrows():
        sigma_k = float(row["J_min_boot_sd"])
        tol = max(sigma_k, sigma_star)
        ok = (float(row["J_min_mean"]) - J_star) <= tol
        in_window.append(bool(ok))
    df["in_window"] = in_window
    window_ks = [int(k) for k, ok in zip(df["k"].tolist(), df["in_window"].tolist()) if ok]
    return window_ks, k_star


def summarize_across_k(
    k_results: Dict[int, Dict[str, Any]],
    N: int,
    rho: float,
    v0: float,
    out_dir: str,
):
    ensure_dir(out_dir)
    rows = []

    D_union = nearest_sorted_unique_float([
        float(D)
        for k in k_results
        for D in k_results[k]["summary_df"]["noise"].values.tolist()
    ])

    heat_rows = []
    for k in sorted(k_results.keys()):
        summary_df = k_results[k]["summary_df"].sort_values("noise").reset_index(drop=True)
        metrics = k_results[k]["metrics"]
        b = metrics["base"]
        bs = metrics["bootstrap_summary"]

        rows.append({
            "N": N,
            "rho": rho,
            "v0": v0,
            "k": k,
            "D_opt_mean": float(b["D_opt"]),
            "D_opt_ci_lo": float(bs["D_opt"]["ci_lo"]),
            "D_opt_ci_hi": float(bs["D_opt"]["ci_hi"]),
            "D_c_mean": float(b["D_c"]),
            "D_c_ci_lo": float(bs["D_c"]["ci_lo"]),
            "D_c_ci_hi": float(bs["D_c"]["ci_hi"]),
            "J_min_mean": float(b["J_min"]),
            "J_min_ci_lo": float(bs["J_min"]["ci_lo"]),
            "J_min_ci_hi": float(bs["J_min"]["ci_hi"]),
            "J_min_boot_sd": float(bs["J_min"]["boot_sd"]),
            "Delta_mean": float(b["Delta"]),
            "Delta_ci_lo": float(bs["Delta"]["ci_lo"]),
            "Delta_ci_hi": float(bs["Delta"]["ci_hi"]),
            "Delta_boot_sd": float(bs["Delta"]["boot_sd"]),
            "valley_depth_mean": float(b["valley_depth"]),
            "valley_depth_ci_lo": float(bs["valley_depth"]["ci_lo"]),
            "valley_depth_ci_hi": float(bs["valley_depth"]["ci_hi"]),
            "opt_on_boundary": bool(b["opt_on_boundary"]),
            "opt_on_boundary_boot_mean": float(bs["opt_on_boundary"]["mean"]),
            "critical_on_boundary": bool(b["critical_on_boundary"]),
            "critical_on_boundary_boot_mean": float(bs["critical_on_boundary"]["mean"]),
        })

        D_obs = summary_df["noise"].values.astype(float)
        J_obs = summary_df["J_mean"].values.astype(float)
        J_interp = np.interp(D_union, D_obs, J_obs)
        for D, J in zip(D_union, J_interp):
            heat_rows.append({
                "N": N,
                "rho": rho,
                "v0": v0,
                "k": k,
                "noise": float(D),
                "J_interp": float(J),
            })

    k_metrics_df = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)
    window_ks, k_star = compute_window_from_k_metrics(k_metrics_df)
    k_metrics_df["in_window"] = k_metrics_df["k"].isin(window_ks)

    heat_df = pd.DataFrame(heat_rows).sort_values(["k", "noise"]).reset_index(drop=True)

    k_metrics_path = os.path.join(out_dir, f"exp4_k_metrics_summary_N{N}.csv")
    heat_path = os.path.join(out_dir, f"exp4_heatmap_table_N{N}.csv")
    k_metrics_df.to_csv(k_metrics_path, index=False)
    heat_df.to_csv(heat_path, index=False)
    print(f"saved: {k_metrics_path}")
    print(f"saved: {heat_path}")

    summary_payload = {
        "N": N,
        "rho": rho,
        "v0": v0,
        "k_star": k_star,
        "window_ks": window_ks,
        "window_definition": "J_min(k) - J_min* <= max(sigma_k, sigma_k*) using bootstrap SD",
    }
    with open(os.path.join(out_dir, f"exp4_window_summary_N{N}.json"), "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, ensure_ascii=False)

    plot_exp4_heatmap(heat_df, k_metrics_df, title=f"Exp4 heatmap, N={N}", out_path=os.path.join(out_dir, f"exp4_heatmap_N{N}.png"))
    plot_exp4_jmin(k_metrics_df, title=f"Exp4 Jmin window, N={N}", out_path=os.path.join(out_dir, f"exp4_Jmin_window_N{N}.png"))
    plot_exp4_delta(k_metrics_df, title=f"Exp4 Delta alignment, N={N}", out_path=os.path.join(out_dir, f"exp4_Delta_N{N}.png"))
    plot_exp4_valley(k_metrics_df, title=f"Exp4 valley depth, N={N}", out_path=os.path.join(out_dir, f"exp4_valley_N{N}.png"))

    return {
        "k_metrics_df": k_metrics_df,
        "heat_df": heat_df,
        "window_ks": window_ks,
        "k_star": k_star,
    }


# =========================================================
# Plots
# =========================================================

def plot_point_curves(summary_df: pd.DataFrame, metrics: Dict[str, Any], title: str = "", out_path: Optional[str] = None):
    D = summary_df["noise"].values.astype(float)
    chi = summary_df["chi_mean"].values.astype(float)
    chi_se = summary_df["chi_se"].values.astype(float)
    J = summary_df["J_mean"].values.astype(float)
    J_se = summary_df["J_se"].values.astype(float)
    phi = summary_df["phi_mean"].values.astype(float)
    sdot = summary_df["sdot_proxy_mean"].values.astype(float)

    base = metrics["base"]

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))

    ax = axes[0]
    ax.errorbar(D, chi, yerr=chi_se, marker="o", lw=1.8, capsize=3)
    ax.axvline(base["D_c"], ls="--", lw=1.2)
    ax.set_xscale("log")
    ax.set_title(r"$\chi(D;k)$")
    ax.set_xlabel("D")
    ax.set_ylabel(r"$\chi$")

    ax = axes[1]
    ax.errorbar(D, J, yerr=J_se, marker="o", lw=1.8, capsize=3)
    ax.axvline(base["D_opt"], ls="--", lw=1.2)
    ax.set_xscale("log")
    ax.set_title(r"$J(D;k)$")
    ax.set_xlabel("D")
    ax.set_ylabel("J")
    if base["opt_on_boundary"]:
        ax.text(0.03, 0.94, "boundary opt", transform=ax.transAxes, va="top")

    ax = axes[2]
    ax.plot(D, phi, "o-", label=r"$\Phi$")
    ax.plot(D, sdot, "s--", label=r"$\dot S$ proxy")
    ax.set_xscale("log")
    ax.set_title("order and strict cost")
    ax.set_xlabel("D")
    ax.legend()

    if title:
        fig.suptitle(title, y=1.02, fontsize=13)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=160, bbox_inches="tight")
        plt.close()
        print(f"saved: {out_path}")
    else:
        plt.show()


def plot_exp4_heatmap(heat_df: pd.DataFrame, k_metrics_df: pd.DataFrame, title: str = "", out_path: Optional[str] = None):
    ks = sorted(heat_df["k"].unique())
    Ds = sorted(heat_df["noise"].unique())
    pivot = heat_df.pivot(index="k", columns="noise", values="J_interp").reindex(index=ks, columns=Ds)

    X, Y = np.meshgrid(np.array(Ds, dtype=float), np.array(ks, dtype=float))
    Z = pivot.values.astype(float)

    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    pc = ax.pcolormesh(X, Y, Z, shading="auto")
    plt.colorbar(pc, ax=ax, label="J")

    ax.plot(k_metrics_df["D_c_mean"].values, k_metrics_df["k"].values, "w--", lw=2.0, label=r"$D_c(k)$")
    ax.plot(k_metrics_df["D_opt_mean"].values, k_metrics_df["k"].values, "w-", lw=2.0, label=r"$D_{opt}(k)$")
    bad = k_metrics_df["opt_on_boundary"].values.astype(bool)
    if np.any(bad):
        ax.scatter(
            k_metrics_df.loc[bad, "D_opt_mean"].values,
            k_metrics_df.loc[bad, "k"].values,
            marker="x",
            s=80,
            linewidths=2.0,
            label="boundary opt",
        )
    ax.set_xscale("log")
    ax.set_xlabel("D")
    ax.set_ylabel("k")
    ax.legend(loc="best")
    if title:
        ax.set_title(title)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=170, bbox_inches="tight")
        plt.close()
        print(f"saved: {out_path}")
    else:
        plt.show()


def _err_from_ci(mean_arr: np.ndarray, lo_arr: np.ndarray, hi_arr: np.ndarray) -> np.ndarray:
    lower = mean_arr - lo_arr
    upper = hi_arr - mean_arr
    return np.vstack([lower, upper])


def plot_exp4_jmin(k_metrics_df: pd.DataFrame, title: str = "", out_path: Optional[str] = None):
    df = k_metrics_df.sort_values("k").reset_index(drop=True)
    k = df["k"].values.astype(float)
    y = df["J_min_mean"].values.astype(float)
    lo = df["J_min_ci_lo"].values.astype(float)
    hi = df["J_min_ci_hi"].values.astype(float)

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.errorbar(k, y, yerr=_err_from_ci(y, lo, hi), marker="o", lw=1.8, capsize=3)

    for _, row in df[df["in_window"]].iterrows():
        ax.axvspan(float(row["k"]) - 0.45, float(row["k"]) + 0.45, alpha=0.15)

    boundary_mask = df["opt_on_boundary"].values.astype(bool)
    if np.any(boundary_mask):
        ax.scatter(k[boundary_mask], y[boundary_mask], marker="x", s=70, linewidths=2.0, label="boundary opt")

    ax.set_xlabel("k")
    ax.set_ylabel(r"$J_{min}(k)$")
    ax.set_xticks(k)
    if np.any(boundary_mask):
        ax.legend()
    if title:
        ax.set_title(title)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=170, bbox_inches="tight")
        plt.close()
        print(f"saved: {out_path}")
    else:
        plt.show()


def plot_exp4_delta(k_metrics_df: pd.DataFrame, title: str = "", out_path: Optional[str] = None):
    df = k_metrics_df.sort_values("k").reset_index(drop=True)
    k = df["k"].values.astype(float)
    y = df["Delta_mean"].values.astype(float)
    lo = df["Delta_ci_lo"].values.astype(float)
    hi = df["Delta_ci_hi"].values.astype(float)

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.errorbar(k, y, yerr=_err_from_ci(y, lo, hi), marker="o", lw=1.8, capsize=3)
    ax.set_xlabel("k")
    ax.set_ylabel(r"$\Delta(k)=|D_c-D_{opt}|$")
    ax.set_xticks(k)
    if title:
        ax.set_title(title)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=170, bbox_inches="tight")
        plt.close()
        print(f"saved: {out_path}")
    else:
        plt.show()


def plot_exp4_valley(k_metrics_df: pd.DataFrame, title: str = "", out_path: Optional[str] = None):
    df = k_metrics_df.sort_values("k").reset_index(drop=True)
    k = df["k"].values.astype(float)
    y = df["valley_depth_mean"].values.astype(float)
    lo = df["valley_depth_ci_lo"].values.astype(float)
    hi = df["valley_depth_ci_hi"].values.astype(float)

    mask = ~np.isnan(y)

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.errorbar(k[mask], y[mask], yerr=_err_from_ci(y[mask], lo[mask], hi[mask]), marker="o", lw=1.8, capsize=3)
    ax.set_xlabel("k")
    ax.set_ylabel("valley depth")
    ax.set_xticks(k)
    if title:
        ax.set_title(title)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=170, bbox_inches="tight")
        plt.close()
        print(f"saved: {out_path}")
    else:
        plt.show()


def plot_drift_comparison(all_drift_summaries: Dict[str, Dict[int, Dict[str, Any]]], out_dir: str):
    ensure_dir(out_dir)
    labels = list(all_drift_summaries.keys())

    N_values = sorted({N for label in labels for N in all_drift_summaries[label].keys()})
    for N in N_values:
        fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.6))

        ax = axes[0]
        for label in labels:
            if N not in all_drift_summaries[label]:
                continue
            df = all_drift_summaries[label][N]["k_metrics_df"].sort_values("k")
            ax.plot(df["k"], df["J_min_mean"], marker="o", lw=1.8, label=label)
        ax.set_xlabel("k")
        ax.set_ylabel(r"$J_{min}(k)$")
        ax.set_title(f"Jmin drift, N={N}")
        ax.legend()

        ax = axes[1]
        for label in labels:
            if N not in all_drift_summaries[label]:
                continue
            df = all_drift_summaries[label][N]["k_metrics_df"].sort_values("k")
            ax.plot(df["k"], df["Delta_mean"], marker="o", lw=1.8, label=label)
        ax.set_xlabel("k")
        ax.set_ylabel(r"$\Delta(k)$")
        ax.set_title(f"Delta drift, N={N}")
        ax.legend()

        plt.tight_layout()
        path = os.path.join(out_dir, f"exp4_drift_compare_N{N}.png")
        plt.savefig(path, dpi=170, bbox_inches="tight")
        plt.close()
        print(f"saved: {path}")


# =========================================================
# Full suites
# =========================================================

def run_exp4_suite_for_paramset(
    label: str,
    N_list: List[int],
    k_list: List[int],
    seeds: List[int],
    sim_base: SimConfig,
    exp_cfg: Exp4Config,
    grid_cfg: TwoStageGridConfig,
    out_root: str,
    verbose: bool = False,
):
    root = os.path.join(out_root, label)
    ensure_dir(root)

    combined_point_rows = []
    N_summaries: Dict[int, Dict[str, Any]] = {}

    for N in N_list:
        N_dir = os.path.join(root, f"N{N}")
        ensure_dir(N_dir)

        k_results = {}
        for k in k_list:
            k_dir = os.path.join(N_dir, f"k{k}")
            ensure_dir(k_dir)
            result = run_k_sweep_two_stage(
                N=N,
                k=k,
                seeds=seeds,
                sim_base=sim_base,
                exp_cfg=exp_cfg,
                grid_cfg=grid_cfg,
                out_dir=k_dir,
                verbose=verbose,
            )
            k_results[k] = result
            combined_point_rows.append(result["summary_df"])

        N_summary = summarize_across_k(
            k_results=k_results,
            N=N,
            rho=sim_base.rho,
            v0=sim_base.v0,
            out_dir=N_dir,
        )
        N_summaries[N] = N_summary

    if combined_point_rows:
        combined_df = pd.concat(combined_point_rows, axis=0).reset_index(drop=True)
        combined_path = os.path.join(root, "exp4_all_point_summaries.csv")
        combined_df.to_csv(combined_path, index=False)
        print(f"saved: {combined_path}")

    return N_summaries


def run_exp4_drift_suite(
    drift_cfg: DriftConfig,
    N_list: List[int],
    k_list: List[int],
    seeds: List[int],
    exp_cfg: Exp4Config,
    grid_cfg: TwoStageGridConfig,
    out_root: str,
    include_self_in_alignment: bool = False,
    verbose: bool = False,
):
    ensure_dir(out_root)
    all_summaries = {}

    for p in drift_cfg.parameter_sets:
        label = str(p["label"])
        sim_base = SimConfig(
            N=N_list[0],
            k=k_list[0],
            rho=float(p["rho"]),
            v0=float(p["v0"]),
            noise=grid_cfg.coarse_D[0],
            dt=1.0,
            include_self_in_alignment=include_self_in_alignment,
        )
        print(f"\n==================== PARAM SET: {label} ====================")
        print(f"rho={sim_base.rho}, v0={sim_base.v0}")
        all_summaries[label] = run_exp4_suite_for_paramset(
            label=label,
            N_list=N_list,
            k_list=k_list,
            seeds=seeds,
            sim_base=sim_base,
            exp_cfg=exp_cfg,
            grid_cfg=grid_cfg,
            out_root=out_root,
            verbose=verbose,
        )

    if drift_cfg.enabled:
        plot_drift_comparison(all_summaries, out_dir=os.path.join(out_root, "drift_comparison"))

    return all_summaries


# =========================================================
# Optional regression helper
# =========================================================

def run_k7_anchor_regression(
    out_root: str,
    N: int = 1024,
    seeds: Optional[List[int]] = None,
    verbose: bool = False,
):
    if seeds is None:
        seeds = list(range(5))

    ensure_dir(out_root)
    exp_cfg = Exp4Config(
        burnin_steps=20_000,
        measure_steps=12_000,
        n_boot=1000,
        use_strict_entropy_proxy=True,
    )
    grid_cfg = TwoStageGridConfig(
        coarse_D=[0.005, 0.02, 0.05, 0.0738, 0.10, 0.15],
        refine_half_width_decades=0.12,
        refine_points_each_target=9,
        include_midpoint_refine=True,
        D_min=0.003,
        D_max=0.20,
        boundary_guard_points=7,
    )
    sim_base = SimConfig(
        N=N,
        k=7,
        rho=1.0,
        v0=0.05,
        noise=0.05,
        dt=1.0,
        include_self_in_alignment=False,
    )
    return run_k_sweep_two_stage(
        N=N,
        k=7,
        seeds=seeds,
        sim_base=sim_base,
        exp_cfg=exp_cfg,
        grid_cfg=grid_cfg,
        out_dir=out_root,
        verbose=verbose,
    )


# =========================================================
# Main
# =========================================================

def main():
    BASE_OUT = "exp4_low_degree_window_outputs_strict"
    ensure_dir(BASE_OUT)

    print("JAX devices:", [str(d) for d in jax.devices()])

    # -------------------------
    # smoke test
    # -------------------------
    print("\n===== SMOKE TEST =====")
    smoke_sim = SimConfig(N=256, k=7, rho=1.0, v0=0.05, noise=0.05, dt=1.0, include_self_in_alignment=False)
    smoke_exp = Exp4Config(burnin_steps=2000, measure_steps=3000, n_boot=200, use_strict_entropy_proxy=True)
    smoke_grid = TwoStageGridConfig(coarse_D=[0.01, 0.03, 0.06, 0.10], D_min=0.005, D_max=0.15)
    smoke_dir = os.path.join(BASE_OUT, "smoke")
    ensure_dir(smoke_dir)

    smoke_result = run_k_sweep_two_stage(
        N=256,
        k=7,
        seeds=[0, 1],
        sim_base=smoke_sim,
        exp_cfg=smoke_exp,
        grid_cfg=smoke_grid,
        out_dir=smoke_dir,
        verbose=True,
    )
    print("SMOKE base metrics:")
    print(json.dumps(to_jsonable(smoke_result["metrics"]["base"]), indent=2, ensure_ascii=False))

    # -------------------------
    # optional anchor regression
    # -------------------------
    print("\n===== OPTIONAL K=7 ANCHOR REGRESSION =====")
    anchor_dir = os.path.join(BASE_OUT, "anchor_k7")
    ensure_dir(anchor_dir)
    _ = run_k7_anchor_regression(out_root=anchor_dir, N=256, seeds=[0, 1], verbose=True)

    # -------------------------
    # formal experiment 4
    # -------------------------
    print("\n===== EXPERIMENT 4: LOW-DEGREE WINDOW (STRICT) =====")
    exp_cfg = Exp4Config(
        burnin_steps=20_000,
        measure_steps=12_000,
        n_boot=1000,
        use_strict_entropy_proxy=True,
    )
    grid_cfg = TwoStageGridConfig(
        coarse_D=[0.005, 0.02, 0.035, 0.05, 0.0738, 0.10, 0.15],
        refine_half_width_decades=0.18,
        refine_points_each_target=7,
        include_midpoint_refine=True,
        D_min=0.003,
        D_max=0.20,
        boundary_guard_points=6,
    )
    drift_cfg = DriftConfig(
        enabled=True,
        parameter_sets=[
            {"label": "base", "rho": 1.0, "v0": 0.05},
            {"label": "rho_low", "rho": 0.8, "v0": 0.05},
            {"label": "rho_high", "rho": 1.2, "v0": 0.05},
            {"label": "v0_low", "rho": 1.0, "v0": 0.04},
            {"label": "v0_high", "rho": 1.0, "v0": 0.06},
        ],
    )

    N_list = [1024, 2048]
    k_list = list(range(2, 15))
    seeds = list(range(5))

    all_summaries = run_exp4_drift_suite(
        drift_cfg=drift_cfg,
        N_list=N_list,
        k_list=k_list,
        seeds=seeds,
        exp_cfg=exp_cfg,
        grid_cfg=grid_cfg,
        out_root=BASE_OUT,
        include_self_in_alignment=False,
        verbose=False,
    )

    if "base" in all_summaries:
        for N in N_list:
            if N in all_summaries["base"]:
                summary = all_summaries["base"][N]
                print(f"N={N}, k*={summary['k_star']}, window={summary['window_ks']}")


if __name__ == "__main__":
    main()

from __future__ import annotations

"""
实验 1：3D Topological Vicsek Model — JAX GPU 版（Notebook 直接运行）

设计目标
--------
1. 真正走 GPU：主时间推进用 JAX + lax.scan，避免 Python for-loop 调度开销
2. 直接扔进 Jupyter Notebook 单元格即可运行
3. 保留完整主张网络、逐点 checkpoint、resume、raw time series 落盘、绘图
4. 每个 (N,k,D) 跑完立刻保存，断点续跑
5. 启动时打印 jax.devices()，没有 GPU 就立刻报错，避免“租了 5090 还在 CPU 跑”

说明
----
- 这个版本不再使用 scipy.cKDTree，而是在 JAX 中每步计算 PBC 下的全对距离并做 topological k-NN。
- 对你当前的 N = 512/1024/2048，这样写虽然仍然很重，但会真正把核心算子放到 GPU 上。
- 统计聚合 / bootstrap / 绘图仍在主机端完成，这部分相对模拟主核要轻得多。

Notebook 用法
------------
1. 整段代码粘进一个单元格
2. 保持底部 RUN_MODE = "formal"
3. 运行单元格即可开始正式实验

终端用法
--------
python exp1_vicsek3d_jax_gpu_notebook.py --quick
python exp1_vicsek3d_jax_gpu_notebook.py --formal
"""

import os
# 关闭 JAX 默认 75% GPU 显存预分配，避免刚启动就吃满整张卡
# 如需更高吞吐，可改成 true 或直接删除这行
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import Dict, List, Optional, Tuple

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax, random as jr

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kwargs):  # type: ignore
        return it


# ─────────────────────────────────────────────────────────────
# 0. 配置与结果数据类
# ─────────────────────────────────────────────────────────────

@dataclass
class SimConfig:
    # 物理参数
    v0: float = 0.05
    dt: float = 1.0
    rho: float = 1.0

    # 时间尺度
    t_burn: int = 5_000
    t_measure: int = 20_000
    n_seeds: int = 5

    # 数值截断
    chi_cutoff: float = 1e-6

    # 关联函数
    corr_nbins: int = 32
    corr_sample_size: int = 512
    corr_every: int = 1_000

    # bootstrap
    bootstrap_reps: int = 400
    bootstrap_block_len: int = 200

    # 输出控制
    store_timeseries: bool = True
    store_correlation_profiles: bool = True
    resume: bool = True
    output_dir: str = "results_exp1_gpu_jax"

    # GPU 安全开关
    require_gpu: bool = True


@dataclass
class SeedResult:
    seed: int
    phi_mean: float
    chi: float
    sdot_mean: float
    xi: float
    phi_ts: Optional[np.ndarray] = field(default=None, repr=False)
    sdot_ts: Optional[np.ndarray] = field(default=None, repr=False)
    corr_r: Optional[np.ndarray] = field(default=None, repr=False)
    corr_C: Optional[np.ndarray] = field(default=None, repr=False)


@dataclass
class PointSummary:
    N: int
    k: int
    D: float
    n_seeds: int

    Phi: float
    Phi_se: float
    Phi_ci_lo: float
    Phi_ci_hi: float

    chi: float
    chi_se: float
    chi_ci_lo: float
    chi_ci_hi: float

    Sdot: float
    Sdot_se: float
    Sdot_ci_lo: float
    Sdot_ci_hi: float

    xi: float
    xi_se: float
    xi_ci_lo: float
    xi_ci_hi: float

    J: float
    J_ci_lo: float
    J_ci_hi: float


# ─────────────────────────────────────────────────────────────
# 1. 环境 / 基础工具
# ─────────────────────────────────────────────────────────────

def report_jax_backend(require_gpu: bool = True) -> List[str]:
    devices = jax.devices()
    desc = [f"{d.platform}:{getattr(d, 'device_kind', type(d).__name__)}" for d in devices]
    print("JAX devices:", desc)
    if require_gpu:
        has_gpu = any(d.platform == "gpu" for d in devices)
        if not has_gpu:
            raise RuntimeError(
                "没有检测到 JAX GPU 设备。当前 jax.devices() 只看到 CPU。"
                "请确认服务器上安装的是 GPU-enabled JAX，并且当前 notebook 内核就是那个环境。"
            )
    return desc


def cfg_signature(cfg: SimConfig) -> Tuple:
    return (
        cfg.v0, cfg.dt, cfg.rho,
        cfg.t_burn, cfg.t_measure,
        cfg.corr_nbins, cfg.corr_sample_size, cfg.corr_every,
    )


def system_size(N: int, rho: float) -> float:
    return (N / rho) ** (1.0 / 3.0)


def stable_boot_seed(N: int, k: int, D: float) -> int:
    token = f"{N}|{k}|{D:.10f}".encode("utf-8")
    digest = hashlib.blake2b(token, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % (2**32)


def point_key(N: int, k: int, D: float) -> str:
    return f"N{N}_k{k}_D{D:.8f}"


def ensure_dirs(output_dir: str) -> Dict[str, str]:
    paths = {
        "root": output_dir,
        "points": os.path.join(output_dir, "points"),
        "timeseries": os.path.join(output_dir, "timeseries"),
        "correlations": os.path.join(output_dir, "correlations"),
        "figures": os.path.join(output_dir, "figures"),
        "meta": os.path.join(output_dir, "meta"),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths


def point_paths(output_dir: str, N: int, k: int, D: float) -> Dict[str, str]:
    dirs = ensure_dirs(output_dir)
    key = point_key(N, k, D)
    return {
        "json": os.path.join(dirs["points"], key + ".json"),
        "ts_npz": os.path.join(dirs["timeseries"], key + ".npz"),
        "corr_npz": os.path.join(dirs["correlations"], key + ".npz"),
    }


def normalize_rows(x: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    norms = jnp.linalg.norm(x, axis=1, keepdims=True)
    return x / jnp.where(norms < eps, 1.0, norms)


def minimum_image(dp: jnp.ndarray, L: float) -> jnp.ndarray:
    return dp - L * jnp.round(dp / L)


# ─────────────────────────────────────────────────────────────
# 2. JAX GPU 主核
# ─────────────────────────────────────────────────────────────

_RUNNER_CACHE: Dict[Tuple, callable] = {}


def make_seed_runner(N: int, k: int, cfg: SimConfig):
    """
    构造并缓存一个 jitted 单-seed 运行器：
      inputs: (seed:int, D:float)
      outputs: dict-like pytree containing phi_ts, sdot_ts, corr profile, summary
    """
    key = (N, k) + cfg_signature(cfg)
    if key in _RUNNER_CACHE:
        return _RUNNER_CACHE[key]

    L = float(system_size(N, cfg.rho))
    r_bins = jnp.linspace(0.0, L / 2.0, cfg.corr_nbins + 1, dtype=jnp.float32)
    r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
    sample_size = int(min(N, cfg.corr_sample_size))
    eye_mask = jnp.eye(N, dtype=bool)

    def knn_indices_pbc(pos: jnp.ndarray) -> jnp.ndarray:
        dp = minimum_image(pos[:, None, :] - pos[None, :, :], L)
        dist2 = jnp.sum(dp * dp, axis=-1)
        dist2 = jnp.where(eye_mask, jnp.inf, dist2)
        nbrs = jnp.argpartition(dist2, kth=k - 1, axis=1)[:, :k]
        return nbrs

    def step_once(pos: jnp.ndarray, u: jnp.ndarray, D: jnp.ndarray, key: jnp.ndarray):
        nbrs = knn_indices_pbc(pos)
        A = jnp.mean(u[nbrs], axis=1)
        eta = jr.normal(key, shape=u.shape, dtype=u.dtype)
        u_new = normalize_rows(A + jnp.sqrt(2.0 * D * cfg.dt) * eta)
        pos_new = jnp.mod(pos + cfg.v0 * u_new * cfg.dt, L)
        return pos_new, u_new, A

    def connected_correlation(pos: jnp.ndarray, u: jnp.ndarray, key: jnp.ndarray):
        mean_u = jnp.mean(u, axis=0)
        du = u - mean_u
        denom = jnp.mean(jnp.sum(du * du, axis=1))

        def aligned_case(_):
            sums = jnp.ones(cfg.corr_nbins, dtype=jnp.float32)
            counts = jnp.ones(cfg.corr_nbins, dtype=jnp.int32)
            return sums, counts

        def general_case(subkey):
            idx = jr.permutation(subkey, N)[:sample_size]
            pos_s = pos[idx]
            du_s = du[idx]

            dp = minimum_image(pos_s[:, None, :] - pos_s[None, :, :], L)
            rmat = jnp.linalg.norm(dp, axis=-1)
            dotmat = du_s @ du_s.T

            iu = jnp.triu_indices(sample_size, k=1)
            r_flat = rmat[iu]
            dot_flat = dotmat[iu]

            bin_idx = jnp.searchsorted(r_bins, r_flat, side="right") - 1
            valid = (bin_idx >= 0) & (bin_idx < cfg.corr_nbins) & jnp.isfinite(dot_flat)
            bin_idx = jnp.clip(bin_idx, 0, cfg.corr_nbins - 1)

            values = jnp.where(valid, dot_flat / denom, 0.0)
            counts = jnp.where(valid, 1, 0).astype(jnp.int32)

            sums = jnp.zeros(cfg.corr_nbins, dtype=jnp.float32).at[bin_idx].add(values)
            hits = jnp.zeros(cfg.corr_nbins, dtype=jnp.int32).at[bin_idx].add(counts)
            return sums, hits

        return lax.cond(denom < 1e-15, aligned_case, general_case, key)

    def burn_body(carry, _):
        pos, u, key, D = carry
        key, step_key = jr.split(key)
        pos, u, _ = step_once(pos, u, D, step_key)
        return (pos, u, key, D), None

    def measure_body(carry, t):
        pos, u, key, D, corr_sum, corr_hits = carry
        key, step_key, corr_key = jr.split(key, 3)
        pos, u, A = step_once(pos, u, D, step_key)

        phi = jnp.linalg.norm(jnp.mean(u, axis=0))
        sdot = jnp.mean(jnp.sum((u - A) ** 2, axis=1)) / (2.0 * D * cfg.dt)

        def do_corr(args):
            pos, u, corr_key, corr_sum, corr_hits = args
            sums, hits = connected_correlation(pos, u, corr_key)
            return corr_sum + sums, corr_hits + hits

        def skip_corr(args):
            _, _, _, corr_sum, corr_hits = args
            return corr_sum, corr_hits

        corr_sum, corr_hits = lax.cond(
            jnp.equal(jnp.mod(t, cfg.corr_every), 0),
            do_corr,
            skip_corr,
            (pos, u, corr_key, corr_sum, corr_hits),
        )

        new_carry = (pos, u, key, D, corr_sum, corr_hits)
        obs = (phi, sdot)
        return new_carry, obs

    @jax.jit
    def run_seed(seed: int, D: float):
        key = jr.PRNGKey(seed)
        key, pos_key, u_key = jr.split(key, 3)

        pos0 = jr.uniform(pos_key, shape=(N, 3), minval=0.0, maxval=L, dtype=jnp.float32)
        u0 = normalize_rows(jr.normal(u_key, shape=(N, 3), dtype=jnp.float32))
        D = jnp.asarray(D, dtype=jnp.float32)

        burn_carry = (pos0, u0, key, D)
        burn_carry, _ = lax.scan(burn_body, burn_carry, xs=None, length=cfg.t_burn)
        pos_b, u_b, key_b, D_b = burn_carry

        corr_sum0 = jnp.zeros(cfg.corr_nbins, dtype=jnp.float32)
        corr_hits0 = jnp.zeros(cfg.corr_nbins, dtype=jnp.int32)
        meas_carry = (pos_b, u_b, key_b, D_b, corr_sum0, corr_hits0)

        meas_carry, (phi_ts, sdot_ts) = lax.scan(
            measure_body, meas_carry, xs=jnp.arange(cfg.t_measure, dtype=jnp.int32)
        )
        _, _, _, _, corr_sum, corr_hits = meas_carry

        phi_mean = jnp.mean(phi_ts)
        chi = N * (jnp.mean(phi_ts ** 2) - phi_mean ** 2)
        sdot_mean = jnp.nanmean(sdot_ts)

        corr_C = jnp.where(
            corr_hits > 0,
            corr_sum / corr_hits.astype(corr_sum.dtype),
            jnp.nan,
        )

        return {
            "phi_ts": phi_ts,
            "sdot_ts": sdot_ts,
            "phi_mean": phi_mean,
            "chi": chi,
            "sdot_mean": sdot_mean,
            "r": r_centers,
            "corr_C": corr_C,
        }

    _RUNNER_CACHE[key] = run_seed
    return run_seed


# ─────────────────────────────────────────────────────────────
# 3. 后处理：xi / bootstrap / 聚合
# ─────────────────────────────────────────────────────────────

def compute_xi_zero_crossing(r: np.ndarray, C: np.ndarray) -> float:
    valid = np.isfinite(C)
    if valid.sum() < 2:
        return float(r[-1])

    r_v, C_v = r[valid], C[valid]

    exact = np.where(np.isclose(C_v, 0.0))[0]
    if len(exact) > 0:
        return float(r_v[exact[0]])

    crossings = np.where(np.sign(C_v[:-1]) * np.sign(C_v[1:]) < 0)[0]
    if len(crossings) == 0:
        return float(r_v[-1])

    i = int(crossings[0])
    r0, r1 = r_v[i], r_v[i + 1]
    c0, c1 = C_v[i], C_v[i + 1]
    return float(r0 - c0 * (r1 - r0) / (c1 - c0))


def mean_and_se(x: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan"), float("nan")
    if len(x) == 1:
        return float(x[0]), 0.0
    return float(np.mean(x)), float(np.std(x, ddof=1) / math.sqrt(len(x)))


def block_bootstrap_indices(T: int, block_len: int, rng: np.random.Generator) -> np.ndarray:
    n_blocks = math.ceil(T / block_len)
    max_start = max(T - block_len + 1, 1)
    starts = rng.integers(0, max_start, size=n_blocks)
    idx = np.concatenate([np.arange(s, min(s + block_len, T)) for s in starts])
    return idx[:T]


def bootstrap_summary(
    seed_results: List[SeedResult],
    N: int,
    cfg: SimConfig,
    boot_seed: int,
) -> Dict[str, Tuple[float, float]]:
    rng = np.random.default_rng(boot_seed)
    n_seeds = len(seed_results)
    B = cfg.bootstrap_reps

    Phi_star = np.empty(B, dtype=float)
    chi_star = np.empty(B, dtype=float)
    Sdot_star = np.empty(B, dtype=float)
    xi_star = np.empty(B, dtype=float)
    J_star = np.empty(B, dtype=float)

    has_ts = all(sr.phi_ts is not None and sr.sdot_ts is not None for sr in seed_results)

    if not has_ts:
        s_phi = np.array([sr.phi_mean for sr in seed_results], dtype=float)
        s_chi = np.array([sr.chi for sr in seed_results], dtype=float)
        s_sdot = np.array([sr.sdot_mean for sr in seed_results], dtype=float)
        s_xi = np.array([sr.xi for sr in seed_results], dtype=float)

        for b in range(B):
            idx = rng.integers(0, n_seeds, size=n_seeds)
            phi_b = float(np.nanmean(s_phi[idx]))
            chi_b = float(np.nanmean(s_chi[idx]))
            sdot_b = float(np.nanmean(s_sdot[idx]))
            xi_b = float(np.nanmean(s_xi[idx]))

            Phi_star[b] = phi_b
            chi_star[b] = chi_b
            Sdot_star[b] = sdot_b
            xi_star[b] = xi_b
            J_star[b] = sdot_b / chi_b if (np.isfinite(chi_b) and chi_b > cfg.chi_cutoff) else float("nan")
    else:
        for b in range(B):
            chosen = rng.integers(0, n_seeds, size=n_seeds)
            phi_segs = []
            sdot_segs = []
            xi_vals = []

            for j in chosen:
                sr = seed_results[int(j)]
                assert sr.phi_ts is not None and sr.sdot_ts is not None
                T = len(sr.phi_ts)
                idx = block_bootstrap_indices(T, cfg.bootstrap_block_len, rng)
                phi_segs.append(sr.phi_ts[idx])
                sdot_segs.append(sr.sdot_ts[idx])
                xi_vals.append(sr.xi)

            phi_cat = np.concatenate(phi_segs)
            sdot_cat = np.concatenate(sdot_segs)
            phi_b = float(np.mean(phi_cat))
            chi_b = float(N * (np.mean(phi_cat ** 2) - phi_b ** 2))
            sdot_b = float(np.nanmean(sdot_cat))
            xi_b = float(np.nanmean(np.asarray(xi_vals, dtype=float)))

            Phi_star[b] = phi_b
            chi_star[b] = chi_b
            Sdot_star[b] = sdot_b
            xi_star[b] = xi_b
            J_star[b] = sdot_b / chi_b if (np.isfinite(chi_b) and chi_b > cfg.chi_cutoff) else float("nan")

    def ci(arr: np.ndarray) -> Tuple[float, float]:
        a = arr[np.isfinite(arr)]
        if len(a) == 0:
            return float("nan"), float("nan")
        return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))

    return {
        "Phi": ci(Phi_star),
        "chi": ci(chi_star),
        "Sdot": ci(Sdot_star),
        "xi": ci(xi_star),
        "J": ci(J_star),
    }


def summarize_point(
    N: int,
    k: int,
    D: float,
    seed_results: List[SeedResult],
    cfg: SimConfig,
    boot_seed: int,
) -> PointSummary:
    phi_v = np.array([sr.phi_mean for sr in seed_results], dtype=float)
    chi_v = np.array([sr.chi for sr in seed_results], dtype=float)
    sdot_v = np.array([sr.sdot_mean for sr in seed_results], dtype=float)
    xi_v = np.array([sr.xi for sr in seed_results], dtype=float)

    Phi, Phi_se = mean_and_se(phi_v)
    chi, chi_se = mean_and_se(chi_v)
    Sdot, Sdot_se = mean_and_se(sdot_v)
    xi, xi_se = mean_and_se(xi_v)

    boot = bootstrap_summary(seed_results, N, cfg, boot_seed=boot_seed)

    chi_safe = chi if (np.isfinite(chi) and chi > cfg.chi_cutoff) else float("nan")
    J = Sdot / chi_safe if np.isfinite(chi_safe) else float("nan")

    J_lo, J_hi = boot["J"]
    if not np.isfinite(J_lo) and np.isfinite(J):
        J_lo = J
        J_hi = J

    return PointSummary(
        N=N,
        k=k,
        D=D,
        n_seeds=len(seed_results),
        Phi=Phi,
        Phi_se=Phi_se,
        Phi_ci_lo=boot["Phi"][0],
        Phi_ci_hi=boot["Phi"][1],
        chi=chi,
        chi_se=chi_se,
        chi_ci_lo=boot["chi"][0],
        chi_ci_hi=boot["chi"][1],
        Sdot=Sdot,
        Sdot_se=Sdot_se,
        Sdot_ci_lo=boot["Sdot"][0],
        Sdot_ci_hi=boot["Sdot"][1],
        xi=xi,
        xi_se=xi_se,
        xi_ci_lo=boot["xi"][0],
        xi_ci_hi=boot["xi"][1],
        J=J,
        J_ci_lo=J_lo,
        J_ci_hi=J_hi,
    )


# ─────────────────────────────────────────────────────────────
# 4. 单点运行 / checkpoint / resume
# ─────────────────────────────────────────────────────────────

def save_point_checkpoint(
    output_dir: str,
    summary: PointSummary,
    seed_results: List[SeedResult],
    cfg: SimConfig,
) -> None:
    paths = point_paths(output_dir, summary.N, summary.k, summary.D)

    payload = {
        "summary": asdict(summary),
        "seed_level": [
            {
                "seed": sr.seed,
                "phi_mean": sr.phi_mean,
                "chi": sr.chi,
                "sdot_mean": sr.sdot_mean,
                "xi": sr.xi,
            }
            for sr in seed_results
        ],
        "config_snapshot": asdict(cfg),
        "artifacts": {
            "timeseries_npz": os.path.basename(paths["ts_npz"]),
            "correlation_npz": os.path.basename(paths["corr_npz"]),
        },
    }
    with open(paths["json"], "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    if cfg.store_timeseries:
        ts_payload: Dict[str, np.ndarray] = {}
        for sr in seed_results:
            if sr.phi_ts is not None:
                ts_payload[f"phi_seed{sr.seed}"] = sr.phi_ts
            if sr.sdot_ts is not None:
                ts_payload[f"sdot_seed{sr.seed}"] = sr.sdot_ts
        if len(ts_payload) > 0:
            np.savez_compressed(paths["ts_npz"], **ts_payload)

    if cfg.store_correlation_profiles:
        corr_payload: Dict[str, np.ndarray] = {}
        shared_r_saved = False
        for sr in seed_results:
            if sr.corr_r is not None and not shared_r_saved:
                corr_payload["r"] = sr.corr_r
                shared_r_saved = True
            if sr.corr_C is not None:
                corr_payload[f"C_seed{sr.seed}"] = sr.corr_C
        if len(corr_payload) > 0:
            np.savez_compressed(paths["corr_npz"], **corr_payload)


def load_point_summary(output_dir: str, N: int, k: int, D: float) -> Optional[PointSummary]:
    paths = point_paths(output_dir, N, k, D)
    if not os.path.exists(paths["json"]):
        return None
    with open(paths["json"], "r", encoding="utf-8") as f:
        payload = json.load(f)
    return PointSummary(**payload["summary"])


def run_single_seed(N: int, k: int, D: float, seed: int, cfg: SimConfig) -> SeedResult:
    runner = make_seed_runner(N, k, cfg)
    out = runner(seed, D)
    out = jax.tree_util.tree_map(lambda x: np.asarray(jax.device_get(x)), out)

    phi_ts = out["phi_ts"].astype(np.float32)
    sdot_ts = out["sdot_ts"].astype(np.float32)
    corr_r = out["r"].astype(np.float32)
    corr_C = out["corr_C"].astype(np.float32)

    phi_mean = float(out["phi_mean"])
    chi = float(out["chi"])
    sdot_mean = float(out["sdot_mean"])
    xi = compute_xi_zero_crossing(corr_r, corr_C)

    return SeedResult(
        seed=seed,
        phi_mean=phi_mean,
        chi=chi,
        sdot_mean=sdot_mean,
        xi=xi,
        phi_ts=phi_ts if cfg.store_timeseries else None,
        sdot_ts=sdot_ts if cfg.store_timeseries else None,
        corr_r=corr_r if cfg.store_correlation_profiles else None,
        corr_C=corr_C if cfg.store_correlation_profiles else None,
    )


def run_parameter_point(
    N: int,
    k: int,
    D: float,
    cfg: SimConfig,
    seeds: Optional[List[int]] = None,
) -> Tuple[PointSummary, List[SeedResult]]:
    if seeds is None:
        seeds = list(range(cfg.n_seeds))

    seed_results = []
    for seed in seeds:
        sr = run_single_seed(N, k, D, seed, cfg)
        seed_results.append(sr)

    boot_seed = stable_boot_seed(N, k, D)
    summary = summarize_point(N, k, D, seed_results, cfg, boot_seed=boot_seed)
    return summary, seed_results


def run_or_resume_parameter_point(
    N: int,
    k: int,
    D: float,
    cfg: SimConfig,
    seeds: Optional[List[int]] = None,
) -> PointSummary:
    if cfg.resume:
        cached = load_point_summary(cfg.output_dir, N, k, D)
        if cached is not None:
            return cached

    summary, seed_results = run_parameter_point(N, k, D, cfg, seeds=seeds)
    save_point_checkpoint(cfg.output_dir, summary, seed_results, cfg)
    return summary


# ─────────────────────────────────────────────────────────────
# 5. 扫描 / 全局保存 / 绘图
# ─────────────────────────────────────────────────────────────

def default_coarse_scan() -> np.ndarray:
    return np.unique(np.round(
        np.concatenate([
            np.logspace(-4, -1, 10),
            np.linspace(0.1, 0.5, 8),
        ]),
        8,
    ))


def refine_scan_grid(D_peak: float, n_fine: int = 12, D_min: float = 1e-4, D_max: float = 0.5) -> np.ndarray:
    return np.linspace(max(D_min, 0.5 * D_peak), min(D_max, 2.0 * D_peak), n_fine)


def scan_D(
    N: int,
    k: int,
    cfg: SimConfig,
    D_coarse: Optional[np.ndarray] = None,
    n_fine: int = 12,
    verbose: bool = True,
) -> List[PointSummary]:
    if D_coarse is None:
        D_coarse = default_coarse_scan()

    summaries: List[PointSummary] = []
    done: set = set()

    def run_one(D_val: float) -> None:
        key = round(float(D_val), 10)
        if key in done:
            return
        done.add(key)
        summary = run_or_resume_parameter_point(N, k, float(D_val), cfg)
        summaries.append(summary)

    tag = f"N={N}, k={k}"
    for D in tqdm(D_coarse, desc=f"{tag} 粗扫", disable=not verbose):
        run_one(float(D))

    chi_arr = np.array([s.chi for s in summaries], dtype=float)
    D_arr = np.array([s.D for s in summaries], dtype=float)
    D_peak = float(D_arr[np.nanargmax(chi_arr)])

    D_fine = refine_scan_grid(D_peak, n_fine=n_fine)
    for D in tqdm(D_fine, desc=f"{tag} 细扫", disable=not verbose):
        run_one(float(D))

    summaries.sort(key=lambda s: s.D)
    return summaries


def extract_critical_points(summaries: List[PointSummary], chi_cutoff: float = 1e-6) -> Dict[str, float]:
    D = np.array([s.D for s in summaries], dtype=float)
    chi = np.array([s.chi for s in summaries], dtype=float)
    J = np.array([s.J for s in summaries], dtype=float)

    D_c = float(D[np.nanargmax(chi)])

    valid = np.isfinite(J) & np.isfinite(chi) & (chi > chi_cutoff)
    if np.any(valid):
        D_opt = float(D[valid][np.nanargmin(J[valid])])
        Delta = float(abs(D_c - D_opt))
    else:
        D_opt = float("nan")
        Delta = float("nan")

    return {"D_c": D_c, "D_opt": D_opt, "Delta": Delta}


def save_manifest(cfg: SimConfig, formal_configs: List[Tuple[int, int]], output_dir: str) -> None:
    dirs = ensure_dirs(output_dir)
    manifest = {
        "config": asdict(cfg),
        "formal_configs": [{"N": N, "k": k} for (N, k) in formal_configs],
        "coarse_scan": default_coarse_scan().tolist(),
        "jax_devices": report_jax_backend(require_gpu=False),
    }
    with open(os.path.join(dirs["meta"], "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def save_results(
    output_dir: str,
    all_summaries: Dict[str, List[PointSummary]],
    all_critical: Dict[str, Dict[str, float]],
) -> None:
    ensure_dirs(output_dir)
    payload = {
        "summaries": {label: [asdict(s) for s in summaries] for label, summaries in all_summaries.items()},
        "critical_points": all_critical,
    }
    path = os.path.join(output_dir, "experiment1_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"汇总结果已保存：{path}")


def print_critical_table(all_critical: Dict[str, Dict[str, float]]) -> None:
    print("\n" + "=" * 66)
    print(f"{'Key':>16} {'D_c':>12} {'D_opt':>12} {'Delta':>12}")
    print("=" * 66)
    for label, crit in sorted(all_critical.items()):
        D_c = crit.get("D_c", float("nan"))
        D_opt = crit.get("D_opt", float("nan"))
        Delta = crit.get("Delta", float("nan"))
        D_opt_str = f"{D_opt:.6f}" if np.isfinite(D_opt) else "N/A"
        Delta_str = f"{Delta:.6f}" if np.isfinite(Delta) else "N/A"
        print(f"{label:>16} {D_c:>12.6f} {D_opt_str:>12} {Delta_str:>12}")
    print("=" * 66)


def plot_phase_diagram(
    all_summaries: Dict[str, List[PointSummary]],
    all_critical: Dict[str, Dict[str, float]],
    output_dir: str,
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    dirs = ensure_dirs(output_dir)

    colors = plt.cm.tab20(np.linspace(0, 1, max(len(all_summaries), 1)))
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.28)
    axs = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(4)]

    panel_cfg = [
        ("Phi", "Phi_ci_lo", "Phi_ci_hi", r"$\Phi$ (Order parameter)", r"$\Phi$"),
        ("chi", "chi_ci_lo", "chi_ci_hi", r"$\chi$ (Susceptibility)", r"$\chi$"),
        ("Sdot", "Sdot_ci_lo", "Sdot_ci_hi", r"$\dot{S}$ (Entropy proxy)", r"$\dot{S}$"),
        ("J", "J_ci_lo", "J_ci_hi", r"$J = \dot{S}/\chi$ (Cost per response)", r"$J$"),
    ]

    for ax_i, (value_key, lo_key, hi_key, title, ylabel) in enumerate(panel_cfg):
        ax = axs[ax_i]
        for i, (label, summaries) in enumerate(sorted(all_summaries.items())):
            col = colors[i % len(colors)]
            D = np.array([s.D for s in summaries], dtype=float)
            val = np.array([getattr(s, value_key) for s in summaries], dtype=float)
            lo = np.array([getattr(s, lo_key) for s in summaries], dtype=float)
            hi = np.array([getattr(s, hi_key) for s in summaries], dtype=float)
            mask = np.isfinite(D) & np.isfinite(val)
            if not np.any(mask):
                continue
            ax.plot(D[mask], val[mask], "o-", markersize=4, linewidth=1.3, color=col, label=label)

            lo_m = np.where(np.isfinite(lo[mask]), lo[mask], val[mask])
            hi_m = np.where(np.isfinite(hi[mask]), hi[mask], val[mask])
            ax.fill_between(D[mask], lo_m, hi_m, color=col, alpha=0.10)

            crit = all_critical.get(label, {})
            if ax_i == 1 and np.isfinite(crit.get("D_c", float("nan"))):
                ax.axvline(crit["D_c"], color=col, ls="--", lw=1, alpha=0.55)
            if ax_i == 3 and np.isfinite(crit.get("D_opt", float("nan"))):
                ax.axvline(crit["D_opt"], color=col, ls=":", lw=1, alpha=0.55)

        ax.set_xscale("log")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(r"$D$ (rotational diffusion)", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, ncol=2)

    fig.suptitle("3D Topological Vicsek Model — JAX GPU Phase Diagram & J Landscape", fontsize=13, fontweight="bold")
    out = os.path.join(dirs["figures"], "phase_diagram.png")
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"图像已保存：{out}")


def plot_J_alignment(
    all_summaries: Dict[str, List[PointSummary]],
    all_critical: Dict[str, Dict[str, float]],
    output_dir: str,
) -> None:
    import matplotlib.pyplot as plt

    dirs = ensure_dirs(output_dir)

    colors = plt.cm.viridis(np.linspace(0, 0.95, max(len(all_summaries), 1)))
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

    for i, (label, summaries) in enumerate(sorted(all_summaries.items())):
        col = colors[i % len(colors)]
        D = np.array([s.D for s in summaries], dtype=float)
        chi = np.array([s.chi for s in summaries], dtype=float)
        J = np.array([s.J for s in summaries], dtype=float)
        crit = all_critical.get(label, {})

        mJ = np.isfinite(J)
        mC = np.isfinite(chi)
        if np.any(mJ):
            axes[0].plot(D[mJ], J[mJ], "o-", color=col, markersize=4, label=label)
        if np.isfinite(crit.get("D_opt", float("nan"))):
            axes[0].axvline(crit["D_opt"], color=col, ls=":", lw=1, alpha=0.5)

        if np.any(mC):
            axes[1].plot(D[mC], chi[mC], "s-", color=col, markersize=4, label=label)
        if np.isfinite(crit.get("D_c", float("nan"))):
            axes[1].axvline(crit["D_c"], color=col, ls="--", lw=1, alpha=0.5)

    axes[0].set_xscale("log")
    axes[0].set_xlabel(r"$D$")
    axes[0].set_ylabel(r"$J = \dot{S}/\chi$")
    axes[0].set_title(r"$J$ Landscape (vertical dotted lines = $D_{opt}$)")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=7, ncol=2)

    axes[1].set_xscale("log")
    axes[1].set_xlabel(r"$D$")
    axes[1].set_ylabel(r"$\chi$")
    axes[1].set_title(r"Susceptibility (vertical dashed lines = $D_c$)")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=7, ncol=2)

    out = os.path.join(dirs["figures"], "J_alignment.png")
    plt.tight_layout()
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"图像已保存：{out}")


# ─────────────────────────────────────────────────────────────
# 6. 主入口
# ─────────────────────────────────────────────────────────────

def main_quick(output_dir: str = "results_exp1_gpu_jax") -> None:
    cfg = SimConfig(
        t_burn=500,
        t_measure=1500,
        n_seeds=2,
        bootstrap_reps=100,
        bootstrap_block_len=100,
        corr_every=150,
        corr_sample_size=256,
        store_timeseries=True,
        store_correlation_profiles=True,
        resume=True,
        output_dir=output_dir,
        require_gpu=True,
    )

    report_jax_backend(require_gpu=cfg.require_gpu)
    print("=== Quick test (JAX GPU): N=128, k=5, D=0.08 ===")
    summary = run_or_resume_parameter_point(N=128, k=5, D=0.08, cfg=cfg)
    print(json.dumps(asdict(summary), indent=2, ensure_ascii=False))


def main_formal(output_dir: str = "results_exp1_gpu_jax") -> None:
    cfg = SimConfig(
        t_burn=5_000,
        t_measure=20_000,
        n_seeds=5,
        bootstrap_reps=400,
        bootstrap_block_len=200,
        corr_every=1_000,
        corr_sample_size=512,
        store_timeseries=True,
        store_correlation_profiles=True,
        resume=True,
        output_dir=output_dir,
        require_gpu=True,
    )

    report_jax_backend(require_gpu=cfg.require_gpu)

    N_LIST = [512, 1024, 2048]
    K_LIST = [3, 5, 7, 9, 12]
    FORMAL_CONFIGS: List[Tuple[int, int]] = [(N, k) for N in N_LIST for k in K_LIST]

    save_manifest(cfg, FORMAL_CONFIGS, output_dir)

    D_coarse = default_coarse_scan()
    all_summaries: Dict[str, List[PointSummary]] = {}
    all_critical: Dict[str, Dict[str, float]] = {}

    for N, k in FORMAL_CONFIGS:
        label = f"N={N},k={k}"
        print(f"\n>>> {label}")
        summaries = scan_D(N=N, k=k, cfg=cfg, D_coarse=D_coarse, n_fine=12, verbose=True)
        all_summaries[label] = summaries
        all_critical[label] = extract_critical_points(summaries, chi_cutoff=cfg.chi_cutoff)
        print(f"    {label} -> {all_critical[label]}")

        save_results(output_dir, all_summaries, all_critical)

    print_critical_table(all_critical)
    save_results(output_dir, all_summaries, all_critical)
    plot_phase_diagram(all_summaries, all_critical, output_dir)
    plot_J_alignment(all_summaries, all_critical, output_dir)


# ─────────────────────────────────────────────────────────────
# 7. 终端 / Notebook 双入口
# ─────────────────────────────────────────────────────────────

def main_cli() -> None:
    parser = argparse.ArgumentParser(
        description="3D Topological Vicsek Model — JAX GPU notebook-ready version"
    )
    parser.add_argument("--quick", action="store_true", help="快速验证模式")
    parser.add_argument("--formal", action="store_true", help="正式实验模式")
    parser.add_argument("--output-dir", default="results_exp1_gpu_jax", help="输出目录")
    args, unknown = parser.parse_known_args()

    if unknown:
        print(f"检测到附加参数，已自动忽略： {unknown}")

    if args.quick:
        main_quick(args.output_dir)
    elif args.formal:
        main_formal(args.output_dir)
    else:
        print("请指定 --quick 或 --formal")
        parser.print_help()


def _running_in_ipython() -> bool:
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except Exception:
        return False


RUN_MODE = "formal"          # 改成 "quick" 可先做快速验证；改成 None 则不自动运行
OUTPUT_DIR = "results_exp1_gpu_jax"

if _running_in_ipython():
    if RUN_MODE == "quick":
        print(">>> Notebook mode: running QUICK test")
        main_quick(OUTPUT_DIR)
    elif RUN_MODE == "formal":
        print(">>> Notebook mode: running FORMAL experiment")
        main_formal(OUTPUT_DIR)
    else:
        print("RUN_MODE 设为 None，不自动运行。")
        print("你也可以手动调用：")
        print("  main_quick(OUTPUT_DIR)")
        print("  main_formal(OUTPUT_DIR)")
elif __name__ == "__main__":
    main_cli()

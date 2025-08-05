# plot_iqm_discrete.py
# -----------------------------------------------------------
# Author: Your Name – 2025‑07‑30
# Purpose: IQM plots for MinAtar discrete‑action experiments
# -----------------------------------------------------------

from __future__ import annotations
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# ---------- matplotlib defaults for TMLR -------------------
mpl.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Computer Modern"],
    "axes.labelsize":    7,
    "xtick.labelsize":   6,
    "ytick.labelsize":   6,
    "legend.fontsize":   6,
    "axes.linewidth": 0.75,
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
})
sns.set_style("whitegrid", {'axes.edgecolor': '.8'})

# ---------- constants --------------------------------------
FIGSIZE        = (3.25, 2.1)          # single‑column TMLR
MINATAR_ENVS   = ("Asterix", "Breakout", "Freeway",
                  "Seaquest", "SpaceInvaders")
MAX_STEPS      = 3_000_000
EVAL_STEPS     = np.linspace(0, MAX_STEPS, MAX_STEPS // 100)
colors         = ["#6a6a6a", "#007D81", "#810f7c",
                  "#008fd5", "#fc4f30", "#e5ae38", "#6d904f"]

# ---------- utility functions ------------------------------
def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)

def interpolate_run(steps, returns):
    """Left‑continuous interpolation onto global EVAL_STEPS grid."""
    return np.interp(EVAL_STEPS,
                     steps,
                     returns,
                     left=returns[0],
                     right=returns[-1])

def detect_variant(run_dir_name: str) -> str:
    """Same heuristic as your original script."""
    try:
        descriptor = run_dir_name.split("__", 1)[1].lower()
    except IndexError:
        descriptor = run_dir_name.lower()

    if descriptor.startswith("sac"):
        return "SAC"

    if descriptor.startswith("klac"):
        flags = []
        if "klac_bias" in descriptor:
            flags.append("bonus")
        if "annealing" in descriptor:
            flags.append("anneal")
        if "non_uniform_prior" in descriptor:
            flags.append("prior")
        return "KLAC" + (("+" + "+".join(flags)) if flags else "")

    return "UNKNOWN"

# ---------- data aggregation -------------------------------
def collect_interpolated_returns(root: Path
                                 ) -> Dict[str, List[np.ndarray]]:
    """
    Returns:
        per_algo_runs[algo] = [np.ndarray(shape=(T,)),  ...]  # one per run
    where T = len(EVAL_STEPS).
    """
    per_algo_runs: Dict[str, List[np.ndarray]] = {}

    for run_dir in root.iterdir():
        if not run_dir.is_dir():
            continue
        algo = detect_variant(run_dir.name)
        pkl_file = run_dir / "episodic_return.pkl"
        if not pkl_file.exists():
            continue

        blob = load_pickle(pkl_file)
        for steps, vals in zip(blob["steps"], blob["vals"]):
            interp_vals = interpolate_run(np.asarray(steps),
                                          np.asarray(vals))
            per_algo_runs.setdefault(algo, []).append(interp_vals)

    return per_algo_runs

# ---------- IQM & CI ---------------------------------------
def iqm(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """Inter‑quartile mean along given axis (Agarwal et al., 2021)."""
    q25 = np.nanpercentile(x, 25, axis=axis, keepdims=True)
    q75 = np.nanpercentile(x, 75, axis=axis, keepdims=True)
    mask = (x >= q25) & (x <= q75)
    return np.nanmean(np.where(mask, x, np.nan), axis=axis)

def bootstrap_iqm_curves(runs: np.ndarray,
                         B: int = 1000,
                         alpha: float = .05
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Args:
        runs: array shape (N_runs, T)
    Returns:
        mean, lo, hi   (all shape (T,))
    """
    N, T = runs.shape
    mean = iqm(runs, axis=0)

    boot_stats = np.empty((B, T), dtype=np.float32)
    rng = np.random.default_rng(0)          # deterministic seed

    for b in range(B):
        sample_idx = rng.integers(0, N, size=N)   # resample runs
        boot_stats[b] = iqm(runs[sample_idx], axis=0)

    lo, hi = np.percentile(
        boot_stats,
        [100 * alpha / 2, 100 * (1 - alpha / 2)],
        axis=0
    )
    return mean, lo, hi

# ---------- plotting ---------------------------------------
def plot_iqm_curves(per_algo_runs: Dict[str, List[np.ndarray]],
                    outdir: Path):
    fig, ax = plt.subplots(figsize=FIGSIZE)

    for i, (algo, run_list) in enumerate(sorted(per_algo_runs.items())):
        runs = np.vstack(run_list)                      # (N_runs, T)
        mean, lo, hi = bootstrap_iqm_curves(runs)
        ax.fill_between(EVAL_STEPS, lo, hi,
                        alpha=0.2, facecolor=colors[i])
        ax.plot(EVAL_STEPS, mean,
                color=colors[i],
                linewidth=1.5,
                label=algo)

    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Inter‑quartile mean return")
    ax.grid(True, linewidth=.3)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    for fmt in ("svg", "png"):
        fig.savefig(outdir / f"fig5_minatar_iqm_curves.{fmt}",
                    dpi=300 if fmt == "png" else None,
                    bbox_inches="tight")
    plt.close(fig)

def plot_final_iqm_bar(per_algo_runs: Dict[str, List[np.ndarray]],
                       outdir: Path):
    algo_names, iqm_means, err_lo, err_hi = [], [], [], []
    for algo, run_list in sorted(per_algo_runs.items()):
        runs = np.vstack(run_list)
        final_returns = runs[:, -1]          # last eval point
        mean, lo, hi  = bootstrap_iqm_curves(
            final_returns[:, None])          # shape (N,1)
        algo_names.append(algo)
        iqm_means.append(mean.item())
        err_lo.append(mean - lo)
        err_hi.append(hi - mean)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar(algo_names, iqm_means,
           yerr=[err_lo, err_hi],
           color=[colors[i] for i in range(len(algo_names))],
           capsize=2, width=.6)
    ax.set_ylabel("Final IQM return")
    ax.set_xlabel("")
    ax.grid(axis='y', linewidth=.3)
    fig.tight_layout()
    for fmt in ("svg", "png"):
        fig.savefig(outdir / f"fig6_minatar_iqm_final.{fmt}",
                    dpi=300 if fmt == "png" else None,
                    bbox_inches="tight")
    plt.close(fig)

# ---------- main -------------------------------------------
if __name__ == "__main__":
    ROOT   = Path("/hri/rawstreams/project/klac_2026-01/MinAtar/")
    OUTDIR = Path("../paper_plots/")
    OUTDIR.mkdir(parents=True, exist_ok=True)

    runs_by_algo = collect_interpolated_returns(ROOT)
    print(f"✓ Loaded {sum(map(len, runs_by_algo.values()))} runs "
          f"from {len(runs_by_algo)} algorithms.")

    plot_iqm_curves(runs_by_algo, OUTDIR)
    plot_final_iqm_bar(runs_by_algo, OUTDIR)
    print("✓ IQM figures written to", OUTDIR.resolve())

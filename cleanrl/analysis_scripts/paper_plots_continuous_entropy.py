#!/usr/bin/env python3
"""
Figure builder for: *Maximum Relative Entropy Reinforcement Learning*
========================================================
Generates the four plots that are referenced in the draft:

    Fig. 1  Learning curves on **MinAtar** (SAC vs. KLAC)      (§6.5.1)
    Fig. 2  Learning curves on MuJoCo (all 4 envs aggregated) (§6.5.2)
    Fig. 3  Q–values over training time                      (§6.5.3)
    Fig. 4  AUC (sample-efficiency) comparison               (§6.5.4)

The script follows the figure–formatting rules implicit in the
TMLR LaTeX template:

  * single-column width  = 3.25 inch  (≈ 8.25 cm)
  * max-height           = 2.1 inch   (≈ 5.3 cm)
  * serif text (Computer Modern) to match the template
  * 7 pt axis labels / 6 pt tick labels (never < 5 pt)
  * colour-blind-safe palette (`matplotlib.tab10`)
  * 0.9 pt lines, 0.75 pt box-spines

Requirements
------------
* Python ≥ 3.9
* numpy, pandas, matplotlib, seaborn (optional for smoothing)

Example
-------
$ python make_tmlr_figures.py --root /hri/rawstreams/project/klac_2026-01/
"""
from __future__ import annotations
import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.interpolate import interp1d

# ---------- matplotlib defaults for TMLR -------------------------------------------------
mpl.rcParams.update({
    "font.size": 8,                # base font  ↔  ≥9 pt
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 9,
    "font.family": "serif",
    "font.serif": [
        "Times New Roman",      # Windows / macOS
        "Nimbus Roman",         # Linux (URW)
        "TeX Gyre Termes",      # TeX Live
        "Liberation Serif",     # free replacement
        "DejaVu Serif"          # ships with matplotlib, always present
    ],      # matches TMLR template
    "pdf.fonttype": 42,            # embed as editable text, not paths
})

FIGSIZE = (6.8, 4.2)        # inches, single-column in TMLR
MUJOCO_ENVS = ("Hopper", "Walker2d", "HalfCheetah", "Ant", "InvertedPendulum", "Humanoid", "Swimmer", "Reacher")
N_COLS, N_ROWS = 4, 2
MAX_STEPS = 1000000
EVAL_STEPS = np.linspace(0, MAX_STEPS, num=MAX_STEPS // 200)
SMOOTH_WINDOW = 100
colors = ["#6a6a6a", "#810f7c", "#e5ae38", "#007D81", "#008fd5", "#007D81", "#810f7c", "#fc4f30", "#e5ae38", "#6d904f"]
algorithm_order = ["SAC", "KLAC+bonus+anneal", "KLAC+no_bonus+anneal", "KLAC+bonus", "KLAC+no_bonus"]
algorithms_label_map = {"KLAC+bonus+anneal": r"KLAC", "KLAC+no_bonus": r"KLAC$_{-ab}$", "SAC": "SAC",
                        "KLAC+bonus": r"KLAC$_{-a}$", "KLAC+no_bonus+anneal": r"KLAC$_{-b}$"}
algorithms = [algorithms_label_map[algo] for algo in algorithm_order]
algorithm_color_map = {algorithms_label_map[alg]: colors[i] for i, alg in enumerate(algorithm_order)}
colour_map = dict(zip(algorithms, colors))
_LABEL_FS = 10
_TICK_FS = 9
_TITLE_FS = 10
_LEGEND_FS = 10

# ---------- helpers ----------------------------------------------------------------------
def load_pickle(path: Path) -> Dict[str, List[np.ndarray]]:
    """Load one of the *_metric.pkl files written by extract_tb_scalars.py"""
    with path.open("rb") as f:
        blob = pickle.load(f)
    return blob

def aggregate_runs(steps_list: List[np.ndarray],
                   vals_list:  List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    runs = []
    for s, v in zip(steps_list, vals_list):
        _, interp_vals = interpolate_run(s, v, EVAL_STEPS)
        smoothed = smooth(interp_vals, SMOOTH_WINDOW)
        runs.append(smoothed)
    mean, lo, hi = bootstrap_ci_vectorized(np.array(runs))
    mean = smooth(mean, SMOOTH_WINDOW)
    lo = smooth(lo, SMOOTH_WINDOW)
    hi = smooth(hi, SMOOTH_WINDOW)
    return mean, lo, hi


def bootstrap_ci_vectorized(S, B=1000, alpha=0.05):
    N, T = S.shape
    mean = np.mean(S, axis=0)
    indices = np.random.randint(0, N, size=(B, N, T))
    boot_samples = S[indices, np.arange(T)]
    boot_means = np.mean(boot_samples, axis=1)
    ci_lower, ci_upper = np.percentile(
        boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)], axis=0
    )
    return mean, ci_lower, ci_upper

def smooth(x, weight):
    y = np.ones(weight)
    z = np.ones(len(x))
    return np.convolve(x, y, "same") / np.convolve(z, y, "same")

def interpolate_run(steps, returns, eval_steps):
    interp_fn = interp1d(
        steps,
        returns,
        kind="previous",
        bounds_error=False,
        fill_value=(returns[0], returns[-1]),
    )
    return eval_steps, interp_fn(eval_steps)


def cumtrapz_np(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Vectorised cumulative trapezoidal integral with NumPy only
    (equivalent to scipy.integrate.cumtrapz, but returns the same
     length as the input by prepending a zero).
    """
    dx   = np.diff(x)
    avg  = 0.5 * (y[1:] + y[:-1])
    area = np.cumsum(dx * avg)
    return np.concatenate(([0.0], area))


def detect_variant(run_dir_name: str) -> str:
    try:
        descriptor = run_dir_name.split("__", maxsplit=2)[1].lower()
    except IndexError:
        descriptor = run_dir_name.lower()

    if descriptor.startswith("sac"):
        return "SAC"

    if descriptor.startswith("klac"):
        flags = []
        if "with_bonus" in descriptor:
            flags.append("bonus")
        if "no_bonus" in descriptor:
            flags.append("no_bonus")
        if "with_annealing" in descriptor:
            flags.append("anneal")
        if "with_prior" in descriptor:
            flags.append("prior")
        label = "KLAC" + (("+" + "+".join(flags)) if flags else "")
        return label

    return "UNKNOWN"

def collect_metrics(root: Path):
    """
    Walk KLAC experiment directory and gather:
        metrics[variant][env][metric] -> (steps_list, vals_list)
    """
    metrics: Dict[str, Dict[str, Dict[str, List[List[np.ndarray]]]]] = {}
    for run_dir in root.glob("*"):
        if not run_dir.is_dir():
            continue
        parts = run_dir.name.split("_", 1)         # e.g. HopperKLAC_42 -> ['HopperKLAC', '42']
        env_tag = parts[0].split('-', 1)[0]          # HopperKLAC -> Hopper
        # heuristic: detect variant token anywhere in folder name
        variant = detect_variant(run_dir.name)
        pkl_map = {"entropy.pkl": "entropy"}

        for pkl_file, key in pkl_map.items():
            fp = run_dir / pkl_file
            if not fp.exists():
                continue
            blob = load_pickle(fp)
            steps_list = blob["steps"]
            vals_list  = blob["vals"]

            metrics.setdefault(variant, {}).setdefault(env_tag, {}) \
                   .setdefault(key, ([], []))
            metrics[variant][env_tag][key][0].extend(steps_list)
            metrics[variant][env_tag][key][1].extend(vals_list)
    return metrics


def plot_entropy_per_env(metrics, out_path):
    fig, axes = plt.subplots(N_ROWS, N_COLS,
                             figsize=(FIGSIZE[0], FIGSIZE[1]))
    axes = axes.flatten()
    variants = sorted(metrics.keys())
    markers = {
        "SAC": "o",          # circle
        "KLAC": "s",         # square
        r"KLAC$_{-b}$": "D", # diamond
        r"KLAC$_{-a}$": "^", # triangle
        r"KLAC$_{-ab}$": "v" # inverted triangle
    }
    marker_offsets = np.linspace(-2.5, 0, len(algorithms))

    for env_ax, env in zip(axes, MUJOCO_ENVS):
        for i, variant in enumerate(algorithm_order):
            if env not in metrics[variant]:
                continue
            mean, lo, hi = aggregate_runs(*metrics[variant][env]["entropy"])
            label = algorithms_label_map[variant]
            color = colour_map[label]
            env_ax.fill_between(EVAL_STEPS, lo, hi,
                                alpha=0.2, facecolor=color)
            env_ax.plot(EVAL_STEPS, mean, linewidth=1.0,
                        color=color, label=label)
            x_pos = EVAL_STEPS[-1] * (1 + marker_offsets[i] * 0.05)
            y_pos = mean[-1]
            env_ax.plot(x_pos, y_pos,
                        marker=markers[label],
                        color=algorithm_color_map[label],
                        markersize=4,
                        linestyle="None")
        env_ax.set_title(env + "-v5")
        env_ax.set_xlabel("Env steps")
        env_ax.grid(True, linewidth=.3)

    axes[0].set_ylabel("Policy entropy")
    axes[4].set_ylabel("Policy entropy")
    labels = [algorithms_label_map[variant] for variant in algorithm_order]
    handles = [
        plt.Line2D([0], [0],
                   color=colour_map[alg],
                   marker=markers[alg],
                   linewidth=1.0,
                   markersize=4,
                   label=alg)
        for alg in algorithms
    ]
    fig.legend(handles, algorithms,
               loc="upper center",
               bbox_to_anchor=(0.5, 1.04),
               ncol=len(algorithms),
               frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path / "fig7_mujoco_entropy_per_env.svg",
                format="svg", bbox_inches="tight")
    fig.savefig(out_path / "fig7_mujoco_entropy_per_env.png",
                format="png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    root = Path("/hri/rawstreams/project/klac_2026-01/")
    outdir = Path("../paper_plots/")

    outdir.mkdir(parents=True, exist_ok=True)
    metrics = collect_metrics(root)

    # plot_learning_curves_minatar(metrics, outdir / "fig1_minatar_curves.pdf")
    plot_entropy_per_env(metrics, outdir)
    print("✓ All figures written to", outdir.resolve())
    plt.show()

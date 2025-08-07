
from __future__ import annotations
import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from scipy.interpolate import interp1d

# ---------- matplotlib defaults for TMLR -------------------------------------------------
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": [
        "Times New Roman",
        "Nimbus Roman",
        "TeX Gyre Termes",
        "Liberation Serif",
        "DejaVu Serif"
    ],
    "axes.labelsize":    12,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   12,
    "axes.linewidth": 0.75,
    "pdf.fonttype":      42,   # editable text in the PDF
    "ps.fonttype":       42,
})
sns.set_style("whitegrid", {'axes.edgecolor': '.8'})

FIGSIZE = (3.25, 2.1)        # inches, single-column in TMLR
MINATAR_ENVS = ("Seaquest")
N_COLS, N_ROWS = 2, 1
MAX_STEPS = 3000000
EVAL_STEPS = np.linspace(0, MAX_STEPS, num=MAX_STEPS // 500)
SMOOTH_WINDOW = 100
colors = ["#6a6a6a", "#810f7c", "#007D81", "#008fd5", "#fc4f30", "#e5ae38", "#6d904f"]

# ---------- helpers ----------------------------------------------------------------------
def load_pickle(path: Path) -> Dict[str, List[np.ndarray]]:
    """Load one of the *_metric.pkl files written by extract_tb_scalars.py"""
    with path.open("rb") as f:
        blob = pickle.load(f)
    return blob

def add_global_legend(fig, variant_labels, colour_cycle):
    """Place a centred legend above all subplots."""
    handles = [plt.Line2D([0], [0],
                          color=colour_cycle[i % len(colour_cycle)],
                          linewidth=1.0)
               for i, _ in enumerate(variant_labels)]
    fig.legend(handles, variant_labels,
               loc="upper center",
               bbox_to_anchor=(0.5, 1.04),
               ncol=len(variant_labels),
               frameon=False)

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
        pkl_map = {"episodic_return.pkl": "return",
                   "q_values.pkl":        "q"}

        for pkl_file, key in pkl_map.items():
            fp = run_dir / pkl_file
            if not fp.exists():
                continue
            blob = load_pickle(fp)
            steps_list = blob["steps"]
            vals_list = blob["vals"]

            metrics.setdefault(variant, {}).setdefault(env_tag, {}).setdefault(key, ([], []))
            metrics[variant][env_tag][key][0].extend(steps_list)
            metrics[variant][env_tag][key][1].extend(vals_list)
    return metrics

# ---------- plotting routines ------------------------------------------------------------
def plot_learning_curves_minatar(metrics, out_path):
    fig_w = FIGSIZE[0] * N_COLS  # keep each panel single-column width
    fig_h = FIGSIZE[1] * N_ROWS
    fig, axes = plt.subplots(
        N_ROWS, N_COLS,
        figsize=(fig_w, fig_h),
        sharey=False
    )
    axes = axes.flatten()
    variants = sorted(metrics.keys())
    algorithms_label_map = {"KLAC+bonus+anneal": r"KLAC", "KLAC+no_bonus": r"KLAC$_{-ab}$", "SAC": "SAC",
                            "KLAC+bonus": r"KLAC$_{-a}$", "KLAC+no_bonus+anneal": r"KLAC$_{-b}$"}
    algorithms_list = ["SAC", "KLAC", r"KLAC$_{-b}$", r"KLAC$_{-a}$", r"KLAC$_{-ab}$"]
    algorithm_color_map = {alg: colors[i] for i, alg in enumerate(algorithms_list)}

    env_ax = axes[0]
    env = MINATAR_ENVS
    for i, variant in enumerate(variants):
        if env not in metrics[variant]:
            continue
        mean, lo, hi = aggregate_runs(*metrics[variant][env]["return"])
        env_ax.fill_between(EVAL_STEPS, lo, hi,
                            alpha=0.2, facecolor=algorithm_color_map[algorithms_label_map[variant]])
        env_ax.plot(EVAL_STEPS, mean, linewidth=1.0,
                    color=algorithm_color_map[algorithms_label_map[variant]],
                    label=algorithms_label_map[variant])
    env_ax.set_title(env, fontsize=10)
    env_ax.set_xlabel("Env steps")
    env_ax.grid(True, linewidth=.3, linestyle='--')

    env_ax = axes[1]
    for i, variant in enumerate(variants):
        if env not in metrics[variant]:
            continue
        mean, lo, hi = aggregate_runs(*metrics[variant][env]["q"])
        env_ax.fill_between(EVAL_STEPS, lo, hi,
                            alpha=0.2, facecolor=algorithm_color_map[algorithms_label_map[variant]])
        env_ax.plot(EVAL_STEPS, mean, linewidth=1.0,
                    color=algorithm_color_map[algorithms_label_map[variant]],
                    label=algorithms_label_map[variant])
    env_ax.set_title(env, fontsize=10)
    env_ax.set_xlabel("Env steps")
    env_ax.grid(True, linewidth=.3, linestyle='--')

    axes[0].set_ylabel("Episodic return")
    axes[1].set_ylabel("Q values")
    add_global_legend(fig, algorithms_list, colors)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_file_svg = out_path / "seaquest_plot.svg"
    out_file_png = out_path / "seaquest_plot.png"
    fig.savefig(out_file_svg, format="svg", bbox_inches="tight")
    fig.savefig(out_file_png, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    root = Path("/hri/rawstreams/project/klac_2026-01/MinAtar")
    outdir = Path("../paper_plots/")

    outdir.mkdir(parents=True, exist_ok=True)
    metrics = collect_metrics(root)

    # plot_learning_curves_minatar(metrics, outdir / "fig1_minatar_curves.pdf")
    plot_learning_curves_minatar(metrics, outdir)
    print("✓ All figures written to", outdir.resolve())

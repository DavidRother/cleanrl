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
import seaborn as sns
from scipy.interpolate import interp1d

# ---------- matplotlib defaults for TMLR -------------------------------------------------
mpl.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Computer Modern"],
    "axes.labelsize":    7,
    "xtick.labelsize":   6,
    "ytick.labelsize":   6,
    "legend.fontsize":   6,
    "axes.linewidth": 0.75,
    "pdf.fonttype":      42,   # editable text in the PDF
    "ps.fonttype":       42,
})
sns.set_style("whitegrid", {'axes.edgecolor': '.8'})

FIGSIZE = (3.25, 2.1)        # inches, single-column in TMLR
MUJOCO_ENVS = ("Hopper", "Walker2d", "HalfCheetah", "Ant", "InvertedPendulum", "Humanoid", "Swimmer", "Reacher")
N_COLS, N_ROWS = 4, 2
MAX_STEPS = 1000000
EVAL_STEPS = np.linspace(0, MAX_STEPS, num=MAX_STEPS // 100)
SMOOTH_WINDOW = 400
colors = ["#6a6a6a", "#007D81", "#810f7c", "#008fd5", "#fc4f30", "#e5ae38", "#6d904f"]

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
        descriptor = run_dir_name.split("v5_", maxsplit=2)[1].lower()
    except IndexError:
        descriptor = run_dir_name.lower()

    if descriptor.startswith("sac"):
        return "SAC"

    if descriptor.startswith("klac"):
        flags = []
        if "with_bonus"      in descriptor:
            flags.append("bonus")
        if "with_annealing"  in descriptor:
            flags.append("anneal")
        if "with_prior"      in descriptor:
            flags.append("prior")
        label = "KLAC" + (("+" + "+".join(flags)) if flags else "")
        return label

    return "UNKNOWN"

def collect_metrics(root: Path) -> Dict[str, Dict[str, Dict[str, List[np.ndarray]]]]:
    """
    Walk KLAC experiment directory and gather:
        metrics[variant][env][metric] -> (steps_list, vals_list)
    """
    metrics: Dict[str, Dict[str, Dict[str, List[List[np.ndarray]]]]] = {}
    for run_dir in root.glob("*"):
        if not run_dir.is_dir():
            continue
        parts   = run_dir.name.split("_", 1)         # e.g. HopperKLAC_42 -> ['HopperKLAC', '42']
        env_tag = parts[0].split('-', 1)[0]          # HopperKLAC -> Hopper
        # heuristic: detect variant token anywhere in folder name
        variant = detect_variant(run_dir.name)
        pkl_map = {"episodic_return.pkl": "return",
                   "entropy.pkl":         "entropy",
                   "q_values.pkl":        "q"}

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

# ---------- plotting routines ------------------------------------------------------------
def plot_learning_curves_minatar(metrics, out_path):
    """Fig 1: per-game learning curves on MinAtar (5 panels, SAC vs KLAC)."""
    tasks   = ("Breakout", "Asterix", "Seaquest", "Freeway", "SpaceInvaders")
    colours = {"SAC": sns.color_palette("tab10")[0],
               "KLAC": sns.color_palette("tab10")[1]}
    fig, axes = plt.subplots(1, len(tasks), figsize=(FIGSIZE[0]*len(tasks)/2, FIGSIZE[1]),
                             sharey=True)
    for env, ax in zip(tasks, axes):
        for variant in ("SAC", "KLAC"):
            if env not in metrics[variant]:
                continue
            steps, mean, ci = aggregate_runs(*metrics[variant][env]["return"])
            ax.plot(steps, mean, label=variant,
                    color=colours[variant], linewidth=0.9)
            ax.fill_between(steps, mean-ci, mean+ci,
                            color=colours[variant], alpha=0.25, linewidth=0)
        ax.set_title(env, fontsize=7)
        ax.set_xlabel("Environment steps")
        ax.grid(True, linewidth=0.3)
    axes[0].set_ylabel("Episodic return")
    axes[-1].legend(frameon=False, bbox_to_anchor=(1.02, 1.0))
    plt.tight_layout()
    fig.savefig(out_path, format="svg", bbox_inches="tight")

def plot_learning_curves_mujoco(metrics, out_path):
    fig_w = FIGSIZE[0] * N_COLS          # keep each panel single-column width
    fig_h = FIGSIZE[1] * N_ROWS
    fig, axes = plt.subplots(
        N_ROWS, N_COLS,
        figsize=(fig_w, fig_h),
        sharey=False
    )
    axes = axes.flatten()
    variants = sorted(metrics.keys())

    for env_ax, env in zip(axes, MUJOCO_ENVS):
        for i, variant in enumerate(variants):
            if env not in metrics[variant]:
                continue
            mean, lo, hi = aggregate_runs(*metrics[variant][env]["return"])
            env_ax.fill_between(EVAL_STEPS, lo, hi,
                                alpha=0.2, facecolor=colors[i])
            env_ax.plot(EVAL_STEPS, mean, linewidth=2.5,
                        color=colors[i], label=variant)
        env_ax.set_title(env, fontsize=7)
        env_ax.set_xlabel("Env steps")
        env_ax.grid(True, linewidth=.3)

    axes[0].set_ylabel("Episodic return")
    add_global_legend(fig, variants, colors)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_file_svg = out_path / "fig2_mujoco_sparse_curves_per_env.svg"
    out_file_png = out_path / "fig2_mujoco_sparse_curves_per_env.png"
    fig.savefig(out_file_svg, format="svg", bbox_inches="tight")
    fig.savefig(out_file_png, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_q_values_per_env(metrics, out_path):
    fig_w, fig_h = FIGSIZE[0] * N_COLS, FIGSIZE[1] * N_ROWS
    fig, axes = plt.subplots(N_ROWS, N_COLS,
                             figsize=(fig_w, fig_h),
                             sharey=False)
    axes = axes.flatten()
    variants = sorted(metrics.keys())

    for env_ax, env in zip(axes, MUJOCO_ENVS):
        for i, variant in enumerate(variants):
            if env not in metrics[variant] or "q" not in metrics[variant][env]:
                continue
            steps, mean, ci = aggregate_runs(*metrics[variant][env]["q"])
            mean, lo, hi = aggregate_runs(*metrics[variant][env]["return"])
            env_ax.fill_between(EVAL_STEPS, lo, hi,
                                alpha=0.2, facecolor=colors[i])
            env_ax.plot(EVAL_STEPS, mean, linewidth=2.5,
                        color=colors[i], label=variant)
        env_ax.set_title(env, fontsize=7)
        env_ax.set_xlabel("Env steps")
        env_ax.grid(True, linewidth=.3)

    axes[0].set_ylabel(r"$Q_t$ (critic)")
    add_global_legend(fig, variants, colors)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_file_svg = out_path / "fig3_mujoco_sparse_q_values_per_env.svg"
    out_file_png = out_path / "fig3_mujoco_sparse_q_values_per_env.png"
    fig.savefig(out_file_svg, format="svg", bbox_inches="tight")
    fig.savefig(out_file_png, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_auc(metrics, out_path):
    """Fig 4: bar-plot of AUC across envs (higher = better sample-efficiency)."""
    records = []
    for variant, env_dict in metrics.items():
        for env, m in env_dict.items():
            steps, mean, _ = aggregate_runs(*m["return"])
            auc = np.trapz(mean, steps) / steps[-1]
            records.append({"Variant": variant, "Env": env, "AUC": auc})
    df = pd.DataFrame(records)
    order = df.groupby("Variant")["AUC"].median().sort_values(ascending=False).index
    plt.figure(figsize=FIGSIZE)
    sns.barplot(data=df, x="Variant", y="AUC", order=order,
                palette="tab10", width=0.75, capsize=.02, errcolor=".3")
    plt.ylabel("Normalised AUC")
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(out_path, format="svg", bbox_inches="tight")


def plot_auc_curves_per_env(metrics, out_path):
    fig_w, fig_h = FIGSIZE[0] * N_COLS, FIGSIZE[1] * N_ROWS
    fig, axes = plt.subplots(N_ROWS, N_COLS,
                             figsize=(fig_w, fig_h),
                             sharey=True,
                             sharex="none")
    axes = axes.flatten()
    variants = sorted(metrics.keys())

    for env_ax, env in zip(axes, MUJOCO_ENVS):
        for i, variant in enumerate(variants):
            if env not in metrics[variant]:
                continue
            steps, mean, _ = aggregate_runs(*metrics[variant][env]["return"])
            auc_cum = cumtrapz_np(mean, steps) / steps
            env_ax.plot(steps, auc_cum,
                        label=variant,
                        color=colors[i],
                        linewidth=0.9)
        env_ax.set_title(env, fontsize=7)
        env_ax.set_xlabel("Env steps")
        env_ax.grid(True, linewidth=.3)

    axes[0].set_ylabel("Cumulative AUC / step")
    add_global_legend(fig, variants, colors)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_file_svg = out_path / "fig4_mujoco_sparse_auc_per_env.svg"
    out_file_png = out_path / "fig4_mujoco_sparse_auc_per_env.png"
    fig.savefig(out_file_svg, format="svg", bbox_inches="tight")
    fig.savefig(out_file_png, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    root = Path("/hri/rawstreams/project/klac_2026-01/SparseContinuous")
    outdir = Path("../paper_plots/")

    outdir.mkdir(parents=True, exist_ok=True)
    metrics = collect_metrics(root)

    # plot_learning_curves_minatar(metrics, outdir / "fig1_minatar_curves.pdf")
    plot_learning_curves_mujoco(metrics, outdir)
    plot_q_values_per_env(metrics, outdir)
    plot_auc_curves_per_env(metrics, outdir)
    print("✓ All figures written to", outdir.resolve())

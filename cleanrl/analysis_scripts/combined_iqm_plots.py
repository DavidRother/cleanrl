#!/usr/bin/env python3
"""
Combine the Mujoco and MinAtar aggregate bar-plots
into one 2×3 figure (row 0: Mujoco, row 1: MinAtar).

The code is a straight merge of your two stand-alone
scripts with minimal refactoring:

* identical data-loading / normalisation logic
* identical bootstrap settings (5 000 reps, 95 % CI)
* identical colour palette and rcParams
* identical tick logic for each row
"""

from pathlib import Path
from typing import Dict, List, Tuple

import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from rliable import library as rly
from rliable import metrics, plot_utils

# --------------------------------------------------------------------------- #
# Matplotlib style   (unchanged)
# --------------------------------------------------------------------------- #
mpl.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "font.family": "serif",
    "font.serif": [
        "Times New Roman",
        "Nimbus Roman",
        "TeX Gyre Termes",
        "Liberation Serif",
        "DejaVu Serif"
    ],
    "pdf.fonttype": 42,
})
# consistent colours across both rows
COLORS = ["#008fd5", "#6a6a6a", "#007D81",
          "#810f7c", "#fc4f30", "#e5ae38", "#6d904f"]

# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def _detect_variant_minatar(run_dir_name: str) -> str:
    # same heuristic as in your MinAtar script
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


def _detect_variant_mujoco(run_dir_name: str) -> str:
    # same heuristic as in your Mujoco script
    try:
        descriptor = run_dir_name.split("__", maxsplit=2)[1].lower()
    except IndexError:
        descriptor = run_dir_name.lower()

    if descriptor.startswith("sac"):
        return "SAC"

    if descriptor.startswith("klac"):
        flags = []
        if "with_bonus"     in descriptor: flags.append("bonus")
        if "with_annealing" in descriptor: flags.append("anneal")
        if "with_prior"     in descriptor: flags.append("prior")
        return "KLAC" + (("+" + "+".join(flags)) if flags else "")
    return "UNKNOWN"


def _collect_metrics(root: Path,
                     detect_variant_fn) -> Dict[str, Dict[str, Dict[str, List]]]:
    """Returns a nested dict[alg][env]['return'] = list[runs]"""
    out = {}
    for run_dir in root.iterdir():
        if not run_dir.is_dir():
            continue
        env_tag  = run_dir.name.split("_", 1)[0].split('-', 1)[0]
        variant  = detect_variant_fn(run_dir.name)
        pkl_file = run_dir / "episodic_return.pkl"
        if not pkl_file.exists():
            continue
        blob = _load_pickle(pkl_file)
        # average over last 10 evaluations
        vals = [np.mean(v[-10:]) for v in blob["vals"]]
        out.setdefault(variant, {}).setdefault(env_tag, {}) \
           .setdefault("return", []).extend(vals)
    return out


def _normalise_to_sac(score_dict: dict,
                      sac_key: str = "SAC",
                      ref_func=np.mean,
                      label_map=None):
    if sac_key not in score_dict:
        raise KeyError(f"SAC baseline '{sac_key}' not found")
    sac_ref = ref_func(score_dict[sac_key], axis=0)
    sac_ref = np.where(sac_ref == 0, 1e-8, sac_ref)

    norm = {}
    for alg, arr in score_dict.items():
        name = label_map[alg] if label_map else alg
        norm[name] = arr / sac_ref
    return norm


def _bootstrap(score_dict_norm, reps=5000):
    agg_vec = lambda x: np.array([
        metrics.aggregate_mean(x),
        metrics.aggregate_iqm(x),
        metrics.aggregate_median(x)
    ])
    return rly.get_interval_estimates(
        score_dict_norm, agg_vec, reps=reps, confidence_interval_size=0.95)


def _plot_row(point_est, ci_bounds,
              algorithms: List[str],
              aggregators: List[str],
              colour_map: Dict[str, str],
              axes_row,
              row_name: str,
              xtick_strategy: str):
    """
    Draw one row of 3 small multiples (Mean, IQM, Median).
    """
    for j, metric in enumerate(aggregators):
        ax = axes_row[j]
        # bar-plot with asymmetric CI error bars
        y_pos = np.arange(len(algorithms))[::-1]  # top = first algo
        vals  = [point_est[a][j]      for a in algorithms]
        lows  = [ci_bounds[a][0][j]   for a in algorithms]
        highs = [ci_bounds[a][1][j]   for a in algorithms]
        # bars
        for k, algo in enumerate(algorithms):
            ax.barh(y_pos[k], vals[k],
                    color=colour_map[algo], height=0.6)
            ax.errorbar(x=vals[k], y=y_pos[k],
                        xerr=np.array([[vals[k]-lows[k]],
                                       [highs[k]-vals[k]]]),
                        fmt='none', ecolor='black', capsize=3, linewidth=1)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(algorithms if j == 0 else ['']*len(algorithms))
        ax.set_title(metric)
        ax.grid(axis='x', linestyle=':', linewidth=0.4)
        ax.set_xlabel('SAC normalised score')

        # custom x-ticks exactly as in your originals
        if xtick_strategy == "mujoco":
            ax.set_xticks([0.9, 1.0, 1.1])
        else:  # minatar
            v_full = int(ci_bounds["KLAC"][1][j]) + 1
            ticks  = sorted({1, (1+v_full)//2, v_full})
            ax.set_xticks(ticks)

    # row label ( centred above the three sub-axes )
    axes_row[1].annotate(row_name,
                         xy=(0.5, 1.15),
                         xycoords='axes fraction',
                         ha='center', va='bottom',
                         fontsize=13, fontweight='bold')


# --------------------------------------------------------------------------- #
#                             Path configuration                              #
# --------------------------------------------------------------------------- #
ROOT_MUJ   = Path("/hri/rawstreams/project/klac_2026-01/")
ROOT_MINA  = Path("/hri/rawstreams/project/klac_2026-01/MinAtar/")
OUTDIR     = Path("../paper_plots/")
OUTDIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
#                ----------   MUJOCO (row 0)   ----------                     #
# --------------------------------------------------------------------------- #
metrics_mj  = _collect_metrics(ROOT_MUJ, _detect_variant_mujoco)
envs_mj     = sorted(next(iter(metrics_mj.values())).keys())
score_mj    = {alg: np.stack([np.asarray(env_dict[e]['return'], dtype=np.float32)
                              for e in envs_mj], axis=1)
               for alg, env_dict in metrics_mj.items()}

label_map   = {"KLAC+bonus+anneal": r"KLAC",
               "KLAC": r"KLAC$_{-ab}$",
               "SAC": "SAC",
               "KLAC+bonus": r"KLAC$_{-a}$"}
alg_order_mj    = ["KLAC", "KLAC+bonus", "SAC", "KLAC+bonus+anneal"]
score_mj_norm   = _normalise_to_sac(score_mj, label_map=label_map)
point_mj, ci_mj = _bootstrap(score_mj_norm)
alg_mj          = [label_map[a] for a in alg_order_mj]
colour_map_mj   = dict(zip(alg_mj, COLORS))

# --------------------------------------------------------------------------- #
#                ----------   MINATAR (row 1)   ----------                    #
# --------------------------------------------------------------------------- #
metrics_ma  = _collect_metrics(ROOT_MINA, _detect_variant_minatar)
envs_ma     = sorted(next(iter(metrics_ma.values())).keys())
score_ma    = {alg: np.stack([np.asarray(env_dict[e]['return'], dtype=np.float32)
                              for e in envs_ma], axis=1)
               for alg, env_dict in metrics_ma.items()}

label_map_ma = {"KLAC+bonus+anneal": r"KLAC",
                "KLAC": r"KLAC$_{-ab}$",
                "SAC": "SAC",
                "KLAC+bonus": r"KLAC$_{-a}$"}
score_ma_norm   = _normalise_to_sac(score_ma, label_map=label_map_ma)
point_ma, ci_ma = _bootstrap(score_ma_norm)
alg_ma          = [label_map_ma[a] for a in score_ma.keys()]
colour_map_ma   = dict(zip(alg_ma, COLORS))

# --------------------------------------------------------------------------- #
#                              Final figure                                   #
# --------------------------------------------------------------------------- #
aggregators = ["Mean", "IQM", "Median"]
two_col_w   = 6.8
row_h       = two_col_w * 0.45
fig, axes = plt.subplots(nrows=2, ncols=3,
                         figsize=(two_col_w, 2*row_h + 0.4),
                         sharey=False)
plt.subplots_adjust(hspace=0.45)

# row 0 – Mujoco
_plot_row(point_mj, ci_mj,
          algorithms=alg_mj,
          aggregators=aggregators,
          colour_map=colour_map_mj,
          axes_row=axes[0],
          row_name="Mujoco Environments",
          xtick_strategy="mujoco")

# row 1 – MinAtar
_plot_row(point_ma, ci_ma,
          algorithms=alg_ma,
          aggregators=aggregators,
          colour_map=colour_map_ma,
          axes_row=axes[1],
          row_name="MinAtar Environments",
          xtick_strategy="minatar")

fig.tight_layout()
fig.savefig(OUTDIR / "combined_mujoco_minatar.svg", format="svg",
            bbox_inches="tight")
fig.savefig(OUTDIR / "combined_mujoco_minatar.png", format="png", dpi=300,
            bbox_inches="tight")
plt.show()

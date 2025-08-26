from rliable import library as rly
from rliable import metrics
from rliable import plot_utils
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl


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
# colors = ["#008fd5", "#007D81", "#810f7c", "#6a6a6a", "#fc4f30", "#e5ae38", "#6d904f"]
colors = ["#008fd5", "#007D81", "#e5ae38", "#810f7c", "#6a6a6a", "#fc4f30", "#e5ae38", "#6d904f"]


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


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


def top10_mean(v: np.ndarray, k: int = 10) -> float:
    """Mean of the k highest values in v (works even if len(v) < k)."""
    k = min(k, v.size)                 # clamp k to available samples
    if k == 0:
        return np.nan                  # or 0.0, whichever you prefer
    # grab the k largest elements without a full sort
    top_k = np.partition(v, -k)[-k:]   # O(n) instead of O(n log n)
    return np.mean(top_k)


def collect_metrics(root: Path):
    """Return nested dict[algo][env]['return'] → list[float]."""
    metrics: Dict[str, Dict[str, Dict[str, List[np.ndarray]]]] = {}
    for run_dir in root.iterdir():
        if not run_dir.is_dir():
            continue
        env_tag = run_dir.name.split("_", 1)[0].split('-', 1)[0]
        variant = detect_variant(run_dir.name)
        pkl_file = run_dir / "episodic_return.pkl"
        if not pkl_file.exists():
            continue
        blob = load_pickle(pkl_file)
        steps_list, vals_list = blob["steps"], blob["vals"]
        metrics.setdefault(variant, {}).setdefault(env_tag, {}).setdefault("return", [])
        # last-10-episode average per run
        metrics[variant][env_tag]["return"].extend([top10_mean(np.asarray(vals, dtype=np.float32))
                                                    for vals in vals_list])
    return metrics


def normalise_to_sac(score_dict: dict, sac_key: str = "SAC", ref_func=np.mean, label_map=None) -> dict:
    if sac_key not in score_dict:
        raise KeyError(f"SAC baseline '{sac_key}' not found in score_dict")

    # 1) reference per task (shape = [tasks])
    sac_scores = score_dict[sac_key]            # (runs, tasks)
    sac_reference = ref_func(sac_scores, axis=0)   # → (tasks,)

    # avoid divide-by-zero for very poor or un-trained SAC runs
    sac_reference  = np.where(sac_reference == 0, 1e-8, sac_reference)

    # 2) create a normalised copy
    norm_dict = {}
    for algo, arr in score_dict.items():
        new_name = label_map[algo] if label_map is not None else algo
        norm_dict[new_name] = arr / sac_reference       # broadcasting (runs, tasks)/(tasks,)

    return norm_dict


def relabel_dict(score_dict: dict, label_map=None) -> dict:
    norm_dict = {}
    for algo, arr in score_dict.items():
        new_name = label_map[algo] if label_map is not None else algo
        norm_dict[new_name] = arr      # broadcasting (runs, tasks)/(tasks,)

    return norm_dict


if __name__ == "__main__":
    ROOT = Path("/hri/rawstreams/project/klac_2026-01/")
    OUTDIR = Path("../paper_plots/")
    OUTDIR.mkdir(parents=True, exist_ok=True)

    metrics_collected = collect_metrics(ROOT)

    envs = sorted(next(iter(metrics_collected.values())).keys())  # keep a fixed order
    score_dict = {}

    for algo, env_dict in metrics_collected.items():
        per_env_arrays = [np.asarray(env_dict[e]['return'], dtype=np.float32)
                          for e in envs]
        score_dict[algo] = np.stack(per_env_arrays, axis=1)

    algorithm_order = ["KLAC+no_bonus", "KLAC+bonus", "KLAC+no_bonus+anneal", "KLAC+bonus+anneal", "SAC"]
    algorithms_label_map = {"KLAC+bonus+anneal": r"KLAC", "KLAC+no_bonus": r"KLAC$_{-ab}$", "SAC": "SAC",
                            "KLAC+bonus": r"KLAC$_{-a}$", "KLAC+no_bonus+anneal": r"KLAC$_{-b}$"}
    score_dict_norm = normalise_to_sac(score_dict, sac_key="SAC", ref_func=np.mean, label_map=algorithms_label_map)
    relabeled_dict = relabel_dict(score_dict, algorithms_label_map)

    aggregate_vec = lambda x: np.array([
        metrics.aggregate_mean(x),
        metrics.aggregate_iqm(x),
        metrics.aggregate_median(x)
    ])

    # point_est, ci_bounds = rly.get_interval_estimates(
    #     score_dict,
    #     aggregate_vec,
    #     reps=5000,
    #     confidence_interval_size=0.95
    # )

    point_est, ci_bounds = rly.get_interval_estimates(
        relabeled_dict,  # << NO normalisation here
        aggregate_vec,  # [Mean, IQM, Median]
        reps=1000, confidence_interval_size=0.95
    )

    baseline = point_est['SAC']
    baseline_ci_bounds = ci_bounds["SAC"]

    for algo in point_est:
        point_est[algo] = point_est[algo] / baseline
        ci_bounds[algo] = ci_bounds[algo] / baseline

    aggregators = ["Mean", "IQM", "Median"]

    algorithms_old = list(score_dict.keys())
    algorithms = [algorithms_label_map[algo] for algo in algorithm_order]
    # palette = sns.color_palette("colorblind", len(algorithms))
    # hatches = ["", "//", "xx", "\\\\"]  # SAC, KLAC, KLAC+bonus, KLAC+all

    colour_map = dict(zip(algorithms, colors))

    fig, axes = plot_utils.plot_interval_estimates(
        point_est, ci_bounds,
        metric_names=aggregators,
        algorithms=algorithms,
        # REMOVE per-axes xlabel to avoid duplicates:
        xlabel=None,
        colors=colour_map
    )

    # Figure size
    two_col_width = 6.5
    fig.set_size_inches(two_col_width, 2.1)

    # Font sizes
    _LABEL_FS = 10
    _TICK_FS = 9
    _TITLE_FS = 10
    _LEGEND_FS = 10

    # Your tick/xlim/grid edits (unchanged)
    klac_vals = ci_bounds["KLAC"][1]
    for i, ax in enumerate(np.ravel(axes)):
        ticks = [0.9, 1.0, 1.1]
        ax.set_xticks(ticks)
        ax.set_xlim(ticks[0], ticks[-1])
        ax.grid(axis="x", linestyle=":", linewidth=1.5, color="#666666")
        ax.set_ylabel("")  # keep y-labels empty per your plot style

    # Apply font sizes once per axes
    for ax in fig.get_axes():
        # ensure no per-axes xlabels (super label will be used)
        ax.set_xlabel("")
        ax.set_title(ax.get_title(), fontsize=_TITLE_FS)
        ax.tick_params(axis="both", labelsize=_TICK_FS)

    # One figure-level x label (controls size centrally)
    # fig.supxlabel('SAC normalised Scores (±95 % bootstrap CI)',
    #               fontsize=_LABEL_FS, y=0.03)  # tweak y if needed
    axes[1].set_xlabel('SAC normalised Scores (±95 % bootstrap CI)',
                       fontsize=_LABEL_FS, labelpad=8)

    # If there is a legend and you want to control its size globally:
    # leg = fig.legends[0] if fig.legends else None
    # if leg:
    #     leg.set_title(leg.get_title().get_text(), prop={'size': _LEGEND_FS})
    #     for t in leg.get_texts():
    #         t.set_fontsize(_LEGEND_FS)

    # Layout AFTER adding supxlabel
    fig.tight_layout(rect=(0, 0.05, 1, 1))  # leave a little room for the super label

    fig.savefig(OUTDIR / 'mujoco_iqm_barplot.svg', format="svg", bbox_inches="tight")
    fig.savefig(OUTDIR / 'mujoco_iqm_barplot.png', format="png", dpi=300, bbox_inches="tight")
    plt.show()

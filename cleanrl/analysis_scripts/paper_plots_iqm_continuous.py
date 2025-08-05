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
    "font.size": 12,                # base font  ↔  ≥9 pt
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
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
colors = ["#008fd5", "#007D81", "#6a6a6a", "#810f7c", "#fc4f30", "#e5ae38", "#6d904f"]


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
        if "with_bonus"      in descriptor:
            flags.append("bonus")
        if "with_annealing"  in descriptor:
            flags.append("anneal")
        if "with_prior"      in descriptor:
            flags.append("prior")
        label = "KLAC" + (("+" + "+".join(flags)) if flags else "")
        return label

    return "UNKNOWN"


def collect_metrics(root: Path):
    metrics: Dict[str, Dict[str, Dict[str, List[List[np.ndarray]]]]] = {}
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
        metrics[variant][env_tag]["return"].extend([np.mean(vals[-10:]) for vals in vals_list])
    return metrics


def normalise_to_sac(score_dict: dict, sac_key: str = "SAC", ref_func=np.mean, label_map=None) -> dict:
    if sac_key not in score_dict:
        raise KeyError(f"SAC baseline '{sac_key}' not found in score_dict")

    # 1) reference per task (shape = [tasks])
    sac_scores     = score_dict[sac_key]            # (runs, tasks)
    sac_reference  = ref_func(sac_scores, axis=0)   # → (tasks,)

    # avoid divide-by-zero for very poor or un-trained SAC runs
    sac_reference  = np.where(sac_reference == 0, 1e-8, sac_reference)

    # 2) create a normalised copy
    norm_dict = {}
    for algo, arr in score_dict.items():
        new_name = label_map[algo] if label_map is not None else algo
        norm_dict[new_name] = arr / sac_reference       # broadcasting (runs, tasks)/(tasks,)

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

    algorithm_order = ["KLAC", "KLAC+bonus", "SAC", "KLAC+bonus+anneal"]
    algorithms_label_map = {"KLAC+bonus+anneal": r"KLAC", "KLAC": r"KLAC$_{-ab}$", "SAC": "SAC",
                            "KLAC+bonus": r"KLAC$_{-a}$"}
    score_dict_norm = normalise_to_sac(score_dict, sac_key="SAC", ref_func=np.mean, label_map=algorithms_label_map)

    aggregate_vec = lambda x: np.array([
        metrics.aggregate_mean(x),
        metrics.aggregate_iqm(x),
        metrics.aggregate_median(x)
    ])

    point_est, ci_bounds = rly.get_interval_estimates(
        score_dict_norm,
        aggregate_vec,
        reps=5000,
        confidence_interval_size=0.95
    )

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
        xlabel='SAC normalised Scores (±95 % bootstrap CI)',
        colors=colour_map
    )

    klac_vals = ci_bounds["KLAC"][1]  # KLAC = full “KLAC+bonus+anneal” after label_map
    for i, ax in enumerate(np.ravel(axes)):
        v_full = int(klac_vals[i]) + 1  # KLAC value for this metric (Mean, IQM, Median)
        ticks = [0.9, 1.0, 1.1]  # 1 ↔ mid-point ↔ KLAC
        ticks.sort()  # makes sure they’re in ascending order
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t}" for t in ticks])  # pretty printing (optional)
        ax.set_xlim(ticks[0], ticks[-1])  # tight bounds

        ax.grid(axis="x", linestyle=":", linewidth=0.4)
        ax.set_ylabel("")

    two_col_width = 6.8
    aspect = 0.45
    fig.set_size_inches(two_col_width, two_col_width * aspect)

    # fig.supxlabel('SAC normalised Scores (±95 % bootstrap CI)', y=0.04, fontsize=14, va='center')

    fig.tight_layout()
    fig.savefig(OUTDIR / 'mujoco_iqm_barplot.svg', format="svg", bbox_inches="tight")
    fig.savefig(OUTDIR / 'mujoco_iqm_barplot.png', format="png", dpi=300, bbox_inches="tight")
    plt.show()
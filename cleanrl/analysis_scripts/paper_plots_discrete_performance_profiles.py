from pathlib import Path
from typing import Dict, List

import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from rliable import library as rly
from rliable import metrics
from rliable import plot_utils
from matplotlib.scale import FuncScale

# ---------------------------------------------------------------------------
# Matplotlib / seaborn style (matches TMLR template)
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
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
colors = ["#007D81", "#6a6a6a", "#008fd5", "#810f7c", "#e5ae38", "#fc4f30", "#e5ae38", "#6d904f"]
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


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
        if "klac_no_bias" in descriptor:
            flags.append("no_bonus")
        if "annealing" in descriptor:
            flags.append("anneal")
        if "non_uniform_prior" in descriptor:
            flags.append("prior")
        return "KLAC" + (("+" + "+".join(flags)) if flags else "")

    return "UNKNOWN"


def last10_nz_mean(v):
    nz = v[v != 0]
    return np.mean(nz[-10:]) if nz.size else 0.0


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
    sac_reference = np.where(sac_reference == 0, 1e-8, sac_reference)

    # 2) create a normalised copy
    norm_dict = {}
    for algo, arr in score_dict.items():
        new_name = label_map[algo] if label_map is not None else algo
        a = arr / sac_reference
        norm_dict[new_name] = a

    return norm_dict


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ROOT   = Path("/hri/rawstreams/project/klac_2026-01/MinAtar/")
    OUTDIR = Path("../paper_plots/")
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load returns → build (runs × tasks) matrices
    # -----------------------------------------------------------------------
    metrics_collected = collect_metrics(ROOT)
    algorithms_old_order = ["KLAC+bonus", "SAC", "KLAC+no_bonus", "KLAC+bonus+anneal", "KLAC+no_bonus+anneal"]
    algorithms_label_map = {"KLAC+bonus+anneal": r"KLAC", "KLAC+no_bonus": r"KLAC$_{-ab}$", "SAC": "SAC",
                            "KLAC+bonus": r"KLAC$_{-a}$", "KLAC+no_bonus+anneal": r"KLAC$_{-b}$"}
    envs = sorted(next(iter(metrics_collected.values())).keys())  # fixed order
    score_dict = {}

    for algo, env_dict in metrics_collected.items():
        per_env_arrays = [np.asarray(env_dict[e]['return'], dtype=np.float32)
                          for e in envs]
        score_dict[algo] = np.stack(per_env_arrays, axis=1)  # (runs, tasks)

    # SAC-normalise all scores
    score_dict_norm = normalise_to_sac(score_dict, sac_key="SAC", ref_func=np.mean, label_map=algorithms_label_map)

    print(list(score_dict_norm.keys()))
    all_scores = np.concatenate([arr.flatten() for arr in score_dict_norm.values()])
    tau_max = np.ceil(all_scores.max() * 2) / 2           # round up to 0.5-step
    print("tau_max: ", tau_max)
    taus = np.linspace(0.1, tau_max, 700)                  # 0.0 … τ_max (80 bins)

    perf_prof, perf_prof_cis = rly.create_performance_profile(
        score_dict_norm,
        taus,
        reps=5_000,
        confidence_interval_size=0.95
    )
    two_col_width = 6.8
    algorithms = [algorithms_label_map[algo] for algo in algorithms_old_order]
    colour_map = dict(zip(algorithms, colors))

    fig_pp, ax_pp = plt.subplots(figsize=(two_col_width, two_col_width * 0.55))

    plot_utils.plot_performance_profiles(
        perf_prof,
        taus,
        performance_profile_cis=perf_prof_cis,
        colors=colour_map,
        xlabel=r'SAC Normalised return $\tau$ (log scale)',
        ax=ax_pp
    )

    ax_pp.set_xscale('log')
    ax_pp.set_xlim(0.4, 40)  # start at 1, end at 70
    ax_pp.set_xticks([0.4, 1, 10, 35])
    ax_pp.set_xticklabels(['0.4', '1', '10', '35'])
    ax_pp.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax_pp.tick_params(axis='x', which='minor', bottom=False)
    handles, labels = ax_pp.get_legend_handles_labels()
    label_to_handle = dict(zip(labels, handles))

    legend_order = [
        "SAC",
        r"KLAC",
        r"KLAC$_{-b}$",
        r"KLAC$_{-a}$",
        r"KLAC$_{-ab}$"
    ]

    ordered_labels = [lab for lab in legend_order if lab in label_to_handle]
    ordered_handles = [label_to_handle[lab] for lab in ordered_labels]
    ax_pp.legend(
        ordered_handles,
        ordered_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
        fontsize=12,
        ncol=len(ordered_labels),
        handlelength=1.5,
        columnspacing=0.8,
    )

    lin_boundary = 1.0  # x-value where scaling switches
    ax_pp.axvline(lin_boundary,
                  color='lightgrey',
                  linestyle='--',
                  linewidth=1.0,
                  zorder=0)

    ax_pp.set_ylabel(r'Fraction of runs with score ≥ $\tau$')
    ax_pp.grid(linestyle=':', linewidth=0.4)
    fig_pp.subplots_adjust(top=0.90)
    fig_pp.tight_layout()

    fig_pp.savefig(OUTDIR / 'minatar_perf_profile2.svg', format='svg', bbox_inches='tight')
    fig_pp.savefig(OUTDIR / 'minatar_perf_profile2.png', format='png', dpi=300, bbox_inches='tight')

    plt.show()

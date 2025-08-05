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
colors = ["#008fd5", "#6a6a6a", "#007D81", "#810f7c", "#fc4f30", "#e5ae38", "#6d904f"]
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
        if "annealing" in descriptor:
            flags.append("anneal")
        if "non_uniform_prior" in descriptor:
            flags.append("prior")
        return "KLAC" + (("+" + "+".join(flags)) if flags else "")

    return "UNKNOWN"


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
        metrics[variant][env_tag]["return"].extend([np.mean(vals[-10:]) for vals in vals_list])
    return metrics


def normalise_to_sac(score_dict: dict, sac_key: str = "SAC", ref_func=np.mean) -> dict:
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
        norm_dict[algo] = arr / sac_reference       # broadcasting (runs, tasks)/(tasks,)

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

    envs = sorted(next(iter(metrics_collected.values())).keys())  # fixed order
    score_dict = {}

    for algo, env_dict in metrics_collected.items():
        per_env_arrays = [np.asarray(env_dict[e]['return'], dtype=np.float32)
                          for e in envs]
        score_dict[algo] = np.stack(per_env_arrays, axis=1)  # (runs, tasks)

    # SAC-normalise all scores
    score_dict_norm = normalise_to_sac(score_dict, sac_key="SAC", ref_func=np.mean)

    all_scores = np.concatenate([arr.flatten() for arr in score_dict_norm.values()])
    tau_max = np.ceil(all_scores.max() * 2) / 2           # round up to 0.5-step
    taus = np.linspace(0.0, tau_max, 81)                  # 0.0 … τ_max (80 bins)

    perf_prof, perf_prof_cis = rly.create_performance_profile(
        score_dict_norm,
        taus,
        reps=5_000,
        confidence_interval_size=0.95
    )
    two_col_width = 6.8
    algorithms = list(score_dict.keys())
    colour_map = dict(zip(algorithms, colors))

    fig_pp, ax_pp = plt.subplots(figsize=(two_col_width, two_col_width * 0.55))

    plot_utils.plot_performance_profiles(
        perf_prof,
        taus,
        performance_profile_cis=perf_prof_cis,
        colors=colour_map,
        xlabel=r'SAC Normalised return threshold $\tau$',
        ax=ax_pp
    )

    handles, labels = ax_pp.get_legend_handles_labels()  # <- all lines are already labelled
    ax_pp.legend(handles, labels,
                 loc='lower center',  # anchor point of the legend box
                 bbox_to_anchor=(0.5, 1.02), # or 'center left', bbox_to_anchor=(1.02, 0.5)
                 frameon=False,
                 fontsize=8,
                 ncol=len(labels),  # spread all algorithms in one row
                 handlelength=1.5,  # shorten line segments a bit
                 columnspacing=0.8)

    ax_pp.set_ylabel(r'Fraction of runs with score ≥ $\tau$')
    ax_pp.grid(linestyle=':', linewidth=0.4)
    fig_pp.subplots_adjust(top=0.90)
    fig_pp.tight_layout()

    fig_pp.savefig(OUTDIR / 'minatar_perf_profile.svg', format='svg', bbox_inches='tight')
    fig_pp.savefig(OUTDIR / 'minatar_perf_profile.png', format='png', dpi=300, bbox_inches='tight')

    plt.show()

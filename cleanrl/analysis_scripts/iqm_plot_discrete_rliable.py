# iqm_with_rliable.py ---------------------------------------------------------
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

from rliable import library as rly          # pip install -U rliable
from rliable import metrics, plot_utils

FIGSIZE = (3.25, 2.1)        # inches, single-column in TMLR
MUJOCO_ENVS = ("Hopper", "Walker2d", "HalfCheetah", "Ant", "InvertedPendulum", "Humanoid", "Swimmer", "Reacher")
N_COLS, N_ROWS = 4, 2
MAX_STEPS = 1000000
EVAL_STEPS = np.linspace(0, MAX_STEPS, num=MAX_STEPS // 100)
SMOOTH_WINDOW = 400
colors = ["#6a6a6a", "#007D81", "#810f7c", "#008fd5", "#fc4f30", "#e5ae38", "#6d904f"]

ROOT   = Path("/hri/rawstreams/project/klac_2026-01/MinAtar/")
OUTDIR = Path("../paper_plots"); OUTDIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load runs  ➜  dict[algo] -> list(np.ndarray(T,))
# ---------------------------------------------------------------------------
runs_by_algo = collect_interpolated_returns(ROOT)

# ---------------------------------------------------------------------------
# 2. Reshape into (N_runs × N_tasks × T) tensors expected by rliable.
#    Here each MinAtar game is treated as a "task".
# ---------------------------------------------------------------------------
TASKS = ("Asterix", "Breakout", "Freeway", "Seaquest", "SpaceInvaders")
score_dict = {}          # algo -> ndarray [runs, tasks, steps]

for algo, run_arrs in runs_by_algo.items():
    # We have multiple runs for EACH task; stack and split:
    stacked = np.stack(run_arrs)            # shape (runs_total, T)
    # If every run list already groups the 5 tasks contiguously and equally,
    # reshape; else you’ll need a smarter mapping.
    runs_per_task = len(stacked) // len(TASKS)
    score_dict[algo] = stacked.reshape(
        runs_per_task, len(TASKS), -1)      # (runs, tasks, steps)

# ---------------------------------------------------------------------------
# 3. IQM sample‑efficiency curve with CIs
# ---------------------------------------------------------------------------
def iqm_over_time(scores):                  # scores shape (runs, tasks, steps)
    return np.array([metrics.aggregate_iqm(scores[..., t])
                     for t in range(scores.shape[-1])])

iqm_curves, iqm_cis = rly.get_interval_estimates(
    score_dict, iqm_over_time, reps=10_000, axis=0)      # stratified bootstrap

fig, ax = plot_utils.plot_sample_efficiency_curve(
    EVAL_STEPS, iqm_curves, iqm_cis,
    algorithms=sorted(score_dict),
    colors=dict(zip(sorted(score_dict), colors)),
    xlabel="Environment steps",
    ylabel="IQM return")
fig.set_size_inches(*FIGSIZE)
fig.tight_layout()
fig.savefig(OUTDIR / "fig5_minatar_iqm_rliable.svg")
fig.savefig(OUTDIR / "fig5_minatar_iqm_rliable.png", dpi=300)

# ---------------------------------------------------------------------------
# 4. Final‑performance IQM bar‑plot (single point per algo)
# ---------------------------------------------------------------------------
final_iqm, final_cis = rly.get_interval_estimates(
    score_dict,
    lambda x: metrics.aggregate_iqm(x[..., -1]), reps=10_000, axis=0)

fig2, _ = plot_utils.plot_interval_estimates(
    final_iqm[:, None], final_cis[:, None],
    metric_names=["Final IQM"], algorithms=sorted(score_dict),
    colors=dict(zip(sorted(score_dict), colors)))
fig2.set_size_inches(*FIGSIZE)
fig2.tight_layout()
fig2.savefig(OUTDIR / "fig6_minatar_final_iqm_rliable.svg")
fig2.savefig(OUTDIR / "fig6_minatar_final_iqm_rliable.png", dpi=300)

print("✓ rliable figures written to", OUTDIR.resolve())

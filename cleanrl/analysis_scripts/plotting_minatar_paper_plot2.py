import os
import glob
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

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

# ---- your environments ----
env_names = ["Asterix", "Breakout", "Freeway", "Seaquest", "SpaceInvaders"]

# ---- for each env, the run‐dirs to search ----
run_dirs_super_list = [
    [
        f"../runs_paper/MinAtar/{env}-v1__sac_min_atar_multi_run",
        f"../runs_paper/MinAtar/{env}-v1__soft_actor_hard_critic_avg_bias_uniform_prior_min_atar_target_kl_annealing_multi_run",
    ]
    for env in env_names
]

tag_list = ["episodic_return"]
y_labels = {"episodic_return": "Episodic Return"}

output_dir = "../graphs"
os.makedirs(output_dir, exist_ok=True)

# ---- main loop ----
for new_tag in tag_list:              # ⓐ  drop this loop if you only want one tag
    # -----------------------------------------------------------
    # 1)  Figure with "len(env_names)" columns, one per env
    # -----------------------------------------------------------
    n_cols = len(env_names)
    fig, axes = plt.subplots(
        1, n_cols,
        figsize=(6 * n_cols, 5),      #  e.g. 6×5, 12×5, 18×5 …
        sharey=False,                  #  all envs on the same y-scale
        squeeze=False                 #  always returns a 2-D array
    )
    axes = axes.flatten()             #  turn (1, n_cols) → (n_cols,)

    # -----------------------------------------------------------
    # 2)  Loop over environments  →  fill the right column
    # -----------------------------------------------------------
    for col_idx, (env_name, run_dirs) in enumerate(
        zip(env_names, run_dirs_super_list)
    ):
        ax = axes[col_idx]            # ← every column is a different env

        ax.set_title(env_name)
        ax.set_xlabel("Environment Steps")
        if col_idx == 0:              # label y-axis just once, leftmost
            ax.set_ylabel(y_labels.get(new_tag, new_tag))
        ax.grid(True)

        # ----- build labels for this env’s methods ----------------
        label_list = []
        for run_dir in run_dirs:
            base = os.path.basename(run_dir)
            raw = base.split("__", 1)[-1]
            if raw.endswith("_multi_run"):
                raw = raw[:-len("_multi_run")]
            prefix = "sac_min_atar_"
            if raw.startswith(prefix):
                raw = raw[len(prefix):]
            label_list.append(raw.replace("_", " ").title() or "Standard")

        # ----- shared constants -----------------------------------
        max_steps     = 3_000_000
        eval_steps    = np.linspace(0, max_steps, max_steps // 100)
        smooth_window = 200
        colors = ["#6a6a6a", "#007D81", "#810f7c", "#008fd5",
                  "#fc4f30", "#e5ae38", "#6d904f"]

        # ----- loop over *methods* inside this environment --------
        for m_idx, run_dir in enumerate(run_dirs):
            tf_paths = glob.glob(os.path.join(run_dir, "events.out.tfevents.*"))
            if not tf_paths:
                raise FileNotFoundError(f"No event files found in {run_dir}")

            # load all seeds
            all_vals = []
            for path in tf_paths:
                acc = EventAccumulator(path)
                acc.Reload()
                for tag in acc.Tags().get("scalars", []):
                    if tag.rsplit("/", 1)[-1] == new_tag:
                        evts = acc.Scalars(tag)
                        steps  = [e.step   for e in evts]
                        values = [e.value  for e in evts]
                        _, vals = interpolate_run(steps, values, eval_steps)
                        all_vals.append(smooth(vals, smooth_window))

            if not all_vals:      # method had no data for this tag
                continue

            mean, lo, hi = bootstrap_ci_vectorized(np.array(all_vals))
            mean, lo, hi = (smooth(v, smooth_window) for v in (mean, lo, hi))

            ax.fill_between(eval_steps, lo, hi, alpha=0.20,
                            facecolor=colors[m_idx])
            ax.plot(eval_steps, mean, linewidth=2.5,
                    color=colors[m_idx], label=label_list[m_idx])

    # -----------------------------------------------------------
    # 3)  One legend under the whole strip, then save
    # -----------------------------------------------------------
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               ncol=len(labels), fontsize=10, frameon=True)

    fig.tight_layout(rect=[0, 0.08, 1, 1])

    # ── 4)  save as SVG instead of PNG ───────────────────────────────
    fname = f"{new_tag.replace('/', '_')}_plot2.svg"  # ← .svg
    out_path = os.path.join(output_dir, fname)

    fig.savefig(out_path, dpi=300, format="svg")  # ← format
    plt.close(fig)
    print(f"[{new_tag}] → Plot saved: {out_path}")

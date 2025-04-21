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

# ---- for each env, the two run‐dirs to search ----
run_dirs_super_list = [
    [
        f"../runs/MinAtar/{env}-v1__sac_min_atar_max_alpha_multi_run",
        f"../runs/MinAtar/{env}-v1__sac_min_atar_multi_run",
    ]
    for env in env_names
]

tag_list = ["episodic_return", "qf_loss", "qf1_values", "alpha_used", "mean_policy_entropy"]
y_labels = {
    "episodic_return": "Episodic Return",
    "qf_loss": "Q-Function Loss",
    "qf1_values": "Q-Function Values",
    "alpha_used": "Alpha Value",
    "mean_policy_entropy": "Mean Policy Entropy",
}

output_dir = "../graphs"
os.makedirs(output_dir, exist_ok=True)

# ---- main loop ----
for env_name, run_dirs in zip(env_names, run_dirs_super_list):
    for new_tag in tag_list:
        label_list = ["Bounded Alpha", "Standard"]
        max_steps = 3_000_000
        eval_steps = np.linspace(0, max_steps, num=max_steps // 100)
        smooth_window = 200
        title = f"MinAtar {env_name} – {new_tag}"
        colors = ["#007D81", "#6a6a6a"]

        plt.figure()

        # find the two most‐recent event files, one per run-dir
        event_files = []
        for run_dir in run_dirs:
            tf_paths = glob.glob(os.path.join(run_dir, "events.out.tfevents*"))
            if not tf_paths:
                raise FileNotFoundError(f"No event file found in {run_dir}")
            latest = max(tf_paths, key=os.path.getmtime)
            event_files.append(latest)

        for run_num, event_file in enumerate(event_files):
            print(f"[{env_name}][{new_tag}] loading {os.path.basename(event_file)}")
            acc = EventAccumulator(event_file)
            acc.Reload()

            scalars = acc.Tags().get("scalars", [])
            all_vals = []
            for tag in scalars:
                if tag.split("/")[-1] == new_tag:
                    evts = acc.Scalars(tag)
                    steps  = [e.step for e in evts]
                    values = [e.value for e in evts]
                    _, interp_vals = interpolate_run(steps, values, eval_steps)
                    all_vals.append(smooth(interp_vals, smooth_window))

            if not all_vals:
                # no data for this tag/run
                continue

            mean, lo, hi = bootstrap_ci_vectorized(np.array(all_vals))
            mean = smooth(mean, smooth_window)
            lo   = smooth(lo,   smooth_window)
            hi   = smooth(hi,   smooth_window)

            plt.fill_between(eval_steps, lo, hi, alpha=0.2, facecolor=colors[run_num])
            plt.plot(eval_steps, mean, linewidth=2.5, color=colors[run_num],
                     label=label_list[run_num])

        plt.xlabel("Environment Steps")
        plt.ylabel(y_labels.get(new_tag, new_tag))
        plt.title(title)
        plt.legend(loc="upper left", prop={"size": 10}, facecolor="white", fancybox=True)
        plt.grid(True)

        out_path = os.path.join(output_dir, f"{env_name}_{new_tag}_plot.png")
        plt.savefig(out_path, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"→ Plot saved: {out_path}")

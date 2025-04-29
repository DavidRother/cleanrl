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
        f"../runs_requiem/{env}-v1__requiem_v4_min_atar_multi_run",
    ]
    for env in env_names
]

tag_list = ["episodic_return"]
y_labels = {"episodic_return": "Episodic Return"}

output_dir = "../graphs"
os.makedirs(output_dir, exist_ok=True)

# ---- main loop ----
for env_name, run_dirs in zip(env_names, run_dirs_super_list):
    # build labels for each method
    label_list = []
    for run_dir in run_dirs:
        base = os.path.basename(run_dir)
        raw = base.split("__", 1)[-1]
        if raw.endswith("_multi_run"):
            raw = raw[: -len("_multi_run")]
        prefix = "sac_min_atar_"
        if raw.startswith(prefix):
            raw = raw[len(prefix):]
        label = raw.replace("_", " ").title() if raw else "Standard"
        label_list.append(label)

    # shared plotting parameters
    max_steps = 3_000_000
    eval_steps = np.linspace(0, max_steps, num=max_steps // 100)
    smooth_window = 200
    colors = ["#6a6a6a", "#007D81", "#810f7c", "#008fd5", "#fc4f30", "#e5ae38", "#6d904f"]

    for new_tag in tag_list:
        plt.figure(figsize=(8, 5))
        plt.title(f"MinAtar {env_name} – {new_tag}")
        plt.xlabel("Environment Steps")
        plt.ylabel(y_labels.get(new_tag, new_tag))
        plt.grid(True)

        # loop over methods
        for method_idx, run_dir in enumerate(run_dirs):
            # find all tf event files (one per seed)
            tf_paths = glob.glob(os.path.join(run_dir, "events.out.tfevents.*"))
            if not tf_paths:
                raise FileNotFoundError(f"No event files found in {run_dir}")

            # load all accumulators
            accs = []
            for path in tf_paths:
                acc = EventAccumulator(path)
                acc.Reload()
                accs.append(acc)

            # collect all runs for this method
            all_vals = []
            for acc in accs:
                scalar_tags = acc.Tags().get("scalars", [])
                for tag in scalar_tags:
                    if tag.split("/")[-1] == new_tag:
                        evts = acc.Scalars(tag)
                        steps = [e.step for e in evts]
                        values = [e.value for e in evts]
                        _, interp_vals = interpolate_run(steps, values, eval_steps)
                        all_vals.append(smooth(interp_vals, smooth_window))

            if not all_vals:
                continue

            # bootstrap CIs across seeds
            mean, lo, hi = bootstrap_ci_vectorized(np.array(all_vals))
            mean = smooth(mean, smooth_window)
            lo = smooth(lo, smooth_window)
            hi = smooth(hi, smooth_window)

            plt.fill_between(eval_steps, lo, hi,
                             alpha=0.2, facecolor=colors[method_idx])
            plt.plot(eval_steps, mean, linewidth=2.5,
                     color=colors[method_idx], label=label_list[method_idx])

        plt.legend(loc="upper left", prop={"size": 10}, facecolor="white", fancybox=True)
        out_path = os.path.join(output_dir, f"{env_name}_{new_tag}_plot_new.png")
        plt.savefig(out_path, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"[{env_name}][{new_tag}] → Plot saved: {out_path}")

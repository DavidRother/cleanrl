import os
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def bootstrap_ci_vectorized(S, B=1000, alpha=0.05):
    """
    Compute the mean and bootstrapped confidence intervals in a vectorized way.

    Parameters:
        S (np.ndarray): Score matrix of shape (N, T) with returns.
        B (int): Number of bootstrap resamples.
        alpha (float): Significance level (default: 0.05 for 95% CI).

    Returns:
        mean (np.ndarray): Mean for each timestep.
        ci_lower (np.ndarray): Lower bound of the confidence interval.
        ci_upper (np.ndarray): Upper bound of the confidence interval.
    """
    N, T = S.shape
    mean = np.mean(S, axis=0)
    # Bootstrap resampling independently per timestep.
    indices = np.random.randint(0, N, size=(B, N, T))
    boot_samples = S[indices, np.arange(T)]
    boot_means = np.mean(boot_samples, axis=1)
    ci_lower, ci_upper = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)], axis=0)
    return mean, ci_lower, ci_upper


def smooth(x, weight):
    """
    Smooth data with a moving window average.
    """
    y = np.ones(weight)
    z = np.ones(len(x))
    smoothed = np.convolve(x, y, "same") / np.convolve(z, y, "same")
    return smoothed


def interpolate_run(steps, returns, eval_steps):
    # Create an interpolation function using previous values.
    interp_fn = interp1d(
        steps,
        returns,
        kind="previous",
        bounds_error=False,
        fill_value=(returns[0], returns[-1]),
    )
    return eval_steps, interp_fn(eval_steps)


env_name = "SpaceInvaders"
# ====== Configuration ======
# Update this path to point to your combined event file.
combined_event_file = f"../runs/MinAtar/{env_name}-v1__sac_min_atar_max_alpha_multi_run/events.out.tfevents.1744597453.DESKTOP-3KSSRPS.22956.0"
combined_event_file2 = f"../runs/MinAtar/{env_name}-v1__sac_min_atar_multi_run/events.out.tfevents.1744653112.DESKTOP-3KSSRPS.24804.0"
combined_event_file_list = [combined_event_file, combined_event_file2]

label_list = ["Bounded Alpha", "Standard"]

max_steps = 3000000
eval_steps = np.linspace(0, max_steps, num=max_steps // 100)
smooth_window = 200
title = f"MinAtar {env_name}"

# Define a list of colors that will be used for different runs.
colors = ["#6a6a6a", "#007D81", "#810f7c", "#008fd5", "#fc4f30", "#e5ae38", "#6d904f"]

# ====== Load the Combined Event File ======
for run_num, event_file in enumerate(combined_event_file_list):
    print("Loading event file:", event_file)
    event_acc = EventAccumulator(event_file)
    event_acc.Reload()

    # Get all scalar tags from the event file.
    all_scalar_tags = event_acc.Tags().get("scalars", [])

    # Group events by run prefix (assumed to be the first part of the tag, before the '/')
    all_returns = []
    for tag in all_scalar_tags:
        if "episodic_return" == tag.split('/')[-1]:
            # Extract run id from tag; for example: "seed_123456/charts/episodic_return".
            run_id = tag.split('/')[0]
            events = event_acc.Scalars(tag)
            steps = [e.step for e in events]
            returns = [e.value for e in events]
            # Interpolate the returns over a common evaluation step grid.
            interp_steps, interp_returns = interpolate_run(steps, returns, eval_steps)
            # Apply smoothing.
            interp_returns = smooth(interp_returns, smooth_window)
            all_returns.append(interp_returns)
            print(f"Loaded run {run_id} from tag: {tag}")

    # ====== Plotting ======
    mean, lower, higher = bootstrap_ci_vectorized(np.asarray(all_returns))
    mean = smooth(mean, smooth_window)
    lower = smooth(lower, smooth_window)
    higher = smooth(higher, smooth_window)

    plt.fill_between(
        interp_steps,
        lower,
        higher,
        alpha=0.2,
        facecolor=colors[run_num],
        zorder=2,
    )

    plt.plot(
        interp_steps,
        mean,
        linewidth=2.5,
        color=colors[run_num],
        zorder=2.5,
        label=label_list[run_num],
    )

plt.xlabel("Environment Steps")
plt.ylabel("Episodic Return")
plt.title(title)
plt.legend(loc="upper left", prop={"size": 10}, facecolor="white", fancybox=True)
plt.grid(True)
plt.show()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  ***DESCRIPTION***.
#
#  Copyright (C)
#  Honda Research Institute Europe GmbH
#  Carl-Legien-Str. 30
#  63073 Offenbach/Main
#  Germany
#
#  UNPUBLISHED PROPRIETARY MATERIAL.
#  ALL RIGHTS RESERVED.
#
#
import os
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def bootstrap_ci_vectorized(S, B=1000, alpha=0.05):
    """
    Compute the interquartile mean (IQM) and bootstrapped confidence intervals in a vectorized way.

    Parameters:
        S (np.ndarray): Score matrix of shape (N, T).
        B (int): Number of bootstrap resamples.
        alpha (float): Significance level (default: 0.05 for 95% CI).

    Returns:
        iqm (np.ndarray): IQM for each timestep.
        ci_lower (np.ndarray): Lower bound of the confidence interval.
        ci_upper (np.ndarray): Upper bound of the confidence interval.
    """

    N, T = S.shape
    # iqm = stats.trim_mean(S, 0.25, axis=0)  # Compute IQM for original data
    mean = np.mean(S, axis=0)  # Compute IQM for original data

    # Bootstrap resampling **independently per timestep**
    indices = np.random.randint(0, N, size=(B, N, T))  # Generate resampling indices
    boot_samples = S[indices, np.arange(T)]  # Use broadcasting to get resampled data

    # Compute IQM for all resamples in a vectorized manner
    # boot_iqms = stats.trim_mean(boot_samples, 0.25, axis=1)
    boot_iqms = np.mean(boot_samples, axis=1)

    # Compute confidence intervals using the percentile method
    ci_lower, ci_upper = np.percentile(boot_iqms, [100 * alpha / 2, 100 * (1 - alpha / 2)], axis=0)

    return mean, ci_lower, ci_upper


def smooth(x, weight):
    """
    smooth data with moving window average.
    that is,
        smoothed_y[t] = average(y[t-k], y[t-k+1], ..., y[t+k-1], y[t+k])
    where the "smooth" param is width of that window (2k+1)
    """
    y = np.ones(weight)
    z = np.ones(len(x))
    smoothed = np.convolve(x, y, "same") / np.convolve(z, y, "same")

    return smoothed


def interpolate_run(steps, returns, eval_steps):
    # Extrapolate with constant value after last episode -- or 'linear'
    interp_fn = interp1d(
        steps,
        returns,
        kind="previous",
        bounds_error=False,
        fill_value=(returns[0], returns[-1]),
    )
    return eval_steps, interp_fn(eval_steps)


# mount_prefix = "/home/drother/localdisk/mounts"
local_prefix = "../runs"
current_runs = "../cleanrl/runs"

run_folders = []

# run_folders += [("ppo", "../cleanrl/results/Ant/ppo/baseline")]
# run_folders += [("ppo+cpc", "../cleanrl/results/Ant/ppo/cpc_runs_1")]
# title = "Ant"

run_folders += [("ppo", f"{local_prefix}/MinAtar/Asterix-v1__sac_min_atar_max_alpha_multi_run/events.out.tfevents.1744396844.DESKTOP-3KSSRPS.25828.0")]
# run_folders += [("ppo+cpc 10", f"{local_prefix}/Ant/ppo/cpc_tmp10_noise_1")]
title = "MinAtar Asterix"

# run_folders += [("ppo", f"{local_prefix}/HalfCheetah/ppo/baseline")]
# run_folders += [("ppo+cpc 1", f"{local_prefix}/HalfCheetah/ppo/cpc_tmp1_noise_1")]
# run_folders += [("ppo+cpc 10", f"{local_prefix}/HalfCheetah/ppo/cpc_tmp10_noise_1")]
# run_folders += [("ppo+cpc 10 old", f"{local_prefix}/HalfCheetah/ppo/cpc_tmp10_noise_old")]
# run_folders += [("ppo+cpc 5 time", f"{local_prefix}/HalfCheetah/ppo/cpc_tmpMixed5_noise_1")]
# title = "HalfCheetah"

# run_folders += [("ppo", f"{local_prefix}/Walker/ppo/baseline")]
# run_folders += [("ppo+cpc 1", f"{local_prefix}/Walker/ppo/cpc_tmp1_noise_1")]
# run_folders += [("ppo+cpc 5", f"{local_prefix}/Walker/ppo/cpc_tmp5_noise_1")]
# run_folders += [("ppo+cpc 10", f"{local_prefix}/Walker/ppo/cpc_tmp10_noise_1")]
# title = "Walker2d"

max_steps = 3_000_000
eval_steps = np.linspace(0, max_steps, num=max_steps // 100)
smooth_window = 200

colors = ["#6a6a6a", "#007D81", "#810f7c", "#008fd5", "#fc4f30", "#e5ae38", "#6d904f"]

for run_id, run_tuple in enumerate(run_folders):
    run_label, run_folder = run_tuple
    event_files = []

    # Walk through all directories and subdirectories
    for dirpath, dirnames, filenames in os.walk(run_folder):
        # Filter files that start with 'event'
        event_like_files = [f for f in filenames if f.startswith("event")]
        if event_like_files:
            # Add the full path of the first matching file
            event_files.append(os.path.join(dirpath, event_like_files[0]))

    x_steps = None
    all_returns = []
    for event_file in event_files:
        print(event_file)
        event_acc = EventAccumulator(event_file)
        event_acc.Reload()
        # print(event_acc.Tags())

        steps = []
        returns = []
        for event in event_acc.Scalars("charts/episodic_return"):
            steps.append(event.step)
            returns.append(event.value)

        interp_steps, interp_returns = interpolate_run(steps, returns, eval_steps)
        if x_steps is None:
            x_steps = interp_steps
        interp_returns = smooth(interp_returns, smooth_window)
        all_returns.append(interp_returns)

        if 0:
            plt.plot(x_steps, interp_returns, alpha=0.6, linewidth=0.4, linestyle="--", color=colors[run_id])

    mean, lower, higher = bootstrap_ci_vectorized(np.asarray(all_returns))
    mean = smooth(mean, smooth_window)
    lower = smooth(lower, smooth_window)
    higher = smooth(higher, smooth_window)

    plt.fill_between(
        x_steps,
        lower,
        higher,
        alpha=0.2,
        facecolor=colors[run_id],
        zorder=2,
    )

    plt.plot(
        x_steps,
        mean,
        linewidth=2.5,
        color=colors[run_id],
        zorder=2.5,
        label=f"{run_label}",
    )

plt.xlabel("Environment Steps")
plt.ylabel("Episodic Return")
plt.legend(loc="upper left", prop={"size": 10}, facecolor="white", fancybox=True)
plt.title(f"{title}")
plt.grid(True)
plt.show()

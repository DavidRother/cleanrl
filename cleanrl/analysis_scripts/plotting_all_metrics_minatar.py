import os
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
    smoothed = np.convolve(x, y, "same") / np.convolve(z, y, "same")
    return smoothed


def interpolate_run(steps, returns, eval_steps):
    interp_fn = interp1d(
        steps,
        returns,
        kind="previous",
        bounds_error=False,
        fill_value=(returns[0], returns[-1]),
    )
    return eval_steps, interp_fn(eval_steps)


env_names = ["Asterix", "Breakout", "Freeway", "Seaquest", "SpaceInvaders"]
event_file_super_list = [
    [
        "../runs/MinAtar/Asterix-v1__sac_min_atar_max_alpha_multi_run/events.out.tfevents.1744396844.DESKTOP-3KSSRPS.25828.0",
        "../runs/MinAtar/Asterix-v1__sac_min_atar_multi_run/events.out.tfevents.1744718691.cmp2004-04.1574730.0",
    ],
    [
        "../runs/MinAtar/Breakout-v1__sac_min_atar_max_alpha_multi_run/events.out.tfevents.1744541464.DESKTOP-3KSSRPS.18308.0",
        "../runs/MinAtar/Breakout-v1__sac_min_atar_multi_run/events.out.tfevents.1744719793.cmp2004-05.1124806.0",
    ],
    [
        "../runs/MinAtar/Freeway-v1__sac_min_atar_max_alpha_multi_run/events.out.tfevents.1744613275.Lappi.9336.0",
        "../runs/MinAtar/Freeway-v1__sac_min_atar_multi_run/events.out.tfevents.1744718443.cmp2004-03.3938368.0",
    ],
    [
        "../runs/MinAtar/Seaquest-v1__sac_min_atar_max_alpha_multi_run/events.out.tfevents.1744650887.Lappi.7480.0",
        "../runs/MinAtar/Seaquest-v1__sac_min_atar_multi_run/events.out.tfevents.1744718154.cmp2004-02.416801.0",
    ],
    [
        "../runs/MinAtar/SpaceInvaders-v1__sac_min_atar_max_alpha_multi_run/events.out.tfevents.1744597453.DESKTOP-3KSSRPS.22956.0",
        "../runs/MinAtar/SpaceInvaders-v1__sac_min_atar_multi_run/events.out.tfevents.1744653112.DESKTOP-3KSSRPS.24804.0",
    ],
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

for env_name, combined_event_file_list in zip(env_names, event_file_super_list):
    for new_tag in tag_list:
        label_list = ["Bounded Alpha", "Standard"]
        max_steps = 3000000
        eval_steps = np.linspace(0, max_steps, num=max_steps // 100)
        smooth_window = 200
        title = f"MinAtar {env_name} - {new_tag}"

        colors = ["#007D81", "#6a6a6a"]

        plt.figure()

        for run_num, event_file in enumerate(combined_event_file_list):
            event_acc = EventAccumulator(event_file)
            event_acc.Reload()

            all_scalar_tags = event_acc.Tags().get("scalars", [])

            all_returns = []
            for tag in all_scalar_tags:
                if new_tag == tag.split("/")[-1]:
                    events = event_acc.Scalars(tag)
                    steps = [e.step for e in events]
                    returns = [e.value for e in events]
                    interp_steps, interp_returns = interpolate_run(steps, returns, eval_steps)
                    interp_returns = smooth(interp_returns, smooth_window)
                    all_returns.append(interp_returns)

            if not all_returns:
                continue

            mean, lower, higher = bootstrap_ci_vectorized(np.asarray(all_returns))
            mean = smooth(mean, smooth_window)
            lower = smooth(lower, smooth_window)
            higher = smooth(higher, smooth_window)

            plt.fill_between(interp_steps, lower, higher, alpha=0.2, facecolor=colors[run_num])
            plt.plot(interp_steps, mean, linewidth=2.5, color=colors[run_num], label=label_list[run_num])

        plt.xlabel("Environment Steps")
        plt.ylabel(y_labels.get(new_tag, new_tag))
        plt.title(title)
        plt.legend(loc="upper left", prop={"size": 10}, facecolor="white", fancybox=True)
        plt.grid(True)

        plot_filename = os.path.join(output_dir, f"{env_name}_{new_tag}_plot.png")
        plt.savefig(plot_filename, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"Plot saved as: {plot_filename}")
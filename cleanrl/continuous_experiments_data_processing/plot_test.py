import h5py
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# EDIT ME ───────────────────────────────────────────────────────────
h5_path = (
    "Ant-v5__klac_no_bonus_no_annealing_no_prior_continuous_action_multi_run__1__1751453930/"
    "rewards.h5"
)
# ------------------------------------------------------------------


def _load_step_and_value_matrices(h5f):
    """
    Returns
        step_mat  –  shape (runs, max_len)  NaN-padded
        value_mat –  shape (runs, max_len)  NaN-padded
    Falls back to building them if the extractor didn't store them.
    """
    if (
        "episodic_return_step_matrix" in h5f
        and "episodic_return_value_matrix" in h5f
    ):
        return (
            h5f["episodic_return_step_matrix"][()],
            h5f["episodic_return_value_matrix"][()],
        )

    # -------- build on the fly ------------------------------------
    step_series, val_series, max_len = [], [], 0
    for grp in h5f.values():
        if not isinstance(grp, h5py.Group):
            continue
        if "episodic_return_step" in grp and "episodic_return_value" in grp:
            s = grp["episodic_return_step"][()]
            v = grp["episodic_return_value"][()]
            step_series.append(s)
            val_series.append(v)
            max_len = max(max_len, len(s))

    if not step_series:
        raise RuntimeError("No episodic-return data found in the file.")

    def _pad(series, length, dtype):
        mat = np.full((len(series), length), np.nan, dtype=dtype)
        for i, seq in enumerate(series):
            mat[i, : len(seq)] = seq
        return mat

    return _pad(step_series, max_len, np.float32), _pad(val_series, max_len, np.float32)


# ───────────────────────────────────────────────────────────────────
with h5py.File(h5_path, "r") as h5:
    step_mat, val_mat = _load_step_and_value_matrices(h5)

# mean step for each episode-slot (NaNs ignored)
mean_step = np.nanmean(step_mat, axis=0)  # X-axis
mean_ret  = np.nanmean(val_mat,  axis=0)
std_ret   = np.nanstd(val_mat,   axis=0)

# drop columns that were entirely NaN (happen when some runs shorter)
valid = ~np.isnan(mean_step)
mean_step, mean_ret, std_ret = mean_step[valid], mean_ret[valid], std_ret[valid]

# ensure monotonic X (just in case)
order = np.argsort(mean_step)
mean_step, mean_ret, std_ret = mean_step[order], mean_ret[order], std_ret[order]

# --------- Figure 1: Episodic Return vs Training Steps -------------
plt.figure()
plt.plot(mean_step, mean_ret, label="Mean episodic return")
plt.fill_between(
    mean_step, mean_ret - std_ret, mean_ret + std_ret, alpha=0.3, label="±1 SD"
)
plt.title("Episodic Return vs Training Steps\n(mean ± 1 SD across runs)")
plt.xlabel("Training step")
plt.ylabel("Return")
plt.legend()
plt.tight_layout()

# --------- Figure 2: Reward per Step (unchanged) -------------------
# If you also logged per-step reward_step / reward_value, you can add a
# similar block using those datasets. Otherwise you can keep the simple
# step-index plot you already had.
plt.show()
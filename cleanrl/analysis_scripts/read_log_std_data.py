#!/usr/bin/env python3
"""
Load raw_log_std vectors that were dumped with LogStdDumper.

Edit the `H5_PATH` constant below to point to your *.h5* file before running.
"""

from pathlib import Path
import h5py
import numpy as np

# ───────────────────────────────────────────────────────────────────────────────
# >>>>> EDIT THIS LINE to point at your own file <<<<<
H5_PATH = Path("../continuous_experiments/runs_experiment/folder_name__seed_1__1750930601/seed_1_log_std.h5")
# ───────────────────────────────────────────────────────────────────────────────

DATASET_NAME = "log_std"   # don’t touch unless you used a different name

# ── load ───────────────────────────────────────────────────────────────────────
if not H5_PATH.is_file():
    raise FileNotFoundError(f"File not found: {H5_PATH}")

with h5py.File(H5_PATH, "r") as h5:
    if DATASET_NAME not in h5:
        raise KeyError(f"Dataset '{DATASET_NAME}' missing in {H5_PATH}")
    # read entire dataset; dtype is float32
    log_std_array: np.ndarray = h5[DATASET_NAME][...]   # shape = (T, action_dim)

log_std_list: list[list[float]] = log_std_array.tolist()

# ── confirmation ──────────────────────────────────────────────────────────────
num_steps, action_dim = log_std_array.shape
print(f"Loaded {num_steps:,} timesteps (action_dim = {action_dim}) from {H5_PATH}")

# At this point you can use `log_std_list` in any way you like:
# e.g. access step 123:  log_std_list[123]
#      compute mean std per dim:  np.mean(log_std_array, axis=0)

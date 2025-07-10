import os
import re
from collections import defaultdict

import h5py
import numpy as np
from tensorboard.backend.event_processing import event_accumulator
from tqdm import tqdm

# ---------------------------------------------------------------------------
REWARD_TAG_PATTERN = re.compile(r"/charts/(reward|episodic_return)$")
# ---------------------------------------------------------------------------


def find_event_files(logdir):
    for root, _dirs, files in os.walk(logdir):
        for fn in files:
            if fn.startswith("events.out.tfevents"):
                yield os.path.join(root, fn)


def load_scalars(ea, tag):
    evs = ea.Scalars(tag)
    if not evs:
        return np.empty(0, np.int64), np.empty(0, np.float32)
    steps  = np.fromiter((e.step  for e in evs), dtype=np.int64,   count=len(evs))
    vals   = np.fromiter((e.value for e in evs), dtype=np.float32, count=len(evs))
    return steps, vals


# ------------------------------------------------------------------
def collect_scalars(logdir, verbose=True):
    """Return runs[run][tag]['steps'|'values'] = 1-D np.array."""
    runs = defaultdict(lambda: defaultdict(lambda: {"steps": [], "values": []}))

    for ev_path in tqdm(list(find_event_files(logdir)),
                        disable=not verbose, desc="Reading event files"):
        run = os.path.basename(os.path.dirname(ev_path))
        ea  = event_accumulator.EventAccumulator(ev_path, size_guidance={"scalars": 0})
        try:
            ea.Reload()
        except Exception as exc:
            print(f"[WARNING] skip {ev_path}: {exc}")
            continue

        for tag in ea.Tags().get("scalars", []):
            if not REWARD_TAG_PATTERN.search(tag):
                continue
            tag_key          = tag.split("/")[-1]          # reward / episodic_return
            steps, vals      = load_scalars(ea, tag)
            if steps.size == 0:
                continue

            # append; we’ll merge & dedup later
            r = runs[run][tag_key]
            r["steps"].append(steps)
            r["values"].append(vals)
    return runs


def _concat_and_dedup(step_list, val_list):
    """
    Concatenate all arrays in *step_list* / *val_list*, then
    keep the *last* value for each duplicated step.
    Returns two 1-D arrays (steps, values) sorted ascending by step.
    """
    # 1) Flatten
    steps = np.concatenate(step_list)
    vals  = np.concatenate(val_list)

    # 2) Build dict{step: value}; later occurrences overwrite earlier ones
    step_val = {}
    for s, v in zip(steps, vals):
        step_val[int(s)] = float(v)          # cast to Python scalars for dict keys

    # 3) Back to sorted arrays
    uniq_steps = np.fromiter(step_val.keys(), dtype=np.int64)
    uniq_vals  = np.fromiter(step_val.values(), dtype=np.float32)
    order      = np.argsort(uniq_steps)
    return uniq_steps[order], uniq_vals[order]


# ------------------------------------------------------------------
def write_h5(runs, h5_path, verbose=True):
    os.makedirs(os.path.dirname(h5_path) or ".", exist_ok=True)

    # for matrices
    step_series, val_series = [], []
    max_len = 0
    run_names = sorted(runs)

    with h5py.File(h5_path, "w") as h5f:
        for run in run_names:
            grp = h5f.create_group(run)

            # episodic_return
            if "episodic_return" in runs[run]:
                s, v = _concat_and_dedup(runs[run]["episodic_return"]["steps"],
                                         runs[run]["episodic_return"]["values"])

                # ---- store paired compound dataset ------------------------
                dt = np.dtype([("step", "i8"), ("value", "f4")])
                grp.create_dataset("episodic_return_pairs",
                                   data=np.asarray(list(zip(s, v)), dtype=dt))

                # individual scalar datasets (optional)
                grp.create_dataset("episodic_return_step",  data=s)
                grp.create_dataset("episodic_return_value", data=v)

                # collect for matrices
                step_series.append(s)
                val_series.append(v)
                max_len = max(max_len, len(s))

            # reward (unchanged –– steps + values)
            if "reward" in runs[run]:
                s, v = _concat_and_dedup(runs[run]["reward"]["steps"],
                                         runs[run]["reward"]["values"])
                grp.create_dataset("reward_step",  data=s)
                grp.create_dataset("reward_value", data=v)

        # ---- build NaN-padded matrices of steps & values ---------------
        n = len(step_series)
        if n:
            step_mat = np.full((n, max_len), np.nan, dtype=np.float32)
            val_mat  = np.full((n, max_len), np.nan, dtype=np.float32)
            for i, (s, v) in enumerate(zip(step_series, val_series)):
                step_mat[i, :len(s)] = s
                val_mat[i,  :len(v)] = v

            h5f.create_dataset("episodic_return_step_matrix",  data=step_mat)
            h5f.create_dataset("episodic_return_value_matrix", data=val_mat)

    if verbose:
        print(f"[INFO] wrote {len(run_names)} runs ➜ {h5_path}")


if __name__ == "__main__":
    folder_names = [name for name in os.listdir("/hri/rawstreams/project/klac_2026-01/")
                    if name != "MinAtar" and os.path.isdir(os.path.join("/hri/rawstreams/project/klac_2026-01/", name))]

    for folder_name in folder_names:
        if folder_name == "MinAtar" or folder_name == "cleaned_data":
            continue

        logdir = f"/hri/rawstreams/project/klac_2026-01/{folder_name}"
        output = f"/hri/rawstreams/project/klac_2026-01/cleaned_data/{folder_name}/rewards_episodic_return.h5"

        runs = collect_scalars(logdir, verbose=True)
        write_h5(runs, output, verbose=True)

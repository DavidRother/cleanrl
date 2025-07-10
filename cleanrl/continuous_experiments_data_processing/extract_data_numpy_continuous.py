#!/usr/bin/env python3
"""
Extract `/charts/episodic_return`, `/charts/mean_policy_entropy`, and
`/losses/qf1_values` from TensorBoard event files – doing only **one**
EventAccumulator pass per file – and pickle them.

Usage:
    python extract_tb_scalars.py -f <substring>
"""
import argparse
import os
import pickle
import re

import numpy as np
from tensorboard.backend.event_processing import event_accumulator
from tqdm import tqdm


ROOT = "/hri/rawstreams/project/klac_2026-01/"

# ── regexes for the three series we need ──────────────────────────────────────
RET_PATTERN = re.compile(r"/charts/episodic_return$")
ENT_PATTERN = re.compile(r"/charts/mean_policy_entropy$")
Q_PATTERN   = re.compile(r"/losses/qf1_values$")        # adjust if you prefer qf2/min-q/…

PATTERNS = {
    "return":  RET_PATTERN,
    "entropy": ENT_PATTERN,
    "q":       Q_PATTERN,
}

# ── helpers ───────────────────────────────────────────────────────────────────
def find_event_files(root):
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            if fn.startswith("events.out.tfevents"):
                yield os.path.join(dirpath, fn)


def extract_scalars(ev_path):
    """
    Load `ev_path` once and return a dict:
        {"return":  (steps, vals) | None,
         "entropy": (steps, vals) | None,
         "q":       (steps, vals) | None}
    """
    ea = event_accumulator.EventAccumulator(ev_path, size_guidance={"scalars": 0})
    try:
        ea.Reload()
    except Exception as exc:                        # pylint: disable=broad-except
        print(f"[WARNING] skipping {ev_path}: {exc}")
        return {k: None for k in PATTERNS}

    available_tags = ea.Tags().get("scalars", [])
    out = {}
    for key, pat in PATTERNS.items():
        tag = next((t for t in available_tags if pat.search(t)), None)
        if tag is None:
            out[key] = None
            continue
        evs = ea.Scalars(tag)
        if not evs:
            out[key] = None
            continue
        steps = np.fromiter((e.step  for e in evs), dtype=np.int64,   count=len(evs))
        vals  = np.fromiter((e.value for e in evs), dtype=np.float32, count=len(evs))
        out[key] = (steps, vals)
    return out


def maybe_dump_pickle(path, runs_steps, runs_vals, label):
    if not runs_steps:          # nothing collected
        print(f"[INFO] no {label} series found – nothing written.")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"steps": runs_steps, "vals": runs_vals},
                    f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[INFO] wrote {label} → {path}")


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--folder", required=True,
                    help="Substring that must appear in the run-folder name")
    args = ap.parse_args()
    substr = args.folder.lower()

    # substr = "invertedpendulum"

    folder_names = [name for name in os.listdir(ROOT) if substr in name.lower()]

    for folder_name in folder_names:
        if folder_name in {"MinAtar", "cleaned_data"}:
            continue

        # containers for each scalar type
        buckets = {
            "return":  ([], []),
            "entropy": ([], []),
            "q":       ([], []),
        }

        ev_files = list(find_event_files(os.path.join(ROOT, folder_name)))
        for ev_path in tqdm(ev_files, desc=f"[{folder_name}] scanning"):
            series_dict = extract_scalars(ev_path)
            for key, result in series_dict.items():
                if result is None:
                    continue
                steps, vals = result
                buckets[key][0].append(steps)
                buckets[key][1].append(vals)

        run_dir = os.path.join(ROOT, folder_name)
        maybe_dump_pickle(os.path.join(run_dir, "episodic_return.pkl"),
                          *buckets["return"],  "episodic return")
        maybe_dump_pickle(os.path.join(run_dir, "entropy.pkl"),
                          *buckets["entropy"], "entropy")
        maybe_dump_pickle(os.path.join(run_dir, "q_values.pkl"),
                          *buckets["q"],       "q-values")

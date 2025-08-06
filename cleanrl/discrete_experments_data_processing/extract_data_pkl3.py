#!/usr/bin/env python3
"""
Extract scalar series from the *discrete* MinAtar SAC runs and pickle them.

Written to be drop-in compatible with the previous continuous extractor:
   - same pickle structure
   - same three output files per run directory

Usage
-----
python extract_discrete_tb_scalars.py --root <runs_experiment> [--match substr]
"""
from __future__ import annotations

import argparse
import os
import pickle
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing import event_accumulator
from tqdm import tqdm


# ── tag patterns we are interested in ────────────────────────────────────────
RET_RE = re.compile(r"/charts/episodic_length$")        # change if you prefer qf2/…

PATTERNS = {
    "episodic_length":  RET_RE,
}

OUTFILES = {
    "episodic_length":  "episodic_length.pkl",
}


# ── helpers ──────────────────────────────────────────────────────────────────
def find_event_files(root: Path):
    """Yield every TensorBoard event file under *root* (recursive)."""
    for path in root.rglob("events.out.tfevents*"):
        yield path


def collect_scalars(ev_path: Path):
    """
    Read *one* event file and return: {key: {tag: (steps, vals)}}.

    * key  – "return" | "entropy" | "q"
    * tag  – full TB tag string (includes seed_X prefix)
    """

    acc = event_accumulator.EventAccumulator(str(ev_path), size_guidance={"scalars": 0})
    print(f"[INFO] loading {ev_path} …")
    acc.Reload()

    buckets = {
        "return":  ([], []),        # (steps_list, vals_list)
        "entropy": ([], []),
        "q":       ([], []),
    }

    # --- (3) iterate over ALL scalar tags ------------------------------------
    for tag in sorted(acc.Tags().get("scalars", [])):
        # Which scalar is this?
        matched_key = None
        for key, pat in PATTERNS.items():
            if pat.search(tag):
                matched_key = key
                break
        if matched_key is None:
            continue                            # tag we don't care about

        # Which run/seed does it belong to?
        run_id = tag.split("/", 1)[0]          # 'seed_3', 'seed_17', …

        evs = acc.Scalars(tag)
        if not evs:
            continue
        steps = np.fromiter((e.step for e in evs), dtype=np.int64,
                            count=len(evs))
        vals = np.fromiter((e.value for e in evs), dtype=np.float32,
                            count=len(evs))

        buckets[matched_key][0].append(steps)
        buckets[matched_key][1].append(vals)
        print(f"[DEBUG] collected {matched_key:7s}  ← {run_id}  "
              f"({len(steps)} points)")
    return buckets


def dump_pickle(path: Path, steps_list, vals_list, label: str):
    if not steps_list:
        print(f"[INFO] no {label} series found – nothing written.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump({"steps": steps_list, "vals": vals_list},
                    f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[INFO] wrote {label:8s} → {path}")


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="Top-level folder that contains the run directories "
                         "(e.g. runs_experiment).")
    ap.add_argument("--match", default="",
                    help="Only process run folders whose *name* contains this "
                         "substring (case-insensitive).")
    args = ap.parse_args()
    root = args.root
    match = args.match
    #
    # match = "asterix"
    # root = "/hri/rawstreams/project/klac_2026-01/MinAtar/"

    root = Path(root).expanduser().resolve()

    run_dirs = [p for p in root.iterdir()
                if p.is_dir() and (match.lower() in p.name.lower())]

    for run_dir in run_dirs:

        event_files = list(find_event_files(run_dir))
        for ev_path in tqdm(event_files, desc=f"[{run_dir.name}] scanning"):
            buckets = collect_scalars(ev_path)

            # write the three pickles next to the event files
            for key, (steps, vals) in buckets.items():
                dump_pickle(run_dir / OUTFILES[key], steps, vals, key)


if __name__ == "__main__":
    main()

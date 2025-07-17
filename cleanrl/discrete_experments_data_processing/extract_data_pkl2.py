from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path

import numpy as np
import tensorflow as tf
from tqdm import tqdm

# ── tag patterns we are interested in ────────────────────────────────────
RET_RE = re.compile(r"/charts/episodic_return$")
ENT_RE = re.compile(r"/charts/mean_policy_entropy$")
Q_RE = re.compile(r"/losses/qf1_values$")  # change if you prefer qf2/…

PATTERNS = {
    "return": RET_RE,
    "entropy": ENT_RE,
    "q": Q_RE,
}

OUTFILES = {
    "return": "episodic_return.pkl",
    "entropy": "entropy.pkl",
    "q": "q_values.pkl",
}


# ── helpers ──────────────────────────────────────────────────────────────
def find_event_files(root: Path):
    yield from root.rglob("events.out.tfevents*")


def stream_scalars(ev_path: Path):
    buckets = {k: ([], []) for k in PATTERNS}  # key → (steps, vals)

    # iterate over every Event protobuf in the file
    for ev in tf.compat.v1.train.summary_iterator(str(ev_path)):
        if not ev.summary.value:  # skip non‑summary events
            continue
        step = ev.step
        for v in ev.summary.value:  # one Value protobuf per tag
            for key, pat in PATTERNS.items():
                if pat.search(v.tag):
                    buckets[key][0].append(step)
                    buckets[key][1].append(v.simple_value)
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


# ── main ────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="Top-level folder that contains the run directories "
                         "(e.g. runs_experiment).")
    ap.add_argument("--match", default="",
                    help="Only process run folders whose *name* contains this "
                         "substring (case-insensitive).")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    match = args.match.lower()

    run_dirs = [p for p in root.iterdir()
                if p.is_dir() and (match in p.name.lower())]

    if not run_dirs:
        print("[WARN] no matching run directories found.")
        return

    for run_dir in run_dirs:
        print(f"[INFO] processing run directory: {run_dir}")
        event_files = list(find_event_files(run_dir))
        if not event_files:
            print(f"[WARN] no event files in {run_dir}")
            continue

        # aggregate across *all* event files in this run directory
        agg = {k: ([], []) for k in PATTERNS}

        for ev_path in tqdm(event_files, desc=f"[{run_dir.name}] streaming"):
            buckets = stream_scalars(ev_path)
            for key, (s, v) in buckets.items():
                if s:  # avoid empty series
                    agg[key][0].append(np.asarray(s, dtype=np.int64))
                    agg[key][1].append(np.asarray(v, dtype=np.float32))

        # write one pickle per scalar key
        for key, (steps, vals) in agg.items():
            dump_pickle(run_dir / OUTFILES[key], steps, vals, key)


if __name__ == "__main__":
    main()
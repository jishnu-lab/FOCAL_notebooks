#!/usr/bin/env python
"""Cell-state fractions of each episode on the B-cell trajectories.

Writes ``episode_state_fractions_bcell.json`` (next to this script, or to the path
given as the first argument) as ``{trajectory: {episode: {state: fraction}}}``.

An episode is 5 consecutive equidistant sampled points. A single branch is sampled
at 20 points (4 episodes), the entire trajectory at 40 points (8 episodes); node 1
is the root (ActB-1), node 0 the bifurcation (ActB-3/4), node 2 PB, node 3 GC.

The per-window cell-state counts (``prop['sc']['w']`` x ``clusters.csv``) are
smoothed at the sampled points with the same dictys Gaussian kernel that builds the
episodic GRN (``dynamic_network.linspace(start, stop, num_points, dist)``), averaged
over the 5 points of each episode and normalised to sum to 1.

Usage (dictys env, ~1 GB of memory):

    python episode_state_fractions_bcell.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

import focal
from focal.io import DatasetPaths

HERE = Path(__file__).resolve().parent
# memory-lean dynamic.h5 loader (skips the ~35 GB of network arrays)
sys.path.insert(0, str(Path(focal.__file__).resolve().parents[2] / "tests" / "episodic_fix_validation"))
from slim_loader import load_slim  # noqa: E402

POINTS_PER_EPISODE = 5
DIST = 0.001
# name -> (trajectory_range, num_points)
TRAJECTORIES = {
    "ActB3/4_to_PB": ((0, 2), 20),
    "ActB3/4_to_GC": ((0, 3), 20),
    "ActB1/2_to_GC": ((1, 3), 40),
}


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    out_path = Path(argv[0]) if argv else HERE / "episode_state_fractions_bcell.json"

    config = DatasetPaths.from_yaml(str(HERE.parent / "datasets.yaml"))
    net = load_slim(config.DYNAMIC_H5, keep={"sc": ("w",), "ss": ("traj-neighbor",)}, verbose=False)
    onehot = pd.get_dummies(pd.read_csv(config.CELL_LABELS)["Cluster"]).astype(float)
    states = list(onehot.columns)
    counts = net.prop["sc"]["w"].astype(float) @ onehot.values      # windows x states

    out = {"_meta": {
        "dataset": config.DYNAMIC_H5,
        "points_per_episode": POINTS_PER_EPISODE,
        "dist": DIST,
        "method": "per-window cell-state counts smoothed at the sampled points with the dictys "
                  "Gaussian kernel used for the episodic GRN, averaged over the 5 points of each "
                  "episode, normalised to sum to 1",
        "trajectories": {name: {"trajectory_range": list(rng), "num_points": n,
                                "n_episodes": n // POINTS_PER_EPISODE}
                         for name, (rng, n) in TRAJECTORIES.items()},
    }}
    for name, (rng, num_points) in TRAJECTORIES.items():
        pts, fsmooth = net.linspace(rng[0], rng[1], num_points, DIST)
        func_name, args, kwargs = fsmooth.keywords["smoothen_func"]
        smoothed = fsmooth.keywords["pts"].smoothened(
            counts, *args, func_name=func_name, axis=0, **kwargs)(points=pts)   # points x states
        n_episodes = num_points // POINTS_PER_EPISODE
        ep = smoothed.reshape(n_episodes, POINTS_PER_EPISODE, len(states)).mean(axis=1)
        ep /= ep.sum(axis=1, keepdims=True)
        out[name] = {
            f"episode_{i + 1}": {s: round(float(v), 4) for s, v in zip(states, ep[i]) if v >= 5e-5}
            for i in range(n_episodes)
        }

    out_path.write_text(json.dumps(out, indent=1))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

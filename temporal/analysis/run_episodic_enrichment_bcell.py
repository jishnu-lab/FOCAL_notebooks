#!/usr/bin/env python
"""Episodic TF enrichment for the B-cell PB and GC branches (inputs to Fig3_2 / Fig4_2).

Rebuilds the per-episode enrichment tables that ``config.PB['ep1'..'ep4']`` and
``config.GC['ep1'..'ep4']`` in ``../datasets.yaml`` point at, with the *fixed*
``focal.temporal`` code (ISSUES #1 and #3 -- the June 2025 tables on disk were
produced before those fixes and are wrong).

Pipeline per (branch, episode), identical to ``TemporalManager.enrich_episode``:

    smooth the direct-effect network ``w`` at ``num_points`` points on the branch
    -> slice the episode's ``points_per_episode`` points
    -> t-test / direction-invariance filter (p < pval_threshold)
    -> forces = beta x regulator expression *from the episode's own window*
    -> keep |mean force| >= ``percentile``-th percentile  (the episodic GRN)
    -> hypergeometric ORA of the cellular program per TF

The expensive prefix is computed once per episode; only the last step depends on
the cellular program, so several programs can be enriched from one run.

Cellular program: the "Z11" tables of Fig3_2 are in fact the UNION of the two
GC-vs-PB state-discriminative latent factors, Z11 u Z3 with HLA- genes dropped
(57 genes) -- verified 2026-09-11: the committed Fig3_2 outputs are reproduced
exactly by the union and not by Z11 alone. The union is written to the primary
paths; Z11-only tables go to ``z11_only/`` for comparison.

An episode is always 5 equidistant sampled points. A single branch is sampled at
20 points (4 episodes); the entire trajectory at 40 points (8 episodes). Node 1 is
the root (ActB-1), node 0 the bifurcation (ActB-3/4), so the PB / GC tables use the
post-bifurcation branches PB = (0, 2), GC = (0, 3) with num_points = 20,
points_per_episode = 5, dist = 0.001, sparsity = 0.01, percentile = 98,
pval_threshold = 1e-3, network ``w``.

``--programs ko`` (inputs to Fig5) enriches the IRF4-KO (Z4) and BLIMP1-KO (Z5)
programs instead, against the entire root-to-GC range (1, 3) sampled at num_points = 40 -> 8
episodes of 5 points, writing ``config.IRF4['ep1'..'ep8']`` and
``config.BLIMP1[...]`` (``enrichment_episode_{i}.csv``). The tables those groups
pointed at before (Oct 2025) predate the ISSUE #1 / #3 fixes -- every episode after
the first was scaled by episode-1 regulator expression, and most edges by another
TF's expression -- so they are moved to ``legacy_oct2025_prefix_bug/`` next to the
new ones. Run metadata and the 8 GC episodic GRNs go to ``{inputs}/ko_gc_98_run``.

Usage (from any directory; datasets.yaml is found via DatasetPaths.find()):

    python run_episodic_enrichment_bcell.py --n-processes 32          # PB/GC, Fig3_2
    python run_episodic_enrichment_bcell.py --programs ko --n-processes 32   # Fig5
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

# gcpb: post-bifurcation branches (20 points); ko: entire root-to-GC range (40 points)
BRANCHES = {"gcpb": {"pb": (0, 2), "gc": (0, 3)}, "ko": {"gc": (1, 3)}}


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def load_lf_genes(paths: list[str], drop_hla: bool) -> list[str]:
    """Union of SLIDE feature lists (``names`` column), optionally dropping HLA- genes."""
    genes: set[str] = set()
    for path in paths:
        df = pd.read_csv(path, sep="\t", header=0)
        if drop_hla:
            df = df[~df["names"].str.contains("HLA-")]
        new = set(df["names"].tolist())
        log(f"  program file {os.path.basename(path)}: {len(new)} genes (+{len(new - genes)} new)")
        genes |= new
    return sorted(genes)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--datasets-yaml", default=str(HERE.parent / "datasets.yaml"))
    p.add_argument("--programs", choices=["gcpb", "ko"], default="gcpb",
                   help="gcpb: Z11 u Z3 on PB and GC (Fig3_2); "
                        "ko: IRF4-KO (Z4) and BLIMP1-KO (Z5) on GC at 40 points (Fig5)")
    p.add_argument("--branches", default=None,
                   help="comma-separated; default pb,gc (gcpb) or gc (ko)")
    p.add_argument("--num-points", type=int, default=None,
                   help="default 20 (gcpb) or 40 (ko)")
    p.add_argument("--points-per-episode", type=int, default=5)
    p.add_argument("--dist", type=float, default=0.001)
    p.add_argument("--sparsity", type=float, default=0.01)
    p.add_argument("--percentile", type=float, default=98.0)
    p.add_argument("--pval-threshold", type=float, default=0.001)
    p.add_argument("--network-type", default="w")
    p.add_argument("--n-processes", type=int,
                   default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    p.add_argument("--filter-chunk-size", type=int, default=8000)
    p.add_argument("--force-chunk-size", type=int, default=30000)
    p.add_argument("--no-z11-only", action="store_true",
                   help="skip the secondary Z11-only tables")
    args = p.parse_args(argv)
    if args.branches is None:
        args.branches = "pb,gc" if args.programs == "gcpb" else "gc"
    if args.num_points is None:
        args.num_points = 20 if args.programs == "gcpb" else 40

    from focal.io import DatasetPaths
    import dictys
    from focal.temporal._episodes import EpisodeDynamics

    config = DatasetPaths.from_yaml(args.datasets_yaml)
    # programs: name -> (genes, destination dir, table file name for (episode, branch))
    if args.programs == "gcpb":
        out_dir = Path(config.PB["ep1"]).parent          # .../direct_effect_enrichment
        z11_dir = out_dir / "z11_only"
        for d in (out_dir, z11_dir):
            d.mkdir(parents=True, exist_ok=True)
        fname = lambda ep, branch: f"enrichment_ep{ep}_{branch}.csv"

        # Cellular programs. Primary = Z11 u Z3 with HLA- dropped: the program behind
        # the committed Fig3_2 outputs (and the June 2025 tables, see
        # tests/episodic_fix_validation/README.md). Secondary = Z11 alone, for comparison.
        log("Cellular program (primary): Z11 u Z3, HLA- dropped")
        lf_union = load_lf_genes([config.LF_Z11_GC_PB, config.LF_Z3_GC_PB], drop_hla=True)
        programs = {"z11_union_z3": (lf_union, out_dir, fname)}
        if not args.no_z11_only:
            log("Cellular program (secondary): Z11 only")
            lf_z11 = load_lf_genes([config.LF_Z11_GC_PB], drop_hla=True)
            programs["z11_only"] = (lf_z11, z11_dir, fname)
        lf_files = {"lf_z11_file": config.LF_Z11_GC_PB, "lf_z3_file": config.LF_Z3_GC_PB}
    else:
        if args.branches != "gc":
            raise SystemExit("--programs ko enriches the GC branch only (config.IRF4 / "
                             "config.BLIMP1 are single-branch groups); got "
                             f"--branches {args.branches}")
        irf4_dir = Path(config.IRF4["ep1"]).parent       # .../irf4_ko/gc_98
        blimp1_dir = Path(config.BLIMP1["ep1"]).parent   # .../prdm1_ko/gc_98
        out_dir = irf4_dir.parents[1] / "ko_gc_98_run"   # .../intermediate_tmp_files/ko_gc_98_run
        for d in (out_dir, irf4_dir, blimp1_dir):
            d.mkdir(parents=True, exist_ok=True)
        fname = lambda ep, branch: f"enrichment_episode_{ep}.csv"
        if len(config.IRF4) != len(config.BLIMP1):
            raise SystemExit("config.IRF4 and config.BLIMP1 declare different n_episodes")
        if args.num_points // args.points_per_episode != len(config.IRF4):
            raise SystemExit(
                f"num_points={args.num_points} / points_per_episode="
                f"{args.points_per_episode} gives {args.num_points // args.points_per_episode} "
                f"episodes but datasets.yaml declares n_episodes: {len(config.IRF4)}")

        log("Cellular program: IRF4 KO (Z4), HLA- dropped")
        lf_irf4 = load_lf_genes([config.LF_Z4_IRF4_KO], drop_hla=True)
        log("Cellular program: BLIMP1 KO (Z5), HLA- dropped")
        lf_blimp1 = load_lf_genes([config.LF_Z5_PRDM1_KO], drop_hla=True)
        programs = {"irf4_ko": (lf_irf4, irf4_dir, fname),
                    "prdm1_ko": (lf_blimp1, blimp1_dir, fname)}
        lf_files = {"lf_z4_irf4_ko_file": config.LF_Z4_IRF4_KO,
                    "lf_z5_prdm1_ko_file": config.LF_Z5_PRDM1_KO}

        # Keep the pre-fix (Oct 2025) tables for provenance, out of the group's way.
        for name, (_, dest, _) in programs.items():
            legacy = dest / "legacy_oct2025_prefix_bug"
            old = sorted(dest.glob("enrichment_episode_*.csv"))
            if old and not legacy.exists():
                legacy.mkdir()
                for f in old:
                    f.rename(legacy / f.name)
                log(f"{name}: moved {len(old)} pre-fix tables to {legacy}")

    grn_dir = out_dir / "episodic_grn_edges"
    grn_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        **vars(args),
        "dynamic_h5": config.DYNAMIC_H5,
        **lf_files,
        "n_genes": {k: len(v[0]) for k, v in programs.items()},
        "destinations": {k: str(v[1]) for k, v in programs.items()},
        "branches_ranges": BRANCHES[args.programs],
        "started": datetime.now().isoformat(timespec="seconds"),
        "host": os.uname().nodename,
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, default=str))

    t0 = time.time()
    log(f"Loading {config.DYNAMIC_H5}")
    net = dictys.net.dynamic_network.from_file(config.DYNAMIC_H5)
    log(f"network loaded in {time.time() - t0:.0f}s: "
        f"{len(net.nids[0])} TFs, {len(net.ndict)} genes")

    for name, (genes, _, _) in programs.items():
        present = [g for g in genes if g in net.ndict]
        missing = sorted(set(genes) - set(present))
        log(f"program {name}: {len(present)}/{len(genes)} genes in the network"
            + (f"; missing {missing}" if missing else ""))

    n_episodes = args.num_points // args.points_per_episode
    rows = []
    for branch in [b.strip() for b in args.branches.split(",")]:
        rng = BRANCHES[args.programs][branch]
        log(f"===== {branch.upper()} trajectory_range={rng} =====")
        epi = EpisodeDynamics(
            dictys_dynamic_object=net,
            output_folder=str(out_dir),
            trajectory_range=rng,
            num_points=args.num_points,
            dist=args.dist,
            sparsity=args.sparsity,
            network_type=args.network_type,
        )
        t0 = time.time()
        lcpm, dtime = epi.compute_expression_curves()
        log(f"[{branch}] expression curves {lcpm.shape} in {time.time() - t0:.0f}s; "
            f"pseudotime {dtime.iloc[0]:.3f}..{dtime.iloc[-1]:.3f}")

        for ep in range(1, n_episodes + 1):
            sl = slice((ep - 1) * args.points_per_episode, ep * args.points_per_episode)
            tag = f"{branch}_ep{ep}"
            t0 = time.time()
            beta = epi.build_episode_grn(time_slice=sl)      # first call smooths the network
            log(f"[{tag}] episode GRN {beta.shape} (slice {sl.start}:{sl.stop}) "
                f"in {time.time() - t0:.0f}s")

            t0 = time.time()
            filtered = epi.filter_edges(
                n_processes=args.n_processes,
                chunk_size=args.filter_chunk_size,
                pval_threshold=args.pval_threshold,
            )
            log(f"[{tag}] filtered edges {filtered.shape} in {time.time() - t0:.0f}s")
            if len(filtered) == 0:
                log(f"[{tag}] no edges survived the filter; skipping")
                continue

            t0 = time.time()
            epi.compute_tf_expression()
            epi.calculate_forces(n_processes=args.n_processes, chunk_size=args.force_chunk_size)
            edges = epi.select_top_edges(args.percentile)
            log(f"[{tag}] forces + top {args.percentile:g}th-percentile edges "
                f"{edges.shape} in {time.time() - t0:.0f}s")
            edges.drop(columns=[c for c in ("is_in_lf",) if c in edges], errors="ignore") \
                 .to_parquet(grn_dir / f"episode_{ep}_{branch}.parquet")

            row = {
                "branch": branch, "episode": ep, "time_slice": f"{sl.start}:{sl.stop}",
                "pseudotime_start": float(dtime.iloc[sl.start]),
                "pseudotime_end": float(dtime.iloc[sl.stop - 1]),
                "n_edges_episode": int(len(beta)),
                "n_edges_filtered": int(len(filtered)),
                "n_edges_selected": int(len(edges)),
                "n_tfs_in_grn": int(edges.index.get_level_values(0).nunique()),
                "N_targets_in_grn": int(edges.index.get_level_values(1).nunique()),
            }
            for name, (genes, dest, table_name) in programs.items():
                epi.set_lf_genes(genes)
                epi.annotate_lf_in_grn()
                enr = epi.calculate_enrichment()
                out_path = dest / table_name(ep, branch)
                enr.to_csv(out_path, index=False)
                mask = epi.episodic_grn_edges["is_in_lf"]
                k_active = int(epi.episodic_grn_edges[mask].index.get_level_values(1).nunique())
                n_sig = int((enr["p_value"] <= 0.05).sum())
                row[f"{name}_K_lf_active"] = k_active
                row[f"{name}_n_tfs"] = int(len(enr))
                row[f"{name}_n_tfs_p05"] = n_sig
                log(f"[{tag}] {name}: {len(enr)} TFs with >=1 program target, "
                    f"{n_sig} at p<=0.05, K={k_active} program genes active -> {out_path}")
            rows.append(row)
            del beta, filtered
            gc.collect()

        del epi
        gc.collect()

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "run_summary.csv", index=False)
    log("run_summary.csv:\n" + summary.to_string())
    run_config["finished"] = datetime.now().isoformat(timespec="seconds")
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, default=str))
    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

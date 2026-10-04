"""Per-video ground-truth census: node/edge counts, divisions, annotated time window, N_est.

Answers the plan's open items: the number of GT divisions (megayak: 151; forum: ~304),
GT time coverage (one video annotated only t=0-75 of 100), and annotation density.

Usage: python gt_stats.py [--gt /workspace/data/train] [--out gt_stats.csv]
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl

from score import GT_DIR, SCALE, TWINS, embryo, load_geff, n_estimated


def _stats(path: Path) -> dict:
    g = load_geff(path)
    nodes = g.node_attrs(attr_keys=["node_id", "t", "z", "y", "x"])
    edges = g.edge_attrs(attr_keys=[])
    src = edges["source_id"].to_numpy()
    # node ids are large (they encode t), so count out-degree with unique, not bincount
    uniq, outdeg = np.unique(src, return_counts=True) if src.size else (np.zeros(0, int), np.zeros(0, int))
    div_ids = set(uniq[outdeg >= 2].tolist())
    t = nodes["t"].to_numpy()
    pos = {r[0]: np.array(r[2:]) * SCALE for r in nodes.iter_rows()}
    succ: dict[int, list[int]] = {}
    for s, d in zip(src, edges["target_id"].to_numpy()):
        succ.setdefault(int(s), []).append(int(d))
    # daughter geometry, in um
    pd_, sis = [], []
    for p in div_ids:
        ch = succ[p][:2]
        pd_ += [np.linalg.norm(pos[c] - pos[p]) for c in ch]
        sis.append(np.linalg.norm(pos[ch[0]] - pos[ch[1]]))
    return {
        "video": path.stem, "embryo": embryo(path.stem), "twin": path.stem in TWINS,
        "gt_nodes": g.num_nodes(), "gt_edges": g.num_edges(), "divisions": len(div_ids),
        "out_deg_gt2": int((outdeg > 2).sum()),
        "t_min": int(t.min()), "t_max": int(t.max()), "frames_annotated": int(np.unique(t).size),
        "n_est": n_estimated(path),
        "parent_daughter_um": pd_, "sister_um": sis,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", type=Path, default=GT_DIR)
    ap.add_argument("--out", type=Path, default=Path("gt_stats.csv"))
    args = ap.parse_args()
    paths = sorted(args.gt.glob("*.geff"))
    with ProcessPoolExecutor(8, mp_context=mp.get_context("spawn")) as ex:
        rows = list(ex.map(_stats, paths))
    pdist = np.concatenate([r.pop("parent_daughter_um") for r in rows] or [[]])
    sdist = np.concatenate([r.pop("sister_um") for r in rows] or [[]])
    df = pl.DataFrame(rows)
    df.write_csv(args.out)

    print(f"videos: {df.height}  " + "  ".join(
        f"{e}={df.filter(pl.col('embryo') == e).height}" for e in df["embryo"].unique().sort()))
    print(df.group_by("embryo").agg(
        pl.len().alias("videos"), pl.col("gt_nodes").sum(), pl.col("gt_edges").sum(),
        pl.col("divisions").sum(), (pl.col("divisions") > 0).sum().alias("videos_with_div"),
        pl.col("gt_nodes").median().alias("median_nodes"),
        (pl.col("gt_nodes") / pl.col("n_est")).median().alias("median_annot_rate"),
        pl.col("n_est").median().alias("median_n_est"),
    ).sort("embryo"))
    print(f"total GT divisions: {df['divisions'].sum()}  (out-degree > 2 nodes: {df['out_deg_gt2'].sum()})")
    print("annotated t-range (t_min, t_max) -> videos:")
    print(df.group_by("t_min", "t_max").len().sort("len", descending=True).head(10))
    for name, d in (("parent->daughter", pdist), ("sister<->sister", sdist)):
        if d.size:
            q = np.percentile(d, [50, 90, 99])
            print(f"{name} um: median {q[0]:.2f}  p90 {q[1]:.2f}  p99 {q[2]:.2f}  (n={d.size})")


if __name__ == "__main__":
    main()

"""DaughterNet (DivNet v3 recipe, target = GT daughter at its own frame) out-of-fold logits for the attach ranker (09-27).
For every node proposed as a daughter (c or c1, frame t+1) in: all test48 candidates (X1h held-out graphs) and the labelled rows of the three
training populations (all videos; the 48 held-out of x138h8b_hn = the test48 graphs), the fold model that did NOT see that video,
centre crop + 4-view flip TTA; frames loaded once per video.
Out: scratch/v3/dn_test48.parquet and scratch/v3/dn_<pop>.parquet with (video, t, node, dn_logit), t = the node's own frame."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch

sys.path.insert(0, "/workspace/biohub/analysis"); sys.path.insert(0, "/workspace/biohub/harness")
import divnet  # noqa: E402
import div_pipeline as dp  # noqa: E402
import divnet_v3 as v3  # noqa: E402
from divnet_v3_score import coords_from, POPS  # noqa: E402
from score import GT_DIR  # noqa: E402

A = Path("/workspace/biohub/analysis"); OUT = Path("/workspace/biohub/scratch/v3")
HELD = set(open("/workspace/biohub/scratch/heldout48.txt").read().split())


def nodes_of(d):
    """(video, t, node) of every proposed daughter: c and c1 live at frame t+1."""
    return pl.concat([d.select("video", (pl.col("t") + 1).alias("t"), pl.col("c").alias("node")),
                      d.select("video", (pl.col("t") + 1).alias("t"), pl.col("c1").alias("node"))]).unique()


def main():
    vids_all = sorted(p.stem for p in GT_DIR.glob("*_*.geff")); folds = dp._folds(vids_all); fold_of = {v: k for k, f in enumerate(folds) for v in f}
    models = []
    for k in range(5):
        m = divnet.DivNet(); m.load_state_dict(torch.load(A / "divpipe" / f"dnet_v3_s0_fold{k}.pt", map_location="cpu")); models.append(m.cuda().eval())
    jobs = {}   # name -> (need frame, coords)
    te = pl.read_parquet("/workspace/biohub/scratch/sel2/test48.parquet", columns=["video", "t", "c", "c1"])
    need = nodes_of(te); jobs["test48"] = (need, coords_from("/workspace/biohub/caches/stack_x138_head48_heldout", set(need["video"].unique().to_list())))
    for pop, cache in POPS.items():
        parts = []
        for f in sorted((A / f"divpipe_{pop}" / "cands_all").glob("*.parquet")):
            if pop == "x138h8b_hn" and f.stem in HELD:
                continue   # = the test48 graphs (chain32), scored under "test48"
            d = pl.read_parquet(f, columns=["t", "c", "c1", "label"]).filter(pl.col("label") >= 0)
            if d.height:
                parts.append(d.with_columns(pl.lit(f.stem).alias("video")))
        need = nodes_of(pl.concat(parts)); jobs[pop] = (need, coords_from(cache, set(need["video"].unique().to_list())))
        print(pop, "nodes", need.height, flush=True)
    rows = {j: [] for j in jobs}; vids = sorted(set().union(*[set(n["video"].unique().to_list()) for n, _ in jobs.values()])); t0 = time.time()
    for i, v in enumerate(vids):
        frames = v3.norm_frames(v); m = models[fold_of[v]]
        for j, (need, coords) in jobs.items():
            if v not in coords:
                continue
            nd = need.filter(pl.col("video") == v).select("t", "node").sort("t")
            for (t,), g in nd.group_by("t", maintain_order=True):
                ns = g["node"].to_numpy(); X = v3.crops(frames, int(t), coords[v][ns])
                lg = v3.predict(m, X, np.arange(len(ns)), "cuda", bs=256)
                rows[j].extend((v, int(t), int(n), float(l)) for n, l in zip(ns.tolist(), lg.tolist()))
        if i % 10 == 0:
            print(f"  [{i + 1}/{len(vids)}] {v} {time.time() - t0:.0f}s", flush=True)
    for j in jobs:
        pl.DataFrame(rows[j], schema=["video", "t", "node", "dn_logit"], orient="row").write_parquet(OUT / f"dn_{j}.parquet"); print(j, "done", len(rows[j]), flush=True)


if __name__ == "__main__":
    main()

"""All-data division-attach ranker for deployment (same protocol as div_pipeline.stage3all: GEOM + extra feature(s), 300 rounds, PARAMS; fixed seeds).
Usage: python ranker_train_all.py --extra divnet_hardneg_logit --channel orphan --out /workspace/biohub/subs/weights_ds/ranker_all_G_hn_orphan
Feature source per video: ROOT/cands_all/<v>.parquet joined with FEAT_DIRS[extra] (OOF honest logits), label >= 0 rows only."""
from __future__ import annotations
import argparse, json, sys
import numpy as np, polars as pl, lightgbm as lgb
sys.path.insert(0, "/workspace/biohub/analysis"); sys.path.insert(0, "/workspace/biohub/harness")
import div_pipeline as dp
ap = argparse.ArgumentParser(); ap.add_argument("--extra", default="divnet_hardneg_logit"); ap.add_argument("--channel", default="orphan"); ap.add_argument("--out", required=True); ap.add_argument("--exclude", default="", help="comma list or @file of videos left out of training (honest evaluation on them)"); a = ap.parse_args()
extra_cols = [c for c in a.extra.split(",") if c]; feats = dp.GEOM + [c for c in extra_cols if c != "lk_pc"] + (dp.LK_COLS if "lk_pc" in extra_cols else [])
excl = set(open(a.exclude[1:]).read().split()) if a.exclude.startswith("@") else set(a.exclude.split(",")) - {""}
files = [f for f in sorted((dp.ROOT / "cands_all").glob("*.parquet")) if f.stem not in excl]; parts = []; print("excluded videos:", len(excl), "| training videos:", len(files), flush=True)
for f in files:
    d = pl.read_parquet(f)
    for c in extra_cols:
        sub, keys = dp.FEAT_DIRS[c]; ff = dp.ROOT / sub / f"{f.stem}.parquet"
        if ff.exists():
            d = d.join(pl.read_parquet(ff), on=keys, how="left")
    if a.channel == "orphan":
        d = d.filter(pl.col("c_orphan") == 1)
    elif a.channel == "rewire":
        d = d.filter(pl.col("c_orphan") == 0)
    parts.append(d.filter(pl.col("label") >= 0))
ev = pl.concat(parts, how="diagonal_relaxed").sort("video", "t", "p")   # a video without evident rows may lack an extra-feature file
print("rows", ev.height, "pos", int(ev["label"].sum()), "nulls", int(ev.select(feats).null_count().sum_horizontal().item()), flush=True)
params = dict(dp.PARAMS); params["seed"] = 0; params["deterministic"] = True; params["bagging_seed"] = 0; params["feature_fraction_seed"] = 0
m = lgb.train(params, lgb.Dataset(ev.select(feats).to_numpy().astype(np.float32), ev["label"].to_numpy()), 300)
m.save_model(a.out + ".txt"); json.dump(feats, open(a.out + ".feats.json", "w")); print("saved", len(feats), "feats ->", a.out + ".txt")

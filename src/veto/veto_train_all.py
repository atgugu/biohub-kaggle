"""All-data fork-veto classifier for deployment (same params/rounds as the OOF validation in fork_veto.py, fixed seeds).
Usage: python veto_train_all.py --forks divpipe_gf/forks_scored_det_ens3b.parquet --out /workspace/biohub/subs/weights_ds/veto_ens3_all
Trains on every fork with label >= 0 using the honest held-out feature column 'divnet_hn' (as R4's veto_all.txt was)."""
from __future__ import annotations
import argparse, json, sys
import numpy as np, polars as pl, lightgbm as lgb
sys.path.insert(0, "/workspace/biohub/analysis")
import div_pipeline as dp
from fork_veto import FEATS

ap = argparse.ArgumentParser(); ap.add_argument("--forks", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
d = pl.read_parquet(a.forks); ev = d.filter(pl.col("label") >= 0).sort("video", "t", "p")
params = dict(dp.PARAMS); params["scale_pos_weight"] = 1.0; params["seed"] = 0; params["deterministic"] = True; params["bagging_seed"] = 0; params["feature_fraction_seed"] = 0
m = lgb.train(params, lgb.Dataset(ev.select(FEATS).to_numpy().astype(np.float32), ev["label"].to_numpy()), 200)
m.save_model(a.out + ".txt"); json.dump(FEATS, open(a.out + ".feats.json", "w"))
p = m.predict(d.select(FEATS).to_numpy().astype(np.float32))
print(f"trained on {ev.height} forks (TP {int((ev['label'] == 1).sum())}); in-sample vetoes at 0.05/0.07: {int((p < 0.05).sum())}/{int((p < 0.07).sum())} of {d.height}; saved {a.out}.txt")

"""09-28: TripletNet-aware fork veto on the big test (X3h config forks: wide + 3-seed TripletNet blend @0.8). Adds to the fork table:
p_model = veto_all(FEATS) (the deployed veto), tri_fork (3-seed TripletNet, OOF), and p_vt<g> = sigmoid(logit(p_model) + g * tri_fork).
Writes scratch/union/forks_vtri_<br>.parquet for veto_joint.py --col."""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, "/workspace/biohub/analysis")
from fork_veto import FEATS  # noqa: E402

U = Path("/workspace/biohub/scratch/union"); V = Path("/workspace/biohub/scratch/v3")
m = lgb.Booster(model_file="/workspace/biohub/subs/weights_ds/veto_all.txt"); vf = json.load(open("/workspace/biohub/subs/weights_ds/veto_all.feats.json"))
for br in ("oo", "pk"):
    d = pl.read_parquet(U / ("forks_bigk10_v3doo_avg_0.5.parquet" if br == "oo" else "forks_big_v3dpk_avg_0.5.parquet")).join(pl.read_parquet(V / f"trifork_x3e_{br}.parquet"), on=["video", "p", "c1", "c2"], how="left")
    feats = [("divnet_hn_mean" if f == "divnet_hn" else f) for f in vf]   # deployment feature = 5-fold mean (as veto_joint --model)
    pm = np.clip(m.predict(d.select(feats).to_numpy().astype(np.float32)), 1e-6, 1 - 1e-6); tf = d["tri_fork"].fill_null(0.0).to_numpy()
    d = d.with_columns(pl.Series("p_model", pm), *[pl.Series(f"p_vt{str(g).replace('.', '')}", 1 / (1 + np.exp(-(np.log(pm / (1 - pm)) + g * tf)))) for g in (0.5, 1.0)])
    ev = (d["label"] >= 0).to_numpy()
    from sklearn.metrics import roc_auc_score
    y = d["label"].to_numpy()[ev]
    print(br, "forks", d.height, "labelled", int(ev.sum()), "| AUC TP-vs-FP: veto_all %.3f | tri %.3f | vt05 %.3f | vt10 %.3f" % (
        roc_auc_score(y, pm[ev]), roc_auc_score(y, tf[ev]), roc_auc_score(y, d["p_vt05"].to_numpy()[ev]), roc_auc_score(y, d["p_vt10"].to_numpy()[ev])), flush=True)
    d.write_parquet(U / f"forks_vtri_x3e_{br}.parquet")

"""Official joint test from a saved forks parquet (fork_veto.py output), any score column, optional attach first.
Usage: python veto_joint.py --parquet divpipe_gf/forks_scored_det_ens3b.parquet --col p_tp_mean [--attach cands.parquet --k 3 --athr 0] [--thr 0.05,0.07,0.1]"""
from __future__ import annotations
import argparse, sys, warnings
import polars as pl
sys.path.insert(0, "/workspace/biohub/analysis"); sys.path.insert(0, "/workspace/biohub/harness"); warnings.filterwarnings("ignore")
import div_pipeline as dp
from fork_veto import apply_attach, joint_test, print_agg
ap = argparse.ArgumentParser(); ap.add_argument("--parquet", required=True); ap.add_argument("--col", default="p_tp_mean"); ap.add_argument("--attach", default=None)
ap.add_argument("--k", type=int, default=3); ap.add_argument("--athr", type=float, default=0.0); ap.add_argument("--thr", default="0.03,0.05,0.07,0.1")
ap.add_argument("--model", default=None, help="LightGBM veto model: score the forks with it (deployment feature = divnet_hn_mean) into column p_model"); ap.add_argument("--per-video", default=None, help="write per-video counts (setting, video, etp..nodes) to this CSV"); a = ap.parse_args()
subs = dict(sorted(dp.submissions(), key=lambda kv: kv[0]))
if a.attach:
    subs, n = apply_attach(subs, pl.read_parquet(a.attach), a.k, a.athr); print(f"attach: {n} edges added", flush=True)
d = pl.read_parquet(a.parquet)
if a.model:
    import json, lightgbm as lgb, numpy as np
    from fork_veto import FEATS
    m = lgb.Booster(model_file=a.model); X = d.select([f if f != "divnet_hn" else "divnet_hn_mean" for f in FEATS]).to_numpy().astype(np.float32)
    d = d.with_columns(pl.Series("p_model", m.predict(X))); a.col = "p_model"
    ev = d.filter(pl.col("label") >= 0)
    for th in [float(x) for x in a.thr.split(",")]:
        print(f"  model {a.model.split('/')[-1]} thr {th}: vetoes {int((d['p_model'] < th).sum())}/{d.height} forks; TP forks vetoed {int(((d['p_model'] < th) & (d['label'] == 1)).sum())}/{int((d['label'] == 1).sum())}, FP vetoed {int(((d['p_model'] < th) & (d['label'] == 0)).sum())}/{int((d['label'] == 0).sum())}")
print(f"{a.parquet} col={a.col}: {d.height} forks", flush=True)
print_agg(joint_test(subs, d, [float(x) for x in a.thr.split(",")], a.col), a.per_video)

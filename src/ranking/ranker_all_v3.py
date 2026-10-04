"""09-27: all-data pooled attach rankers with the DivNet v3 feature, for deployment. Same protocol as v3rank.py (3 populations:
divpipe_x138h8b_hn, divpipe_x138, divpipe_gf; labelled rows; PARAMS; 300 rounds) but on ALL videos incl. the 48 held-out (their v3 logits are
out-of-fold too: v3_train_<pop>_held.parquet). Seeds 0-2 -> weights_ds/ranker_all_u3<arm>_s<seed>.txt + one shared .feats.json.
Usage: python ranker_all_v3.py --arm v3|v3b"""
import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, "/workspace/biohub/analysis")
import div_pipeline as dp  # noqa: E402

A = Path("/workspace/biohub/analysis"); V = Path("/workspace/biohub/scratch/v3"); W = Path("/workspace/biohub/subs/weights_ds")
ap = argparse.ArgumentParser(); ap.add_argument("--arm", required=True, choices=["v3", "v3b", "v3d", "v3ed", "v3d01"]); a = ap.parse_args()
F = dp.GEOM + {"v3": ["divnet_v3_logit"], "v3b": ["divnet_hardneg_logit", "divnet_v3_logit"], "v3d": ["divnet_v3_logit", "dn_c", "dn_c1"], "v3ed": ["divnet_v3_logit", "dn_c", "dn_c1"], "v3d01": ["divnet_v3_logit", "dn_c", "dn_c1"]}[a.arm] + dp.LK_COLS


def load_pop(pop):
    root = A / f"divpipe_{pop}"
    if a.arm == "v3ed":   # 3-seed ensemble logits (v3e_train_<pop> covers all videos except x138h8b's 48 = v3e_test48)
        v3tr = pl.concat([pl.read_parquet(V / f"v3e_train_{pop}.parquet")] + ([pl.read_parquet(V / "v3e_test48.parquet")] if pop == "x138h8b_hn" else [])).unique(subset=["video", "t", "p"])
    else:
        held = V / ("v3_test48.parquet" if pop == "x138h8b_hn" else f"v3_train_{pop}_held.parquet")   # x138h8b's 48 held-out cands = the test48 graphs (chain32)
        v3tr = pl.concat([pl.read_parquet(V / f"v3_train_{pop}.parquet"), pl.read_parquet(held)]).unique(subset=["video", "t", "p"])
    ps = []
    for f in sorted((root / "cands_all").glob("*.parquet")):
        v = f.stem; d = pl.read_parquet(f).filter(pl.col("label") >= 0)
        if d.height == 0:
            continue
        if "video" not in d.columns:
            d = d.with_columns(pl.lit(v).alias("video"))
        for sub, keys in (("divnet_hardneg_all", ["video", "p"]), ("linker_feats", ["video", "t", "p", "c1", "c"])):
            ff = root / sub / f"{v}.parquet"
            if ff.exists():
                d = d.join(pl.read_parquet(ff), on=keys, how="left")
        ps.append(d)
    d = pl.concat(ps, how="diagonal_relaxed").join(v3tr, on=["video", "t", "p"], how="left")
    if a.arm in ("v3d", "v3ed", "v3d01"):
        tg = "dns01" if a.arm == "v3d01" else "dn"   # v3d01: DaughterNet seeds 0+1 averaged (OOF)
        dn = pl.concat([pl.read_parquet(V / f"{tg}_{pop}.parquet")] + ([pl.read_parquet(V / f"{tg}_test48.parquet")] if pop == "x138h8b_hn" else [])).unique(subset=["video", "t", "node"])
        d = d.with_columns((pl.col("t") + 1).alias("_t1"))
        for col, node in (("dn_c", "c"), ("dn_c1", "c1")):
            d = d.join(dn.rename({"t": "_t1", "node": node, "dn_logit": col}), on=["video", "_t1", node], how="left")
        d = d.drop("_t1"); print(pop, "dn missing", d["dn_c"].null_count(), d["dn_c1"].null_count(), flush=True)
    print(pop, "videos", d["video"].n_unique(), "rows", d.height, "pos", int((d["label"] == 1).sum()), "| v3 missing", d["divnet_v3_logit"].null_count(), flush=True)
    return d


tr = pl.concat([load_pop(p) for p in ("x138h8b_hn", "x138", "gf")], how="diagonal_relaxed")
X = tr.select(F).to_numpy().astype(np.float32); y = tr["label"].to_numpy()
for seed in (0, 1, 2):
    p = dict(dp.PARAMS); p.update(seed=seed, deterministic=True, bagging_seed=seed, feature_fraction_seed=seed)
    m = lgb.train(p, lgb.Dataset(X, y), 300); out = W / f"ranker_all_u3{a.arm}_s{seed}"
    m.save_model(str(out) + ".txt"); print("saved", out, flush=True)
json.dump(F, open(W / f"ranker_all_u3{a.arm}_s0.feats.json", "w")); print("feats", len(F), F[-8:], flush=True)

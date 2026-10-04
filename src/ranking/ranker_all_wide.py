"""09-28: all-data WIDE ranker ensemble for deployment (the members validated on the big test, bigtest_ens.py): on the three populations'
labelled rows with out-of-fold v3 + DaughterNet features (same data as ranker_all_v3.py --arm v3d):
  base_s0..s2 = ranker_all_u3v3d_s0..s2 (already trained)      deep, shallow, nolk (no linker features), pop_<p> (one population each)
  logit = standardized logistic regression (C 0.1), exported as JSON {feats, mean, scale, coef, intercept} (NaN -> 0 as in training).
Each LightGBM member gets its own <name>.feats.json (nolk uses GEOM + v3 + dn only)."""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.argv = [sys.argv[0], "--arm", "v3d"]
src = open("/workspace/biohub/scratch/v3/ranker_all_v3.py").read().split("\ntr = pl.concat(")[0]
exec(src)   # imports, F (v3d features), load_pop()
W = Path("/workspace/biohub/subs/weights_ds")
parts = [load_pop(p).with_columns(pl.lit(p).alias("_pop")) for p in ("x138h8b_hn", "x138", "gf")]
tr = pl.concat(parts, how="diagonal_relaxed"); y = tr["label"].to_numpy()
FN = dp.GEOM + ["divnet_v3_logit", "dn_c", "dn_c1"]


def lgbp(seed, **kw):
    p = dict(dp.PARAMS); p.update(seed=seed, deterministic=True, bagging_seed=seed, feature_fraction_seed=seed); p.update(kw); return p


def save(name, m, feats):
    m.save_model(str(W / f"ranker_all_w_{name}.txt")); json.dump(feats, open(W / f"ranker_all_w_{name}.feats.json", "w")); print("saved", name, flush=True)


save("deep", lgb.train(lgbp(0, num_leaves=16, learning_rate=0.03, feature_fraction=0.6, min_child_samples=20), lgb.Dataset(tr.select(F).to_numpy().astype(np.float32), y), 500), F)
save("shallow", lgb.train(lgbp(0, num_leaves=4, learning_rate=0.1), lgb.Dataset(tr.select(F).to_numpy().astype(np.float32), y), 200), F)
save("nolk", lgb.train(lgbp(0), lgb.Dataset(tr.select(FN).to_numpy().astype(np.float32), y), 300), FN)
for p in ("x138h8b_hn", "x138", "gf"):
    t = tr.filter(pl.col("_pop") == p)
    save(f"pop_{p}", lgb.train(lgbp(0), lgb.Dataset(t.select(F).to_numpy().astype(np.float32), t["label"].to_numpy()), 300), F)
X = np.nan_to_num(tr.select(F).to_numpy().astype(np.float64)); sc = StandardScaler().fit(X)
lr = LogisticRegression(C=0.1, max_iter=2000).fit(sc.transform(X), y)
json.dump({"feats": F, "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(), "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0])},
          open(W / "ranker_all_w_logit.json", "w")); print("saved logit", flush=True)
for s in range(3):   # base members: sidecar feats per file for the per-member loader
    json.dump(F, open(W / f"ranker_all_u3v3d_s{s}.feats.json", "w"))

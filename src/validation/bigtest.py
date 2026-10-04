"""09-27 21:30 UTC: 4x larger honest attach test, step 2. All 195 videos of population divpipe_x138h8b_hn (X3's graphs: 147 head8_all +
48 held-out), ALL candidate rows. Rankers 5-fold by video (dp._folds over the 199 GT videos): for fold f, train on the labelled rows of the three
populations excluding fold-f videos, predict fold-f rows; seeds 0-2 averaged (the deployed form). Arms: u3 (GEOM + v2 hardneg DivNet + LK = X3c),
v3 (GEOM + v3 + LK = X3d), v3d (GEOM + v3 + dn_c + dn_c1 + LK = X3e). Brackets: pk (pilkwang linker features) and oo (fold-A linker for 6bba,
fold-B for 44b6 — both never saw that embryo). All image features are out-of-fold. Writes scratch/union/cands195_<arm><br>_avg.parquet."""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, "/workspace/biohub/analysis"); sys.path.insert(0, "/workspace/biohub/harness")
import div_pipeline as dp  # noqa: E402
import linker_feats as lf  # noqa: E402

A = Path("/workspace/biohub/analysis"); V = Path("/workspace/biohub/scratch/v3"); U = Path("/workspace/biohub/scratch/union")
R = A / "divpipe_x138h8b_hn"; HELD = set(open("/workspace/biohub/scratch/heldout48.txt").read().split())
KEYS = ["video", "t", "p", "c1", "c", "q", "label"]
ARMS = {"u3": dp.GEOM + ["divnet_hardneg_logit"] + dp.LK_COLS, "v3": dp.GEOM + ["divnet_v3_logit"] + dp.LK_COLS,
        "v3d": dp.GEOM + ["divnet_v3_logit", "dn_c", "dn_c1"] + dp.LK_COLS}


def add_dn(d, dn):
    d = d.with_columns((pl.col("t") + 1).alias("_t1"))
    for col, node in (("dn_c", "c"), ("dn_c1", "c1")):
        d = d.join(dn.rename({"t": "_t1", "node": node, "dn_logit": col}), on=["video", "_t1", node], how="left")
    return d.drop("_t1")


# ---- test table: all rows of the 195 videos
V3 = pl.concat([pl.read_parquet(V / "allrows_v3_x138h8b.parquet"), pl.read_parquet(V / "v3_test48.parquet")]).unique(subset=["video", "t", "p"])
DN = pl.concat([pl.read_parquet(V / "allrows_dn_x138h8b.parquet"), pl.read_parquet(V / "dn_test48.parquet")]).unique(subset=["video", "t", "node"])
parts = []
for f in sorted((R / "cands_all").glob("*.parquet")):
    v = f.stem; d = pl.read_parquet(f)
    if "video" not in d.columns:
        d = d.with_columns(pl.lit(v).alias("video"))
    for sub, keys in (("divnet_hardneg_all", ["video", "p"]), ("linker_feats", ["video", "t", "p", "c1", "c"])):
        ff = R / sub / f"{v}.parquet"
        if ff.exists():
            d = d.join(pl.read_parquet(ff), on=keys, how="left")
    parts.append(d.select([c for c in KEYS + dp.GEOM + ["divnet_hardneg_logit"] + dp.LK_COLS if c in d.columns]))
te = pl.concat(parts, how="diagonal_relaxed").join(V3, on=["video", "t", "p"], how="left"); te = add_dn(te, DN)
print("test rows", te.height, "videos", te["video"].n_unique(), "pos", int((te["label"] == 1).sum()),
      "| missing v3", te["divnet_v3_logit"].null_count(), "dn", te["dn_c"].null_count(), "hn", te["divnet_hardneg_logit"].null_count(), "lk", te["lk_pc"].null_count(), flush=True)

# ---- out-of-sample linker features (cached)
oo_path = V / "lk_oo_195.parquet"
if not oo_path.exists():
    import os
    dp.CACHE = Path("/workspace/biohub/scratch/v3/cache195")   # dp reads BIOHUB_CACHE at import time: set the module global
    subs = {v: g for v, g in dp.submissions()}; outs = []
    for emb, cache in (("6bba", "/workspace/biohub/caches/foldA_run1_e025"), ("44b6", "/workspace/biohub/caches/foldB_e011")):
        lf.LK = Path(cache)
        for v in sorted(x for x in te["video"].unique().to_list() if x.startswith(emb)):
            if not (lf.LK / f"{v}.npz").exists() or v not in subs:
                print("no oo linker for", v, flush=True); continue
            outs.append(lf.video_feats(v, subs[v], te.filter(pl.col("video") == v).select("t", "p", "c1", "c", "q")))
    pl.concat(outs).unique(subset=["video", "t", "p", "c1", "c"]).write_parquet(oo_path)
oo = pl.read_parquet(oo_path).rename({c: c + "_oo" for c in dp.LK_COLS})
te_oo = te.join(oo, on=["video", "t", "p", "c1", "c"], how="left").with_columns([pl.col(c + "_oo").alias(c) for c in dp.LK_COLS])
print("oo coverage", float(te_oo["lk_matched"].mean()), "pk", float(te["lk_matched"].mean()), flush=True)


# ---- training rows (labelled rows of the three populations, all videos; out-of-fold image features)
def load_pop(pop):
    root = A / f"divpipe_{pop}"
    v3tr = pl.concat([pl.read_parquet(V / f"v3_train_{pop}.parquet")] + [pl.read_parquet(V / ("v3_test48.parquet" if pop == "x138h8b_hn" else f"v3_train_{pop}_held.parquet"))]).unique(subset=["video", "t", "p"])
    dn = pl.concat([pl.read_parquet(V / f"dn_{pop}.parquet")] + ([pl.read_parquet(V / "dn_test48.parquet")] if pop == "x138h8b_hn" else [])).unique(subset=["video", "t", "node"])
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
    d = add_dn(pl.concat(ps, how="diagonal_relaxed").join(v3tr, on=["video", "t", "p"], how="left"), dn)
    print(pop, "train rows", d.height, "pos", int((d["label"] == 1).sum()), flush=True)
    return d


tr = pl.concat([load_pop(p) for p in ("x138h8b_hn", "x138", "gf")], how="diagonal_relaxed")
from score import GT_DIR  # noqa: E402
folds = dp._folds(sorted(p.stem for p in GT_DIR.glob("*_*.geff")))
for arm, F in ARMS.items():
    s2 = {"pk": np.zeros(te.height), "oo": np.zeros(te.height)}
    for kf, fold in enumerate(folds):
        fset = set(fold); mtr = ~tr["video"].is_in(list(fset)); mte = te["video"].is_in(list(fset)).to_numpy()
        if not mte.any():
            continue
        Xtr = tr.filter(mtr).select(F).to_numpy().astype(np.float32); ytr = tr.filter(mtr)["label"].to_numpy()
        for seed in (0, 1, 2):
            p = dict(dp.PARAMS); p.update(seed=seed, deterministic=True, bagging_seed=seed, feature_fraction_seed=seed)
            m = lgb.train(p, lgb.Dataset(Xtr, ytr), 300)
            for br, table in (("pk", te), ("oo", te_oo)):
                s2[br][mte] += m.predict(table.filter(pl.Series(mte)).select(F).to_numpy().astype(np.float32)) / 3
        print(f"{arm} fold {kf} done", flush=True)
    for br in ("pk", "oo"):
        te.select(KEYS).with_columns(pl.Series("s2", s2[br])).write_parquet(U / f"cands195_{arm}{br}_avg.parquet")
    print(arm, "written", flush=True)

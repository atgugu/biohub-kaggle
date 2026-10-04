"""DivNet v2 arms (GPU), same honest 5-fold-by-video protocol as divnet_oof.py:
  --arm hardneg : + hard negatives = parents of the ranker's OOF false-positive candidates (label 0, s2 >= 0.3)
  --arm crop48  : larger context, crop 24 x 48 x 48 (pooled xy) instead of 16 x 32 x 32
Outputs: divpipe/divnet_<arm>_fold{k}.pt, divpipe/divnet_<arm>_all/<video>.parquet (video, p, divnet_<arm>_logit)
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import polars as pl
import torch

sys.path.insert(0, "/workspace/biohub/analysis"); sys.path.insert(0, "/workspace/biohub/harness")
import divnet  # noqa: E402
import divnet_fast  # noqa: E402
import div_pipeline as dp  # noqa: E402
import divnet_oof as base  # noqa: E402
from score import GT_DIR  # noqa: E402


def set_crop(cz, cyx):
    divnet.CZ, divnet.CYX = cz, cyx; divnet.MARK = divnet._marker()
    divnet_fast.CZ, divnet_fast.CYX = cz, cyx


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--arm", required=True, choices=["hardneg", "crop48", "hardneg2", "hardneg_s1", "hardneg_s2", "hardneg_s3", "hardneg_s4", "hardneg3"]); ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
    t0 = time.time(); device = torch.device("cuda"); arm = a.arm
    if arm.startswith("hardneg_s"):
        seed = int(arm[-1]); base.rng = np.random.default_rng(seed); torch.manual_seed(seed)
    if arm == "hardneg3":   # fixed miner: exclusion by GT geometry, not by the truncated scored pool; seed via --seed
        base.rng = np.random.default_rng(a.seed); torch.manual_seed(a.seed); arm = f"hardneg3_s{a.seed}"
    cz, cyx = (24, 48) if arm == "crop48" else (16, 32); set_crop(cz, cyx)
    OUT = dp.ROOT / f"divnet_{arm}_all"; OUT.mkdir(exist_ok=True)
    vids = sorted(p.stem for p in GT_DIR.glob("*.geff")); samples = []
    for v in vids:
        samples += base.gt_samples(v)
    coords = {v: grp.filter(pl.col("row_type") == "node").select("z", "y", "x").to_numpy().astype(np.float64) for v, grp in dp.submissions()}
    if arm.startswith("hardneg"):
        src = dp.ROOT / ("stage3all_scored_B_gio_lk.parquet" if arm == "hardneg2" else "stage3all_scored.parquet")
        allc = pl.read_parquet(src)
        mitotic = allc.filter(pl.col("label") == 1).select("video", "p").unique()   # a wrong daughter at a TRUE division parent is not a negative parent
        pool = allc.filter((pl.col("label") == 0) & (pl.col("s2") >= (0.2 if arm == "hardneg2" else 0.3))).select("video", "t", "p").unique().join(mitotic, on=["video", "p"], how="anti")
        hard = [(r["video"], int(r["t"]), coords[r["video"]][r["p"]], 0, False) for r in pool.iter_rows(named=True) if r["video"] in coords]
        if arm.startswith("hardneg3"):   # drop any mined negative within 6 um of a GT dividing parent at t-1..t+1 (full GT truth, not the scored pool)
            from score import SCALE
            SC = np.array(SCALE); gtp = {}
            for (v, t, c, lab, *_) in samples:
                if lab == 1:
                    gtp.setdefault(v, []).append((t, c * SC))
            n0 = len(hard); kept = []
            for h in hard:
                near = any(abs(t - h[1]) <= 1 and np.linalg.norm(c - h[2] * SC) <= 6.0 for t, c in gtp.get(h[0], []))
                if not near:
                    kept.append(h)
            hard = kept; print(f"hardneg3: removed {n0 - len(hard)} mined negatives near GT dividing parents", flush=True)
        print(f"hard negatives: {len(hard)}", flush=True); samples += hard
    samples = [(s[0], s[1], s[2], s[3], True) if len(s) == 4 else s for s in samples]
    samples.sort(key=lambda s: (s[0], s[1])); y = np.array([s[3] for s in samples], np.int8); sv = np.array([s[0] for s in samples])
    print(f"arm {arm}: samples {len(y)} positives {int(y.sum())} crop {cz}x{cyx}", flush=True)
    X = np.zeros((len(y), 5, cz, cyx, cyx), np.float16)
    for v in vids:
        idx = np.flatnonzero(sv == v)
        if len(idx) == 0:
            continue
        frames = divnet_fast.norm_frames(v)
        for t in sorted({samples[i][1] for i in idx}):
            ii = [i for i in idx if samples[i][1] == t]
            X[ii] = divnet_fast.crops(frames, t, np.stack([samples[i][2] for i in ii])).astype(np.float16)
    print(f"crops {X.shape} in {time.time() - t0:.0f}s", flush=True)
    folds = dp._folds(vids); oof = np.full(len(y), np.nan, np.float32); models = []
    for kf, fold in enumerate(folds):
        te = np.flatnonzero(np.isin(sv, list(fold))); tr = np.flatnonzero(~np.isin(sv, list(fold)))
        print(f"fold {kf}: train {len(tr)} (pos {int(y[tr].sum())}) test {len(te)}", flush=True)
        m = base.train(X, y, tr, device, bs=32 if arm == "crop48" else 64); oof[te] = base.predict(m, X, te, device, bs=128); models.append(m)
        torch.save(m.state_dict(), dp.ROOT / f"divnet_{arm}_fold{kf}.pt")
    from sklearn.metrics import roc_auc_score, average_precision_score
    gt_only = np.array([s[4] for s in samples])
    print(f"OOF AUC {roc_auc_score(y, oof):.4f} AP {average_precision_score(y, oof):.4f} (all samples) | GT-node subset AUC {roc_auc_score(y[gt_only], oof[gt_only]):.4f}", flush=True)
    fold_of = {v: kf for kf, fold in enumerate(folds) for v in fold}
    for i, f in enumerate(sorted((dp.ROOT / "cands_all").glob("*.parquet"))):
        v = f.stem
        if (OUT / f"{v}.parquet").exists() or v not in fold_of:
            continue
        m = models[fold_of[v]]; par = pl.read_parquet(f, columns=["t", "p"]).unique().sort("t"); frames = divnet_fast.norm_frames(v); rows = []
        with torch.no_grad():
            for (t,), g in par.group_by("t", maintain_order=True):
                ps = g["p"].to_numpy(); Xc = divnet_fast.crops(frames, int(t), coords[v][ps])
                for j in range(0, len(ps), 128):
                    lg = m(torch.from_numpy(Xc[j:j + 128]).to(device)).float().cpu().numpy(); rows.extend(zip(ps[j:j + 128].tolist(), lg.tolist()))
        pl.DataFrame(rows, schema=["p", f"divnet_{arm}_logit"], orient="row").with_columns(pl.lit(v).alias("video")).write_parquet(OUT / f"{v}.parquet")
        if i % 20 == 0:
            print(f"  scored {i + 1}/199", flush=True)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

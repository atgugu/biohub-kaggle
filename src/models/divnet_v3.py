"""DivNet v3 (09-27): same honest 5-fold-by-video protocol and the SAME samples as divnet_oof_v2 --arm hardneg3 (151 GT dividing
parents, <=150 one-child GT nodes per video, mined hard negatives with the GT-geometry exclusion), changed input and training only:
  - xy max-pool 2 instead of 4 (0.81 um/px; a nucleus spans ~12 px instead of ~6), crop 16 x 48 x 48 (26 x 39 x 39 um)
  - centre jitter +-1 z / +-3 xy px (inference runs on PREDICTED parent coordinates), z/y/x flips + transpose, intensity scale/shift
  - 30 epochs
Prints OOF AUC/AP (all samples, GT-node subset) to compare with v2 hardneg3 (AUC 0.7548, AP 0.0586).
Outputs: <ROOT>/divnet_v3_s<seed>_fold{k}.pt (+ oof npz). Candidate scoring is a separate step (divnet_v3_score.py).
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import zarr
import numcodecs.blosc; numcodecs.blosc.use_threads = False

sys.path.insert(0, "/workspace/biohub/analysis"); sys.path.insert(0, "/workspace/biohub/harness")
import divnet  # noqa: E402
import div_pipeline as dp  # noqa: E402
import divnet_oof as base  # noqa: E402
from score import GT_DIR, SCALE  # noqa: E402

POOL, CZ, CYX, JZ, JXY = 2, 16, 48, 1, 3
SZ, SXY = CZ + 2 * JZ, CYX + 2 * JXY


def marker():
    z, y, x = np.meshgrid(np.arange(CZ) - CZ // 2, np.arange(CYX) - CYX // 2, np.arange(CYX) - CYX // 2, indexing="ij")
    return np.exp(-0.5 * ((z / 1.5) ** 2 + (y / 4.0) ** 2 + (x / 4.0) ** 2)).astype(np.float32)


MARK = marker()


def norm_frames(video):
    arr = zarr.open_group(f"/workspace/data/train/{video}.zarr", mode="r")["0"]
    T, Z, Y, X = arr.shape; out = np.zeros((T, Z, Y // POOL, X // POOL), np.float16)
    for t in range(T):
        x = torch.from_numpy(np.asarray(arr[t]).astype(np.float32))
        x = F.max_pool2d(x.unsqueeze(1), POOL).squeeze(1).numpy()
        lo, hi = np.percentile(x, 50.0), np.percentile(x, 99.5)
        out[t] = np.clip((x - lo) / (hi - lo + 1e-6), -0.5, 6.0)
    return out


def crops(frames, t, zyx, sz=SZ, sxy=SXY):
    """(n, 4, sz, sxy, sxy) image crops (lags -1..2) around full-res voxel coords zyx; zero outside the volume."""
    T, Z, Y, X = frames.shape; n = len(zyx); x = np.zeros((n, 4, sz, sxy, sxy), np.float16)
    cz = np.rint(zyx[:, 0]).astype(int); cy = np.rint(zyx[:, 1] / POOL).astype(int); cx = np.rint(zyx[:, 2] / POOL).astype(int)
    for li, lag in enumerate(divnet.LAGS):
        fr = frames[int(np.clip(t + lag, 0, T - 1))]
        for i in range(n):
            z0, y0, x0 = cz[i] - sz // 2, cy[i] - sxy // 2, cx[i] - sxy // 2
            zs, ys, xs = slice(max(z0, 0), min(z0 + sz, Z)), slice(max(y0, 0), min(y0 + sxy, Y)), slice(max(x0, 0), min(x0 + sxy, X))
            if zs.stop > zs.start and ys.stop > ys.start and xs.stop > xs.start:
                x[i, li, zs.start - z0:zs.stop - z0, ys.start - y0:ys.stop - y0, xs.start - x0:xs.stop - x0] = fr[zs, ys, xs]
    return x


def daughter_samples(v):
    """DaughterNet samples: positives = GT daughters (children of a GT node with 2 children), at their own frame; negatives = GT nodes whose
    parent has exactly one child (ordinary continuations), <= 150 per video (base.rng)."""
    g = base.load_geff(base.GT_DIR / f"{v}.geff"); na = g.node_attrs(attr_keys=["node_id", "t", "z", "y", "x"]); ea = g.edge_attrs(attr_keys=[])
    src = ea["source_id"].to_numpy(); tgt = ea["target_id"].to_numpy(); u, cnt = np.unique(src, return_counts=True); two = set(u[cnt == 2].tolist())
    rows = {r[0]: r[1:] for r in na.iter_rows()}
    dau = [int(b) for a_, b in zip(src, tgt) if a_ in two]; cont = np.array([int(b) for a_, b in zip(src, tgt) if a_ not in two])
    pos = [(v, int(rows[n][0]), np.array(rows[n][1:], np.float64), 1) for n in dau]
    neg = [(v, int(rows[n][0]), np.array(rows[n][1:], np.float64), 0) for n in base.rng.choice(cont, size=min(150, len(cont)), replace=False)] if len(cont) else []
    return pos + neg


def to_input(xb, train):
    """xb: (b, 4, SZ, SXY, SXY) half on device -> (b, 5, CZ, CYX, CYX) float with marker; random jitter/aug when training."""
    b = xb.shape[0]; mk = torch.from_numpy(MARK).to(xb.device)
    if train:
        dz = np.random.randint(0, 2 * JZ + 1); dy, dx = np.random.randint(0, 2 * JXY + 1, 2)
        x = xb[:, :, dz:dz + CZ, dy:dy + CYX, dx:dx + CYX].float()
        x = x * torch.empty(b, 1, 1, 1, 1, device=x.device).uniform_(0.85, 1.15) + torch.empty(b, 1, 1, 1, 1, device=x.device).uniform_(-0.1, 0.1)
        x = torch.cat([x, mk.expand(b, 1, CZ, CYX, CYX)], 1)
        if np.random.rand() < 0.5: x = x.flip(-1)
        if np.random.rand() < 0.5: x = x.flip(-2)
        if np.random.rand() < 0.5: x = x.flip(-3)
        if np.random.rand() < 0.5: x = x.transpose(-1, -2)
        return x
    x = xb[:, :, JZ:JZ + CZ, JXY:JXY + CYX, JXY:JXY + CYX].float()
    return torch.cat([x, mk.expand(b, 1, CZ, CYX, CYX)], 1)


def train(X, y, tr, device, epochs=30, bs=32, lr=3e-4):
    m = divnet.DivNet().to(device); m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * ((len(tr) + bs - 1) // bs))
    pos = max(1, int(y[tr].sum())); lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(min(30.0, (len(tr) - pos) / pos), device=device))
    for ep in range(epochs):
        perm = np.random.permutation(tr); tot = 0.0
        for i in range(0, len(perm), bs):
            idx = np.sort(perm[i:i + bs])
            xb = to_input(torch.from_numpy(X[idx]).to(device), True); yb = torch.from_numpy(y[idx].astype(np.float32)).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = lossf(m(xb).float(), yb)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += float(loss) * len(idx)
        if ep % 5 == 4:
            print(f"    epoch {ep} loss {tot / len(perm):.4f}", flush=True)
    m.eval(); return m


@torch.no_grad()
def predict(m, X, idx, device, bs=128, tta=True):
    out = np.zeros(len(idx), np.float32)
    for i in range(0, len(idx), bs):
        x = to_input(torch.from_numpy(X[idx[i:i + bs]]).to(device), False)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = m(x).float()
            if tta:
                lg = (lg + m(x.flip(-1)).float() + m(x.flip(-2)).float() + m(x.flip(-3)).float()) / 4
        out[i:i + bs] = lg.cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--all", action="store_true"); ap.add_argument("--target", default="parent", choices=["parent", "daughter"]); a = ap.parse_args()
    t0 = time.time(); device = torch.device("cuda"); np.random.seed(a.seed); torch.manual_seed(a.seed)
    base.rng = np.random.default_rng(0)          # the SAME sample draw as v2 hardneg3 (its gt_samples used rng seed 0 before mining)
    vids = sorted(p.stem for p in GT_DIR.glob("*_*.geff")); samples = []
    for v in vids:
        samples += (base.gt_samples(v) if a.target == "parent" else daughter_samples(v))
    coords = {v: grp.filter(pl.col("row_type") == "node").select("z", "y", "x").to_numpy().astype(np.float64) for v, grp in dp.submissions()}
    allc = pl.read_parquet(dp.ROOT / "stage3all_scored.parquet")
    mitotic = allc.filter(pl.col("label") == 1).select("video", "p").unique()
    pool = allc.filter((pl.col("label") == 0) & (pl.col("s2") >= 0.3)).select("video", "t", "p").unique().join(mitotic, on=["video", "p"], how="anti")
    if a.target == "parent":
        hard = [(r["video"], int(r["t"]), coords[r["video"]][r["p"]], 0) for r in pool.iter_rows(named=True) if r["video"] in coords]
    else:   # daughter: the proposed (wrong) daughter c of the ranker's false positives, at frame t+1
        pool = allc.filter((pl.col("label") == 0) & (pl.col("s2") >= 0.3)).select("video", "t", "c").unique()
        hard = [(r["video"], int(r["t"]) + 1, coords[r["video"]][r["c"]], 0) for r in pool.iter_rows(named=True) if r["video"] in coords]
    SC = np.array(SCALE); gtp = {}
    for (v, t, c, lab) in samples:
        if lab == 1:
            gtp.setdefault(v, []).append((t, c * SC))
    hard = [h for h in hard if not any(abs(t - h[1]) <= 1 and np.linalg.norm(c - h[2] * SC) <= 6.0 for t, c in gtp.get(h[0], []))]
    print(f"hard negatives: {len(hard)}", flush=True)
    samples = [(s[0], s[1], s[2], s[3], True) for s in samples] + [(h[0], h[1], h[2], h[3], False) for h in hard]
    samples.sort(key=lambda s: (s[0], s[1])); y = np.array([s[3] for s in samples], np.int8); sv = np.array([s[0] for s in samples])
    print(f"v3: samples {len(y)} positives {int(y.sum())} crop {CZ}x{CYX} pool {POOL} (stored {SZ}x{SXY})", flush=True)
    X = np.zeros((len(y), 4, SZ, SXY, SXY), np.float16)
    for v in vids:
        idx = np.flatnonzero(sv == v)
        if len(idx) == 0:
            continue
        frames = norm_frames(v)
        for t in sorted({samples[i][1] for i in idx}):
            ii = [i for i in idx if samples[i][1] == t]
            X[ii] = crops(frames, t, np.stack([samples[i][2] for i in ii]))
    print(f"crops {X.shape} in {time.time() - t0:.0f}s", flush=True)
    if a.all:   # deployment model: every sample, same recipe
        m = train(X, y, np.arange(len(y)), device, epochs=a.epochs); torch.save(m.state_dict(), dp.ROOT / (f"divnet_v3_all_s{a.seed}.pt" if a.target == "parent" else f"dnet_v3_all_s{a.seed}.pt"))
        print(f"all-data model saved in {time.time() - t0:.0f}s", flush=True); return
    folds = dp._folds(vids); oof = np.full(len(y), np.nan, np.float32)
    for kf, fold in enumerate(folds):
        te = np.flatnonzero(np.isin(sv, list(fold))); tr = np.flatnonzero(~np.isin(sv, list(fold)))
        print(f"fold {kf}: train {len(tr)} (pos {int(y[tr].sum())}) test {len(te)} (pos {int(y[te].sum())})  {time.time() - t0:.0f}s", flush=True)
        m = train(X, y, tr, device, epochs=a.epochs); oof[te] = predict(m, X, te, device)
        torch.save(m.state_dict(), dp.ROOT / (f"divnet_v3_s{a.seed}_fold{kf}.pt" if a.target == "parent" else f"dnet_v3_s{a.seed}_fold{kf}.pt"))
    from sklearn.metrics import roc_auc_score, average_precision_score
    gt_only = np.array([s[4] for s in samples])
    print(f"OOF AUC {roc_auc_score(y, oof):.4f} AP {average_precision_score(y, oof):.4f} (all samples) | GT-node subset AUC "
          f"{roc_auc_score(y[gt_only], oof[gt_only]):.4f} AP {average_precision_score(y[gt_only], oof[gt_only]):.4f}", flush=True)
    np.savez(dp.ROOT / (f"divnet_v3_s{a.seed}_oof.npz" if a.target == "parent" else f"dnet_v3_s{a.seed}_oof.npz"), oof=oof, y=y, sv=sv, t=np.array([s[1] for s in samples]))
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

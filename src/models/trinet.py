"""TripletNet (09-28): is (parent p at t -> children c1, c at t+1) a real division? DivNet v3 recipe (xy max-pool 2, 16x48x48 crop centred on
the parent, 4 lags, jitter / flips / intensity aug, 30 epochs) with 6 input channels: 4 image lags + parent marker (centre) + CHILD marker
(gaussians at both proposed children, rendered after the jitter shift so markers follow the crop). Honest 5-fold by video (dp._folds).
Samples (all coordinates full-res voxels):
  GT positives: every GT division (P, D1, D2).
  GT negatives: (a) one-child GT node P' with its child and a fake sister = another GT node at t+1, 4-14 um from P';
                (b) a GT division with one true daughter and a wrong sister (GT node at t+1, 4-14 um from P, not a daughter).
  Pipeline rows (predicted coordinates, the inference distribution): labelled candidate rows of the three populations
                (label 1 = positive; label 0 = negative, <= 25 per video sampled).
Out: <ROOT>/trinet_s<seed>_fold{k}.pt, trinet_s<seed>_oof.npz; --all trains the deployment model trinet_all_s<seed>.pt."""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import polars as pl
import torch
import torch.nn as nn

sys.path.insert(0, "/workspace/biohub/analysis"); sys.path.insert(0, "/workspace/biohub/harness")
import divnet  # noqa: E402
import div_pipeline as dp  # noqa: E402
import divnet_v3 as v3  # noqa: E402
from divnet_v3_score import coords_from  # noqa: E402
from score import GT_DIR, SCALE, load_geff  # noqa: E402

SC = np.array(SCALE); POOL, CZ, CYX, JZ, JXY = v3.POOL, v3.CZ, v3.CYX, v3.JZ, v3.JXY
POPS = {"x138h8b_hn": "/workspace/biohub/caches/stack_x138_head8_all", "x138": "/workspace/biohub/caches/stack_x138", "gf": "/workspace/biohub/caches/stack_gf"}


def gt_samples(v, rng):
    g = load_geff(GT_DIR / f"{v}.geff"); na = g.node_attrs(attr_keys=["node_id", "t", "z", "y", "x"]); ea = g.edge_attrs(attr_keys=[])
    src = ea["source_id"].to_numpy(); tgt = ea["target_id"].to_numpy(); rows = {r[0]: (r[1], np.array(r[2:], np.float64)) for r in na.iter_rows()}
    ch = {}
    for a, b in zip(src.tolist(), tgt.tolist()):
        ch.setdefault(a, []).append(b)
    byt = {}
    for n, (t, xyz) in rows.items():
        byt.setdefault(t, []).append((n, xyz))
    out = []

    def near(t, xyz, excl):
        cand = [(n, x) for n, x in byt.get(t, []) if n not in excl and 4.0 <= np.linalg.norm((x - xyz) * SC) <= 14.0]
        return cand[rng.integers(len(cand))][1] if cand else None
    one = [a for a, c in ch.items() if len(c) == 1]
    for a, c in ch.items():
        t, p = rows[a]
        if len(c) == 2:
            out.append((v, t, p, rows[c[0]][1], rows[c[1]][1], 1))
            for keep in c:
                x = near(t + 1, p, set(c))
                if x is not None:
                    out.append((v, t, p, rows[keep][1], x, 0))
    for a in rng.choice(one, size=min(60, len(one)), replace=False) if one else []:
        t, p = rows[a]; c1 = ch[a][0]; x = near(t + 1, p, {c1})
        if x is not None:
            out.append((v, t, p, rows[c1][1], x, 0))
    return out


def pipeline_samples(rng, vids, negcap=25):
    out = []
    for pop, cache in POPS.items():
        parts = []
        for f in sorted((dp.ROOT.parent / f"divpipe_{pop}" / "cands_all").glob("*.parquet")):
            if f.stem not in vids:
                continue
            d = pl.read_parquet(f, columns=["t", "p", "c1", "c", "label"]).filter(pl.col("label") >= 0)
            pos = d.filter(pl.col("label") == 1); neg = d.filter(pl.col("label") == 0)
            if neg.height > negcap:
                neg = neg.sample(negcap, seed=int(rng.integers(1 << 30)))
            parts.append(pl.concat([pos, neg]).with_columns(pl.lit(f.stem).alias("video")))
        d = pl.concat(parts); co = coords_from(cache, set(d["video"].unique().to_list()))
        for r in d.iter_rows(named=True):
            if r["video"] in co:
                P = co[r["video"]]; out.append((r["video"], int(r["t"]), P[r["p"]], P[r["c1"]], P[r["c"]], int(r["label"])))
        print(pop, "pipeline samples", len(out), flush=True)
    return out


def offsets(p, a, b):
    """child offsets relative to the parent in crop units (z slices, pooled xy px)."""
    return np.array([[a[0] - p[0], (a[1] - p[1]) / POOL, (a[2] - p[2]) / POOL], [b[0] - p[0], (b[1] - p[1]) / POOL, (b[2] - p[2]) / POOL]], np.float32)


_zz, _yy, _xx = [torch.arange(n, dtype=torch.float32) for n in (CZ, CYX, CYX)]


def render(off, sh, dev):
    """off (b, 2, 3) child offsets; sh (3,) crop-start shift relative to the centred crop -> (b, 1, CZ, CYX, CYX) child-marker channel."""
    zz, yy, xx = _zz.to(dev), _yy.to(dev), _xx.to(dev)
    c = torch.tensor([CZ // 2, CYX // 2, CYX // 2], dtype=torch.float32, device=dev) - sh.to(dev)
    pos = off + c                                                              # (b, 2, 3)
    g = 0
    for k in range(2):
        dz = (zz[None, :] - pos[:, k, 0:1]) / 1.5; dy = (yy[None, :] - pos[:, k, 1:2]) / 3.0; dx = (xx[None, :] - pos[:, k, 2:3]) / 3.0
        g = g + torch.exp(-0.5 * (dz[:, :, None, None] ** 2 + dy[:, None, :, None] ** 2 + dx[:, None, None, :] ** 2))
    return g[:, None]


def to_input(xb, off, train):
    b = xb.shape[0]; dev = xb.device; mk = torch.from_numpy(v3.MARK).to(dev)
    if train:
        dz = np.random.randint(0, 2 * JZ + 1); dy, dx = np.random.randint(0, 2 * JXY + 1, 2)
        x = xb[:, :, dz:dz + CZ, dy:dy + CYX, dx:dx + CYX].float()
        x = x * torch.empty(b, 1, 1, 1, 1, device=dev).uniform_(0.85, 1.15) + torch.empty(b, 1, 1, 1, 1, device=dev).uniform_(-0.1, 0.1)
        sh = torch.tensor([dz - JZ, dy - JXY, dx - JXY], dtype=torch.float32)
    else:
        x = xb[:, :, JZ:JZ + CZ, JXY:JXY + CYX, JXY:JXY + CYX].float(); sh = torch.zeros(3)
    x = torch.cat([x, mk.expand(b, 1, CZ, CYX, CYX), render(off, sh, dev)], 1)
    if train:
        if np.random.rand() < 0.5: x = x.flip(-1)
        if np.random.rand() < 0.5: x = x.flip(-2)
        if np.random.rand() < 0.5: x = x.flip(-3)
        if np.random.rand() < 0.5: x = x.transpose(-1, -2)
    return x


def train(X, O, y, tr, device, epochs=30, bs=32, lr=3e-4, width=16):
    m = divnet.DivNet(cin=6, b=width).to(device); m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * ((len(tr) + bs - 1) // bs))
    pos = max(1, int(y[tr].sum())); lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(min(30.0, (len(tr) - pos) / pos), device=device))
    for ep in range(epochs):
        perm = np.random.permutation(tr); tot = 0.0
        for i in range(0, len(perm), bs):
            idx = np.sort(perm[i:i + bs])
            xb = to_input(torch.from_numpy(X[idx]).to(device), torch.from_numpy(O[idx]).to(device), True); yb = torch.from_numpy(y[idx].astype(np.float32)).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = lossf(m(xb).float(), yb)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += float(loss) * len(idx)
        if ep % 5 == 4:
            print(f"    epoch {ep} loss {tot / len(perm):.4f}", flush=True)
    m.eval(); return m


@torch.no_grad()
def predict(m, X, O, idx, device, bs=128):
    out = np.zeros(len(idx), np.float32)
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]; x = to_input(torch.from_numpy(X[j]).to(device), torch.from_numpy(O[j]).to(device), False)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = (m(x).float() + m(x.flip(-1)).float() + m(x.flip(-2)).float() + m(x.flip(-3)).float()) / 4
        out[i:i + len(j)] = lg.cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--all", action="store_true"); ap.add_argument("--width", type=int, default=16); ap.add_argument("--tag", default=""); ap.add_argument("--negcap", type=int, default=25); ap.add_argument("--fork-tables", default=""); ap.add_argument("--fork-rep", type=int, default=3); ap.add_argument("--loeo", action="store_true", help="09-29: leave-one-EMBRYO-out folds (fold 0 = 44b6 test / trained on 6bba; fold 1 = 6bba test / trained on 44b6)"); a = ap.parse_args()
    t0 = time.time(); device = torch.device("cuda"); np.random.seed(a.seed); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    vids = sorted(p.stem for p in GT_DIR.glob("*_*.geff")); samples = []
    for v in vids:
        samples += gt_samples(v, rng)
    ng = len(samples); samples += pipeline_samples(rng, set(vids), a.negcap)
    if a.fork_tables:   # 09-28 hard-negative round 2: labelled pipeline FORKS (TP / FP after the X3h attach) at predicted coordinates, FP repeated
        fk = pl.concat([pl.read_parquet(p, columns=["video", "t", "p", "c1", "c2", "label"]) for p in a.fork_tables.split(",")]).filter(pl.col("label") >= 0).unique(subset=["video", "p", "c1", "c2"])
        co = coords_from("/workspace/biohub/scratch/v3/cache195", set(fk["video"].unique().to_list())); nf = 0
        for r in fk.iter_rows(named=True):
            if r["video"] not in co:
                continue
            P = co[r["video"]]; smp = (r["video"], int(r["t"]), P[r["p"]], P[r["c1"]], P[r["c2"]], int(r["label"]))
            for _ in range(a.fork_rep if r["label"] == 0 else 1):
                samples.append(smp); nf += 1
        print(f"fork samples added {nf} (TP {int((fk['label'] == 1).sum())}, FP {int((fk['label'] == 0).sum())} x{a.fork_rep})", flush=True)
    samples.sort(key=lambda s: (s[0], s[1])); y = np.array([s[5] for s in samples], np.int8); sv = np.array([s[0] for s in samples])
    print(f"samples {len(y)} (GT {ng}) positives {int(y.sum())}", flush=True)
    X = np.zeros((len(y), 4, v3.SZ, v3.SXY, v3.SXY), np.float16); O = np.stack([offsets(s[2], s[3], s[4]) for s in samples])
    for v in vids:
        idx = np.flatnonzero(sv == v)
        if len(idx) == 0:
            continue
        frames = v3.norm_frames(v)
        for t in sorted({samples[i][1] for i in idx}):
            ii = [i for i in idx if samples[i][1] == t]
            X[ii] = v3.crops(frames, t, np.stack([samples[i][2] for i in ii]))
    print(f"crops {X.shape} in {time.time() - t0:.0f}s", flush=True)
    if a.all:
        m = train(X, O, y, np.arange(len(y)), device, epochs=a.epochs, width=a.width); torch.save(m.state_dict(), dp.ROOT / f"trinet{a.tag}_all_s{a.seed}.pt"); print("all-data saved", flush=True); return
    folds = [set(v for v in vids if v.startswith("44b6")), set(v for v in vids if v.startswith("6bba"))] if a.loeo else dp._folds(vids); oof = np.full(len(y), np.nan, np.float32)
    for kf, fold in enumerate(folds):
        te = np.flatnonzero(np.isin(sv, list(fold))); tr = np.flatnonzero(~np.isin(sv, list(fold)))
        print(f"fold {kf}: train {len(tr)} (pos {int(y[tr].sum())}) test {len(te)} (pos {int(y[te].sum())})  {time.time() - t0:.0f}s", flush=True)
        m = train(X, O, y, tr, device, epochs=a.epochs, width=a.width); oof[te] = predict(m, X, O, te, device); torch.save(m.state_dict(), dp.ROOT / f"trinet{a.tag}_s{a.seed}_fold{kf}.pt")
    from sklearn.metrics import roc_auc_score, average_precision_score
    print(f"OOF AUC {roc_auc_score(y, oof):.4f} AP {average_precision_score(y, oof):.4f}", flush=True)
    np.savez(dp.ROOT / f"trinet{a.tag}_s{a.seed}_oof.npz", oof=oof, y=y, sv=sv, t=np.array([s[1] for s in samples]))
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

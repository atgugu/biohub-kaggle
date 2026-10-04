"""Train our own V1284 coordinate-refinement head from the x138 capture (caches/v1284_capture/<video>/<frame>.npz).

Capture format (written by the notebook's own module in V1284_MODE=capture): coords (N,4) int16 = (t, z, y, x) at the detector's
(1,4,4) downsample, where a voxel is isotropic 1.625 um; features (N,224) float32 = 32 UNet channels at the centre + 6 directional
differences. Head (same architecture the module loads): Linear(224,32)-SiLU-Linear(32,3), output bounded to < 2 um by
2*d/(1+|d|); input standardised by saved mean/scale. Target: displacement from the detector centre to the Hungarian-matched GT
node (<= 7 um), in um. Validation: by-movie folds, mean centre error before/after. Output: {'state_dict','mean','scale'}.

Usage: python v1284_train.py [--folds 5] [--epochs 30] [--out /workspace/biohub/subs/weights_ds/biohub-v1284-head-s075/v1284_head.pt]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, "/workspace/biohub/harness")
from score import GT_DIR, load_geff  # noqa: E402

CAP = Path("/workspace/biohub/caches/v1284_capture"); VOX = 1.625; DS = np.array([1.0, 4.0, 4.0]); MAX_UM = 7.0


def make_head():
    head = torch.nn.Sequential(torch.nn.Linear(224, 32), torch.nn.SiLU(), torch.nn.Linear(32, 3))
    torch.nn.init.zeros_(head[-1].weight); torch.nn.init.zeros_(head[-1].bias)
    return head


def bounded(head, x):
    delta = head(x)
    return 2.0 * delta / (1.0 + torch.linalg.vector_norm(delta, dim=-1, keepdim=True))


def gt_by_frame(v):
    g = load_geff(GT_DIR / f"{v}.geff"); na = g.node_attrs(attr_keys=["t", "z", "y", "x"])
    t = na["t"].to_numpy(); p = np.stack([na["z"].to_numpy(), na["y"].to_numpy(), na["x"].to_numpy()], 1).astype(np.float64) / DS   # downsampled voxels
    return {int(k): p[t == k] for k in np.unique(t)}


def load_video(v):
    """Returns X (n,224), D (n,3) displacement det->GT in um, and the per-detection frame ids."""
    gt = gt_by_frame(v); X, D, T = [], [], []
    for f in sorted((CAP / v).glob("*.npz")):
        z = np.load(f); c = z["coords"].astype(np.float64); feats = z["features"]
        if not len(c):
            continue
        t = int(c[0, 0]); g = gt.get(t)
        if g is None or not len(g):
            continue
        det = c[:, 1:]; cost = np.linalg.norm((det[:, None, :] - g[None, :, :]) * VOX, axis=-1)
        r, k = linear_sum_assignment(np.where(cost <= MAX_UM, cost, 1e6))
        ok = cost[r, k] <= MAX_UM
        if not ok.any():
            continue
        X.append(feats[r[ok]]); D.append((g[k[ok]] - det[r[ok]]) * VOX); T.append(np.full(int(ok.sum()), t))
    if not X:
        return None
    return np.concatenate(X), np.concatenate(D).astype(np.float32), np.concatenate(T)


def train(X, D, mean, scale, epochs, device, seed=0):
    torch.manual_seed(seed); head = make_head().to(device)
    x = torch.from_numpy((X - mean) / scale).float().to(device); y = torch.from_numpy(D).to(device)
    n = np.linalg.norm(D, axis=1); w = torch.from_numpy(np.where(n < 1.9, 1.0, 0.0).astype(np.float32)).to(device)   # unreachable (>= 2 um) targets carry no gradient
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-3); bs = 512; steps = epochs * ((len(x) + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps); head.train()
    for _ in range(epochs):
        perm = torch.randperm(len(x), device=device)
        for i in range(0, len(x), bs):
            j = perm[i:i + bs]; loss = ((bounded(head, x[j]) - y[j]) ** 2).sum(1).mul(w[j]).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    head.eval(); return head


def evaluate(head, X, D, mean, scale, device):
    with torch.no_grad():
        s = bounded(head, torch.from_numpy((X - mean) / scale).float().to(device)).cpu().numpy()
    before = np.linalg.norm(D, axis=1); after = np.linalg.norm(D - s, axis=1)
    return before.mean(), after.mean(), np.median(before), np.median(after)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--folds", type=int, default=5); ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--out", default="/workspace/biohub/subs/weights_ds/biohub-v1284-head-s075/v1284_head.pt"); ap.add_argument("--min-videos", type=int, default=8); ap.add_argument("--exclude", default="", help="comma list or @file of videos to leave out of training (for an honest replay test on them)"); ap.add_argument("--cap", default="/workspace/biohub/caches/v1284_capture", help="capture directory (default: 8-view captures; z-TTA captures: caches/v1284_capture_zt)")
    a = ap.parse_args(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global CAP
    CAP = Path(a.cap)
    excl = set(open(a.exclude[1:]).read().split()) if a.exclude.startswith("@") else set(a.exclude.split(",")) - {""}
    vids = sorted(p.name for p in CAP.iterdir() if p.is_dir() and p.name not in excl and (GT_DIR / f"{p.name}.geff").exists() and any(p.glob("*.npz")))
    data = {}
    for v in vids:
        r = load_video(v)
        if r is not None:
            data[v] = r
    vids = sorted(data); print(f"videos {len(vids)}; matched pairs {sum(len(d[1]) for d in data.values())}", flush=True)
    if len(vids) < a.min_videos:
        print("not enough videos captured yet"); return
    Xall = np.concatenate([data[v][0] for v in vids]); mean = Xall.mean(0); scale = Xall.std(0) + 1e-6
    folds = [vids[k::a.folds] for k in range(a.folds)]; rows = []
    for k, held in enumerate(folds):
        tr = [v for v in vids if v not in held]
        head = train(np.concatenate([data[v][0] for v in tr]), np.concatenate([data[v][1] for v in tr]), mean, scale, a.epochs, device)
        for v in held:
            b, af, bm, am = evaluate(head, data[v][0], data[v][1], mean, scale, device); rows.append((v, len(data[v][1]), b, af, bm, am))
    rows.sort(); imp = [r for r in rows if r[3] < r[2]]
    for v, n, b, af, bm, am in rows:
        print(f"  {v}  n={n:6d}  mean err {b:.3f} -> {af:.3f} um ({(af / b - 1) * 100:+.1f}%)  median {bm:.3f} -> {am:.3f}")
    tot_n = sum(r[1] for r in rows); pb = sum(r[1] * r[2] for r in rows) / tot_n; pa = sum(r[1] * r[3] for r in rows) / tot_n
    print(f"HELD-OUT BY MOVIE: pooled mean centre error {pb:.3f} -> {pa:.3f} um ({(pa / pb - 1) * 100:+.1f}%); {len(imp)}/{len(rows)} movies improve", flush=True)
    for e in ("44b6", "6bba"):
        sub = [r for r in rows if r[0].startswith(e)]
        if sub:
            n = sum(r[1] for r in sub); print(f"  {e}: {sum(r[1] * r[2] for r in sub) / n:.3f} -> {sum(r[1] * r[3] for r in sub) / n:.3f} um, {sum(1 for r in sub if r[3] < r[2])}/{len(sub)} improve")
    # Leave-one-embryo-out (both directions) + constant-offset baselines: the GT-minus-detection z offset differs by embryo
    # (~+0.85 um on 6bba vs ~+0.15 um on 44b6), so a by-movie CV inside both embryos could look good by learning an offset.
    print("LEAVE-ONE-EMBRYO-OUT:", flush=True)
    for tr_e, te_e in (("44b6", "6bba"), ("6bba", "44b6")):
        tr = [v for v in vids if v.startswith(tr_e)]; te = [v for v in vids if v.startswith(te_e)]
        if not tr or not te:
            continue
        Xtr = np.concatenate([data[v][0] for v in tr]); Dtr = np.concatenate([data[v][1] for v in tr]); Xte = np.concatenate([data[v][0] for v in te]); Dte = np.concatenate([data[v][1] for v in te])
        head = train(Xtr, Dtr, mean, scale, a.epochs, device); b, af, _, _ = evaluate(head, Xte, Dte, mean, scale, device)
        off_tr = Dtr.mean(0); off_te = Dte.mean(0)
        const_tr = np.linalg.norm(Dte - off_tr, axis=1).mean(); const_te = np.linalg.norm(Dte - off_te, axis=1).mean()
        print(f"  train {tr_e} ({len(tr)} videos) -> test {te_e} ({len(te)} videos): mean err {b:.3f} -> head {af:.3f} um ({(af / b - 1) * 100:+.1f}%) | "
              f"constant-offset baselines: train-embryo offset {const_tr:.3f}, oracle test-embryo offset {const_te:.3f} | mean GT-det offset train {off_tr.round(2)} test {off_te.round(2)}", flush=True)
    head = train(Xall, np.concatenate([data[v][1] for v in vids]), mean, scale, a.epochs, device)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {k: v.cpu() for k, v in head.state_dict().items()}, "mean": torch.from_numpy(mean).float(), "scale": torch.from_numpy(scale).float(),
                "videos": vids, "pairs": int(tot_n)}, out)
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()

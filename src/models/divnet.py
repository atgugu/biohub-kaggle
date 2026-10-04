"""giorgosi DivNet v2, reconstructed from the checkpoint (arch 'divnet_unet3d_gap', 5-ch input).

The public notebooks load this checkpoint into a different 1-channel net with strict=False (random weights).
The training source is not public, so the preprocessing details not pinned by the checkpoint's config
(xy pooling type, percentile scope, channel order) are calibrated by AUC on GT divisions vs GT non-dividing nodes:
    python divnet.py calibrate
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import zarr
import numcodecs.blosc; numcodecs.blosc.use_threads = False   # blosc threads race under load -> "decompression: -1"

CKPT = Path("/workspace/kaggle_inputs/biohub-divnet-v2/best_overall.pt")
DATA = Path("/workspace/data/train")
LAGS = (-1, 0, 1, 2); CZ, CYX, POOL = 16, 32, 4


NORM = "instance"   # the checkpoint has affine norm layers without running stats: instance or group norm


def _norm_layer(c):
    return nn.InstanceNorm3d(c, affine=True) if NORM == "instance" else nn.GroupNorm(int(NORM[2:]), c)


class Block(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.block = nn.Sequential(nn.Conv3d(i, o, 3, padding=1, bias=False), _norm_layer(o), nn.ReLU(inplace=True),
                                   nn.Conv3d(o, o, 3, padding=1, bias=False), _norm_layer(o), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.block(x)


class DivNet(nn.Module):
    def __init__(self, cin=5, b=16):
        super().__init__()
        self.enc1, self.enc2, self.enc3 = Block(cin, b), Block(b, 2 * b), Block(2 * b, 4 * b)
        self.bottleneck = Block(4 * b, 8 * b)
        self.up3, self.dec3 = nn.ConvTranspose3d(8 * b, 4 * b, 2, 2), Block(8 * b, 4 * b)
        self.up2, self.dec2 = nn.ConvTranspose3d(4 * b, 2 * b, 2, 2), Block(4 * b, 2 * b)
        self.up1, self.dec1 = nn.ConvTranspose3d(2 * b, b, 2, 2), Block(2 * b, b)
        self.head = nn.Linear(b, 1)

    def forward(self, x):
        e1 = self.enc1(x); e2 = self.enc2(F.max_pool3d(e1, 2)); e3 = self.enc3(F.max_pool3d(e2, 2))
        bt = self.bottleneck(F.max_pool3d(e3, 2))
        d3 = self.dec3(torch.cat([self.up3(bt), e3], 1)); d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.head(d1.mean(dim=(2, 3, 4))).squeeze(-1)


def load_divnet(device="cuda") -> DivNet:
    m = DivNet(); sd = torch.load(CKPT, map_location="cpu", weights_only=False)["model_state"]
    m.load_state_dict(sd, strict=True)   # strict: every tensor must match
    return m.to(device).eval()


@lru_cache(maxsize=64)
def _frame(video: str, t: int, pool: str) -> np.ndarray:
    arr = zarr.open_group(str(DATA / f"{video}.zarr"), mode="r")["0"]
    x = torch.from_numpy(np.asarray(arr[t]).astype(np.float32))
    if pool == "stride":
        x = x[:, ::POOL, ::POOL]
    else:
        f = F.avg_pool2d if pool == "mean" else F.max_pool2d
        x = f(x.unsqueeze(1), POOL).squeeze(1)
    return x.numpy()


def _norm(x, ref):
    lo, hi = np.percentile(ref, 50.0), np.percentile(ref, 99.5)
    return np.clip((x - lo) / (hi - lo + 1e-6), -0.5, 6.0)


def _marker():
    z, y, x = np.meshgrid(np.arange(CZ) - CZ // 2, np.arange(CYX) - CYX // 2, np.arange(CYX) - CYX // 2, indexing="ij")
    return np.exp(-0.5 * ((z / 1.5) ** 2 + (y / 2.0) ** 2 + (x / 2.0) ** 2)).astype(np.float32)


MARK = _marker()


def make_input(video: str, t: int, zyx, pool="mean", scope="frame", order="img_first", T=100) -> np.ndarray:
    """(5, 16, 32, 32) input for a node at frame t, full-res voxel coords zyx."""
    cz, cy, cx = int(round(zyx[0])), int(round(zyx[1] / POOL)), int(round(zyx[2] / POOL))
    chans = []
    for lag in LAGS:
        fr = _frame(video, int(np.clip(t + lag, 0, T - 1)), pool)
        Z, Y, X = fr.shape
        out = np.zeros((CZ, CYX, CYX), np.float32)
        z0, y0, x0 = cz - CZ // 2, cy - CYX // 2, cx - CYX // 2
        zs, ys, xs = slice(max(z0, 0), min(z0 + CZ, Z)), slice(max(y0, 0), min(y0 + CYX, Y)), slice(max(x0, 0), min(x0 + CYX, X))
        crop = fr[zs, ys, xs]
        crop = _norm(crop, fr if scope == "frame" else crop)
        out[zs.start - z0:zs.stop - z0, ys.start - y0:ys.stop - y0, xs.start - x0:xs.stop - x0] = crop
        chans.append(out)
    chans = chans + [MARK] if order == "img_first" else [MARK] + chans
    return np.stack(chans)


@torch.no_grad()
def predict(model, inputs: np.ndarray, bs=64) -> np.ndarray:
    out = []
    for i in range(0, len(inputs), bs):
        out.append(torch.sigmoid(model(torch.from_numpy(inputs[i:i + bs]).cuda())).cpu().numpy())
    return np.concatenate(out)


def calibrate():
    import polars as pl
    from sklearn.metrics import roc_auc_score
    sys.path.insert(0, "/workspace/biohub/harness")
    from score import GT_DIR, load_geff
    rng = np.random.default_rng(0); stats = pl.read_csv("/workspace/biohub/harness/gt_stats.csv")
    vids = stats.filter(pl.col("divisions") > 0)["video"].to_list()[:40]
    samples = []
    for v in vids:
        g = load_geff(GT_DIR / f"{v}.geff"); na = g.node_attrs(attr_keys=["node_id", "t", "z", "y", "x"]); ea = g.edge_attrs(attr_keys=[])
        src, cnt = np.unique(ea["source_id"].to_numpy(), return_counts=True); div = set(src[cnt >= 2].tolist()); one = src[cnt == 1]
        rows = {r[0]: r[1:] for r in na.iter_rows()}
        for n in div:
            samples.append((v, rows[n][0], rows[n][1:], 1))
        for n in rng.choice(one, size=min(12, len(one)), replace=False):
            samples.append((v, rows[n][0], rows[n][1:], 0))
    y = np.array([s[3] for s in samples]); print("samples", len(y), "positives", int(y.sum()), flush=True)
    variants = [(p, s, "img_first") for p in ("mean", "stride", "max") for s in ("frame", "crop")]
    samples.sort(key=lambda s: (s[0], s[1])); y = np.array([s[3] for s in samples])
    X = {k: [] for k in variants}
    for v, t, zyx, _ in samples:
        for k in variants:
            X[k].append(make_input(v, t, zyx, *k))
    global NORM
    res = []
    for NORM in ("instance", "gn8", "gn4", "gn2", "gn1"):
      model = load_divnet()
      for k in variants:
        p = predict(model, np.stack(X[k]))
        res.append((roc_auc_score(y, p), NORM, k, p[y == 1].mean(), p[y == 0].mean(), np.percentile(p, [1, 50, 99]).round(3).tolist()))
    for r in sorted(res, reverse=True)[:12]:
        print("AUC %.4f %s %s | mean p pos %.3f neg %.3f | p pct1/50/99 %s" % r, flush=True)


if __name__ == "__main__":
    calibrate()

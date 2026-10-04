"""DivNet v3 inference for the attach (09-27): the same preprocessing as analysis/divnet_v3.py (xy max-pool 2, per-frame 50/99.5
percentile normalisation, lags -1..2, centre crop 16 x 48 x 48, gaussian marker sigma 1.5 z / 4 xy px), DivNet(cin=5, b=16).
Logit = mean over models of the mean over TTA views (1 = identity; 4 = identity + x/y/z flips, as in the out-of-fold evaluation)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import zarr

POOL, CZ, CYX = 2, 16, 48
LAGS = (-1, 0, 1, 2)


def _marker():
    z, y, x = np.meshgrid(np.arange(CZ) - CZ // 2, np.arange(CYX) - CYX // 2, np.arange(CYX) - CYX // 2, indexing="ij")
    return np.exp(-0.5 * ((z / 1.5) ** 2 + (y / 4.0) ** 2 + (x / 4.0) ** 2)).astype(np.float32)


MARK = _marker()


def norm_frames(zarr_path: Path) -> np.ndarray:
    arr = zarr.open_group(str(zarr_path), mode="r")["0"]
    T, Z, Y, X = arr.shape; out = np.zeros((T, Z, Y // POOL, X // POOL), np.float16)
    for t in range(T):
        x = torch.from_numpy(np.asarray(arr[t]).astype(np.float32))
        x = F.max_pool2d(x.unsqueeze(1), POOL).squeeze(1).numpy()
        lo, hi = np.percentile(x, 50.0), np.percentile(x, 99.5)
        out[t] = np.clip((x - lo) / (hi - lo + 1e-6), -0.5, 6.0)
    return out


def crops(frames: np.ndarray, t: int, zyx: np.ndarray) -> np.ndarray:
    T, Z, Y, X = frames.shape; n = len(zyx); x = np.zeros((n, 5, CZ, CYX, CYX), np.float32); x[:, 4] = MARK
    cz = np.rint(zyx[:, 0]).astype(int); cy = np.rint(zyx[:, 1] / POOL).astype(int); cx = np.rint(zyx[:, 2] / POOL).astype(int)
    for li, lag in enumerate(LAGS):
        fr = frames[int(np.clip(t + lag, 0, T - 1))]
        for i in range(n):
            z0, y0, x0 = cz[i] - CZ // 2, cy[i] - CYX // 2, cx[i] - CYX // 2
            zs, ys, xs = slice(max(z0, 0), min(z0 + CZ, Z)), slice(max(y0, 0), min(y0 + CYX, Y)), slice(max(x0, 0), min(x0 + CYX, X))
            if zs.stop > zs.start and ys.stop > ys.start and xs.stop > xs.start:
                x[i, li, zs.start - z0:zs.stop - z0, ys.start - y0:ys.stop - y0, xs.start - x0:xs.stop - x0] = fr[zs, ys, xs]
    return x


@torch.no_grad()
def crops_gpu(Fp, tt, zyx, T):
    """Same crops as crops(), gathered on the GPU from the zero-padded frame stack Fp (T, Z+CZ, Y+CYX, X+CYX); tt = per-item frame."""
    dev = Fp.device; n = len(zyx); tt = torch.as_tensor(np.asarray(tt, np.int64), device=dev)
    cz = torch.as_tensor(np.rint(zyx[:, 0]).astype(np.int64), device=dev); cy = torch.as_tensor(np.rint(zyx[:, 1] / POOL).astype(np.int64), device=dev)
    cx = torch.as_tensor(np.rint(zyx[:, 2] / POOL).astype(np.int64), device=dev)
    Zp, Yp, Xp = Fp.shape[1:]
    cz = cz.clamp(0, Zp - CZ); cy = cy.clamp(0, Yp - CYX); cx = cx.clamp(0, Xp - CYX)   # padded index of the crop start = centre (pad = half size)
    ar_z = torch.arange(CZ, device=dev); ar_y = torch.arange(CYX, device=dev)
    zz = (cz[:, None] + ar_z)[:, :, None, None]; yy = (cy[:, None] + ar_y)[:, None, :, None]; xx = (cx[:, None] + ar_y)[:, None, None, :]
    out = torch.empty((n, 5, CZ, CYX, CYX), dtype=torch.float32, device=dev)
    for li, lag in enumerate(LAGS):
        out[:, li] = Fp[(tt + lag).clamp(0, T - 1)[:, None, None, None], zz, yy, xx].float()
    out[:, 4] = torch.from_numpy(MARK).to(dev)
    return out


@torch.no_grad()
def logits_gpu(models, frames, cand_tp, Pv, device, tta=1, bs=256) -> dict:
    """logits() with GPU crop gathering in fixed-size batches across frames, and fp16 on pre-Ampere GPUs (T4: bf16 would be EMULATED)."""
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.get_device_capability(device)[0] >= 8) else torch.float16
    T = frames.shape[0]
    Fp = torch.nn.functional.pad(torch.from_numpy(frames).to(device), (CYX // 2, CYX // 2, CYX // 2, CYX // 2, CZ // 2, CZ // 2))
    d = cand_tp.sort("t"); ts = d["t"].to_numpy(); ps = d["p"].to_numpy(); out = {}
    for j in range(0, len(ps), bs):
        xb = crops_gpu(Fp, ts[j:j + bs], Pv[ps[j:j + bs]], T); acc = torch.zeros(xb.shape[0], device=device)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            for m in models:
                lg = m(xb).float()
                if tta == 4:
                    lg = (lg + m(xb.flip(-1)).float() + m(xb.flip(-2)).float() + m(xb.flip(-3)).float()) / 4
                acc += lg / len(models)
        out.update(zip(ps[j:j + bs].tolist(), acc.cpu().numpy().tolist()))
    del Fp
    return out


@torch.no_grad()
def logits(models, frames, cand_tp, Pv, device, tta=1, bs=256) -> dict:
    """cand_tp: polars frame with columns t, p (unique); Pv: node voxel coords. Returns {p: logit}."""
    out = {}; use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.get_device_capability(device)[0] >= 8) else torch.float16   # T4: bf16 only emulated
    for (t,), g in cand_tp.sort("t").group_by("t", maintain_order=True):
        ps = g["p"].to_numpy(); X = crops(frames, int(t), Pv[ps])
        for j in range(0, len(ps), bs):
            xb = torch.from_numpy(X[j:j + bs]).to(device); acc = torch.zeros(xb.shape[0], device=device)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                for m in models:
                    lg = m(xb).float()
                    if tta == 4:
                        lg = (lg + m(xb.flip(-1)).float() + m(xb.flip(-2)).float() + m(xb.flip(-3)).float()) / 4
                    acc += lg / len(models)
            out.update(zip(ps[j:j + bs].tolist(), acc.cpu().numpy().tolist()))
    return out


# ---------------------------------------------------------------------------------------------- TripletNet (09-28)
def _child_marker(off, dev):
    """off (b, 2, 3) child offsets from the parent in crop units (z slices, pooled xy px) -> (b, 1, CZ, CYX, CYX); the parent is the crop centre."""
    zz = torch.arange(CZ, dtype=torch.float32, device=dev); yy = torch.arange(CYX, dtype=torch.float32, device=dev)
    pos = off + torch.tensor([CZ // 2, CYX // 2, CYX // 2], dtype=torch.float32, device=dev)
    g = 0
    for k in range(2):
        dz = (zz[None, :] - pos[:, k, 0:1]) / 1.5; dy = (yy[None, :] - pos[:, k, 1:2]) / 3.0; dx = (yy[None, :] - pos[:, k, 2:3]) / 3.0
        g = g + torch.exp(-0.5 * (dz[:, :, None, None] ** 2 + dy[:, None, :, None] ** 2 + dx[:, None, None, :] ** 2))
    return g[:, None]


@torch.no_grad()
def tri_logits_gpu(models, frames, rows, Pv, device, tta=4, bs=256) -> list:
    """rows: polars frame with t, p, c1, c (node indices into Pv). TripletNet logit per row (same order): crop at the parent (frame t),
    4 lags + parent marker + child marker at c1 and c (frame t+1)."""
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.get_device_capability(device)[0] >= 8) else torch.float16
    T = frames.shape[0]
    Fp = torch.nn.functional.pad(torch.from_numpy(frames).to(device), (CYX // 2, CYX // 2, CYX // 2, CYX // 2, CZ // 2, CZ // 2))
    ts = rows["t"].to_numpy(); ps = rows["p"].to_numpy(); c1 = rows["c1"].to_numpy(); cc = rows["c"].to_numpy(); out = []
    for j in range(0, len(ps), bs):
        P = Pv[ps[j:j + bs]]; A = Pv[c1[j:j + bs]]; B = Pv[cc[j:j + bs]]
        off = np.stack([np.stack([A[:, 0] - P[:, 0], (A[:, 1] - P[:, 1]) / POOL, (A[:, 2] - P[:, 2]) / POOL], 1),
                        np.stack([B[:, 0] - P[:, 0], (B[:, 1] - P[:, 1]) / POOL, (B[:, 2] - P[:, 2]) / POOL], 1)], 1).astype(np.float32)
        xb = torch.cat([crops_gpu(Fp, ts[j:j + bs], P, T), _child_marker(torch.from_numpy(off).to(device), device)], 1)
        acc = torch.zeros(xb.shape[0], device=device)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            for m in models:
                lg = m(xb).float()
                if tta == 4:
                    lg = (lg + m(xb.flip(-1)).float() + m(xb.flip(-2)).float() + m(xb.flip(-3)).float()) / 4
                acc += lg / len(models)
        out.extend(acc.cpu().numpy().tolist())
    del Fp
    return out

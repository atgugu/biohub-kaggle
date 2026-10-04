"""Post-hoc division attachment for a submission CSV (self-contained: numpy, scipy, polars, torch, lightgbm).

For every dataset in the CSV: generate second-daughter candidates (orphan or rewire) around single-child
parents, score them with geometry + the reconstructed DivNet, attach the top-K per video above a threshold
(dedupe parent and daughter; a rewire removes the daughter's old incoming edge), rewrite the CSV.

Usage (local or Kaggle):
  python apply_divisions.py --csv submission.csv --zarr-dir <dir with <dataset>.zarr> --divnet best_overall.pt
                            --ranker ranker_all.txt [--k 10] [--thr 0.5] [--out submission.csv]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import zarr
import numcodecs.blosc; numcodecs.blosc.use_threads = False   # blosc threads race under load -> "decompression: -1"
from scipy.spatial import cKDTree

SC = np.array([1.625, 0.40625, 0.40625])
R_PARENT, R_SISTER = 13.0, 17.0
GEOM = ["d_pc", "d_pc1", "d_sis", "sym", "cosang", "mid", "dz_sis", "c_orphan", "d_qc", "q_children", "q_alt",
        "len_p_back", "len_c_fwd", "len_c1_fwd", "len_q_back", "speed_p", "c1_vs_vel", "c_vs_vel", "density",
        "n_frame", "n_next", "n_cand_same_p", "rank_d_pc"]
FEATS = GEOM + ["divnet_logit"]
LAGS = (-1, 0, 1, 2); CZ, CYX, POOL = 16, 32, 4


# ----------------------------------------------------------------------------- DivNet (giorgosi v2, reconstructed)
class _Block(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.block = nn.Sequential(nn.Conv3d(i, o, 3, padding=1, bias=False), nn.InstanceNorm3d(o, affine=True), nn.ReLU(inplace=True),
                                   nn.Conv3d(o, o, 3, padding=1, bias=False), nn.InstanceNorm3d(o, affine=True), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.block(x)


class DivNet(nn.Module):
    def __init__(self, cin=5, b=16):
        super().__init__()
        self.enc1, self.enc2, self.enc3 = _Block(cin, b), _Block(b, 2 * b), _Block(2 * b, 4 * b)
        self.bottleneck = _Block(4 * b, 8 * b)
        self.up3, self.dec3 = nn.ConvTranspose3d(8 * b, 4 * b, 2, 2), _Block(8 * b, 4 * b)
        self.up2, self.dec2 = nn.ConvTranspose3d(4 * b, 2 * b, 2, 2), _Block(4 * b, 2 * b)
        self.up1, self.dec1 = nn.ConvTranspose3d(2 * b, b, 2, 2), _Block(2 * b, b)
        self.head = nn.Linear(b, 1)

    def forward(self, x):
        e1 = self.enc1(x); e2 = self.enc2(F.max_pool3d(e1, 2)); e3 = self.enc3(F.max_pool3d(e2, 2))
        bt = self.bottleneck(F.max_pool3d(e3, 2))
        d3 = self.dec3(torch.cat([self.up3(bt), e3], 1)); d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.head(d1.mean(dim=(2, 3, 4))).squeeze(-1)


def load_divnet(path, device):
    m = DivNet(); sd = torch.load(path, map_location="cpu", weights_only=False)
    m.load_state_dict(sd["model_state"] if "model_state" in sd else sd, strict=True)
    return m.to(device).eval()


def _marker():
    z, y, x = np.meshgrid(np.arange(CZ) - CZ // 2, np.arange(CYX) - CYX // 2, np.arange(CYX) - CYX // 2, indexing="ij")
    return np.exp(-0.5 * ((z / 1.5) ** 2 + (y / 2.0) ** 2 + (x / 2.0) ** 2)).astype(np.float32)


MARK = _marker()


def norm_frames(zarr_path: Path) -> np.ndarray:
    arr = zarr.open_group(str(zarr_path), mode="r")["0"]
    out = np.zeros((arr.shape[0], arr.shape[1], arr.shape[2] // POOL, arr.shape[3] // POOL), np.float32)
    for t in range(arr.shape[0]):
        x = torch.from_numpy(np.asarray(arr[t]).astype(np.float32))
        x = F.max_pool2d(x.unsqueeze(1), POOL).squeeze(1).numpy()
        lo, hi = np.percentile(x, 50.0), np.percentile(x, 99.5)
        out[t] = np.clip((x - lo) / (hi - lo + 1e-6), -0.5, 6.0)
    return out


def crops(frames, t, zyx):
    T, Z, Y, X = frames.shape
    n = len(zyx); x = np.zeros((n, 5, CZ, CYX, CYX), np.float32); x[:, 4] = MARK
    cz = np.rint(zyx[:, 0]).astype(int); cy = np.rint(zyx[:, 1] / POOL).astype(int); cx = np.rint(zyx[:, 2] / POOL).astype(int)
    for li, lag in enumerate(LAGS):
        fr = frames[int(np.clip(t + lag, 0, T - 1))]
        for i in range(n):
            z0, y0, x0 = cz[i] - CZ // 2, cy[i] - CYX // 2, cx[i] - CYX // 2
            zs, ys, xs = slice(max(z0, 0), min(z0 + CZ, Z)), slice(max(y0, 0), min(y0 + CYX, Y)), slice(max(x0, 0), min(x0 + CYX, X))
            x[i, li, zs.start - z0:zs.stop - z0, ys.start - y0:ys.stop - y0, xs.start - x0:xs.stop - x0] = fr[zs, ys, xs]
    return x


# ----------------------------------------------------------------------------- candidates
def track_len(start, nxt, cap=30):
    n, cur = 0, start
    while cur in nxt and n < cap:
        cur = nxt[cur]; n += 1
    return n


def candidates(T, P, S, D):
    """T (N,) frame, P (N,3) um, S/D edge index arrays -> polars DataFrame of candidates with GEOM features."""
    N = len(T); children = [[] for _ in range(N)]; parent = np.full(N, -1)
    for a, b in zip(S, D):
        children[a].append(b); parent[b] = a
    nxt = {a: c[0] for a, c in enumerate(children) if len(c) == 1}; prv = {b: a for b, a in enumerate(parent) if a >= 0}
    byt = {t: np.flatnonzero(T == t) for t in np.unique(T)}; trees = {t: cKDTree(P[i]) for t, i in byt.items()}
    rows = []
    for t, idx in byt.items():
        if t + 1 not in byt:
            continue
        nxt_idx = byt[t + 1]; tr1 = trees[t + 1]
        dens = trees[t].query_ball_point(P[idx], 10.0, return_length=True)
        for k, p in enumerate(idx):
            if len(children[p]) != 1:
                continue
            c1 = children[p][0]
            for j in tr1.query_ball_point(P[p], R_PARENT):
                c = nxt_idx[j]
                if c == c1:
                    continue
                d_pc, d_pc1, d_sis = np.linalg.norm(P[p] - P[c]), np.linalg.norm(P[p] - P[c1]), np.linalg.norm(P[c1] - P[c])
                if d_sis > R_SISTER:
                    continue
                v1, v2 = P[c1] - P[p], P[c] - P[p]
                q = parent[c]; pp = prv.get(p); vel = P[p] - P[pp] if pp is not None else np.zeros(3)
                rows.append(dict(
                    t=int(t), p=int(p), c1=int(c1), c=int(c), q=int(q),
                    d_pc=d_pc, d_pc1=d_pc1, d_sis=d_sis, sym=abs(d_pc - d_pc1) / (0.5 * (d_pc + d_pc1) + 1e-9),
                    cosang=float(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)), mid=np.linalg.norm((P[c1] + P[c]) / 2 - P[p]),
                    dz_sis=abs(P[c1][0] - P[c][0]), c_orphan=int(q < 0), d_qc=float(np.linalg.norm(P[q] - P[c])) if q >= 0 else -1.0,
                    q_children=len(children[q]) if q >= 0 else 0,
                    q_alt=(float(tr1.query(P[q], k=2)[0][1]) if q >= 0 and len(nxt_idx) > 1 else -1.0),
                    len_p_back=track_len(p, prv), len_c_fwd=track_len(c, nxt), len_c1_fwd=track_len(c1, nxt),
                    len_q_back=track_len(q, prv) if q >= 0 else 0,
                    speed_p=float(np.linalg.norm(vel)), c1_vs_vel=float(np.linalg.norm(P[c1] - (P[p] + vel))),
                    c_vs_vel=float(np.linalg.norm(P[c] - (P[p] + vel))),
                    density=int(dens[k]), n_frame=len(idx), n_next=len(nxt_idx)))
    if not rows:
        return None
    return pl.DataFrame(rows).with_columns(pl.len().over("p").alias("n_cand_same_p"), pl.col("d_pc").rank().over("p").alias("rank_d_pc"))


# ----------------------------------------------------------------------------- per-dataset apply
def process(grp: pl.DataFrame, zarr_path: Path, divnet, ranker, device, k: int, thr: float, log) -> pl.DataFrame:
    nodes = grp.filter(pl.col("row_type") == "node"); edges = grp.filter(pl.col("row_type") == "edge")
    nid = nodes["node_id"].to_numpy(); T = nodes["t"].to_numpy(); Pv = nodes.select("z", "y", "x").to_numpy().astype(np.float64)
    P = Pv * SC; pos = {int(n): i for i, n in enumerate(nid)}
    S = np.array([pos[a] for a in edges["source_id"].to_numpy()], dtype=np.int64); D = np.array([pos[b] for b in edges["target_id"].to_numpy()], dtype=np.int64)
    t0 = time.time(); cand = candidates(T, P, S, D)
    if cand is None:
        return grp
    frames = norm_frames(zarr_path)
    par = cand.select("t", "p").unique().sort("t"); logit = {}
    with torch.no_grad():
        for (t,), g in par.group_by("t", maintain_order=True):
            ps = g["p"].to_numpy(); X = crops(frames, int(t), Pv[ps])
            for j in range(0, len(ps), 256):
                lg = divnet(torch.from_numpy(X[j:j + 256]).to(device)).float().cpu().numpy()
                logit.update(zip(ps[j:j + 256].tolist(), lg.tolist()))
    cand = cand.with_columns(pl.Series("divnet_logit", [logit[p] for p in cand["p"].to_list()], dtype=pl.Float64))
    s = ranker.predict(cand.select(FEATS).to_numpy().astype(np.float32))
    cand = cand.with_columns(pl.Series("score", s)).sort("score", descending=True)
    used_p, used_c, add, rm = set(), set(), [], set()
    for r in cand.filter(pl.col("score") >= thr).iter_rows(named=True):
        if len(add) >= k:
            break
        if r["p"] in used_p or r["c"] in used_c:
            continue
        used_p.add(r["p"]); used_c.add(r["c"]); add.append((int(nid[r["p"]]), int(nid[r["c"]])))
        if r["q"] >= 0:
            rm.add((int(nid[r["q"]]), int(nid[r["c"]])))
    log(f"  {cand.height} candidates, {len(add)} attached ({sum(1 for a in add if a not in rm)} orphan / {len(rm)} rewire), {time.time() - t0:.0f}s")
    if not add:
        return grp
    keep = [(a, b) not in rm for a, b in zip(edges["source_id"].to_list(), edges["target_id"].to_list())]
    new_e = edges.head(len(add)).with_columns(pl.Series("source_id", [a for a, _ in add]), pl.Series("target_id", [b for _, b in add]))
    return pl.concat([nodes, edges.filter(pl.Series(keep)), new_e])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True); ap.add_argument("--zarr-dir", required=True); ap.add_argument("--divnet", required=True)
    ap.add_argument("--ranker", required=True); ap.add_argument("--k", type=int, default=10); ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--out", default=None)
    ap.add_argument("--deadline-s", type=float, default=0.0, help="stop attaching (keep remaining videos unchanged) after this many seconds")
    a = ap.parse_args(); out = a.out or a.csv; t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    divnet = load_divnet(a.divnet, device); ranker = lgb.Booster(model_file=a.ranker)
    df = pl.read_csv(a.csv); cols = df.columns; parts = []
    for (v,), grp in df.group_by("dataset", maintain_order=True):
        if a.deadline_s > 0 and time.time() - t_start > a.deadline_s:
            print(f"[divisions] {v}: SKIPPED (deadline {a.deadline_s:.0f}s reached)", flush=True); parts.append(grp); continue
        print(f"[divisions] {v}", flush=True)
        try:
            parts.append(process(grp, Path(a.zarr_dir) / f"{v}.zarr", divnet, ranker, device, a.k, a.thr, lambda m: print(m, flush=True)))
        except Exception as exc:  # never lose a dataset: keep the original rows
            print(f"  FAILED ({type(exc).__name__}: {exc}); keeping original graph", flush=True); parts.append(grp)
    res = pl.concat([p.select(cols) for p in parts]).drop("id").with_row_index("id")
    res.write_csv(out); print(f"[divisions] wrote {out}: {res.height} rows", flush=True)


if __name__ == "__main__":
    main()

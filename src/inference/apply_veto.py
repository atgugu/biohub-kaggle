"""Fork veto for a submission CSV: score every existing fork (node with 2 children) with geometry + honest DivNet
(mean of fold models) -> LightGBM P(true division); remove the edge to the farther daughter when P < thr.
Self-contained with apply_divisions.py (DivNet, crops, norm_frames) shipped alongside.

Usage: python apply_veto.py --csv sub.csv --zarr-dir DIR --veto veto_all.txt --feats veto_all.feats.json
                            --divnet-hardneg f0.pt,f1.pt,... [--thr 0.05] [--deadline-s S] [--out sub.csv]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
import torch
from scipy.spatial import cKDTree

import apply_divisions as v1

SC = v1.SC


def fork_rows(nodes: pl.DataFrame, edges: pl.DataFrame):
    nid = nodes["node_id"].to_numpy(); T = nodes["t"].to_numpy(); Pv = nodes.select("z", "y", "x").to_numpy().astype(np.float64); P = Pv * SC
    pos = {int(n): i for i, n in enumerate(nid)}
    S = [pos[a] for a in edges["source_id"].to_numpy()]; D = [pos[b] for b in edges["target_id"].to_numpy()]
    children = {}; parent = {}
    for a, b in zip(S, D):
        children.setdefault(a, []).append(b); parent[b] = a
    nxt = {a: c[0] for a, c in children.items() if len(c) == 1}
    trees = {t: cKDTree(P[T == t]) for t in np.unique(T)}

    def back(x):
        n = 0
        while x in parent and n < 30:
            x = parent[x]; n += 1
        return n

    def fwd(x):
        n = 0
        while x in nxt and n < 30:
            x = nxt[x]; n += 1
        return n
    rows = []
    for p, ch in children.items():
        if len(ch) != 2:
            continue
        c1, c2 = ch; d1, d2 = np.linalg.norm(P[c1] - P[p]), np.linalg.norm(P[c2] - P[p]); ds = np.linalg.norm(P[c1] - P[c2])
        v1_, v2_ = P[c1] - P[p], P[c2] - P[p]; pp = parent.get(p); sp = float(np.linalg.norm(P[p] - P[pp])) if pp is not None else 0.0
        rows.append(dict(p=int(p), c1=int(c1), c2=int(c2), t=int(T[p]), nid_p=int(nid[p]), nid_far=int(nid[c2] if d2 >= d1 else nid[c1]),
                         d1=d1, d2=d2, d_sis=ds, sym=abs(d1 - d2) / (0.5 * (d1 + d2) + 1e-9), cosang=float(v1_ @ v2_ / (np.linalg.norm(v1_) * np.linalg.norm(v2_) + 1e-9)),
                         speed_p=sp, len_p_back=back(p), len_c1_fwd=fwd(c1), len_c2_fwd=fwd(c2),
                         density=int(trees[T[p]].query_ball_point(P[p], 10.0, return_length=True)), n_frame=int((T == T[p]).sum()), dz_sis=abs(P[c1][0] - P[c2][0])))
    return rows, Pv


def process(grp, zarr_path, feats, veto, models, device, thr, log, tri_models=(), tri_gamma=1.0, tri_tta=4):
    nodes = grp.filter(pl.col("row_type") == "node"); edges = grp.filter(pl.col("row_type") == "edge")
    t0 = time.time(); rows, Pv = fork_rows(nodes, edges)
    if not rows:
        log("  no forks"); return grp
    frames = v1.norm_frames(zarr_path); d = pl.DataFrame(rows)
    logit = {}
    with torch.no_grad():
        for (t,), g in d.group_by("t", maintain_order=True):
            ps = g["p"].to_numpy(); X = torch.from_numpy(v1.crops(frames, int(t), Pv[ps])).to(device)
            lg = np.mean([m(X).float().cpu().numpy() for m in models], axis=0); logit.update(zip(ps.tolist(), lg.tolist()))
    d = d.with_columns(pl.Series("divnet_hn", [logit[p] for p in d["p"].to_list()], dtype=pl.Float64))
    p_tp = veto.predict(d.select(feats).to_numpy().astype(np.float32))
    if tri_models:   # 09-28 TripletNet-aware veto: P = sigmoid(logit(p_veto) + gamma * TripletNet(parent -> both children))
        import divnet_v3_infer as v3i
        f3 = v3i.norm_frames(zarr_path)
        tl = np.asarray(v3i.tri_logits_gpu(tri_models, f3, d.select("t", "p", pl.col("c1"), pl.col("c2").alias("c")), Pv, device, tta=tri_tta)); del f3
        pc = np.clip(p_tp, 1e-6, 1 - 1e-6); p_tp = 1.0 / (1.0 + np.exp(-(np.log(pc / (1 - pc)) + tri_gamma * tl)))
    drop = d.filter(pl.Series(p_tp < thr))
    rm = set(zip(drop["nid_p"].to_list(), drop["nid_far"].to_list()))
    log(f"  {d.height} forks, {len(rm)} vetoed (P<{thr}), {time.time() - t0:.0f}s")
    if not rm:
        return grp
    keep = [(a, b) not in rm for a, b in zip(edges["source_id"].to_list(), edges["target_id"].to_list())]
    return pl.concat([nodes, edges.filter(pl.Series(keep))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True); ap.add_argument("--zarr-dir", required=True); ap.add_argument("--veto", required=True); ap.add_argument("--feats", required=True)
    ap.add_argument("--divnet-hardneg", required=True); ap.add_argument("--thr", type=float, default=0.05); ap.add_argument("--deadline-s", type=float, default=0.0); ap.add_argument("--out", default=None)
    ap.add_argument("--trinet", default=None, help="09-28: TripletNet .pt list -> TripletNet-aware veto"); ap.add_argument("--tri-gamma", type=float, default=1.0); ap.add_argument("--tri-tta", type=int, default=4)
    ap.add_argument("--fallback-csv", default=None, help="CSV whose rows replace a video's graph when the veto fails or is skipped for it (e.g. the pre-attach submission)")
    a = ap.parse_args(); out = a.out or a.csv; t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feats = json.load(open(a.feats)); veto = lgb.Booster(model_file=a.veto); models = []
    for f in a.divnet_hardneg.split(","):
        m = v1.DivNet(); m.load_state_dict(torch.load(f, map_location="cpu")); models.append(m.to(device).eval())
    trm = []
    if a.trinet:
        for f in a.trinet.split(","):
            m = v1.DivNet(cin=6); m.load_state_dict(torch.load(f, map_location="cpu")); trm.append(m.to(device).eval())
    print(f"[veto] feats={feats} thr={a.thr} models={len(models)} trinet={len(trm)} gamma={a.tri_gamma}", flush=True)
    df = pl.read_csv(a.csv); cols = df.columns; parts = []
    fb = pl.read_csv(a.fallback_csv) if a.fallback_csv else None
    def fallback(v, grp):
        if fb is None:
            return grp
        g = fb.filter(pl.col("dataset") == v); return g.select(cols) if g.height else grp
    for (v,), grp in df.group_by("dataset", maintain_order=True):
        if a.deadline_s > 0 and time.time() - t_start > a.deadline_s:
            print(f"[veto] {v}: SKIPPED (deadline){' -> fallback graph' if fb is not None else ''}", flush=True); parts.append(fallback(v, grp)); continue
        print(f"[veto] {v}", flush=True)
        try:
            parts.append(process(grp, Path(a.zarr_dir) / f"{v}.zarr", feats, veto, models, device, a.thr, lambda m: print(m, flush=True), trm, a.tri_gamma, a.tri_tta))
        except Exception as exc:
            print(f"  FAILED ({type(exc).__name__}: {exc}); {'fallback graph' if fb is not None else 'keeping original graph'}", flush=True); parts.append(fallback(v, grp))
    res = pl.concat([p.select(cols) for p in parts]).drop("id").with_row_index("id")
    res.write_csv(out); print(f"[veto] wrote {out}: {res.height} rows", flush=True)


if __name__ == "__main__":
    main()

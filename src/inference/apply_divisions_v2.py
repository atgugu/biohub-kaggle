"""Division attach v2 for a submission CSV: geometry + linker evidence (+ DivNet features) -> LightGBM ranker -> attach top-K.

Self-contained (numpy, scipy, polars, torch, lightgbm). Feature list comes from the ranker's sidecar JSON.
Linker evidence: per-video npz written by predict_cache.py (every candidate edge logit within a radius, softmax-over-
sources normaliser), stack nodes matched to cache nodes at <= 2.5 um in the same frame.
DivNet features: 'divnet_logit' = giorgosi model (one .pt); 'divnet_hardneg_logit' = mean of fold models (--divnet-hardneg a,b,c,...).

Usage:
  python apply_divisions_v2.py --csv sub.csv --zarr-dir DIR --ranker ranker.txt --feats ranker.feats.json
        [--linker-cache DIR] [--divnet best_overall.pt] [--divnet-hardneg f0.pt,f1.pt,...] [--k 10] [--thr 0.0]
        [--deadline-s S] [--out sub.csv]
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

import apply_divisions as v1   # same directory (shipped together)
import divnet_v3_infer as v3i   # 09-27: DivNet v3 feature (same directory)

SC = v1.SC; GATE = 2.5
LK_COLS = ["lk_pc", "lk_qc", "lk_pc1", "lk_c_best", "lk_c_margin", "lk_matched"]


def linker_feats(npz_path: Path, T, Pv, cand: pl.DataFrame) -> pl.DataFrame:
    """Return cand with LK_COLS appended (zeros/-1 when no cache)."""
    if npz_path is None or not npz_path.exists():
        return cand.with_columns([pl.lit(0.0).alias(c) for c in LK_COLS[:5]] + [pl.lit(0).alias("lk_matched")])
    z = np.load(npz_path); cn = z["nodes"]; ct = cn[:, 0]; cpos = cn[:, 1:].astype(np.float64) * SC
    P = Pv * SC; s2c = np.full(len(T), -1, np.int64)
    for t in np.unique(T):
        ci = np.flatnonzero(ct == t); si = np.flatnonzero(T == t)
        if len(ci) == 0:
            continue
        d, j = cKDTree(cpos[ci]).query(P[si]); ok = d <= GATE; s2c[si[ok]] = ci[j[ok]]
    prob = np.exp(z["e_logit"] - z["lse_src"][z["e_tgt"]]).astype(np.float32); es, et = z["e_src"], z["e_tgt"]
    pair = dict(zip(zip(es.tolist(), et.tolist()), prob.tolist()))
    order = np.lexsort((-prob, et)); et_o, pr_o = et[order], prob[order]; first = np.r_[True, et_o[1:] != et_o[:-1]]
    best = dict(zip(et_o[first].tolist(), pr_o[first].tolist())); second = {}
    for i in np.flatnonzero(~first):
        t_ = int(et_o[i])
        if t_ not in second:
            second[t_] = float(pr_o[i])
    rows = []
    for p, c1, c, q in cand.select("p", "c1", "c", "q").iter_rows():
        cp, cc1, cc, cq = s2c[p], s2c[c1], s2c[c], (s2c[q] if q >= 0 else -1)
        lk_pc = pair.get((cp, cc), 0.0) if cp >= 0 and cc >= 0 else 0.0
        lk_qc = (pair.get((cq, cc), 0.0) if cq >= 0 and cc >= 0 else 0.0) if q >= 0 else -1.0
        lk_pc1 = pair.get((cp, cc1), 0.0) if cp >= 0 and cc1 >= 0 else 0.0
        b = best.get(int(cc), 0.0) if cc >= 0 else 0.0
        other = (second.get(int(cc), 0.0) if abs(b - lk_pc) < 1e-9 else b) if cc >= 0 else 0.0
        rows.append((lk_pc, lk_qc, lk_pc1, b, lk_pc - other, int(cp >= 0 and cc >= 0 and cc1 >= 0)))
    lk = pl.DataFrame(rows, schema=LK_COLS, orient="row")
    return pl.concat([cand, lk], how="horizontal")


def divnet_logits(models, frames, cand, Pv, device):
    par = cand.select("t", "p").unique().sort("t"); out = {}
    with torch.no_grad():
        for (t,), g in par.group_by("t", maintain_order=True):
            ps = g["p"].to_numpy(); X = v1.crops(frames, int(t), Pv[ps])
            for j in range(0, len(ps), 256):
                xb = torch.from_numpy(X[j:j + 256]).to(device)
                lg = np.mean([m(xb).float().cpu().numpy() for m in models], axis=0)
                out.update(zip(ps[j:j + 256].tolist(), lg.tolist()))
    return out


def _load_member(path, feats):
    """09-28 wide ensemble: a LightGBM .txt (own sidecar <stem>.feats.json if present) or a logistic-regression .json."""
    p = Path(path)
    if p.suffix == ".json":
        return ("logit", json.load(open(p)), None)
    side = p.with_suffix(".feats.json")
    return ("lgb", lgb.Booster(model_file=str(p)), json.load(open(side)) if side.exists() else feats)


def _member_predict(r, cand, feats):
    kind, m, fs = r
    if kind == "logit":
        X = np.nan_to_num(cand.select(m["feats"]).to_numpy().astype(np.float64))
        z = ((X - np.array(m["mean"])) / np.array(m["scale"])) @ np.array(m["coef"]) + m["intercept"]
        return 1.0 / (1.0 + np.exp(-z))
    return m.predict(cand.select(fs).to_numpy().astype(np.float32))


def process(grp, zarr_path, npz_path, feats, ranker, dn_models, hn_models, device, k, thr, log, channel="all", v3_models=(), v3_tta=1, dnet_models=(), tri_models=(), tri_top=200, tri_alpha=1.0):
    nodes = grp.filter(pl.col("row_type") == "node"); edges = grp.filter(pl.col("row_type") == "edge")
    nid = nodes["node_id"].to_numpy(); T = nodes["t"].to_numpy(); Pv = nodes.select("z", "y", "x").to_numpy().astype(np.float64)
    pos = {int(n): i for i, n in enumerate(nid)}
    S = np.array([pos[a] for a in edges["source_id"].to_numpy()], dtype=np.int64); D = np.array([pos[b] for b in edges["target_id"].to_numpy()], dtype=np.int64)
    t0 = time.time(); cand = v1.candidates(T, Pv * SC, S, D)
    if cand is None:
        return grp
    if channel == "orphan":
        cand = cand.filter(pl.col("c_orphan") == 1)
    elif channel == "rewire":
        cand = cand.filter(pl.col("c_orphan") == 0)
    if cand.height == 0:
        return grp
    if any(c.startswith("lk_") for c in feats):
        cand = linker_feats(npz_path, T, Pv, cand)
    need_frames = ("divnet_logit" in feats) or ("divnet_hardneg_logit" in feats)
    frames = v1.norm_frames(zarr_path) if need_frames else None
    if "divnet_logit" in feats:
        lg = divnet_logits(dn_models, frames, cand, Pv, device); cand = cand.with_columns(pl.Series("divnet_logit", [lg[p] for p in cand["p"].to_list()], dtype=pl.Float64))
    if "divnet_hardneg_logit" in feats:
        lg = divnet_logits(hn_models, frames, cand, Pv, device); cand = cand.with_columns(pl.Series("divnet_hardneg_logit", [lg[p] for p in cand["p"].to_list()], dtype=pl.Float64))
    frames3 = v3i.norm_frames(zarr_path) if ("divnet_v3_logit" in feats or "dn_c" in feats or "dn_c1" in feats or tri_models) else None
    _lf = v3i.logits_gpu if device.type == "cuda" else v3i.logits
    if "divnet_v3_logit" in feats:
        lg = _lf(v3_models, frames3, cand.select("t", "p").unique(), Pv, device, tta=v3_tta)
        cand = cand.with_columns(pl.Series("divnet_v3_logit", [lg[p] for p in cand["p"].to_list()], dtype=pl.Float64))
    if "dn_c" in feats or "dn_c1" in feats:   # 09-27 DaughterNet: newborn-daughter logit of the proposed children at frame t+1
        nd = pl.concat([cand.select((pl.col("t") + 1).alias("t"), pl.col("c").alias("p")), cand.select((pl.col("t") + 1).alias("t"), pl.col("c1").alias("p"))]).unique()
        lg = _lf(dnet_models, frames3, nd, Pv, device, tta=v3_tta)
        cand = cand.with_columns(pl.Series("dn_c", [lg[c] for c in cand["c"].to_list()], dtype=pl.Float64), pl.Series("dn_c1", [lg[c] for c in cand["c1"].to_list()], dtype=pl.Float64))
    s = np.mean([_member_predict(r, cand, feats) for r in ranker], axis=0) if isinstance(ranker, list) else ranker.predict(cand.select(feats).to_numpy().astype(np.float32))   # 09-27/28: averaged (wide) ranker members
    if tri_models:   # 09-28 TripletNet blend: top-N rows by the stage-1 score get sigmoid(logit(s) + alpha * tri_logit); all other rows 0
        order = np.argsort(-s, kind="stable")[:tri_top]
        tl = np.asarray(v3i.tri_logits_gpu(tri_models, frames3, cand[order.tolist()].select("t", "p", "c1", "c"), Pv, device, tta=v3_tta))
        sc = np.clip(s[order], 1e-6, 1 - 1e-6); s2 = np.zeros_like(s); s2[order] = 1.0 / (1.0 + np.exp(-(np.log(sc / (1 - sc)) + tri_alpha * tl))); s = s2
    del frames3
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
    log(f"  {cand.height} candidates (linker matched {cand['lk_matched'].mean() if 'lk_matched' in cand.columns else float('nan'):.2f}), "
        f"{len(add)} attached ({len(rm)} rewires), {time.time() - t0:.0f}s")
    if not add:
        return grp
    keep = [(a, b) not in rm for a, b in zip(edges["source_id"].to_list(), edges["target_id"].to_list())]
    new_e = edges.head(len(add)).with_columns(pl.Series("source_id", [a for a, _ in add]), pl.Series("target_id", [b for _, b in add]))
    return pl.concat([nodes, edges.filter(pl.Series(keep)), new_e])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True); ap.add_argument("--zarr-dir", required=True); ap.add_argument("--ranker", required=True); ap.add_argument("--feats", required=True)
    ap.add_argument("--linker-cache", default=None); ap.add_argument("--divnet", default=None); ap.add_argument("--divnet-hardneg", default=None); ap.add_argument("--divnet-v3", default=None); ap.add_argument("--divnet-v3-tta", type=int, default=1); ap.add_argument("--dnet", default=None); ap.add_argument("--trinet", default=None); ap.add_argument("--tri-top", type=int, default=200); ap.add_argument("--tri-alpha", type=float, default=1.0)
    ap.add_argument("--k", type=int, default=10); ap.add_argument("--thr", type=float, default=0.0); ap.add_argument("--channel", default="all", choices=["all", "orphan", "rewire"]); ap.add_argument("--deadline-s", type=float, default=0.0); ap.add_argument("--out", default=None)
    ap.add_argument("--part-dir", default=None, help="09-28: write every processed video's rows to <dir>/<video>.parquet as soon as it is done (a killed shard keeps its finished videos)")
    ap.add_argument("--shard", default="0/1", help="i/n: process only the datasets whose order index %% n == i; pass the others through unchanged (09-27: 2-GPU attach)")
    a = ap.parse_args(); out = a.out or a.csv; t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feats = json.load(open(a.feats)); rk = [_load_member(f, feats) for f in a.ranker.split(",")]; ranker = rk
    v3m = []
    if a.divnet_v3 and "divnet_v3_logit" in feats:
        for f in a.divnet_v3.split(","):
            m = v1.DivNet(); m.load_state_dict(torch.load(f, map_location="cpu")); v3m.append(m.to(device).eval())
    dnm = []
    if a.dnet and ("dn_c" in feats or "dn_c1" in feats):
        for f in a.dnet.split(","):
            m = v1.DivNet(); m.load_state_dict(torch.load(f, map_location="cpu")); dnm.append(m.to(device).eval())
    trm = []
    if a.trinet:
        for f in a.trinet.split(","):
            m = v1.DivNet(cin=6); m.load_state_dict(torch.load(f, map_location="cpu")); trm.append(m.to(device).eval())
    if ("dn_c" in feats or "dn_c1" in feats) and not dnm:
        raise SystemExit("ranker needs dn_c/dn_c1 but no --dnet models were given")
    dn = [v1.load_divnet(a.divnet, device)] if a.divnet and "divnet_logit" in feats else []
    hn = []
    if a.divnet_hardneg and "divnet_hardneg_logit" in feats:
        for f in a.divnet_hardneg.split(","):
            m = v1.DivNet(); m.load_state_dict(torch.load(f, map_location="cpu")); hn.append(m.to(device).eval())
    print(f"[divisions v2] feats={feats} k={a.k} thr={a.thr} channel={a.channel} rankers={len(rk)} divnet_v3={len(v3m)} (tta {a.divnet_v3_tta}) trinet={len(trm) if a.trinet else 0} top {a.tri_top} alpha {a.tri_alpha}", flush=True)
    if "divnet_v3_logit" in feats and not v3m:
        raise SystemExit("ranker needs divnet_v3_logit but no --divnet-v3 models were given")
    df = pl.read_csv(a.csv); cols = df.columns; parts = []; si, sn = (int(x) for x in a.shard.split("/"))
    for di, ((v,), grp) in enumerate(df.group_by("dataset", maintain_order=True)):
        if di % sn != si:
            parts.append(grp); continue
        if a.deadline_s > 0 and time.time() - t_start > a.deadline_s:
            print(f"[divisions v2] {v}: SKIPPED (deadline)", flush=True); parts.append(grp); continue
        print(f"[divisions v2] {v}", flush=True)
        try:
            npz = Path(a.linker_cache) / f"{v}.npz" if a.linker_cache else None
            parts.append(process(grp, Path(a.zarr_dir) / f"{v}.zarr", npz, feats, ranker, dn, hn, device, a.k, a.thr, lambda m: print(m, flush=True), a.channel, v3m, a.divnet_v3_tta, dnm, trm, a.tri_top, a.tri_alpha))
            if a.part_dir:
                import os as _os
                Path(a.part_dir).mkdir(parents=True, exist_ok=True); _tmp = Path(a.part_dir) / f".{v}.parquet.tmp"
                parts[-1].select(cols).write_parquet(_tmp); _os.replace(_tmp, Path(a.part_dir) / f"{v}.parquet")   # atomic: a kill mid-write leaves no partial part file
        except Exception as exc:
            print(f"  FAILED ({type(exc).__name__}: {exc}); keeping original graph", flush=True); parts.append(grp)
    res = pl.concat([p.select(cols) for p in parts]).drop("id").with_row_index("id")
    res.write_csv(out); print(f"[divisions v2] wrote {out}: {res.height} rows", flush=True)


if __name__ == "__main__":
    main()

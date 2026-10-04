"""Linker-evidence features for division candidates, from the all-video pilkwang cache (every candidate edge logit).

For each candidate (p, c1, c, q) of a video: stack nodes are mapped to cache nodes (nearest within 2.5 um, same frame);
prob(a->b) = exp(logit - logsumexp_over_sources[b]) (the pack's softmax over sources).
Columns: lk_pc (p->c), lk_qc (q->c, -1 if orphan), lk_pc1 (p->c1), lk_c_best (best source prob into c),
lk_c_margin (lk_pc - best OTHER source into c), lk_matched (all of p, c1, c mapped).
Env: BIOHUB_DIVROOT (cands_all/ in, linker_feats/ out), BIOHUB_CACHE (stack CSVs), BIOHUB_LINKER_CACHE (npz dir).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.spatial import cKDTree

sys.path.insert(0, "/workspace/biohub/analysis")
import div_pipeline as dp  # noqa: E402

SC = np.array([1.625, 0.40625, 0.40625]); GATE = 2.5
LK = Path(os.environ.get("BIOHUB_LINKER_CACHE", "/workspace/biohub/caches/pilk_144_all"))
OUT = dp.ROOT / "linker_feats"; OUT.mkdir(parents=True, exist_ok=True)


def video_feats(v, grp, cand):
    z = np.load(LK / f"{v}.npz"); cn = z["nodes"]; ct = cn[:, 0]; cpos = cn[:, 1:].astype(np.float64) * SC
    nodes = grp.filter(pl.col("row_type") == "node"); T = nodes["t"].to_numpy(); P = nodes.select("z", "y", "x").to_numpy().astype(np.float64) * SC
    s2c = np.full(len(T), -1, np.int64)
    for t in np.unique(T):
        ci = np.flatnonzero(ct == t); si = np.flatnonzero(T == t)
        if len(ci) == 0:
            continue
        d, j = cKDTree(cpos[ci]).query(P[si]); ok = d <= GATE; s2c[si[ok]] = ci[j[ok]]
    prob = np.exp(z["e_logit"] - z["lse_src"][z["e_tgt"]]).astype(np.float32)
    es, et = z["e_src"], z["e_tgt"]
    pair = {}
    for a, b, pr in zip(es.tolist(), et.tolist(), prob.tolist()):
        pair[(a, b)] = pr
    # best and second-best source prob per cache target
    order = np.lexsort((-prob, et)); et_o, pr_o = et[order], prob[order]
    first = np.r_[True, et_o[1:] != et_o[:-1]]
    best = dict(zip(et_o[first].tolist(), pr_o[first].tolist()))
    idx2 = np.flatnonzero(~first); second = {}
    for i in idx2:
        t_ = int(et_o[i])
        if t_ not in second:
            second[t_] = float(pr_o[i])
    rows = []
    for r in cand.select("t", "p", "c1", "c", "q").iter_rows():
        t, p, c1, c, q = r; cp, cc1, cc, cq = s2c[p], s2c[c1], s2c[c], (s2c[q] if q >= 0 else -1)
        lk_pc = pair.get((cp, cc), 0.0) if cp >= 0 and cc >= 0 else 0.0
        lk_qc = (pair.get((cq, cc), 0.0) if cq >= 0 and cc >= 0 else 0.0) if q >= 0 else -1.0
        lk_pc1 = pair.get((cp, cc1), 0.0) if cp >= 0 and cc1 >= 0 else 0.0
        b = best.get(int(cc), 0.0) if cc >= 0 else 0.0
        other = (second.get(int(cc), 0.0) if abs(b - lk_pc) < 1e-9 else b) if cc >= 0 else 0.0
        rows.append((t, p, c1, c, lk_pc, lk_qc, lk_pc1, b, lk_pc - other, int(cp >= 0 and cc >= 0 and cc1 >= 0)))
    return pl.DataFrame(rows, schema=["t", "p", "c1", "c", "lk_pc", "lk_qc", "lk_pc1", "lk_c_best", "lk_c_margin", "lk_matched"], orient="row").with_columns(pl.lit(v).alias("video"))


def main():
    subs = {v: grp for v, grp in dp.submissions()}
    files = sorted((dp.ROOT / "cands_all").glob("*.parquet"))
    for i, f in enumerate(files):
        v = f.stem
        if (OUT / f"{v}.parquet").exists() or not (LK / f"{v}.npz").exists():
            continue
        cand = pl.read_parquet(f, columns=["t", "p", "c1", "c", "q"])
        video_feats(v, subs[v], cand).write_parquet(OUT / f"{v}.parquet")
        print(f"[{i + 1}/{len(files)}] {v}: {cand.height} candidates", flush=True)


if __name__ == "__main__":
    main()

"""Division candidates on the public stack's final graphs: generate, featurise, label.

Candidate = (parent p at t with exactly one child c1, second daughter c at t+1), c != c1,
d(p,c) <= 13 um, d(c1,c) <= 17 um (GT p99). c may be an orphan track start or already linked to
another parent q (applying the candidate then removes q->c: "rewire").

Label (fast proxy of the official rule, exact-frame): +1 if p matches a GT dividing node and {c1, c}
match its two daughters; 0 (would count as FP) if p matches a GT node with >=1 child otherwise;
-1 neutral (no GT evidence at p: free under the metric, excluded from training).

Usage: python div_candidates.py <out.parquet> <batch dirs...>
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

sys.path.insert(0, "/workspace/biohub/harness")
from score import GT_DIR, load_geff  # noqa: E402

SC = np.array([1.625, 0.40625, 0.40625])
R_PARENT, R_SISTER = 13.0, 17.0
KEEP_NEUTRAL = False
n_all = [0]


def load_pred(grp):
    n = grp.filter(pl.col("row_type") == "node"); e = grp.filter(pl.col("row_type") == "edge")
    nid = n["node_id"].to_numpy(); T = n["t"].to_numpy(); P = n.select("z", "y", "x").to_numpy() * SC
    pos = {k: i for i, k in enumerate(nid)}
    S = np.array([pos[a] for a in e["source_id"].to_numpy()]); D = np.array([pos[b] for b in e["target_id"].to_numpy()])
    return T, P, S, D


def track_len(start, nxt, cap=30):
    n, cur = 0, start
    while cur in nxt and n < cap:
        cur = nxt[cur]; n += 1
    return n


def video_rows(v, T, P, S, D):
    N = len(T)
    children = [[] for _ in range(N)]; parent = np.full(N, -1)
    for a, b in zip(S, D):
        children[a].append(b); parent[b] = a
    nxt = {a: c[0] for a, c in enumerate(children) if len(c) == 1}
    prv = {b: a for b, a in enumerate(parent) if a >= 0}
    byt = {t: np.flatnonzero(T == t) for t in np.unique(T)}
    trees = {t: cKDTree(P[i]) for t, i in byt.items()}

    # GT matching and division lookup
    g = load_geff(GT_DIR / f"{v}.geff"); na = g.node_attrs(attr_keys=["node_id", "t", "z", "y", "x"]); ea = g.edge_attrs(attr_keys=[])
    gid = na["node_id"].to_numpy(); gt = na["t"].to_numpy(); gp = na.select("z", "y", "x").to_numpy() * SC
    gch = {}
    for a, b in zip(ea["source_id"].to_numpy(), ea["target_id"].to_numpy()):
        gch.setdefault(a, []).append(b)
    m_pred2gt = {}
    for t in np.unique(gt):
        gi = np.flatnonzero(gt == t); pi = byt.get(t, np.zeros(0, int))
        if len(pi) == 0:
            continue
        Dm = cdist(gp[gi], P[pi]); r, c = linear_sum_assignment(np.where(Dm <= 7, Dm, 1e6))
        for a, b in zip(r, c):
            if Dm[a, b] <= 7:
                m_pred2gt[pi[b]] = gid[gi[a]]

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
                d_pc, d_pc1, d_sis = (np.linalg.norm(P[p] - P[c]), np.linalg.norm(P[p] - P[c1]), np.linalg.norm(P[c1] - P[c]))
                if d_sis > R_SISTER:
                    continue
                v1, v2 = P[c1] - P[p], P[c] - P[p]
                cosang = float(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))
                mid = np.linalg.norm((P[c1] + P[c]) / 2 - P[p])
                q = parent[c]
                pp = prv.get(p)
                vel = P[p] - P[pp] if pp is not None else np.zeros(3)
                # label (neutral candidates are counted but not stored: memory)
                gpid = m_pred2gt.get(p); lab = -1
                if gpid is not None and len(gch.get(gpid, [])) >= 1:
                    lab = 0
                    kids = gch[gpid]
                    if len(kids) == 2 and {m_pred2gt.get(c1), m_pred2gt.get(c)} == set(kids):
                        lab = 1
                n_all[0] += 1
                if lab < 0 and not KEEP_NEUTRAL:
                    continue
                rows.append(dict(
                    video=v, embryo=v[:4], t=int(t), p=int(p), c1=int(c1), c=int(c), q=int(q), label=lab,
                    d_pc=d_pc, d_pc1=d_pc1, d_sis=d_sis, sym=abs(d_pc - d_pc1) / (0.5 * (d_pc + d_pc1) + 1e-9),
                    cosang=cosang, mid=mid, dz_sis=abs(P[c1][0] - P[c][0]),
                    c_orphan=int(q < 0), d_qc=float(np.linalg.norm(P[q] - P[c])) if q >= 0 else -1.0,
                    q_children=len(children[q]) if q >= 0 else 0,
                    q_alt=(float(tr1.query(P[q], k=2)[0][1]) if q >= 0 and len(nxt_idx) > 1 else -1.0),
                    len_p_back=track_len(p, prv), len_c_fwd=track_len(c, nxt), len_c1_fwd=track_len(c1, nxt),
                    len_q_back=track_len(q, prv) if q >= 0 else 0,
                    speed_p=float(np.linalg.norm(vel)), c1_vs_vel=float(np.linalg.norm(P[c1] - (P[p] + vel))),
                    c_vs_vel=float(np.linalg.norm(P[c] - (P[p] + vel))),
                    density=int(dens[k]), n_frame=len(idx), n_next=len(nxt_idx),
                    n_cand_same_p=0,
                ))
    return rows


def main():
    out = Path(sys.argv[1]); rows = []
    for batch in sys.argv[2:]:
        df = pl.read_csv(Path(batch) / "submission.csv")
        for (v,), grp in df.group_by("dataset"):
            rows += video_rows(v, *load_pred(grp))
        print(batch, "rows so far", len(rows), flush=True)
    d = pl.DataFrame(rows)
    d = d.with_columns(pl.len().over("video", "p").alias("n_cand_same_p"),
                       pl.col("d_pc").rank().over("video", "p").alias("rank_d_pc"))
    d.write_parquet(out)
    lab = d.filter(pl.col("label") >= 0)
    print("candidates", n_all[0], "stored", d.height, "| with GT evidence", lab.height, "| positives", int(lab["label"].sum()),
          "| positives by type: orphan", int(lab.filter(pl.col("c_orphan") == 1)["label"].sum()),
          "rewire", int(lab.filter(pl.col("c_orphan") == 0)["label"].sum()))
    print(lab.group_by("embryo").agg(pl.len(), pl.col("label").sum().alias("pos"), pl.col("video").n_unique().alias("videos")))


if __name__ == "__main__":
    main()

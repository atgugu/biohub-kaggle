"""Fork veto (honest): label the stack's EXISTING forks with the official division metric, learn which to drop.

For every video of the current base (BIOHUB_CACHE): forks = nodes with 2 children in the final graph. Label each fork by
`division_metrics.score_divisions` on the whole graph: TP (paired with a GT division), FP (evaluable false), neutral.
Features: geometry of the fork (parent->daughter distances, sister distance, symmetry, angle, predecessor speed, track
lengths of parent and both daughters, density) + honest DivNet logit of the parent (scored here with the hard-negative
fold models, OOF by video). Classifier: 5-fold-by-video LightGBM on TP vs FP forks. Joint test: remove forks with
OOF P(TP) < thr (removing the edge to the daughter farther from the parent), score officially.

Usage: python fork_veto.py [--thr 0.1,0.2,0.3,0.5]
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import polars as pl
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, "/workspace/biohub/analysis"); sys.path.insert(0, "/workspace/biohub/harness")
warnings.filterwarnings("ignore")
import divnet  # noqa: E402
import divnet_fast  # noqa: E402
import div_pipeline as dp  # noqa: E402
from score import GT_DIR, SCALE, load_geff  # noqa: E402
from tracking_cellmot.division_metrics import score_divisions  # noqa: E402
from tracking_cellmot.metrics import evaluate  # noqa: E402

SC = np.array(SCALE)
FEATS = ["d1", "d2", "d_sis", "sym", "cosang", "speed_p", "len_p_back", "len_c1_fwd", "len_c2_fwd", "density", "n_frame", "dz_sis", "divnet_hn"]


def fork_rows(v, grp):
    nodes = grp.filter(pl.col("row_type") == "node"); edges = grp.filter(pl.col("row_type") == "edge")
    nid = nodes["node_id"].to_numpy(); T = nodes["t"].to_numpy(); Pv = nodes.select("z", "y", "x").to_numpy().astype(np.float64); P = Pv * SC
    pos = {int(n): i for i, n in enumerate(nid)}
    S = np.array([pos[a] for a in edges["source_id"].to_numpy()]); D = np.array([pos[b] for b in edges["target_id"].to_numpy()])
    children = {}; parent = {}
    for a, b in zip(S.tolist(), D.tolist()):
        children.setdefault(a, []).append(b); parent[b] = a
    nxt = {a: c[0] for a, c in children.items() if len(c) == 1}
    trees = {t: cKDTree(P[T == t]) for t in np.unique(T)}
    rows = []
    for p, ch in children.items():
        if len(ch) != 2:
            continue
        c1, c2 = ch; d1, d2 = np.linalg.norm(P[c1] - P[p]), np.linalg.norm(P[c2] - P[p]); ds = np.linalg.norm(P[c1] - P[c2])
        v1, v2 = P[c1] - P[p], P[c2] - P[p]; pp = parent.get(p); sp = float(np.linalg.norm(P[p] - P[pp])) if pp is not None else 0.0
        rows.append(dict(video=v, p=int(p), c1=int(c1), c2=int(c2), t=int(T[p]), nid_p=int(nid[p]), nid_c1=int(nid[c1]), nid_c2=int(nid[c2]),
                         d1=d1, d2=d2, d_sis=ds, sym=abs(d1 - d2) / (0.5 * (d1 + d2) + 1e-9), cosang=float(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)),
                         speed_p=sp, len_p_back=0, len_c1_fwd=0, len_c2_fwd=0,
                         density=int(trees[T[p]].query_ball_point(P[p], 10.0, return_length=True)), n_frame=int((T == T[p]).sum()), dz_sis=abs(P[c1][0] - P[c2][0])))
        # track lengths
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
        rows[-1]["len_p_back"] = back(p); rows[-1]["len_c1_fwd"] = fwd(c1); rows[-1]["len_c2_fwd"] = fwd(c2)
    return rows, Pv


def fork_labels(v, grp):
    """TP / FP fork node ids from the official division metric (needs a matched graph)."""
    from score import graph_from_rows
    nodes = grp.filter(pl.col("row_type") == "node"); edges = grp.filter(pl.col("row_type") == "edge")
    g = graph_from_rows(nodes, edges); gt = load_geff(GT_DIR / f"{v}.geff")
    evaluate(g, gt, scale=SCALE, max_distance=7.0)             # writes matching attrs onto g
    res = score_divisions(g, gt, scale=SCALE, max_distance=7.0)
    # tracksdata assigned fresh node ids: map back through the row order (bulk_add_nodes preserves order)
    ids = g.node_attrs(attr_keys=["node_id"])["node_id"].to_numpy(); csv_ids = nodes["node_id"].to_numpy()
    back = dict(zip(ids.tolist(), csv_ids.tolist()))
    tp = {back[i] for i in getattr(res, "tp_forks", set()) if i in back}; fp = {back[i] for i in getattr(res, "fp_forks", set()) if i in back}
    return tp, fp, res


def apply_attach(subs, top, k, athr):
    """Honest orphan attach: top-k candidates per video by OOF score s2 (same greedy semantics as div_pipeline._joint_test). Returns (subs, n_added)."""
    n_add = 0
    for v, grp in subs.items():
        nodes = grp.filter(pl.col("row_type") == "node"); edges = grp.filter(pl.col("row_type") == "edge"); nid = nodes["node_id"].to_numpy()
        cand = top.filter(pl.col("video") == v).sort("s2", descending=True); used_p, used_c, add, rm = set(), set(), [], set()
        for r in cand.filter(pl.col("s2") >= athr).iter_rows(named=True):
            if len(add) >= k:
                break
            if r["p"] in used_p or r["c"] in used_c:
                continue
            used_p.add(r["p"]); used_c.add(r["c"]); add.append((int(nid[r["p"]]), int(nid[r["c"]])))
            if r["q"] >= 0:
                rm.add((int(nid[r["q"]]), int(nid[r["c"]])))
        if not add:
            continue
        keep = [(x, y) not in rm for x, y in zip(edges["source_id"].to_list(), edges["target_id"].to_list())]
        e2 = pl.concat([edges.filter(pl.Series(keep)), edges.head(len(add)).with_columns(pl.Series("source_id", [x for x, _ in add]), pl.Series("target_id", [y for _, y in add]))])
        subs[v] = pl.concat([nodes, e2]); n_add += len(add)
    return subs, n_add


def joint_test(subs, d, thrs, col="p_tp"):
    """Official joint test: drop forks with d[col] < thr (edge to the farther daughter). Returns {setting: {video: counts}}."""
    from div_pipeline import _eval
    agg = {"base": {}}
    for th in thrs:
        agg[th] = {}
    for v, grp in subs.items():
        nodes = grp.filter(pl.col("row_type") == "node"); edges = grp.filter(pl.col("row_type") == "edge")
        base = _eval(nodes, edges, v); agg["base"][v] = base
        fv = d.filter(pl.col("video") == v)
        for th in thrs:
            drop = fv.filter(pl.col(col) < th)
            if drop.height == 0:
                agg[th][v] = base; continue
            rm = {(r["nid_p"], r["nid_c2"] if r["d2"] >= r["d1"] else r["nid_c1"]) for r in drop.iter_rows(named=True)}
            keep = [(s_, t_) not in rm for s_, t_ in zip(edges["source_id"].to_list(), edges["target_id"].to_list())]
            agg[th][v] = _eval(nodes, edges.filter(pl.Series(keep)), v)
    return agg


def print_agg(agg, out_csv=None):
    from div_pipeline import _official
    print("\nsetting     edge TP/FP/FN           div TP/FP/FN    OFFICIAL  adjJ   divJ   | 44b6 score | 6bba score")
    out_rows = []
    for key, rows_by in agg.items():
        tot = np.sum(list(rows_by.values()), axis=0); sc, aj, dj = _official(rows_by)
        per = {}
        for e in ("44b6", "6bba"):
            sub = {v: r for v, r in rows_by.items() if v.startswith(e)}
            per[e] = _official(sub)[0] if sub else float("nan")
        print(f"{str(key):10s} {int(tot[0])}/{int(tot[1])}/{int(tot[2])}   {int(tot[3])}/{int(tot[4])}/{int(tot[5])}   {sc:.4f}  {aj:.4f}  {dj:.4f}   | {per['44b6']:.4f} | {per['6bba']:.4f}", flush=True)
        for v, r in rows_by.items():
            out_rows.append((str(key), v, *[int(x) for x in r]))
    if out_csv:
        pl.DataFrame(out_rows, schema=["setting", "video", "etp", "efp", "efn", "dtp", "dfp", "dfn", "nodes"], orient="row").write_csv(out_csv)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--thr", default="0.05,0.1,0.2,0.3,0.5"); ap.add_argument("--mode", default="single", choices=["single", "geom"]); ap.add_argument("--tag", default=""); ap.add_argument("--arms", default="hardneg", help="comma list of DivNet arms; feature = mean of held-out fold logits over arms"); ap.add_argument("--attach", default=None, help="OOF-scored candidate parquet (s2) to attach top-k per video BEFORE the veto"); ap.add_argument("--extra", default=None, help="parquet (video, nid_p, <feat>) of extra honest per-parent features to join, e.g. mitosis_logit"); ap.add_argument("--drop-divnet", action="store_true"); ap.add_argument("--k", type=int, default=3); ap.add_argument("--athr", type=float, default=0.0); a = ap.parse_args()
    device = torch.device("cuda"); vids = sorted(p.stem for p in GT_DIR.glob("*.geff")); folds = dp._folds(vids); fold_of = {v: k for k, f in enumerate(folds) for v in f}
    arms = a.arms.split(","); hn = []; hn_arms = []
    for arm in arms:
        ms = []
        for k in range(5):
            m = divnet.DivNet(); m.load_state_dict(torch.load(f"/workspace/biohub/analysis/divpipe/divnet_{arm}_fold{k}.pt", map_location="cpu")); ms.append(m.to(device).eval())
        hn_arms.append(ms)
    hn = hn_arms[0]; print("arms", arms, flush=True)
    subs = dict(sorted(dp.submissions(), key=lambda kv: kv[0]))   # deterministic order
    if a.attach:   # honest orphan attach first (same greedy semantics as div_pipeline._joint_test), then the veto sees the attached forks too
        subs, n_add = apply_attach(subs, pl.read_parquet(a.attach), a.k, a.athr)
        print(f"attach: {n_add} edges added over {len(subs)} videos (k={a.k}, thr={a.athr}); 'base' below = attached graph", flush=True)
    allrows = []; base_counts = {}
    for i, (v, grp) in enumerate(subs.items()):
        rows, Pv = fork_rows(v, grp)
        tp, fp, res = fork_labels(v, grp); base_counts[v] = res
        if not rows:
            continue
        frames = divnet_fast.norm_frames(v); m = hn[fold_of[v]]
        with torch.no_grad():
            for r in rows:
                x = torch.from_numpy(divnet_fast.crops(frames, r["t"], Pv[[r["p"]]])).to(device)
                r["divnet_hn"] = float(np.mean([ms[fold_of[v]](x).float().cpu()[0].item() for ms in hn_arms]))   # held-out fold model(s) (validation; mean over arms)
                r["divnet_hn_mean"] = float(np.mean([mm(x).float().cpu()[0].item() for ms in hn_arms for mm in ms]))   # mean over ALL fold models of all arms (deployment path: R4 = 5, ens = 5 x arms)
                r["label"] = 1 if r["nid_p"] in tp else (0 if r["nid_p"] in fp else -1)
        allrows += rows
        if i % 20 == 0:
            print(f"[{i + 1}/{len(subs)}] {v}: forks {len(rows)} TP {len(tp)} FP {len(fp)}", flush=True)
    d = pl.DataFrame(allrows)
    extra_cols = []
    if a.extra:
        ex = pl.read_parquet(a.extra); extra_cols = [c for c in ex.columns if c not in ("video", "nid_p")]
        d = d.join(ex, on=["video", "nid_p"], how="left"); print(f"extra features {extra_cols}: {int(d.select(extra_cols[0]).null_count().item())} forks without a value", flush=True)
    d.write_parquet(dp.ROOT / f"forks{a.tag}.parquet")
    ev = d.filter(pl.col("label") >= 0); print(f"forks {d.height}: TP {int((d['label'] == 1).sum())} FP {int((d['label'] == 0).sum())} neutral {int((d['label'] < 0).sum())}", flush=True)
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    params = dict(dp.PARAMS); params["scale_pos_weight"] = 1.0; params["seed"] = 0; params["deterministic"] = True; params["bagging_seed"] = 0; params["feature_fraction_seed"] = 0
    feats = [f for f in FEATS if f != "divnet_hn"] if (a.mode == "geom" or a.drop_divnet) else list(FEATS)
    feats = feats + extra_cols
    print("mode", a.mode, "feats", feats, flush=True)
    oof = np.full(d.height, np.nan); oof_mean = np.full(d.height, np.nan)
    for fold in folds:
        tr = ev.filter(~pl.col("video").is_in(list(fold))); te_idx = np.flatnonzero(d["video"].is_in(list(fold)).to_numpy())
        if tr.height == 0 or tr["label"].n_unique() < 2:
            continue
        m = lgb.train(params, lgb.Dataset(tr.select(feats).to_numpy().astype(np.float32), tr["label"].to_numpy()), 200)
        oof[te_idx] = m.predict(d[te_idx].select(feats).to_numpy().astype(np.float32))
        if a.mode == "single":   # same classifier, deployed feature (5-fold mean) instead of the single-fold logit
            cols = [f if f != "divnet_hn" else "divnet_hn_mean" for f in feats]
            oof_mean[te_idx] = m.predict(d[te_idx].select(cols).to_numpy().astype(np.float32))
    d = d.with_columns(pl.Series("p_tp", oof), pl.Series("p_tp_mean", oof_mean)); evm = d.filter(pl.col("label") >= 0)
    print(f"OOF AUC TP vs FP forks: {roc_auc_score(evm['label'], evm['p_tp']):.4f}", flush=True)
    if a.mode == "single":
        for th in (0.03, 0.05, 0.07):
            vs, vm = d["p_tp"] < th, d["p_tp_mean"] < th
            both = int((vs & vm).sum()); print(f"  decision agreement at {th}: single vetoes {int(vs.sum())}, mean-feature vetoes {int(vm.sum())}, both {both} | TP forks vetoed single {int((vs & (d['label'] == 1)).sum())} mean {int((vm & (d['label'] == 1)).sum())}", flush=True)
    d.write_parquet(dp.ROOT / f"forks_scored{a.tag}.parquet")
    # joint official test: drop forks with p_tp < thr (remove the edge to the daughter farther from the parent)
    thrs = [float(x) for x in a.thr.split(",")]
    print_agg(joint_test(subs, d, thrs, "p_tp"), dp.ROOT / f"veto_per_video{a.tag}.csv")
    if a.mode == "single":
        print("\n[deployment path: same classifier, mean-over-all-fold-models feature]")
        print_agg(joint_test(subs, d, thrs, "p_tp_mean"))


if __name__ == "__main__":
    main()

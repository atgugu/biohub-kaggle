"""Two-stage division ranker with an OFFICIAL-metric joint test.

stage0  candidates for every cached video incl. neutral ones -> cands_all/<video>.parquet
stage1  geometry LightGBM (trained on GT-evident rows, 5-fold by video) scores ALL candidates of the
        held-out videos; keep top N per video -> stage1_top/<video>.parquet
stage2  DivNet logit for every (video, parent) in stage1_top (GPU) -> divnet_top.parquet
stage3  final ranker (geometry + DivNet), OOF by video; apply top-K per video to the stack's final graph
        (rewire q->c to p->c, or attach the orphan c) and score with the official scorer:
        division TP/FP/FN, edge TP/FP/FN and score, summed over videos, vs the untouched graphs.

Usage: python div_pipeline.py stage0|stage1|stage2|stage3 [--topn 300] [--ks 1,2,3,5,10]
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "/workspace/biohub/analysis")
sys.path.insert(0, "/workspace/biohub/harness")
warnings.filterwarnings("ignore")

ROOT = Path(os.environ.get("BIOHUB_DIVROOT", "/workspace/biohub/analysis/divpipe")); CACHE = Path(os.environ.get("BIOHUB_CACHE", "/workspace/biohub/caches/stack_s2"))
GEOM = ["d_pc", "d_pc1", "d_sis", "sym", "cosang", "mid", "dz_sis", "c_orphan", "d_qc", "q_children", "q_alt",
        "len_p_back", "len_c_fwd", "len_c1_fwd", "len_q_back", "speed_p", "c1_vs_vel", "c_vs_vel", "density",
        "n_frame", "n_next", "n_cand_same_p", "rank_d_pc"]
KEY = ["video", "embryo", "t", "p", "c1", "c", "q", "label"]
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=8, min_child_samples=10, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, scale_pos_weight=20, verbose=-1, num_threads=8)


def batches():
    """Cache layout: batch_*/ (+ test4/ for the s2/geofusion caches) or xbatch_*/ (x138 replay); only dirs with a submission.csv."""
    return [b for b in sorted(CACHE.glob("batch_*")) + sorted(CACHE.glob("xbatch_*")) + [CACHE / "test4"] if (b / "submission.csv").exists()]


def submissions():
    for b in batches():
        df = pl.read_csv(b / "submission.csv")
        for (v,), grp in df.group_by("dataset"):
            yield v, grp


def stage0():
    import div_candidates as dc
    dc.KEEP_NEUTRAL = True
    out = ROOT / "cands_all"; out.mkdir(parents=True, exist_ok=True)
    for v, grp in submissions():
        if (out / f"{v}.parquet").exists():
            continue
        rows = dc.video_rows(v, *dc.load_pred(grp))
        if not rows:
            continue
        d = pl.DataFrame(rows).with_columns(pl.len().over("p").alias("n_cand_same_p"), pl.col("d_pc").rank().over("p").alias("rank_d_pc"))
        d.write_parquet(out / f"{v}.parquet")
        print(v, d.height, "cands", int((d["label"] >= 0).sum()), "evident", int((d["label"] == 1).sum()), "pos", flush=True)


def _folds(videos, k=5):
    rng = np.random.default_rng(0); vs = np.array(sorted(videos)); rng.shuffle(vs)
    return [set(vs[i::k].tolist()) for i in range(k)]


def stage1(topn):
    import lightgbm as lgb
    files = sorted((ROOT / "cands_all").glob("*.parquet"))
    ev = pl.concat([pl.read_parquet(f).filter(pl.col("label") >= 0) for f in files])
    print("evident rows", ev.height, "pos", int(ev["label"].sum()), flush=True)
    out = ROOT / "stage1_top"; out.mkdir(exist_ok=True)
    for i, fold in enumerate(_folds([f.stem for f in files])):
        tr = ev.filter(~pl.col("video").is_in(list(fold)))
        m = lgb.train(PARAMS, lgb.Dataset(tr.select(GEOM).to_numpy().astype(np.float32), tr["label"].to_numpy()), 300)
        for v in sorted(fold):
            d = pl.read_parquet(ROOT / "cands_all" / f"{v}.parquet")
            s = m.predict(d.select(GEOM).to_numpy().astype(np.float32))
            d = d.with_columns(pl.Series("s1", s)).sort("s1", descending=True).head(topn)
            d.write_parquet(out / f"{v}.parquet")
        print("fold", i, "done", flush=True)
    top = pl.concat([pl.read_parquet(f) for f in out.glob("*.parquet")])
    print("stage1 kept", top.height, "| positives kept", int((top["label"] == 1).sum()), "of", int(ev["label"].sum()), flush=True)


def stage2():
    import torch, divnet
    top = pl.concat([pl.read_parquet(f) for f in (ROOT / "stage1_top").glob("*.parquet")]).select("video", "t", "p").unique().sort("video", "t")
    coords = {v: grp.filter(pl.col("row_type") == "node").select("z", "y", "x").to_numpy() for v, grp in submissions()}
    model = divnet.load_divnet(); out, buf, keys = [], [], []

    def flush():
        nonlocal buf, keys
        if buf:
            with torch.no_grad():
                lg = model(torch.from_numpy(np.stack(buf)).cuda()).cpu().numpy()
            out.extend((v, p, float(l)) for (v, p), l in zip(keys, lg)); buf, keys = [], []
    for i, (v, t, p) in enumerate(top.iter_rows()):
        buf.append(divnet.make_input(v, t, coords[v][p], "max", "frame", "img_first")); keys.append((v, p))
        if len(buf) == 128:
            flush()
        if i % 5000 == 0:
            print(i, "/", top.height, flush=True)
    flush()
    pl.DataFrame(out, schema=["video", "p", "divnet_logit"], orient="row").write_parquet(ROOT / "divnet_top.parquet")
    print("divnet done", len(out), flush=True)


def _graph(nodes: pl.DataFrame, edges: pl.DataFrame):
    from score import graph_from_rows
    return graph_from_rows(nodes, edges)


def _eval(nodes, edges, v):
    from score import GT_DIR, SCALE, load_geff
    from tracking_cellmot.metrics import evaluate
    er = evaluate(_graph(nodes, edges), load_geff(GT_DIR / f"{v}.geff"), scale=SCALE, max_distance=7.0)
    return np.array([er.edge_tp, er.edge_fp, er.edge_fn, er.division_tp, er.division_fp, er.division_fn, er.num_pred_nodes])


def _official(rows_by_video):
    """rows_by_video: {video: counts array} -> official summarise() over per-video adjusted rows."""
    from score import GT_DIR, n_estimated
    from tracking_cellmot.metrics import per_sample_metrics, EvaluationResult, summarise
    rows = []
    for v, a in rows_by_video.items():
        er = EvaluationResult(*[int(x) for x in a])
        rows.append(per_sample_metrics(er, n_estimated(GT_DIR / f"{v}.geff"), float("nan")))
    s = summarise(rows)
    return s["score"], s["adj_edge_jaccard"], s["division_jaccard"]


def stage3(ks, thr):
    import lightgbm as lgb
    from score import n_estimated, GT_DIR
    top = pl.concat([pl.read_parquet(f) for f in (ROOT / "stage1_top").glob("*.parquet")])
    top = top.join(pl.read_parquet(ROOT / "divnet_top.parquet"), on=["video", "p"], how="left")
    feats = GEOM + ["s1", "divnet_logit"]
    top = top.with_columns(pl.lit(0.0).alias("s2"))
    ev = top.filter(pl.col("label") >= 0)
    print("stage3 rows", top.height, "evident", ev.height, "pos", int(ev["label"].sum()), flush=True)
    parts = []
    for fold in _folds(top["video"].unique().to_list()):
        tr = ev.filter(~pl.col("video").is_in(list(fold)))
        m = lgb.train(PARAMS, lgb.Dataset(tr.select(feats).to_numpy().astype(np.float32), tr["label"].to_numpy()), 300)
        te = top.filter(pl.col("video").is_in(list(fold)))
        parts.append(te.with_columns(pl.Series("s2", m.predict(te.select(feats).to_numpy().astype(np.float32)))))
    top = pl.concat(parts); top.write_parquet(ROOT / "stage3_scored.parquet")
    # official joint test
    subs = {v: grp for v, grp in submissions()}
    agg = {("base", 0): np.zeros(7)}; per_video = []
    for v, grp in subs.items():
        if v not in set(top["video"].to_list()):
            continue
        nodes = grp.filter(pl.col("row_type") == "node"); edges = grp.filter(pl.col("row_type") == "edge")
        nid = nodes["node_id"].to_numpy()
        base = _eval(nodes, edges, v); agg[("base", 0)] += base
        cand = top.filter(pl.col("video") == v).sort("s2", descending=True)
        for k in ks:
            for th in thr:
                used_p, used_c, add, rm = set(), set(), [], set()
                for r in cand.filter(pl.col("s2") >= th).head(max(k * 3, k)).iter_rows(named=True):
                    if len(add) >= k or r["p"] in used_p or r["c"] in used_c:
                        continue
                    used_p.add(r["p"]); used_c.add(r["c"]); add.append((int(nid[r["p"]]), int(nid[r["c"]])))
                    if r["q"] >= 0:
                        rm.add((int(nid[r["q"]]), int(nid[r["c"]])))
                if not add:
                    agg.setdefault((k, th), np.zeros(7)); agg[(k, th)] += base; continue
                e2 = edges.filter(~pl.struct("source_id", "target_id").map_elements(lambda s: (s["source_id"], s["target_id"]) in rm, return_dtype=pl.Boolean))
                e2 = pl.concat([e2, edges.head(len(add)).with_columns(pl.Series("source_id", [a for a, _ in add]), pl.Series("target_id", [b for _, b in add]))])
                res = _eval(nodes, e2, v); agg.setdefault((k, th), np.zeros(7)); agg[(k, th)] += res
                per_video.append((v, k, th, *(res - base).tolist()))
        print(v, "base div", base[3:6].astype(int).tolist(), flush=True)
    n_est = sum(n_estimated(GT_DIR / f"{v}.geff") for v in subs if v in set(top["video"].to_list()))
    print("\nsetting          edge TP/FP/FN            div TP/FP/FN     edgeJ    divJ    score(J+0.1divJ)")
    for key, a in sorted(agg.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
        ej = a[0] / (a[0] + a[1] + a[2]); dj = a[3] / max(1, a[3] + a[4] + a[5])
        print(f"{str(key):16s} {int(a[0])}/{int(a[1])}/{int(a[2])}   {int(a[3])}/{int(a[4])}/{int(a[5])}   {ej:.4f}  {dj:.4f}  {ej + 0.1 * dj:.4f}")
    pl.DataFrame(per_video, schema=["video", "k", "thr", "d_etp", "d_efp", "d_efn", "d_dtp", "d_dfp", "d_dfn", "d_nodes"], orient="row").write_csv(ROOT / "stage3_per_video.csv")


def rescore(ks, thr):
    """Re-run only the official joint test on the saved stage3all ranking."""
    top = pl.read_parquet(ROOT / "stage3all_scored.parquet"); _joint_test(top, ks, thr)


FEAT_DIRS = {"divnet_logit": ("divnet_all", ["video", "p"]), "divnet_oof_logit": ("divnet_oof_all", ["video", "p"]),
             "divnet_hardneg_logit": ("divnet_hardneg_all", ["video", "p"]), "divnet_crop48_logit": ("divnet_crop48_all", ["video", "p"]),
             "lk_pc": ("linker_feats", ["video", "t", "p", "c1", "c"])}
LK_COLS = ["lk_pc", "lk_qc", "lk_pc1", "lk_c_best", "lk_c_margin", "lk_matched"]


def stage3all(ks, thr, topk_keep=50, extra="divnet_logit", channel="all", tag=""):
    """Single ranker (geometry + extra features on ALL candidates), OOF by video, then the official joint test."""
    import lightgbm as lgb
    from score import n_estimated, GT_DIR
    files = sorted((ROOT / "cands_all").glob("*.parquet"))
    extra_cols = [c for c in extra.split(",") if c]
    feats = GEOM + [c for c in extra_cols if c != "lk_pc"] + (LK_COLS if "lk_pc" in extra_cols else [])
    def load(v):
        d = pl.read_parquet(ROOT / "cands_all" / f"{v}.parquet")
        for c in extra_cols:
            sub, keys = FEAT_DIRS[c]; f = ROOT / sub / f"{v}.parquet"
            if f.exists():
                d = d.join(pl.read_parquet(f), on=keys, how="left")
        if channel == "orphan":
            d = d.filter(pl.col("c_orphan") == 1)
        elif channel == "rewire":
            d = d.filter(pl.col("c_orphan") == 0)
        return d
    ev = pl.concat([load(f.stem).filter(pl.col("label") >= 0) for f in files])
    print("evident rows", ev.height, "pos", int(ev["label"].sum()), "feature nulls", int(ev.select([c for c in feats if c in ev.columns]).null_count().sum_horizontal().item()), flush=True)
    parts = []
    for i, fold in enumerate(_folds([f.stem for f in files])):
        tr = ev.filter(~pl.col("video").is_in(list(fold)))
        m = lgb.train(PARAMS, lgb.Dataset(tr.select(feats).to_numpy().astype(np.float32), tr["label"].to_numpy()), 300)
        for v in sorted(fold):
            d = load(v); s = m.predict(d.select(feats).to_numpy().astype(np.float32))
            parts.append(d.with_columns(pl.Series("s2", s)).sort("s2", descending=True).head(topk_keep))
        print("fold", i, flush=True)
    top = pl.concat(parts); top.write_parquet(ROOT / f"stage3all_scored{tag}.parquet")
    kept_pos = int((top["label"] == 1).sum()); print("feats", feats, "| channel", channel, flush=True)
    print("top", topk_keep, "per video keeps", kept_pos, "of", int(ev["label"].sum()), "positives", flush=True)
    for k in (1, 2, 3, 5, 10, 20, 50):
        sub = top.group_by("video").head(k); print(f"  top-{k}/video: positives {int((sub['label'] == 1).sum())}, evaluable {int((sub['label'] >= 0).sum())} of {sub.height}")
    _joint_test(top, ks, thr, tag)


def _joint_test(top, ks, thr, tag=""):
    from score import n_estimated, GT_DIR
    subs = {v: grp for v, grp in submissions()}; vids = set(top["video"].to_list())
    agg = {("base", 0): np.zeros(7)}; per_video = []; rows_by = {("base", 0): {}}
    for v, grp in subs.items():
        if v not in vids:
            continue
        nodes = grp.filter(pl.col("row_type") == "node"); edges = grp.filter(pl.col("row_type") == "edge")
        nid = nodes["node_id"].to_numpy(); base = _eval(nodes, edges, v); agg[("base", 0)] += base; rows_by[("base", 0)][v] = base
        cand = top.filter(pl.col("video") == v).sort("s2", descending=True)
        for k in ks:
            for th in thr:
                used_p, used_c, add, rm = set(), set(), [], set()
                for r in cand.filter(pl.col("s2") >= th).iter_rows(named=True):
                    if len(add) >= k:
                        break
                    if r["p"] in used_p or r["c"] in used_c:
                        continue
                    used_p.add(r["p"]); used_c.add(r["c"]); add.append((int(nid[r["p"]]), int(nid[r["c"]])))
                    if r["q"] >= 0:
                        rm.add((int(nid[r["q"]]), int(nid[r["c"]])))
                agg.setdefault((k, th), np.zeros(7)); rows_by.setdefault((k, th), {})
                if not add:
                    agg[(k, th)] += base; rows_by[(k, th)][v] = base; continue
                keep = [ (a, b) not in rm for a, b in zip(edges["source_id"].to_list(), edges["target_id"].to_list()) ]
                e2 = pl.concat([edges.filter(pl.Series(keep)),
                                edges.head(len(add)).with_columns(pl.Series("source_id", [a for a, _ in add]), pl.Series("target_id", [b for _, b in add]))])
                res = _eval(nodes, e2, v); agg[(k, th)] += res; rows_by[(k, th)][v] = res; per_video.append((v, k, th, *(res - base).tolist()))
    print("\nsetting          edge TP/FP/FN            div TP/FP/FN     OFFICIAL score   adjJ    divJ")
    for key, a in sorted(agg.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
        sc, aj, dj = _official(rows_by[key])
        print(f"{str(key):16s} {int(a[0])}/{int(a[1])}/{int(a[2])}   {int(a[3])}/{int(a[4])}/{int(a[5])}   {sc:.4f}   {aj:.4f}  {dj:.4f}", flush=True)
    pl.DataFrame(per_video, schema=["video", "k", "thr", "d_etp", "d_efp", "d_efn", "d_dtp", "d_dfp", "d_dfn", "d_nodes"], orient="row").write_csv(ROOT / f"stage3all_per_video{tag}.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("stage"); ap.add_argument("--topn", type=int, default=300)
    ap.add_argument("--ks", default="1,2,3,5,10"); ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--extra", default="divnet_logit"); ap.add_argument("--channel", default="all"); ap.add_argument("--tag", default=""); ap.add_argument("--thr", default="0,0.3,0.5")
    a = ap.parse_args(); ROOT.mkdir(exist_ok=True)
    {"stage0": stage0, "stage1": lambda: stage1(a.topn), "stage2": stage2,
     "stage3": lambda: stage3([int(x) for x in a.ks.split(",")], [float(x) for x in a.thr.split(",")]),
     "rescore": lambda: rescore([int(x) for x in a.ks.split(",")], [float(x) for x in a.thr.split(",")]),
     "stage3all": lambda: stage3all([int(x) for x in a.ks.split(",")], [float(x) for x in a.thr.split(",")], a.topk, a.extra, a.channel, a.tag)}[a.stage]()

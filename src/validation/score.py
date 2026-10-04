"""Local scoring harness: the official Biohub scorer, per video, with embryo folds.

Wraps the host code in research/data_metric/metric_code (commit 075fc5f5) without
changing it. Every run reports, overall and per embryo (44b6 / 6bba):
edge J, adjusted edge J, N_pred/N_est, division TP/FP/FN and score.

The 4 placeholder /test clips are ordinary train videos; they are excluded by
default because in-notebook proxies have scored on them.

Usage:
    python score.py run  --pred <dir of .geff | submission.csv> [--gt /workspace/data/train] [--out rows.csv]
    python score.py diff --a rows_a.csv --b rows_b.csv      # paired per-video comparison
"""
from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

METRIC_CODE = Path("/workspace/biohub/research/data_metric/metric_code")
sys.path[:0] = [str(METRIC_CODE / "src"), str(METRIC_CODE / "scripts")]

import polars as pl  # noqa: E402
import tracksdata as td  # noqa: E402
from geff import GeffMetadata  # noqa: E402

from tracking_cellmot.metrics import (  # noqa: E402
    evaluate, node_recall, per_sample_metrics, summarise,
)

GT_DIR = Path("/workspace/data/train")
SCALE = (1.625, 0.40625, 0.40625)  # (z, y, x) um/voxel, same for every video
TWINS = {"44b6_0113de3b", "44b6_0b24845f", "6bba_05b6850b", "6bba_05db0fb1"}
EMBRYOS = ("44b6", "6bba")


def embryo(name: str) -> str:
    return name.split("_", 1)[0]


def load_geff(path: Path) -> td.graph.BaseGraph:
    g = td.graph.IndexedRXGraph.from_geff(path)
    return g[0] if isinstance(g, tuple) else g


def n_estimated(gt_geff: Path) -> float:
    try:
        val = (GeffMetadata.read(gt_geff).extra or {}).get("estimated_number_of_nodes")
    except Exception:
        return float("nan")
    return float(val) if val is not None else float("nan")


def graph_from_rows(nodes: pl.DataFrame, edges: pl.DataFrame) -> td.graph.BaseGraph:
    from csv_to_geffs import build_graph_from_rows
    return build_graph_from_rows(nodes, edges)


def _score_one(args) -> dict:
    name, gt_dir, pred = args
    gt_geff = Path(gt_dir) / f"{name}.geff"
    row = {"video": name, "embryo": embryo(name)}
    try:
        gt = load_geff(gt_geff)
        g = load_geff(pred) if isinstance(pred, (str, Path)) else graph_from_rows(*pred)
        er = evaluate(g, gt, scale=SCALE, max_distance=7.0)
        rec = node_recall(g, gt) if g.num_edges() > 0 and g.num_nodes() > 0 else 0.0
        n_est = n_estimated(gt_geff)
        row.update(per_sample_metrics(er, n_est, rec))
        row["n_est"] = n_est
        row["n_gt_nodes"] = gt.num_nodes()
    except Exception as exc:  # unreadable geff, missing video, ...
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def _pred_jobs(pred: Path, gt_dir: Path, names: list[str] | None):
    gt_names = {p.stem for p in gt_dir.glob("*.geff")}
    if pred.is_dir():
        preds = {p.stem: p for p in pred.glob("*.geff")}
    else:
        df = pl.read_csv(pred, columns=["dataset", "row_type", "node_id", "t", "z", "y", "x",
                                        "source_id", "target_id"])
        preds = {}
        for (name,), grp in df.group_by("dataset"):
            preds[name] = (grp.filter(pl.col("row_type") == "node"),
                           grp.filter(pl.col("row_type") == "edge"))
    todo = sorted(set(preds) & gt_names)
    if names is not None:
        todo = [n for n in todo if n in set(names)]
    return [(n, str(gt_dir), preds[n]) for n in todo]


def score(pred: Path | str, gt_dir: Path | str = GT_DIR, names: list[str] | None = None,
          workers: int = 8) -> pl.DataFrame:
    """Per-video metric rows for every video present in both pred and GT."""
    jobs = _pred_jobs(Path(pred), Path(gt_dir), names)
    # spawn, not fork: forking after polars starts its thread pool can deadlock
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
        rows = list(ex.map(_score_one, jobs, chunksize=1))
    return pl.DataFrame(rows, infer_schema_length=None)


def summary(rows: pl.DataFrame, include_twins: bool = False) -> dict[str, dict]:
    """Official summarise() overall and per embryo, plus the node ratio."""
    if "error" in rows.columns:
        rows = rows.filter(pl.col("error").is_null())
    if not include_twins:
        rows = rows.filter(~pl.col("video").is_in(list(TWINS)))
    out = {}
    for key, sub in [("all", rows)] + [(e, rows.filter(pl.col("embryo") == e)) for e in EMBRYOS]:
        if sub.height == 0:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = summarise(sub.to_dicts())
        s["n_pred/n_est"] = sub["num_pred_nodes"].sum() / sub["n_est"].sum()
        out[key] = s
    return out


def print_summary(rows: pl.DataFrame, include_twins: bool = False) -> None:
    errs = rows.filter(pl.col("error").is_not_null()) if "error" in rows.columns else rows.clear()
    for r in errs.iter_rows(named=True):
        print(f"  ERROR {r['video']}: {r['error']}")
    for key, s in summary(rows, include_twins).items():
        dj = s["division_jaccard"]
        print(f"{key:>4}: n={s['n']:3d}  score={s['score']:.4f}  adjJ={s['adj_edge_jaccard']:.4f}  "
              f"J={s['edge_jaccard']:.4f}  Npred/Nest={s['n_pred/n_est']:.3f}  "
              f"div J={dj if dj == dj else float('nan'):.3f} "
              f"(TP/FP/FN {s['division_tp']}/{s['division_fp']}/{s['division_fn']})  "
              f"recall={s['node_recall']:.4f}")


def diff(a: pl.DataFrame, b: pl.DataFrame, include_twins: bool = False) -> None:
    """Paired per-video comparison of two runs (b minus a) on adjusted edge J."""
    cols = ["video", "embryo", "adj_edge_jaccard", "edge_tp", "edge_fp", "edge_fn",
            "division_tp", "division_fp", "division_fn", "num_pred_nodes"]
    j = a.select(cols).join(b.select(cols), on=["video", "embryo"], suffix="_b")
    if not include_twins:
        j = j.filter(~pl.col("video").is_in(list(TWINS)))
    j = j.with_columns((pl.col("adj_edge_jaccard_b") - pl.col("adj_edge_jaccard")).alias("d_adj"))
    for key, sub in [("all", j)] + [(e, j.filter(pl.col("embryo") == e)) for e in EMBRYOS]:
        d = sub["d_adj"].drop_nans().drop_nulls()
        if d.len() == 0:
            continue
        sd = d.std() if d.len() > 1 else float("nan")
        se = sd / math.sqrt(d.len()) if sd == sd else float("nan")
        print(f"{key:>4}: n={d.len():3d}  mean d(adjJ)={d.mean():+.5f} ± {se:.5f} (se)  "
              f"wins/ties/losses={(d > 1e-9).sum()}/{(d.abs() <= 1e-9).sum()}/{(d < -1e-9).sum()}")
    sa, sb = summary(a, include_twins), summary(b, include_twins)
    for key in sa:
        if key in sb:
            print(f"{key:>4}: score {sa[key]['score']:.4f} -> {sb[key]['score']:.4f} "
                  f"({sb[key]['score'] - sa[key]['score']:+.4f})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--pred", type=Path, required=True)
    r.add_argument("--gt", type=Path, default=GT_DIR)
    r.add_argument("--out", type=Path)
    r.add_argument("--workers", type=int, default=8)
    r.add_argument("--include-twins", action="store_true")
    d = sub.add_parser("diff")
    d.add_argument("--a", type=Path, required=True)
    d.add_argument("--b", type=Path, required=True)
    d.add_argument("--include-twins", action="store_true")
    args = ap.parse_args()

    if args.cmd == "run":
        rows = score(args.pred, args.gt, workers=args.workers)
        if args.out:
            rows.write_csv(args.out)
        print_summary(rows, args.include_twins)
    else:
        diff(pl.read_csv(args.a), pl.read_csv(args.b), args.include_twins)


if __name__ == "__main__":
    main()

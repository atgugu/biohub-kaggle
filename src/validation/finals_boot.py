"""09-29: finals comparison on the big test under different embryo mixes. Per-video official counts (veto_joint --per-video CSVs) for each
candidate; stratified bootstrap of N-video test sets with a given 44b6 share; official score (per-video adjusted edge J + pooled div J)
per draw; paired differences vs the reference. Usage: python finals_boot.py  (reads scratch/union/pvf_*.csv)"""
import sys

import numpy as np
import polars as pl

sys.path.insert(0, "/workspace/biohub/analysis"); sys.path.insert(0, "/workspace/biohub/harness")
from score import GT_DIR, n_estimated  # noqa: E402
from tracking_cellmot.metrics import EvaluationResult, per_sample_metrics, summarise  # noqa: E402

U = "/workspace/biohub/scratch/union"
# candidate -> (csv stem, setting) ; brackets oo / pk
CANDS = {"X3e-safe": ("pvf_x3e_model", "0.05"), "X3e-tv": ("pvf_x3e_vt10", "0.1"), "X3h-tv5": ("pvf_wb5_vt05", "0.1"), "X3h-tv5-v05": ("pvf_wb5_vt05", "0.05")}
extra = [a for a in sys.argv[1:]]   # name=stem:setting
for e in extra:
    n, rest = e.split("="); s, st = rest.split(":"); CANDS[n] = (s, st)
REF = "X3e-safe"; N = 40; B = 2000; rng = np.random.default_rng(0)
nest = {}


def official(counts, vids):
    rows = [per_sample_metrics(EvaluationResult(*[int(x) for x in counts[v]]), nest[v], float("nan")) for v in vids]
    return summarise(rows)["score"]


for br in ("oo", "pk"):
    C = {}
    for name, (stem, st) in CANDS.items():
        try:
            d = pl.read_csv(f"{U}/{stem}_{br}.csv").filter(pl.col("setting") == st)
        except FileNotFoundError:
            print("missing", stem, br); continue
        C[name] = {r[1]: np.array(r[2:9]) for r in d.iter_rows()}
    vids = sorted(set.intersection(*[set(c) for c in C.values()]))
    for v in vids:
        if v not in nest:
            nest[v] = n_estimated(GT_DIR / f"{v}.geff")
    e44 = [v for v in vids if v.startswith("44b6")]; e6 = [v for v in vids if v.startswith("6bba")]
    print(f"\n== bracket {br}: {len(vids)} videos ({len(e44)} 44b6 / {len(e6)} 6bba); full-set official: " + " | ".join(f"{n} {official(c, vids):.4f}" for n, c in C.items()))
    for w in (0.36, 0.5, 0.75, 1.0):
        n44 = int(round(w * N)); diffs = {n: [] for n in C}
        for _ in range(B):
            s = list(rng.choice(e44, n44, replace=True)) + list(rng.choice(e6, N - n44, replace=True))
            ref = official(C[REF], s)
            for n, c in C.items():
                diffs[n].append(official(c, s) - ref)
        print(f"  44b6 share {w:.2f}: " + " | ".join(f"{n} {np.mean(x):+.4f} (P>0 {np.mean(np.array(x) > 0):.2f})" for n, x in diffs.items() if n != REF))

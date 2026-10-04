# Solution

## 1. Where the points were

The public x138 stack already scored ≈ 0.953 public. Measuring it on in-sample training videos showed
where the remaining loss sat:

* **Edges:** 93.2 % of GT edges recovered; 2.6 % lost to an undetected endpoint, 4.2 % to linking.
  Global frame-to-frame drift, frozen frame pairs and registration did not explain the loss.
* **Divisions** (122 GT divisions in 60 videos, 1,070 emitted forks):

| outcome at a true division | share |
|---|:---:|
| correct fork | 12 % |
| one daughter linked, the other linked to a **different parent** (needs rewiring) | 36 % |
| one daughter linked, the other an **orphan** track start | 20 % |
| a daughter has **no node** in the final graph | 29 % |
| parent undetected / linked to neither | 3 % |

The ILP never creates divisions: every fork comes from a post-hoc "safe division" pass with fixed geometric
gates. Divisions are therefore a *post-processing* problem, solvable on any base's output CSV — and, per
[METRIC.md](METRIC.md), the metric rewards aggressive recall of forks while ignoring forks without GT evidence.

## 2. The stack

```
base graph ──► candidates ──► ranker ──► attach ──► fork veto ──► submission
 (x138)         p→(c1, c)      LightGBM    top-10      LightGBM       (CSV)
                               ×3 seeds    ≥ 0.5       + TripletNet
```

### Candidates ([`src/ranking/div_candidates.py`](../src/ranking/div_candidates.py))
A candidate is `(p, c1, c)`: a parent `p` with exactly one child `c1`, plus a node `c` at `t+1` with
`d(p, c) ≤ 13 µm` and `d(c1, c) ≤ 17 µm` (the p99 of true geometry). `c` is either an orphan track start or is
already linked to another parent `q` (applying the candidate then *rewires* `q→c` to `p→c`). Labels follow a
fast proxy of the official rule: `+1` if `p` matches a GT dividing node and `{c1, c}` its two daughters, `0` if
`p` matches a GT node with children otherwise (would be an FP), and `−1` (neutral, excluded) if there is no GT
evidence at `p`. About 195 videos × 80 k candidates; only 78 positives.

### Features
* **Geometry** — distances, sister symmetry, velocity, track lengths of `p`, `q`, `c`, density.
* **Linker evidence** ([`predict_cache.py`](../src/inference/predict_cache.py),
  [`linker_feats.py`](../src/ranking/linker_feats.py)) — the base linker's logit for `p→c`, `q→c`, `p→c1`, the best
  competing source for `c` and the margin, converted to the pack's softmax-over-sources probability.
* **DivNet v3** ([`divnet_v3.py`](../src/models/divnet_v3.py)) — a 3D U-Net mitosis classifier on a
  16 × 48 × 48 crop (xy max-pool 2, per-frame percentile normalisation, lags −1…+2) with a Gaussian marker on the
  parent; centre jitter, flips/transposes and intensity augmentation; hard negatives mined from the ranker's own
  false positives. The public DivNet v2 checkpoint turned out to be loaded with random weights by the community
  notebooks (0 of 50 tensors matched under `strict=False`); [`divnet.py`](../src/models/divnet.py) reconstructs its
  architecture properly (AUC 0.82 on dividing vs. non-dividing GT nodes).
* **DaughterNet** ([`dnet_score.py`](../src/models/dnet_score.py)) — the same recipe with target "this node is a
  daughter", scored on `c` and `c1`.

### Ranker + attach
Three LightGBM seeds averaged (`ranker_all_u3v3d_s0..s2`, trained on all videos with *out-of-fold* image
features), scored per video; the top 10 with score ≥ 0.5 are attached. Image features are computed by two GPU
shards in the notebook with a wall-clock budget; each finished video is written as a part file so a killed shard
keeps its completed videos.

### Fork veto
Every fork in the post-attach graph is scored by a LightGBM trained on *labelled forks* (TP vs. evaluable FP via
the official metric) with fork geometry + an out-of-fold DivNet logit of the parent. The final model blends in
**TripletNet** ([`trinet.py`](../src/models/trinet.py)) — DivNet v3's recipe with six input channels (four image
lags, a parent marker and a *child* marker at both claimed daughters), trained on GT positives, fake-sister
negatives and real pipeline rows — so it judges the specific hypothesis rather than "is this parent mitotic":

```
p_keep = σ( logit(p_lgbm) + γ·triplet_logit ),   γ = 1,   remove the farther daughter edge if p_keep < 0.1
```

## 3. Validation

Every community weight was trained on the training videos, so a local score on them is in-sample and can
mislead: flip-only test-time augmentation measured +0.0126 over the base's 8-view TTA on training videos, then
scored *below* its base on the leaderboard (0.949 vs. 0.953) and on a held-out-embryo check. Three rules made the
local numbers trustworthy (rules 1–2 held throughout; rule 3, the 195-video big test, was built on 27 Sep and
gated the later candidates — the R-line, `X1h` and `X3d` were decided on smaller held-out sets):

1. **Official metric only** — the host's scorer, per video, per embryo (`src/validation/score.py`).
2. **By-video out-of-fold learning** — every learned component (rankers, DivNets, DaughterNet, TripletNet, veto)
   is trained 5-fold by video and applied only to its held-out videos.
3. **Two brackets on a big test** — all 195 videos, with linker features from a model that *never saw that
   embryo* ("out-of-sample") and from the community model ("in-sample"). A change needed to win in both brackets
   and in both embryo families.

Big-test results (official score; out-of-sample / in-sample):

| candidate | vs. | Δ |
|---|:---:|:---:|
| `X3e` — DivNet v3 + DaughterNet + 3-seed ranker | `X1h` | +0.0055 / +0.0094 |
| `X3e-tv` — `X3e-safe` + TripletNet-aware veto (**final pick**) | `X3e-safe` | +0.0011 / +0.0025 |
| `X3h-tv` — wide ranker ensemble + TripletNet blend + TripletNet-aware veto | `X1h` | +0.0103 / +0.0139 |
| `X3h-tv5` — same with a 5-seed TripletNet blend | `X1h` | +0.0113 / +0.0135 |

The finals were chosen with a bootstrap over simulated hidden sets (43 videos, unknown embryo mix) using the
public reads as a weak prior; [`finals_boot.py`](../src/validation/finals_boot.py) is that bootstrap. The
wide-ensemble family looked best on the big test yet transferred slightly worse than the simpler `X3e` family on
the private board (0.9308–0.9310 vs. 0.9321) — the big test is still only 195 training videos from the same
embryos.

## 4. Operational notes

* The submitted kernel keeps the whole post-processing inside wall-clock guards (attach budgeted to ≈ 10.8 h, veto to
  11.5 h of the 12 h limit). Any failed stage restores the last good graph — and a failed veto restores the
  *pre-attach* graph, never an attached-but-unvetted one.
* The community base's coordinate-refinement head was a private artifact; we trained our own from the base
  notebook's captured features ([`v1284_train.py`](../src/models/v1284_train.py)). The submitted notebook runs with
  it when mounted and in "zero" mode otherwise.

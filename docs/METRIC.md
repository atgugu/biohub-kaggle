# The metric, and why divisions are worth chasing

Source: the host's official scorer ([`royerlab/kaggle-cell-tracking-competition`](https://github.com/royerlab/kaggle-cell-tracking-competition),
commit `075fc5f5`), wrapped without modification by [`src/validation/score.py`](../src/validation/score.py).

```
score = adjusted_edge_J  +  0.1 · division_J
```

## Edges

* Predicted nodes are matched to ground-truth (GT) nodes **per frame** by optimal bipartite assignment on
  scaled distance (µm), with a **7 µm gate** (≈ 17 px in xy, ≈ 4 voxels in z — loose).
* **TP** — a predicted edge whose two endpoints match two GT nodes joined by a GT edge. **FN** — every GT edge
  not so matched.
* **FP** — a non-TP predicted edge counts *only if it contradicts the annotated lineage* (its source matches a GT
  node that has an outgoing edge, or its target matches a GT node with an incoming edge). Edges touching only
  unannotated cells are **free**.
* `J = TP / (TP + FP + FN)`, per video, averaged over videos weighted by `TP + FP + FN`.
* Edges spanning ≠ 1 frame are silently dropped; > 2 children are truncated; duplicates are deduplicated.

### Node-count penalty

```
adj = max(0, J · (1 − 0.1 · (N_pred − N_est) / N_est))
```

`N_est` is the host's estimate of the number of cells in the video (from the geff metadata; ~10 k–33 k). There is
no upper clip. Every extra 1 % of nodes costs 0.1 % of `J` (≈ 0.001 score), so isolated low-confidence
detections are pure cost, while a missed *annotated* edge costs ≈ 2 % of that video's `J`.

## Divisions

A **fork** is any predicted node with two outgoing edges; a **GT division** is a GT node with two children.
A fork is a **TP** if it pairs with a GT division in a local window (grandparent → parent → children →
grandchildren, one frame early or late allowed) with the two daughter lineages landing on two *different*
predicted branches. An unpaired fork is an **FP** only if it has GT evidence against it (its parent matches a
GT node with outgoing edges, it is a failed candidate for a GT division, or its branches land in different GT
components / merge). **A fork with no GT evidence anywhere near it is free.**

Division counts are micro-summed over *all* videos, then `division_J = TP / (TP + FP + FN)`.

## Arithmetic of adding a fork

With split totals around `TP : FP : FN ≈ 5 : 3 : 40` (so `division_J ≈ 0.10`; `T = TP + FP + FN`):

| event | change in `division_J` | change in score |
|---|:---:|:---:|
| one more **correct** fork | + (FP + FN) / T² ≈ **+0.019** | **+0.0019** |
| one more **evaluable wrong** fork | − TP / T² ≈ **−0.0022** | **−0.0002** |
| a fork with **no GT evidence** | 0 | 0 |

A correct fork is worth ≈ 8.6× what a wrong evaluable one costs, so a division ranker pays from
**≈ 10 % precision** among evaluable forks — and unevidenced forks are free. This asymmetry is the whole
reason the *attach* stage can be aggressive (top-10 per video), and the reason the *veto* exists: it removes
the existing forks that are most likely evaluable-and-wrong.

## Data facts that shape validation

| | `44b6` | `6bba` |
|---|:---:|:---:|
| training videos | 71 | 128 |
| median annotated share of cells | 0.8 % | 9.7 % |
| median `N_est` per video | 32.7 k | 9.7 k |
| videos with a division | 21 | 66 |

151 GT divisions in total; 85 % of GT edges are in `6bba`, so a pooled number is dominated by one embryo family —
every comparison here is reported **per embryo**.

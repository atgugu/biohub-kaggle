# Experiments: what moved the score and what didn't

Public/private are the Kaggle scores of the submitted kernel (all 40 are in
[`../data/submissions.csv`](../data/submissions.csv)). "Big test" = the 195-video by-video out-of-fold evaluation
described in [SOLUTION.md](SOLUTION.md#3-validation).

## The ladder

| step | public | private | note |
|---|:---:|:---:|---|
| public notebook baseline (`s2`: validator + sweep off) | 0.94610 | 0.91275 | the floor; the posted 0.947 was one tick of noise |
| geometric-fusion rebase (`R1`) | 0.94731 | 0.91820 | +0.0012 / +0.0054 |
| + fork veto thr 0.05 (`R4`) | 0.94923 | 0.92044 | removes ≈ half of all forks; honest-feature LightGBM |
| + attach v1 and ensemble veto (`R6`) | 0.94729 | 0.92457 | public *down*, private up: first sign the boards disagree on divisions |
| x138 base + our coordinate head + R4 veto (`X1h`) | 0.95380 | 0.92604 | |
| + DivNet v3 attach (`X3d-t50`) | 0.95655 | 0.93141 | first honest-CV winner on both boards |
| + DaughterNet, 3-seed ranker (`X3e-t50`, `X3e-safe`) | 0.95834 | 0.93185 | |
| **+ TripletNet-aware veto (`X3e-tv`)** | **0.96086** | **0.93206** | submitted kernel |

## Levers that worked

| lever | evidence |
|---|---|
| Rebase onto a stronger public base (x138) + our coordinate head + R4 veto (`X1h`) | +0.0046 public / +0.0056 private over `R4` |
| Fork veto (LightGBM on fork geometry + honest DivNet) | +0.002 public / +0.002 private |
| Candidate-attach with *linker evidence* features | the base linker's `p→c` logit separates real second daughters; in a linker stress test the attach kept ≈ 1/3 of its `6bba` gain with out-of-sample linker evidence and ≈ 0 without linker features |
| DivNet v3 (finer xy pooling, jitter, hard negatives) | `X1h` → `X3d-t50`: +0.0028 public / +0.0054 private |
| DaughterNet features | `X3d-t50` → `X3e-t50`: +0.0018 public / +0.0004 private; `X3e-t50` is +0.0078 / +0.0139 vs `X1h` on the big test |
| TripletNet-aware veto | +0.0025 public, ≈ +0.0002 private; raises the safe veto threshold from 0.05 to 0.1 without losing `44b6` |

## Dead ends (each measured, none shipped)

| idea | result |
|---|---|
| Public knobs: detection threshold 0.96, secondary-edge weight 0.20, DeepCenter threshold | 0.946 public — no effect on the 0.946 base |
| Relative-rank bonus on edge logits | 0.946 — no effect |
| **Fine-tuned linker** on all 199 videos | **−0.003 public / −0.004 private** (`s6`, `s8`): a +0.001 cross-embryo gain did not survive the public post-processing, which is tuned to the original linker's calibration |
| Label-cleaned linker fine-tune | identical to plain fine-tune once the cleaner bug was fixed |
| Training the detector from scratch | plateaued at 0.63 on held-out `6bba` vs. 0.905 for the community weights (raw ILP, same protocol) |
| Division attach v1 (geometry + giorgosi DivNet) on the old base (`s11`) | −0.002 public (+0.002 private); geometry alone cannot shortlist (22 of 80 positives in the top 300) |
| Raw-ILP-only output | 0.897 public — the repair passes matter |
| **Flip-only test-time augmentation** | +0.0126 in-sample, **0.949 vs 0.953 public**, negative on a held-out-embryo check |
| D8-complete view policies (`K1`, `K3`) | 0.9496 public vs. 0.9538 for `X1h`; in a held-out check a single transposed view cost 79 edges (−0.0045) |
| Temporal detection TTA | −0.0025 |
| Plain veto at threshold 0.1 (`X3e-v10`) | −0.0019 public / −0.0012 private vs. 0.05 — cuts real `44b6` forks (fixed by the TripletNet term) |
| Learned stage-2 re-ranker | overfit |
| Attach with k > 10 per video | no gain |
| Readmitting more discarded detections (min score 0.94) | ≈ 0 public (0.95656 vs. 0.95643), +0.0014 private — inside the noise, not adopted |
| Wide ranker ensemble (`X3h`) | best on the big test, below `X3e` on both boards |

## Lessons

1. **Distrust anything measured on videos the models were trained on.** The in-sample/out-of-sample brackets
   caught flip-TTA, the fine-tuned linker and the attach-on-raw-graphs failure before they cost more slots.
2. **The two boards measure different things.** The public subset has few divisions; edge-level knobs show on it,
   division work often doesn't. Decide on the big test, use the public board as a smoke test and a prior.
3. **Report per embryo.** `6bba` holds 85 % of GT edges; a pooled gain can hide a `44b6` loss (this is what the
   TripletNet-aware veto exists to prevent).
4. **Exploit the metric's asymmetries** — unevidenced forks are free, a correct fork is worth ≈ 8.6 wrong ones —
   but only with a ranker whose out-of-fold precision you have measured.
5. **Guard the wall clock.** Hidden-set scoring takes 5–11 h; every stage writes part files and falls back to the
   last good graph.

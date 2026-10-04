"""Cache a checkpoint's raw predictions per video, for CPU-side work (node budget, division ranker, ILP sweeps).

Per video -> <out>/<video>.npz:
  nodes      (N, 4) int16   t, z, y, x in original voxels (peaks of the 4-view TTA detection map)
  det_prob   (N,)   float32 sigmoid detection probability at the peak (all peaks above --det-floor)
  e_src,e_tgt (E,)  int32   node indices of candidate edges t -> t+1 with distance <= --radius-um
  e_logit    (E,)   float32 raw linker logit
  e_dist     (E,)   float32 distance in um
  lse_src    (N,)   float32 logsumexp of logits over ALL sources, per target node (NaN for t=0),
                            so p(edge) = exp(e_logit - lse_src[e_tgt]) is the pack's softmax-over-sources
The linker sees every node above --det-floor (its logits depend mildly on the node set via cross-attention).

Usage: python predict_cache.py --ckpt <.pth> --videos a,b | --fold K --which test|heldout  --out <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import os
REPO = Path(os.environ.get("BIOHUB_REPO", "/workspace/biohub/train/repo"))
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import zarr  # noqa: E402
import numcodecs.blosc; numcodecs.blosc.use_threads = False  # noqa: E402  (blosc threads race under load)

DATA = Path("/workspace/data/train")
SCALE = np.array([1.625, 0.40625, 0.40625], dtype=np.float32)


@torch.no_grad()
def cache_video(model, ds_path: Path, device, W: int, downsample, det_floor: float, radius_um: float,
                pool_um: float = 3.0) -> dict:
    from biohub_tracking.io import open_dataset
    from predict_unet_transformer import _load_frame, pool_kernel_from_um
    from train_unet_transformer import extract_pos_features

    ds = open_dataset(ds_path, normalize=False, load_image=False, downsample=downsample)
    zarr_arr = zarr.open_group(str(ds.zarr_path), mode="r")["0"]
    q_low, q_high = float(ds.quantiles["0.001"]), float(ds.quantiles["0.999"])
    T = ds.image_shape[0]
    target_shape = list(ds.image_shape[1:])
    ds_arr = np.array(downsample, dtype=np.float32)
    ds_arr_t = torch.from_numpy(ds_arr).to(device)
    voxel_size = tuple(s * d for s, d in zip(ds.scale, downsample))
    pool_k = pool_kernel_from_um(pool_um, voxel_size)
    pad = tuple(k // 2 for k in pool_k)
    window_shape = (W,) + tuple(target_shape)
    assert W == 2, "cache assumes window_size 2 (one frame pair per window)"

    peaks: dict[int, tuple[np.ndarray, np.ndarray]] = {}   # t -> (idx (n,3) downsampled, prob (n,))
    offset: dict[int, int] = {}
    n_total = 0
    e_src, e_tgt, e_logit, e_dist, lse = [], [], [], [], {}
    for ws in range(0, T - 1):
        fr = [ws, ws + 1]
        imgs = torch.stack([_load_frame(zarr_arr, t, target_shape, downsample) for t in fr])
        imgs = ((imgs - q_low) / (q_high - q_low + 1e-6)).clamp(0.0).unsqueeze(0).to(device)
        unet_out, det = model.encode(imgs)
        for dims in [(-1,), (-2,), (-2, -1)]:
            _, df = model.encode(imgs.flip(dims))
            for f in range(W):
                det[f] = det[f] + df[f].flip(dims)
        det = [d / 4 for d in det]
        for f_idx, t in enumerate(fr):
            if t in peaks:
                continue
            lg = det[f_idx][0].unsqueeze(0)
            prob = torch.sigmoid(lg)
            is_peak = (lg == F.max_pool3d(lg, pool_k, stride=1, padding=pad)) & (prob > det_floor)
            idx = torch.nonzero(is_peak[0, 0])
            peaks[t] = (idx.cpu().numpy().astype(np.int16), prob[0, 0][idx[:, 0], idx[:, 1], idx[:, 2]].cpu().numpy())
            offset[t] = n_total
            n_total += len(idx)
        (cs, _), (ct, _) = peaks[fr[0]], peaks[fr[1]]
        if len(cs) == 0 or len(ct) == 0:
            continue
        ps = torch.from_numpy(cs.astype(np.float32)).unsqueeze(0).to(device)
        pt = torch.from_numpy(ct.astype(np.float32)).unsqueeze(0).to(device)
        cs4 = np.concatenate([np.zeros((len(cs), 1), np.int16), cs], axis=1)
        ct4 = np.concatenate([np.ones((len(ct), 1), np.int16), ct], axis=1)
        pos_s = torch.from_numpy(extract_pos_features(cs4, window_shape)).unsqueeze(0).to(device)
        pos_t = torch.from_numpy(extract_pos_features(ct4, window_shape)).unsqueeze(0).to(device)
        ms = torch.ones(1, len(cs), dtype=torch.bool, device=device)
        mt = torch.ones(1, len(ct), dtype=torch.bool, device=device)
        fs = model._index_features(unet_out[:, 0], ps, ms)
        ft = model._index_features(unet_out[:, 1], pt, mt)
        lo = model.predict_edges(fs, ft, ps * ds_arr_t, pt * ds_arr_t, pos_s, pos_t, ms, mt)[0].float()
        lse[fr[1]] = torch.logsumexp(lo, dim=0).cpu().numpy()
        um = torch.from_numpy(SCALE * ds_arr).to(device)
        d = torch.cdist(ps[0] * um, pt[0] * um)
        ii, jj = torch.nonzero(d <= radius_um, as_tuple=True)
        e_src.append((ii + offset[fr[0]]).cpu().numpy().astype(np.int32))
        e_tgt.append((jj + offset[fr[1]]).cpu().numpy().astype(np.int32))
        e_logit.append(lo[ii, jj].cpu().numpy())
        e_dist.append(d[ii, jj].cpu().numpy())

    nodes = np.concatenate([
        np.concatenate([np.full((len(peaks[t][0]), 1), t, np.int16),
                        (peaks[t][0].astype(np.float32) * ds_arr).astype(np.int16)], axis=1)
        for t in range(T)])
    det_prob = np.concatenate([peaks[t][1] for t in range(T)]).astype(np.float32)
    lse_all = np.full(len(nodes), np.nan, np.float32)
    for t, v in lse.items():
        lse_all[offset[t]:offset[t] + len(v)] = v
    cat = lambda xs, dt: np.concatenate(xs).astype(dt) if xs else np.zeros(0, dt)  # noqa: E731
    return dict(nodes=nodes, det_prob=det_prob, e_src=cat(e_src, np.int32), e_tgt=cat(e_tgt, np.int32),
                e_logit=cat(e_logit, np.float32), e_dist=cat(e_dist, np.float32), lse_src=lse_all)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--videos", default=None)
    ap.add_argument("--fold", type=int, default=None, help="with --which: fold index in embryo_splits.json")
    ap.add_argument("--which", choices=["test", "heldout"], default="heldout",
                    help="test = the monitored subset; heldout = every video of the embryo the fold did NOT train on")
    ap.add_argument("--det-floor", type=float, default=0.5)
    ap.add_argument("--radius-um", type=float, default=20.0)
    ap.add_argument("--downsample", default=None, help="override the checkpoint config, e.g. 1,2,2")
    ap.add_argument("--data-dir", default=None, help="directory holding <video>.zarr (default: train)")
    args = ap.parse_args()
    global DATA
    if args.data_dir:
        DATA = Path(args.data_dir)

    from predict_unet_transformer import load_model
    if args.videos:
        videos = args.videos.split(",")
    else:
        fold = json.load(open("/workspace/biohub/train/embryo_splits.json"))[args.fold]
        if args.which == "test":
            videos = fold["test"]
        else:
            train = set(fold["train"])
            videos = sorted(p.stem for p in DATA.glob("*.geff") if p.stem not in train)
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    model, W, ds = load_model(args.ckpt, device)
    if args.downsample:
        ds = tuple(int(x) for x in args.downsample.split(","))
    for i, v in enumerate(videos):
        dst = args.out / f"{v}.npz"
        if dst.exists():
            continue
        t0 = time.time()
        res = cache_video(model, DATA / v, device, W, ds, args.det_floor, args.radius_um)
        np.savez_compressed(dst, **res)
        print(f"[{i + 1}/{len(videos)}] {v}: nodes {len(res['nodes'])} edges {len(res['e_src'])} "
              f"{time.time() - t0:.0f}s", flush=True)
    (args.out / "meta.json").write_text(json.dumps(dict(ckpt=str(args.ckpt), det_floor=args.det_floor,
                                                        radius_um=args.radius_um, n_videos=len(videos))))


if __name__ == "__main__":
    main()

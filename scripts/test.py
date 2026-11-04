"""Test GSCV at native resolution on PNG test sets (folders with a description.json, see scripts/prepare_testsets.py).

Every sequence is coded with an I-frame (DLF) every --intra_period frames and GSCV P-frames in between; the window
state machine (modeling/gscv/rsm.py) runs on the source frames, restarting at every I-frame. Frames are edge-padded
to multiples of 256; the metrics (scripts/metrics.py) are computed on the original area of the 8-bit
reconstructions, and bpp counts the bits of the I-frames, of the coded tokens and of the side information per
original pixel. The bits are the estimates of the entropy model (as GVC1D).

The paper's setting (the defaults): 96 frames, intra period 32, quality indices 0-7, FP16 weights.

    python scripts/test.py --sets data/testsets/HEVC_F --fid_kid --out results/gscv
    python scripts/test.py --sets ... --shard 0/4 --gpu 0 --out results/gscv     # 4 processes: shards 0/4 .. 3/4
    python scripts/test.py --merge --out results/gscv                             # join results_shard*.json

results.json: {"args": ..., "sequences": {path: {set, category, width, height, frames, rd: {qp: {summary,
per_frame}}}}}; --fid_kid also saves the Inception patch features of every reconstructed frame for
scripts/fid_kid.py, --save_recon the reconstructed frames.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from metrics import FrameScorer, mean_metrics  # noqa: E402
from modeling.gscv.data import load_description  # noqa: E402
from modeling.gscv.rsm import (REFRESH, REVERT, SKIP, TEXTURE, UPDATE, RSMConfig, RSMEncoder,  # noqa: E402
                               block_classes, block_map_to_pixels, pad_to_multiple)

INDEX_MAP = [0, 1, 0, 2, 0, 2, 0, 2]   # frame-type cycle of GVC1D's test.py (rate_gop_size = 8)
RSM_KEYS = ("modes", "revert_slot", "snap_create", "mem_type", "mem_age")
SIG_KEYS = ("config", "ckpt", "iframe_ckpt", "frames", "intra_period", "fp16", "fid_kid")   # resumed / merged
EVAL_CLASSES = RSMConfig()     # fidelity-critical regions of psnr_fid: the default classifier, 64x64 blocks


def resolve(path):
    """A path as given (absolute, or relative to the working directory), else relative to the repository root."""
    if os.path.isabs(path) or os.path.exists(path):
        return os.path.abspath(path)
    return os.path.join(REPO, path)


def fidelity_masks(frames):
    """Per-frame bool maps (H, W) of the fidelity-critical pixels of the source frames, the same for every codec."""
    h, w = frames[0].shape[:2]
    return [block_map_to_pixels(block_classes(f, EVAL_CLASSES) != TEXTURE, h, w, EVAL_CLASSES.block)
            for f in frames]


def read_frames(seq, base, max_frames=None):
    names = seq["frames"] if max_frames is None else seq["frames"][:max_frames]
    out = []
    for name in names:
        with Image.open(os.path.join(base, seq["path"], name)) as im:
            out.append(np.asarray(im.convert("RGB")))
    return out


def to_uint8(x):
    """(1, 3, H, W) float in [0, 1] -> uint8 (H, W, 3)."""
    return x[0].float().clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()


def build(args, device):
    from omegaconf import OmegaConf
    from modeling.gscv.iframe import load_iframe_codec
    from modeling.gscv.model import GSCV
    cfg = OmegaConf.load(args.config)
    model = GSCV(cfg)
    missing, unexpected = model.load_weights(args.ckpt)
    if missing or unexpected:
        print(f"WARNING loading {args.ckpt}: missing {missing}, unexpected {unexpected}")
    model = model.to(device).eval()
    if args.fp16:
        model = model.half()
    rcfg = RSMConfig.from_dict(OmegaConf.to_container(cfg.rsm, resolve=True))
    model.opts["k_snap"] = max(int(rcfg.k_snap), 1)
    return model, load_iframe_codec(args.iframe_ckpt, device), rcfg


def iframe(icodec, frame, args, device):
    x = torch.from_numpy(frame).permute(2, 0, 1)[None].float().div(255).to(device)
    if args.fp16:                        # as GVC1D's test.py: the frames are fp16 before the I-frame codec sees them
        x = x.half().float()
    rec, bpp = icodec.forward_one_frame(x, return_bpp=True)
    return rec, float(bpp) * x.shape[-2] * x.shape[-1]


@torch.no_grad()
def code_sequence(model, icodec, scorer, frames, fid, qp, args, rcfg, device, seq_path):
    h, w = frames[0].shape[:2]
    padded = [pad_to_multiple(f) for f in frames]
    hp, wp = padded[0].shape[:2]
    dtype = torch.float16 if args.fp16 else torch.float32
    t0 = time.time()
    save_dir = None
    if args.save_recon:
        save_dir = os.path.join(args.out, "recon", f"qp{qp}", seq_path.replace("/", "__"))
        os.makedirs(save_dir, exist_ok=True)
    bits, side, rows, feats, i_ts = [], [], [], [], []
    dpb = enc = None
    for t in range(len(frames)):
        if t == 0 or (args.intra_period > 0 and t % args.intra_period == 0):
            # I-frame (DLF); the contexts, the memory and the window state machine start afresh
            rec, i_bits = iframe(icodec, padded[t], args, device)
            enc = RSMEncoder(rcfg, hp, wp, (h, w))
            side_bits = enc.start(padded[t])["side_bits"]
            dpb = {"ref_frame": rec.float(), "ref_feature": None, "long_term_mem": None}
            frame_bits, extra = i_bits + side_bits, {"frame_type": "I"}
            i_ts.append(t)
        else:
            cur = torch.from_numpy(padded[t]).permute(2, 0, 1)[None].to(device).to(dtype).div(255)
            info = enc.step(padded[t])
            qp_win = np.clip(qp + info["qp_offset"], 0, 7).astype(np.int64)
            rsm = {k: torch.from_numpy(np.asarray(info[k])).to(device) for k in RSM_KEYS}
            rsm["fclass"] = torch.from_numpy(info["f_eff"].astype(np.int64))[None].to(device)
            out = model.forward_one_frame(cur, dpb, torch.from_numpy(qp_win).to(device), INDEX_MAP[t % 8], rsm)
            dpb = out["dpb"]
            side_bits = float(info["side_bits"])
            frame_bits = float(out["bits_win"].float().sum()) + side_bits
            m = out["modes"].cpu().numpy()   # without a decoded feature (first P-frame) Skip / Revert are coded
            extra = {"frame_type": "P", "update_ratio": float(np.isin(m, (UPDATE, REFRESH)).mean()),
                     "skip_ratio": float((m == SKIP).mean()), "refresh_ratio": float((m == REFRESH).mean()),
                     "revert_ratio": float((m == REVERT).mean())}
        bits.append(frame_bits)
        side.append(side_bits)
        rec_u8 = to_uint8(dpb["ref_frame"])[:h, :w]
        if scorer.patch_features is not None:
            feats.append(scorer.patch_features(rec_u8).astype(np.float32))
        rows.append(dict(scorer.score(frames[t], rec_u8, fid[t]), bits=frame_bits, **extra))
        if save_dir:
            Image.fromarray(rec_u8).save(os.path.join(save_dir, f"im{t + 1:05d}.png"))

    summary = mean_metrics([{k: v for k, v in r.items() if k not in ("bits", "frame_type")} for r in rows])
    n = h * w * len(frames)
    p_ts = [t for t in range(len(frames)) if t not in set(i_ts)]
    summary.update(bpp=float(np.sum(bits) / n), bpp_side=float(np.sum(side) / n),
                   bpp_i_frame=float(np.mean([bits[t] for t in i_ts]) / (h * w)),
                   bpp_p_frames=float(np.sum([bits[t] for t in p_ts]) / (h * w * max(len(p_ts), 1))),
                   n_i_frames=len(i_ts), seconds=time.time() - t0)
    return summary, rows, (np.concatenate(feats) if feats else None)


def balanced_shards(seqs, n, frames=0):
    """Shard of every sequence: the largest (frames x pixels) first, each onto the least loaded shard."""
    size = [int(s["width"]) * int(s["height"]) * (min(frames, len(s["frames"])) if frames > 0 else len(s["frames"]))
            if "width" in s and "height" in s else 1 for s in seqs]
    load, out = [0] * n, [0] * len(seqs)
    for k in sorted(range(len(seqs)), key=lambda k: (-size[k], k)):
        i = min(range(n), key=lambda i: (load[i], i))
        out[k] = i
        load[i] += size[k]
    return out


def merge(out_dir):
    shards = sorted(glob.glob(os.path.join(out_dir, "results_shard*.json")))
    if not shards:
        print(f"no shard files in {out_dir}")
        return
    results = {"sequences": {}}
    for p in shards:
        with open(p) as f:
            r = json.load(f)
        results.setdefault("args", r.get("args"))
        diff = {k: (results["args"].get(k), r["args"].get(k)) for k in SIG_KEYS
                if results["args"].get(k) != r["args"].get(k)}
        if diff:
            raise SystemExit(f"{p} was written with other settings than {shards[0]}: {diff}")
        for k, v in r["sequences"].items():
            e = results["sequences"].setdefault(k, {kk: vv for kk, vv in v.items() if kk != "rd"} | {"rd": {}})
            e["rd"].update(v["rd"])
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f)
    print(f"merged {len(results['sequences'])} sequences into {out_dir}/results.json")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/gscv.yaml")
    p.add_argument("--ckpt", default="checkpoints/gscv.pt")
    p.add_argument("--iframe_ckpt", default="ckpts/I_frame/DLF.ckpt")
    p.add_argument("--sets", nargs="+", default=[], help="test-set folders containing description.json")
    p.add_argument("--sequences", nargs="*", default=None, help="only these sequence paths")
    p.add_argument("--qps", type=int, nargs="+", default=list(range(8)), help="base quality indices (0-7)")
    p.add_argument("--frames", type=int, default=96, help="frames per sequence (<= 0: all)")
    p.add_argument("--intra_period", type=int, default=32, help="an I-frame every N frames (<= 0: only the first)")
    p.add_argument("--fid_kid", action="store_true", help="save the patch features of FID / KID (scripts/fid_kid.py)")
    p.add_argument("--save_recon", action="store_true", help="save the reconstructed frames as PNG")
    p.add_argument("--no_fp16", dest="fp16", action="store_false", help="fp32 weights (default: fp16)")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--shard", default="0/1", help="i/n: this process codes the i-th of n balanced shards")
    p.add_argument("--allow_short", action="store_true", help="accept sequences with fewer than --frames frames")
    p.add_argument("--out", required=True)
    p.add_argument("--merge", action="store_true", help="join the shard results in --out and exit")
    args = p.parse_args()
    if args.merge:
        merge(os.path.abspath(args.out))
        return
    for k in ("config", "ckpt", "iframe_ckpt"):
        setattr(args, k, resolve(getattr(args, k)))
    args.out = os.path.abspath(args.out)
    args.sets = [resolve(s) for s in args.sets]
    os.chdir(REPO)
    os.makedirs(args.out, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(device)
    model, icodec, rcfg = build(args, device)
    scorer = FrameScorer(device, fid_kid=args.fid_kid)
    shard_i, shard_n = (int(v) for v in args.shard.split("/"))
    results_path = os.path.join(args.out, f"results_shard{shard_i}of{shard_n}.json" if shard_n > 1 else "results.json")
    signature = {k: getattr(args, k) for k in SIG_KEYS}
    if os.path.exists(results_path):     # resume: only with the same settings
        with open(results_path) as f:
            results = json.load(f)
        old = {k: results.get("args", {}).get(k) for k in SIG_KEYS}
        if old != signature:
            raise SystemExit(f"{results_path} was written with other settings "
                             f"{ {k: (old[k], signature[k]) for k in SIG_KEYS if old[k] != signature[k]} }")
    else:
        results = {"args": vars(args), "sequences": {}}
    todo = []
    for set_dir in args.sets:
        seqs, base = load_description(set_dir)
        todo += [(set_dir, base, seq) for seq in seqs if not args.sequences or seq["path"] in args.sequences]
    shard_of = balanced_shards([seq for _, _, seq in todo], shard_n, args.frames)
    for k, (set_dir, base, seq) in enumerate(todo):
        if shard_of[k] != shard_i:
            continue
        frames = read_frames(seq, base, args.frames if args.frames > 0 else None)
        if args.frames > 0 and len(frames) < args.frames:
            msg = f"{seq['path']}: {len(frames)} of {args.frames} frames listed in {set_dir}/description.json"
            if not args.allow_short:
                raise SystemExit(msg + " (--allow_short accepts it)")
            print("WARNING " + msg)
        fid = fidelity_masks(frames)
        entry = results["sequences"].setdefault(seq["path"], {
            "set": os.path.basename(os.path.normpath(set_dir)), "category": seq.get("category", ""),
            "width": int(frames[0].shape[1]), "height": int(frames[0].shape[0]), "frames": len(frames), "rd": {}})
        for qp in args.qps:
            if str(qp) in entry["rd"]:
                continue
            summary, rows, feats = code_sequence(model, icodec, scorer, frames, fid, qp, args, rcfg, device,
                                                 seq["path"])
            entry["rd"][str(qp)] = {"summary": summary, "per_frame": rows}
            if feats is not None:
                os.makedirs(os.path.join(args.out, "fid_kid_features"), exist_ok=True)
                np.save(os.path.join(args.out, "fid_kid_features", f"{seq['path'].replace('/', '__')}_qp{qp}.npy"),
                        feats)
            print(f"{seq['path']:<50} qp={qp} bpp={summary['bpp']:.5f} psnr={summary['psnr']:.2f} "
                  f"lpips={summary['lpips']:.4f} dists={summary['dists']:.4f} siqm={summary['siqm']:.4f} "
                  f"({summary['seconds']:.0f}s)", flush=True)
            with open(results_path + ".tmp", "w") as f:
                json.dump(results, f)
            os.replace(results_path + ".tmp", results_path)
            torch.cuda.empty_cache()
    print(f"saved {results_path}")


if __name__ == "__main__":
    main()

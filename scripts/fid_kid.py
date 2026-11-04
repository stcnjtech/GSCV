"""FID and KID per test set and rate point (as GVC1D).

Features: 2048-d Inception-v3 pool features of overlapped 256x256 patches (metrics.PatchFeatures, GVC1D's
evaluator/inception.py; the patch grid at offset 0 plus the grid shifted by 128 pixels). scripts/test.py --fid_kid
saves them for every reconstructed frame in <results>/fid_kid_features/<sequence>_qp<qp>.npy; this script extracts
the same patches of the source frames (cached in --out), pools source and reconstruction features over all
sequences of a set at each rate point and computes
  FID : as GVC1D's evaluator.VQGANEvaluator (means, covariances with N - 1, scipy sqrtm)
  KID : torch-fidelity's kid_features_to_metric (100 subsets of min(1000, N) features, cubic polynomial kernel)
The bpp of a point is the mean bpp of the set's sequences at that rate point.

    python scripts/fid_kid.py --sets data/testsets/HEVC_F ... --results GSCV=results/gscv VTM=results/vtm \\
        --out results/fid_kid
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy import linalg

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from modeling.gscv.data import load_description  # noqa: E402


def fid_gvc1d(real, fake, eps=1e-6):
    """GVC1D evaluator/evaluator.py VQGANEvaluator.result (rFID) on two feature arrays (N, 2048)."""
    real, fake = real.astype(np.float64), fake.astype(np.float64)
    out = []
    for f in (real, fake):
        n = len(f)
        total, sigma = f.sum(0), f.T @ f
        out.append((total / n, (sigma - np.outer(total, total) / n) / (n - 1)))
    (mu_real, sigma_real), (mu_fake, sigma_fake) = out
    diff = mu_real - mu_fake
    covmean, _ = linalg.sqrtm(sigma_real.dot(sigma_fake), disp=False)
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError(f"Imaginary component {np.max(np.abs(covmean.imag))}")
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    if not np.isfinite(covmean).all():
        tr_covmean = np.sum(np.sqrt(((np.diag(sigma_real) * eps) * (np.diag(sigma_fake) * eps)) / (eps * eps)))
    return float(diff.dot(diff) + np.trace(sigma_real) + np.trace(sigma_fake) - 2 * tr_covmean)


def kid_torch_fidelity(real, fake, subsets=100, subset_size=1000):
    """torch-fidelity's KID (mean, std); the subset size is reduced to the number of features if needed."""
    import torch
    from torch_fidelity.metric_kid import KEY_METRIC_KID_MEAN, KEY_METRIC_KID_STD, kid_features_to_metric
    size = min(subset_size, len(real), len(fake))
    r = kid_features_to_metric(torch.from_numpy(real.astype(np.float32)), torch.from_numpy(fake.astype(np.float32)),
                               kid_subsets=subsets, kid_subset_size=size, verbose=False)
    return float(r[KEY_METRIC_KID_MEAN]), float(r[KEY_METRIC_KID_STD]), size


def feature_file(path, qp):
    return f"{path.replace('/', '__')}_qp{qp}.npy"


def real_features(set_dir, frames, cache_dir, extractor):
    """Source patch features of every sequence of a set (the frames the codecs' features cover), cached."""
    name = os.path.basename(os.path.normpath(set_dir))
    path = os.path.join(cache_dir, f"real_{name}_f{frames}.npz")
    seqs, base = load_description(set_dir)
    if os.path.exists(path):
        z = np.load(path)
        if all(feature_file(s["path"], "src") in z for s in seqs):
            return {s["path"]: z[feature_file(s["path"], "src")] for s in seqs}
    out = {}
    for s in seqs:
        feats = []
        for n in (s["frames"] if frames <= 0 else s["frames"][:frames]):
            with Image.open(os.path.join(base, s["path"], n)) as im:
                feats.append(extractor()(np.asarray(im.convert("RGB"))))
        out[s["path"]] = np.concatenate(feats).astype(np.float32)
        print(f"  source features {s['path']}: {len(out[s['path']])} patches", flush=True)
    os.makedirs(cache_dir, exist_ok=True)
    np.savez(path, **{feature_file(p, "src"): v for p, v in out.items()})
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sets", nargs="+", required=True, help="test-set folders (with description.json)")
    p.add_argument("--results", nargs="+", required=True, help="NAME=results folder (scripts/test.py --fid_kid)")
    p.add_argument("--subsets", type=int, default=100)
    p.add_argument("--subset_size", type=int, default=1000)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    runs = {}
    for spec in args.results:
        name, d = spec.split("=", 1)
        with open(os.path.join(d, "results.json")) as f:
            res = json.load(f)
        if not res.get("args", {}).get("fid_kid"):
            raise SystemExit(f"{d}: run without --fid_kid (no features)")
        runs[name] = (d, res, int(res["args"].get("frames", 0)))
    frames = {f for _, _, f in runs.values()}
    if len(frames) > 1:
        raise SystemExit(f"the runs differ in --frames {frames}")
    frames = frames.pop()
    cache = {}

    def extractor():                    # loaded only when the source features are not cached
        if "x" not in cache:
            from metrics import PatchFeatures
            cache["x"] = PatchFeatures(f"cuda:{args.gpu}")
        return cache["x"]

    table, missing = {}, []
    for set_dir in args.sets:
        set_name = os.path.basename(os.path.normpath(set_dir))
        seqs, _ = load_description(set_dir)
        real = real_features(set_dir, frames, args.out, extractor)
        for name, (d, res, _) in runs.items():
            entries = [res["sequences"].get(s["path"]) for s in seqs]
            if any(e is None for e in entries):
                missing.append(f"{name}/{set_name}: sequences not tested")
                continue
            for qp in sorted(set.intersection(*(set(e["rd"]) for e in entries)), key=float):
                fake, src, bad = [], [], []
                for s in seqs:
                    f = os.path.join(d, "fid_kid_features", feature_file(s["path"], qp))
                    if not os.path.exists(f):
                        bad.append(s["path"])
                        continue
                    x = np.load(f)
                    if len(x) != len(real[s["path"]]):
                        raise SystemExit(f"{f}: {len(x)} patches, the source has {len(real[s['path']])}")
                    fake.append(x)
                    src.append(real[s["path"]])
                if bad:
                    missing.append(f"{name}/{set_name}/qp{qp}: no features for {bad[:3]}")
                    continue
                fake, src = np.concatenate(fake), np.concatenate(src)
                kid, kid_std, size = kid_torch_fidelity(src, fake, args.subsets, args.subset_size)
                row = {"bpp": float(np.mean([e["rd"][qp]["summary"]["bpp"] for e in entries])),
                       "fid": fid_gvc1d(src, fake), "kid": kid, "kid_std": kid_std, "patches": int(len(fake)),
                       "kid_subset_size": size}
                table.setdefault(name, {}).setdefault(set_name, {})[qp] = row
                print(f"{name:<14} {set_name:<16} qp {qp:>3}: bpp {row['bpp']:.5f} FID {row['fid']:8.3f} "
                      f"KID {kid:.5f} +- {kid_std:.5f} ({len(fake)} patches)", flush=True)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "fid_kid.json"), "w") as f:
        json.dump({"frames": frames, "subsets": args.subsets, "subset_size": args.subset_size, "results": table,
                   "missing": missing}, f, indent=2)
    for m in missing:
        print("MISSING", m)


if __name__ == "__main__":
    main()

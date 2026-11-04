"""BD-rates of several codecs against an anchor on the average RD curve of each test set (as in the paper).

Average RD curve of a set: at every rate point, the mean bpp and the mean metric over the set's sequences (psnr_fid:
over the sequences that contain fidelity-critical regions); KID: the set-level value of scripts/fid_kid.py at the
mean bpp. Sequences that a codec did not code at all its rate points are left out for that codec and the anchor. BD-rate (Bjontegaard): log-rate as a function of quality, piecewise-cubic Hermite interpolation (PCHIP),
integrated over the overlapping quality interval; negative = fewer bits than the anchor at equal quality.

Inputs are results folders of scripts/test.py (results.json); an anchor coded by another codec (e.g. VTM) needs a
results.json of the same layout, scored with scripts/metrics.py.

    python scripts/bd_rate.py --anchor VTM=results/vtm --test GSCV=results/gscv \\
        --fid_kid results/fid_kid/fid_kid.json --metrics lpips dists kid psnr siqm
"""
import argparse
import json
import math
import os

import numpy as np
from scipy.interpolate import PchipInterpolator

HIGHER_BETTER = {"psnr": True, "psnr_fid": True, "msssim": True, "siqm": True,
                 "lpips": False, "dists": False, "kid": False, "fid": False}


def _integral(q, logr, lo, hi):
    order = np.argsort(q)
    q, logr = q[order], logr[order]
    q, idx = np.unique(q, return_index=True)          # PCHIP needs strictly increasing x
    logr = logr[idx]
    if len(q) < 2:
        return None
    return float(PchipInterpolator(q, logr).integrate(lo, hi))


def bd_rate(anchor, test, higher_better):
    """Bjontegaard delta rate (%) of `test` vs `anchor`; arrays of (rate, quality)."""
    if len(anchor) < 2 or len(test) < 2:
        return math.nan
    qa, qt = anchor[:, 1], test[:, 1]
    if not higher_better:
        qa, qt = -qa, -qt
    lo, hi = max(qa.min(), qt.min()), min(qa.max(), qt.max())
    if hi <= lo:
        return math.nan
    ia = _integral(qa, np.log(anchor[:, 0]), lo, hi)
    it = _integral(qt, np.log(test[:, 0]), lo, hi)
    if ia is None or it is None:
        return math.nan
    return float((math.exp((it - ia) / (hi - lo)) - 1.0) * 100.0)


def mean_curve(res, seqs, metric):
    """(mean bpp, mean metric) at every rate point that all sequences share."""
    qs = sorted(set.intersection(*(set(res[p]["rd"]) for p in seqs)), key=float)
    pts = []
    for q in qs:
        rows = [res[p]["rd"][q]["summary"] for p in seqs]
        vals = [r[metric] for r in rows if r.get(metric) is not None]
        if vals:
            pts.append((float(np.mean([r["bpp"] for r in rows])), float(np.mean(vals))))
    return np.array(sorted(pts)) if pts else np.zeros((0, 2))


def set_curve(fid_kid, name, set_name, metric):
    """(bpp, set-level KID or FID) of every rate point (scripts/fid_kid.py)."""
    pts = fid_kid.get(name, {}).get(set_name, {})
    return np.array(sorted((v["bpp"], v[metric]) for v in pts.values())) if pts else np.zeros((0, 2))


def load(spec):
    name, d = spec.split("=", 1)
    with open(os.path.join(d, "results.json")) as f:
        return name, json.load(f)["sequences"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--anchor", required=True, help="NAME=results folder")
    p.add_argument("--test", nargs="+", required=True, help="NAME=results folder ...")
    p.add_argument("--fid_kid", default="", help="fid_kid.json of scripts/fid_kid.py (names as above), for KID")
    p.add_argument("--metrics", nargs="+", default=["lpips", "dists", "kid"], choices=sorted(HIGHER_BETTER))
    p.add_argument("--sets", nargs="*", default=None, help="test-set names (default: every set of the anchor)")
    p.add_argument("--out", default="", help="optional json file for the table")
    args = p.parse_args()
    a_name, anchor = load(args.anchor)
    tests = dict(load(s) for s in args.test)
    fk = {}
    if args.fid_kid:
        with open(args.fid_kid) as f:
            fk = json.load(f)["results"]
    sets = args.sets or sorted({e["set"] for e in anchor.values()})
    table = {}
    for s in sets:
        whole = sorted(p for p, e in anchor.items() if e["set"] == s)
        for name, res in tests.items():
            # sequences with every rate point of the codec; the others are left out for the codec and the anchor
            tested = [p for p in whole if p in res]
            points = set().union(*(set(res[p]["rd"]) for p in tested)) if tested else set()
            seqs = [p for p in tested if set(res[p]["rd"]) >= points]
            if len(seqs) < len(whole):
                print(f"WARNING {name}/{s}: {len(whole) - len(seqs)} of {len(whole)} sequences not tested at every "
                      f"rate point, left out (KID / FID: n/a)")
            if not seqs:
                continue
            row = {}
            for m in args.metrics:
                if m in ("kid", "fid"):
                    if len(seqs) < len(whole) or not fk:
                        row[m] = math.nan     # set-level values need every sequence (and fid_kid.json)
                        continue
                    a, t = set_curve(fk, a_name, s, m), set_curve(fk, name, s, m)
                else:
                    a, t = mean_curve(anchor, seqs, m), mean_curve(res, seqs, m)
                row[m] = bd_rate(a, t, HIGHER_BETTER[m])
            table.setdefault(s, {})[name] = row
    print(f"\nBD-rate (%) against {a_name}, average RD curve of each test set (negative = bitrate saving)")
    for s, rows in table.items():
        print(f"\n## {s}\n\n| Method | " + " | ".join(m.upper() for m in args.metrics) + " |")
        print("|---" * (len(args.metrics) + 1) + "|")
        for name, row in rows.items():
            print(f"| {name} | " + " | ".join("n/a" if math.isnan(row[m]) else f"{row[m]:+.1f}" for m in args.metrics)
                  + " |")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"anchor": a_name, "metrics": args.metrics, "bd_rate": table}, f, indent=2)


if __name__ == "__main__":
    main()

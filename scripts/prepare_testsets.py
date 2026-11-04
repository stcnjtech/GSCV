"""Convert test sequences (y4m or planar raw YUV / RGB files) into the PNG + description.json layout of GVC1D.

Output layout:
    <out_root>/<set>/<sequence>/im00001.png ...
    <out_root>/<set>/description.json   (list of {height, width, seq_length, path, fps, frames, source_file, category})

YUV sources are converted with BT.709, limited range by default (a y4m XCOLORRANGE tag wins), nearest-neighbour
chroma upsampling; RGB 4:4:4 sources are only reordered / rescaled. Every codec is evaluated against these RGB PNGs.
A set listed in several calls (e.g. one call per frame size) collects all its sequences in one description.json.

The four test sets of the paper (all frames are converted; the tests use the first 96):
    # AOM CTC Class B (AV2 CTC v8): B1 (10 sequences) and B2 (9), y4m from media.xiph.org/video/aomctc/test_set/
    python scripts/prepare_testsets.py y4m --inputs <b1_syn>/*.y4m <b2_scc>/*.y4m --out_root data/testsets --set AOM_B
    # HEVC Class F (4:2:0 8-bit YUV; one call per size and frame rate)
    python scripts/prepare_testsets.py raw --inputs ChinaSpeed_1024x768_30.yuv --width 1024 --height 768 --fps 30 \
        --out_root data/testsets --set HEVC_F
    # HEVC-SCC (JCT-VC SCC sequences, YCbCr 4:4:4; 10-bit for EBURainFruits and Kimono1)
    python scripts/prepare_testsets.py raw --inputs Console_1920x1080.yuv --width 1920 --height 1080 --chroma 444 \
        --fps 60 --out_root data/testsets --set HEVC_SCC
Check the first PNG of every sequence: a wrong plane order or range shows as wrong colours.
"""
import argparse
import glob
import json
import os
import re

import numpy as np
from PIL import Image

_KR_KB = {"bt709": (0.2126, 0.0722), "bt601": (0.299, 0.114)}


def _ranges(bitdepth, full_range):
    """(y_offset, y_scale, c_offset, c_scale) for integer YCbCr samples."""
    if full_range:
        peak = float((1 << bitdepth) - 1)
        return 0.0, peak, float(1 << (bitdepth - 1)), peak
    s = float(1 << (bitdepth - 8))
    return 16.0 * s, 219.0 * s, 128.0 * s, 224.0 * s


def yuv_to_rgb(y, u, v, bitdepth=8, matrix="bt709", full_range=False):
    """Y, Cb, Cr planes of identical size -> uint8 RGB (H, W, 3)."""
    kr, kb = _KR_KB[matrix]
    kg = 1.0 - kr - kb
    y_off, y_scale, c_off, c_scale = _ranges(bitdepth, full_range)
    luma = (y.astype(np.float64) - y_off) / y_scale
    cb = (u.astype(np.float64) - c_off) / c_scale
    cr = (v.astype(np.float64) - c_off) / c_scale
    r = luma + 2.0 * (1.0 - kr) * cr
    b = luma + 2.0 * (1.0 - kb) * cb
    g = (luma - kr * r - kb * b) / kg
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)


def upsample_chroma(plane, height, width):
    """Nearest-neighbour chroma upsampling to (height, width)."""
    fy = -(-height // plane.shape[0])
    fx = -(-width // plane.shape[1])
    return np.repeat(np.repeat(plane, fy, axis=0), fx, axis=1)[:height, :width]


# ----------------------------------------------------------------------------- readers

def _chroma_size(width, height, chroma):
    if chroma == "420":
        return (width + 1) // 2, (height + 1) // 2
    if chroma == "422":
        return (width + 1) // 2, height
    if chroma == "444":
        return width, height
    if chroma == "400":
        return 0, 0
    raise ValueError(f"unsupported chroma format {chroma}")


def _read_plane(f, width, height, bitdepth):
    if width == 0:
        return None
    dtype = np.uint8 if bitdepth <= 8 else np.dtype("<u2")
    count = width * height
    data = np.fromfile(f, dtype=dtype, count=count)
    if data.size != count:
        raise EOFError
    return data.reshape(height, width)


class Y4MReader:
    """Minimal YUV4MPEG2 reader (4:2:0 / 4:2:2 / 4:4:4 / mono, 8-16 bit)."""

    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            header = f.readline().decode("ascii").split()
            self._data_start = f.tell()
        if not header or header[0] != "YUV4MPEG2":
            raise ValueError(f"{path} is not a y4m file")
        self.width = self.height = None
        self.fps = 30.0
        self.chroma, self.bitdepth = "420", 8
        self.full_range = None          # from an XCOLORRANGE=FULL|LIMITED tag, else the caller's choice
        for tok in header[1:]:
            key, val = tok[0], tok[1:]
            if key == "X" and val.upper().startswith("COLORRANGE="):
                self.full_range = val.split("=", 1)[1].upper() == "FULL"
                continue
            if key == "W":
                self.width = int(val)
            elif key == "H":
                self.height = int(val)
            elif key == "F":
                num, den = val.split(":")
                self.fps = float(num) / float(den)
            elif key == "C":
                m = re.match(r"^(420|422|444|mono)(?:jpeg|paldv|mpeg2|alpha)?(?:p(\d+))?$", val)
                if m is None:
                    raise ValueError(f"unsupported y4m colourspace C{val}")
                self.chroma = "400" if m.group(1) == "mono" else m.group(1)
                self.bitdepth = int(m.group(2)) if m.group(2) else 8

    def frames(self, max_frames=None):
        """Yields (y, u, v) planes at native chroma resolution (u, v are None for mono)."""
        cw, ch = _chroma_size(self.width, self.height, self.chroma)
        with open(self.path, "rb") as f:
            f.seek(self._data_start)
            n = 0
            while max_frames is None or n < max_frames:
                marker = f.readline()
                if not marker:
                    return
                if not marker.startswith(b"FRAME"):
                    raise ValueError(f"corrupt y4m frame marker in {self.path}: {marker[:20]!r}")
                try:
                    y = _read_plane(f, self.width, self.height, self.bitdepth)
                    u = _read_plane(f, cw, ch, self.bitdepth)
                    v = _read_plane(f, cw, ch, self.bitdepth)
                except EOFError:
                    return
                yield y, u, v
                n += 1

    def rgb_frames(self, max_frames=None, matrix="bt709", full_range=False):
        if self.full_range is not None:             # the file says which range it uses
            full_range = self.full_range
        for y, u, v in self.frames(max_frames):
            if u is None:
                u = v = np.full_like(y, 1 << (self.bitdepth - 1))
            u = upsample_chroma(u, self.height, self.width)
            v = upsample_chroma(v, self.height, self.width)
            yield yuv_to_rgb(y, u, v, self.bitdepth, matrix, full_range)


def raw_rgb_frames(path, width, height, chroma="420", bitdepth=8, order="yuv",
                   max_frames=None, matrix="bt709", full_range=False):
    """Planar raw file -> uint8 RGB frames.

    order: "yuv" for YCbCr files, or the plane order of an RGB 4:4:4 file
    ("rgb", "gbr", "bgr"); the JCT-VC SCC .rgb files are stored as R, G, B planes (their CTC
    codes them with InputColourSpaceConvert=RGBtoGBR, which reads file plane 1 as G).
    """
    cw, ch = _chroma_size(width, height, chroma)
    with open(path, "rb") as f:
        n = 0
        while max_frames is None or n < max_frames:
            try:
                p0 = _read_plane(f, width, height, bitdepth)
                p1 = _read_plane(f, cw, ch, bitdepth)
                p2 = _read_plane(f, cw, ch, bitdepth)
            except EOFError:
                return
            if order == "yuv":
                p1 = upsample_chroma(p1, height, width)
                p2 = upsample_chroma(p2, height, width)
                yield yuv_to_rgb(p0, p1, p2, bitdepth, matrix, full_range)
            else:
                if chroma != "444":
                    raise ValueError("RGB plane orders need 4:4:4 data")
                planes = dict(zip(order, (p0, p1, p2)))
                rgb = np.stack([planes["r"], planes["g"], planes["b"]], axis=-1).astype(np.float64)
                rgb = rgb * 255.0 / ((1 << bitdepth) - 1)
                yield np.clip(np.round(rgb), 0, 255).astype(np.uint8)
            n += 1


# ----------------------------------------------------------------------------- conversion

def seq_name_from_file(path):
    return re.sub(r"\.(y4m|yuv|rgb|gbr|raw)$", "", os.path.basename(path), flags=re.I)


def write_sequence(frames, out_dir, max_frames):
    os.makedirs(out_dir, exist_ok=True)
    names, size = [], None
    for idx, rgb in enumerate(frames):
        if max_frames is not None and idx >= max_frames:
            break
        name = f"im{idx + 1:05d}.png"
        Image.fromarray(rgb).save(os.path.join(out_dir, name), compress_level=1)
        names.append(name)
        size = rgb.shape[:2]
    if not names:
        raise RuntimeError(f"no frames written to {out_dir}")
    return names, size


def update_description(set_dir, entry):
    path = os.path.join(set_dir, "description.json")
    seqs = []
    if os.path.exists(path):
        with open(path) as f:
            seqs = json.load(f)
    seqs = [s for s in seqs if s["path"] != entry["path"]] + [entry]
    seqs.sort(key=lambda s: s["path"])
    with open(path, "w") as f:
        json.dump(seqs, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("kind", choices=["y4m", "raw"])
    parser.add_argument("--inputs", nargs="+", required=True, help="files or glob patterns")
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--set", required=True, help="test-set name, e.g. AOM_B, HEVC_F, HEVC_SCC")
    parser.add_argument("--category", default="", help="category label stored in description.json")
    parser.add_argument("--category_map", nargs="*", default=[],
                        help="REGEX=CATEGORY ...: per-sequence categories (first matching regex on the name)")
    parser.add_argument("--frames", type=int, default=96, help="frames to keep (<=0: all)")
    parser.add_argument("--matrix", default="bt709", choices=["bt709", "bt601"])
    parser.add_argument("--full_range", action="store_true", help="YUV source is full range")
    # raw-only options
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--chroma", default="420", choices=["420", "422", "444"])
    parser.add_argument("--bitdepth", type=int, default=8)
    parser.add_argument("--order", default="yuv", help="yuv, or RGB plane order rgb/gbr/bgr")
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    files = []
    for pattern in args.inputs:
        files.extend(sorted(glob.glob(pattern)) or [pattern])
    max_frames = args.frames if args.frames > 0 else None
    set_dir = os.path.join(args.out_root, args.set)

    cat_map = [(re.compile(k), v) for k, v in (m.split("=", 1) for m in args.category_map)]
    for path in files:
        name = seq_name_from_file(path)
        category = next((v for r, v in cat_map if r.search(name)), args.category)
        if args.kind == "y4m":
            reader = Y4MReader(path)
            frames = reader.rgb_frames(max_frames, args.matrix, args.full_range)
            fps, info = reader.fps, f"{reader.chroma} {reader.bitdepth}-bit"
        else:
            if not (args.width and args.height):
                parser.error("raw input needs --width and --height")
            frames = raw_rgb_frames(path, args.width, args.height, args.chroma, args.bitdepth,
                                       args.order, max_frames, args.matrix, args.full_range)
            fps, info = args.fps, f"{args.chroma} {args.bitdepth}-bit {args.order}"
        names, (h, w) = write_sequence(frames, os.path.join(set_dir, name), max_frames)
        update_description(set_dir, {
            "height": int(h), "width": int(w), "seq_length": len(names),
            "path": f"{args.set}/{name}", "fps": fps, "frames": names,
            "source_file": os.path.abspath(path), "category": category,
        })
        print(f"{name}: {w}x{h}, {len(names)} frames, {info} -> {set_dir}/{name}")


if __name__ == "__main__":
    main()

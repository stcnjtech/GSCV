"""Training and validation clips of GSCV (SCVCD-5K PNG folders) with synthetic interaction events and their
window-state annotations.

Training (EntryDataset): one pass over the training entries ({clip, start, end}, data.train_entries; clips longer
than 193 frames are split into near-equal pieces). Each entry gives T frames picked as in GVC1D's
data/openvid_dataset.py (select_frame_indices: random frames, sorted, half of the samples reversed, gaps of at most
max_gap), a crop of side `crop` (multiple of 256) after GVC1D's symmetric zero padding of frames smaller than the
crop, no resize and no flip (glyph scale and orientation are kept).

Interaction events: a long still period (a frame repeated 3-10 times), a switch (another clip spliced in), a
transient element (a rectangle taken from another clip, removed after 2-6 frames), a cursor (an arrow bitmap on a
random path) and a pop-up over a still screen that closes again (exercises the snapshots and REVERT). With
probability rand_f the class map is replaced by a random rectangle map for the whole clip.

Every item carries the window-state annotation of its crop (rsm.RSMEncoder, the same state machine as at test
time): the class map of every frame and, for every P-frame, the window states, revert slots, snapshot slots,
quality offsets and state indices of the memory update.

Validation (TestClips): the first T frames of every sequence of a description.json, symmetric zero padding and a
centre crop, as GVC1D's data/test_dataset.py.
"""
import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .rsm import (REFRESH, UPDATE, RSMConfig, RSMEncoder, TEXTURE, block_classes, random_fmap, valid_blocks,
                  valid_box, valid_pixels)

EVENTS = ("switch", "freeze", "transient", "popup", "cursor")


def _cursor_sprite():
    """Arrow cursor: alpha mask and colour (black body, white outline), 19 x 12."""
    h, w = 19, 12
    body = np.zeros((h, w), bool)
    for r in range(12):                           # arrow head: right triangle, tip at top-left
        body[r, :r + 1] = True
    body[12:17, 4:7] = True                       # tail
    outline = np.zeros_like(body)
    pad = np.pad(body, 1)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            outline |= pad[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
    color = np.where(body, 0, 255).astype(np.uint8)
    return outline, color


_CURSOR = _cursor_sprite()


def select_frame_indices(n, t, rng, max_gap=6):
    """Indices (0..n-1) of the t frames of one training sample, as GVC1D's
    data/openvid_dataset.py ImagePaths._select_frame_indices ('random'): t distinct random frames,
    sorted (in reverse for half of the samples); a gap larger than max_gap is redrawn within max_gap
    of the previous frame (so a sample spans at most (t - 1) * max_gap + 1 frames, starting from its
    earliest frame, or its latest when reversed). A clip of at most t frames is played forward and
    backward from its first frame."""
    if t < n:
        idx = sorted(int(v) for v in rng.choice(n, t, replace=False))
        rev = bool(rng.integers(2))
        if rev:
            idx.reverse()
        for i in range(1, len(idx)):
            pre, cur = idx[i - 1], idx[i]
            if rev and cur < pre - max_gap:
                idx[i] = int(rng.integers(pre - max_gap, pre))            # pre - max_gap .. pre - 1
            elif not rev and cur > pre + max_gap:
                idx[i] = int(rng.integers(pre + 1, pre + max_gap + 1))    # pre + 1 .. pre + max_gap
        return idx
    if n == 1:
        return [0] * t
    out, f, up = [], 0, True
    while len(out) < t:
        out.append(f)
        if up:
            f, up = (f - 1, False) if f == n - 1 else (f + 1, True)
        else:
            f, up = (f + 1, True) if f == 0 else (f - 1, False)
    return out


def pad_symmetric(frames, c):
    """GVC1D's padding of frames smaller than the c x c crop (data/openvid_dataset.py and
    data/test_dataset.py): zeros, split evenly between top / bottom and left / right. Returns the
    padded frames and the (top, left) offset of the original frame."""
    h, w = frames[0].shape[:2]
    ph, pw = max(c - h, 0), max(c - w, 0)
    top, left = ph // 2, pw // 2
    if ph or pw:
        frames = [np.pad(f, ((top, ph - top), (left, pw - left), (0, 0))) for f in frames]
    return frames, top, left


def crop_box(h, w, top, left, y, x, c):
    """(y0, x0, y1, x1) of the original h x w frame inside the c x c crop at (y, x) of the padded
    frame (the valid area of the state machine: padding blocks are TEXTURE, never sent, no change)."""
    return max(top - y, 0), max(left - x, 0), min(top + h - y, c), min(left + w - x, c)


def annotate(frames, fl, cfg, crop, valid, rng=None, cascade=None):
    """Window-state annotation of a clip: rsm.RSMEncoder over the frames with the class maps fl. cascade: P-frames
    per cascade group of the training loop -- every group gets at least one coded window (an encoder choice, so the
    states stay decoder-consistent; DDP needs a gradient path on every GPU)."""
    enc = RSMEncoder(cfg, crop, crop, valid)
    infos = [enc.start(frames[0], fl[0])]
    group_coded = False
    for t in range(1, len(frames)):
        force = None
        last = cascade and (t % cascade == 0 or t == len(frames) - 1)
        if last and not group_coded:
            force = [int(rng.integers(enc.book.n))]
        inf = enc.step(frames[t], fl[t], force=force)
        infos.append(inf)
        group_coded |= bool(np.isin(inf["modes"], (UPDATE, REFRESH)).any())
        if cascade and t % cascade == 0:
            group_coded = False
    p = infos[1:]
    item = {
        "frames": torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).contiguous(),   # uint8 (T, 3, c, c)
        "fclass": torch.from_numpy(np.stack([inf["f_eff"] for inf in infos]).astype(np.int64)),
    }
    for key in ("modes", "revert_slot", "snap_create", "qp_offset", "mem_type", "mem_age"):
        item[key] = torch.from_numpy(np.stack([inf[key] for inf in p]).astype(np.int64))
    item["side_bits"] = torch.tensor([inf["side_bits"] for inf in p], dtype=torch.float32)
    return item


class EntryDataset(Dataset):
    """One training pass: item i is entry order[i] of the training entries ({clip, start, end}, frame numbers
    1-based and inclusive; frames <root>/<clip>/im%05d.png). Its random choices, including its quality index
    qp_base (uniform over the 8 indices, as GVC1D), depend only on (seed, stage_id, pass_index, entry), not on the
    worker or the GPU: a resumed pass draws the same samples, and every stage and pass draws new ones.

    frames: T; crop: side of the square crop (multiple of 256); rsm_cfg: the stage's window-state settings;
    aug: probability of every interaction event; bias_crop: probability of the biased crop position (crop_frames);
    rand_f: probability of a random class map; cascade: P-frames per cascade group of the training loop."""

    def __init__(self, root, entries, order, frames, crop=512, rsm_cfg=None, aug=None, bias_crop=0.5,
                 rand_f=0.0, cascade=None, seed=0, pass_index=0, max_gap=6, stage_id=0):
        assert crop % 256 == 0, crop
        self.root, self.entries, self.order = root, entries, [int(j) for j in order]
        self.frames, self.crop = frames, crop
        self.rsm_cfg = rsm_cfg or RSMConfig()
        self.aug = {"switch": 0.2, "freeze": 0.3, "transient": 0.2, "popup": 0.15, "cursor": 0.3}
        self.aug.update(dict(aug or {}))
        self.bias_crop, self.rand_f, self.cascade = bias_crop, rand_f, cascade
        self.base_seed, self.pass_index, self.max_gap = int(seed), int(pass_index), int(max_gap)
        self.stage_id = int(stage_id)
        paths = sorted({e["clip"] for e in entries})
        self.clip_index = {p: k for k, p in enumerate(paths)}

    def __len__(self):
        return len(self.order)

    # ------------------------------------------------------------------ io
    def read_entry(self, rng, e, t):
        n = int(e["end"]) - int(e["start"]) + 1
        names = [f"im{int(e['start']) + i:05d}.png" for i in select_frame_indices(n, t, rng, self.max_gap)]
        frames = []
        for name in names:
            with Image.open(os.path.join(self.root, e["clip"], name)) as im:
                frames.append(np.asarray(im.convert("RGB")))
        if any(f.shape != frames[0].shape for f in frames):
            raise ValueError(f"{e['clip']}: frame size changes within the clip")
        return frames

    def crop_frames(self, rng, frames, prefer=True):
        """c x c crop after GVC1D's symmetric zero padding; returns the cropped frames and the original area
        inside the crop (y0, x0, y1, x1). With probability bias_crop (prefer=True) the crop is the best of 4
        random positions by (windows that change between the sample's frames) + 0.5 * [the crop mixes
        fidelity-critical and texture blocks] (GVC1D takes a plain random position; most screen windows are still)."""
        c, b = self.crop, self.rsm_cfg.block
        h, w = frames[0].shape[:2]
        frames, top, left = pad_symmetric(frames, c)
        hp, wp = frames[0].shape[:2]
        cands = [(int(rng.integers(0, hp - c + 1)), int(rng.integers(0, wp - c + 1))) for _ in range(4)]
        if prefer and rng.random() < self.bias_crop:
            nw = c // 256

            def score(y, x):
                win = [f[y:y + c, x:x + c] for f in frames]
                moved = np.zeros((nw, nw), bool)
                for a, z in zip(win[:-1], win[1:]):
                    moved |= np.any(a != z, axis=-1).reshape(nw, 256, nw, 256).any(axis=(1, 3))
                box = crop_box(h, w, top, left, y, x, c)
                vb = valid_blocks(c, c, box, b)
                cls = block_classes(win[0], self.rsm_cfg, valid_pixels(c, c, box))
                fid = float(((cls != TEXTURE) & vb).sum() / max(int(vb.sum()), 1))
                return float(moved.mean()) + 0.5 * (0.05 < fid < 0.95)
            cands.sort(key=lambda p: -score(*p))
        y, x = cands[0]
        out = [np.ascontiguousarray(f[y:y + c, x:x + c]) for f in frames]
        return out, crop_box(h, w, top, left, y, x, c)

    def load(self, rng, t, prefer=True, tries=8, avoid=None):
        """(clip index, cropped frames, original area) of a random training entry (for the switch, transient and
        pop-up events); avoid: a clip index not to draw."""
        for attempt in range(tries):
            e = self.entries[int(rng.integers(len(self.entries)))]
            k = self.clip_index[e["clip"]]
            if avoid is not None and k == avoid and len(self.clip_index) > 1:
                continue
            try:
                frames, valid = self.crop_frames(rng, self.read_entry(rng, e, t), prefer)
                return k, frames, valid
            except (OSError, ValueError, KeyError) as err:
                if attempt == tries - 1:
                    raise RuntimeError(f"no readable entry after {tries} tries, last error: {err}") from err
        raise RuntimeError("no readable entry")

    # ------------------------------------------------------------------ interaction events
    # every event maps frames -> frames inside the original area `valid` of the crop (the state machine of a
    # padded crop does not look at its padding)
    def ev_switch(self, rng, frames, k, valid):
        t = len(frames)
        ts = int(rng.integers(2, t)) if t > 2 else 1
        try:
            _, other, other_valid = self.load(rng, t - ts, prefer=False, avoid=k)
        except RuntimeError:
            return frames
        if valid_box(other_valid) != valid_box(valid):     # keep the padding of the first clip (one valid area)
            vp = valid_pixels(self.crop, self.crop, valid)
            if vp is not None:
                other = [np.where(vp[..., None], f, 0).astype(np.uint8) for f in other]
        return frames[:ts] + other

    def ev_freeze(self, rng, frames, k, valid):
        """Long still period: frame t0 is repeated n = 3-10 more times, then the clip goes on from
        t0 + 1 (its last frames drop out of the window)."""
        t = len(frames)
        t0 = int(rng.integers(0, max(t - 1, 1)))
        n = int(rng.integers(3, 11))
        return [frames[i if i <= t0 else max(i - n, t0)] for i in range(t)]

    def ev_transient(self, rng, frames, k, valid, t1=None, d=None):
        """A rectangle from another clip pasted for d frames (pop-up, notification), taken from the donor's
        original area and pasted inside the crop's original area."""
        t = len(frames)
        try:
            _, donor, dbox = self.load(rng, 1, prefer=False, avoid=k)
            donor, dbox = donor[0], valid_box(dbox)
        except RuntimeError:
            donor, dbox = frames[int(rng.integers(0, t))], None
        y0, x0, y1, x1 = valid_box(valid)
        dbox = dbox or (y0, x0, y1, x1)
        ph = min(int(rng.integers(48, 161)), y1 - y0, dbox[2] - dbox[0])
        pw = min(int(rng.integers(64, 225)), x1 - x0, dbox[3] - dbox[1])
        sy, sx = int(rng.integers(dbox[0], dbox[2] - ph + 1)), int(rng.integers(dbox[1], dbox[3] - pw + 1))
        patch = donor[sy:sy + ph, sx:sx + pw].copy()
        y, x = int(rng.integers(y0, y1 - ph + 1)), int(rng.integers(x0, x1 - pw + 1))
        if d is None:                     # shown 2-6 frames and removed inside the clip when it fits
            d = int(rng.integers(2, 7))
            d = min(d, max(t - 2, 1))
        if t1 is None:
            t1 = int(rng.integers(1, max(t - d, 2)))
        frames = list(frames)
        for i in range(t1, min(t, t1 + d)):
            f = frames[i].copy()
            f[y:y + ph, x:x + pw] = patch
            frames[i] = f
        return frames

    def ev_popup(self, rng, frames, k, valid):
        """A pop-up over a screen that has been still long enough to be snapshotted, closed again
        after 2-6 frames: the window returns to its exact earlier content (REVERT)."""
        t = len(frames)
        lead = self.rsm_cfg.a_min + 1                  # the still content must last a_min + 1 frames
        if t < lead + 3:                               # pop-up >= 2 frames, then >= 1 frame closed
            return frames
        t0 = int(rng.integers(0, t - lead - 2))        # the screen is still from t0 on
        t1 = int(rng.integers(t0 + lead, t - 2))       # pop-up start
        d = int(rng.integers(2, min(6, t - 1 - t1) + 1))
        return self.ev_transient(rng, [frames[min(i, t0)] for i in range(t)], k, valid, t1=t1, d=d)

    def ev_cursor(self, rng, frames, k, valid):
        t = len(frames)
        mask, color = _CURSOR
        h, w = mask.shape
        y0, x0, y1, x1 = valid_box(valid)
        if y1 - y0 < h or x1 - x0 < w:
            return frames
        y, x = float(rng.integers(y0, y1 - h + 1)), float(rng.integers(x0, x1 - w + 1))
        vy, vx = rng.uniform(-24, 24, size=2)
        t1 = int(rng.integers(1, max(t - 1, 2)))
        frames = list(frames)
        for i in range(t1, t):
            if rng.random() < 0.25:         # new direction / speed now and then
                vy, vx = rng.uniform(-24, 24, size=2)
            if rng.random() >= 0.3:         # ... and it rests in ~30% of the frames
                y, x = float(np.clip(y + vy, y0, y1 - h)), float(np.clip(x + vx, x0, x1 - w))
            yi, xi = int(y), int(x)
            f = frames[i].copy()
            region = f[yi:yi + h, xi:xi + w]
            region[mask] = color[mask][:, None]
            frames[i] = f
        return frames

    # ------------------------------------------------------------------ item
    def __getitem__(self, i):
        j = self.order[i]
        e = self.entries[j]
        k = self.clip_index[e["clip"]]
        rng = np.random.default_rng((self.base_seed, self.stage_id, self.pass_index, 2, j))   # 2: item keys
        try:
            frames = self.read_entry(rng, e, self.frames)
        except (OSError, ValueError) as err:
            raise RuntimeError(f"training entry {e} cannot be read: {err}") from err
        frames, valid = self.crop_frames(rng, frames, True)
        if self.frames > 2:
            for name in EVENTS:
                if rng.random() < self.aug.get(name, 0.0):
                    frames = getattr(self, f"ev_{name}")(rng, frames, k, valid)
        cfg = self.rsm_cfg
        vpx = valid_pixels(self.crop, self.crop, valid)
        fl = [block_classes(f, cfg, vpx) for f in frames]
        if self.rand_f > 0 and rng.random() < self.rand_f:
            fl = [random_fmap(rng, fl[0])] * len(frames)
        item = annotate(frames, fl, cfg, self.crop, valid, rng, self.cascade)
        item["entry"] = torch.tensor(j)
        item["qp_base"] = torch.tensor(int(rng.integers(0, 8)))
        return item


def load_description(set_dir):
    """Sequences of <set_dir>/description.json and the folder their paths are relative to (the parent
    of set_dir), as GVC1D's data/test_dataset*.py."""
    with open(os.path.join(set_dir, "description.json")) as f:
        cfg = json.load(f)
    seqs = cfg if isinstance(cfg, list) else [dict(s, frames=cfg["frames"]) for s in cfg["seqs"]]
    return seqs, os.path.dirname(os.path.normpath(set_dir))


class TestClips(Dataset):
    """Validation sequences of <set_dir>/description.json for the evaluation during training, as GVC1D's
    data/test_dataset.py (eval_reconstruction): the first `frames` frames, symmetric zero padding to the crop,
    centre crop; true class maps and the stage's window states (no events, no random class map)."""

    def __init__(self, set_dir, frames, crop=512, rsm_cfg=None, states=True):
        self.seqs, self.base = load_description(set_dir)
        self.frames, self.crop = int(frames), int(crop)
        cfg = rsm_cfg or RSMConfig()
        if not states:
            cfg = RSMConfig(**{**cfg.__dict__, "use_skip": False, "use_refresh": False, "use_revert": False,
                               "k_snap": 0})
        self.rsm_cfg = cfg

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        s = self.seqs[i]
        names = list(s["frames"])[:self.frames]
        if len(names) < self.frames:
            raise ValueError(f"{s['path']}: {len(names)} frames listed, {self.frames} needed")
        frames = []
        for n in names:
            with Image.open(os.path.join(self.base, s["path"], n)) as im:
                frames.append(np.asarray(im.convert("RGB")))
        c = self.crop
        h, w = frames[0].shape[:2]
        frames, top, left = pad_symmetric(frames, c)
        hp, wp = frames[0].shape[:2]
        y, x = (hp - c) // 2, (wp - c) // 2
        frames = [np.ascontiguousarray(f[y:y + c, x:x + c]) for f in frames]
        return annotate(frames, [None] * len(frames), self.rsm_cfg, c, crop_box(h, w, top, left, y, x, c))


def collate(items):
    return {k: torch.stack([it[k] for it in items]) for k in items[0]}

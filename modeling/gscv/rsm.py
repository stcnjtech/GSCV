"""Window state machine of GSCV (NumPy only: it runs in the data-loader workers and at test time).

Class map F_t (SFA)  : every 64x64 block gets a class TEXTURE (natural and rendered textures), GRAPHIC (text and
                       graphics) or FLAT (flat regions); fidelity-critical blocks are F != TEXTURE.
Window states S_t    : every 256x256 window (the coding unit of GVC1D) gets a state UPDATE / SKIP / REFRESH /
(IACM)                 REVERT per P-frame.

RSMBook holds everything the decoder derives from the transmitted symbols (window states, revert slots, change
flags, classes of the coded windows): display durations, refresh periods, snapshot slots, the effective class map,
the per-window quality offsets and the state indices of the memory update. RSMEncoder adds the encoder-side
decisions (change detection, revert matching) on top of the same book, so the encoder and decoder bookkeeping
cannot diverge.

Refresh period: when a window shows new content (an UPDATE whose change flag is set, or a REFRESH), the next
n_refine frames form its refresh period, restarted by every further change. The encoder keeps coding the window
there (refinement UPDATEs, change flag 0); both sides raise its quality index by delta_r and mark the memory update
as a refinement; a REFRESH frame is raised as well. A REVERT ends the period (the window is restored exactly).
After an I-frame, windows with >= fid_qp_frac fidelity-critical blocks start in a refresh period. The first P-frame
codes every window (no decoded features exist yet that a SKIP could copy): its states are implied and its change
flags say which windows changed since the I-frame. Snapshots are taken when content that was shown for >= a_min + 1
frames is left.
"""
import math
from dataclasses import dataclass, fields

import numpy as np

UPDATE, SKIP, REFRESH, REVERT = 0, 1, 2, 3
TEXTURE, GRAPHIC, FLAT = 0, 1, 2
WIN = 256


@dataclass
class RSMConfig:
    block: int = 64             # content-map block size (divides 256)
    topk: int = 8               # colours counted for the palette test
    cov_thr: float = 0.85       # top-k colour coverage of palette-like blocks
    edge_thr: float = 48.0      # luma step of a strong edge
    edge_frac: float = 0.005    # share of strong-edge pixels of GRAPHIC blocks
    tol: int = 0                # per-pixel tolerance of "unchanged" (0 = bit-exact)
    static_mad: float = 0.5     # windows with MAD <= this are static (noisy recordings) ...
    static_peak: int = 16       # ... if no pixel changed by more than this (a typed glyph is not noise)
    refresh_frac: float = 0.5   # REFRESH: share of changed pixels ...
    refresh_mad: float = 20.0   # ... and mean absolute difference
    n_refine: int = 2           # N_r: length of the refresh period
    restart_on_change: bool = True   # a change inside a refresh period restarts it
    a_min: int = 8              # content shown for >= a_min + 1 frames is snapshotted when it changes
    k_snap: int = 2             # snapshot slots per window
    revert_mad: float = 0.0     # MAD tolerance of revert matching (0 = bit-exact)
    fid_qp_frac: float = 0.25   # share of fidelity-critical blocks that raises the quality index (rho_k)
    delta_f: int = 1            # quality offset of fidelity-critical windows
    delta_r: int = 1            # quality offset during a refresh period
    iframe_refresh: bool = True  # fidelity windows start in a refresh period after the I-frame
    header_bits: int = 4        # frame header: frame type (1 bit) + qp_base (3 bits)
    use_skip: bool = True       # the window states (all off in the training stages before IACM: every
    use_refresh: bool = True    # window is coded, as GVC1D)
    use_revert: bool = True

    def __post_init__(self):
        # a snapshot is the state before a change and needs a decoded reference: a_min = 0 would
        # snapshot the I-frame state at the first P-frame
        assert self.a_min >= 1, f"a_min must be >= 1, got {self.a_min}"
        assert 256 % self.block == 0 and self.k_snap >= 0, (self.block, self.k_snap)

    @classmethod
    def from_dict(cls, d):
        names = {f.name: f for f in fields(cls)}
        out = {}
        for k, v in dict(d or {}).items():
            if k in names:
                out[k] = parse_value(v, type(getattr(cls(), k)))
        return cls(**out)


def parse_value(v, typ):
    """Config / command-line value -> typ (booleans accept true/false/1/0/yes/no)."""
    if typ is bool:
        return v if isinstance(v, bool) else str(v).strip().lower() in ("1", "true", "yes", "on")
    return typ(v)


# ----------------------------------------------------------------------------- class map

def block_classes(rgb, cfg=None, valid=None):
    """Class of each block of an uint8 (H, W, 3) frame -> uint8 (ceil(H/b), ceil(W/b)).

    Palette-like blocks (top-k colours cover >= cov_thr of the pixels: text, UI, icons, line
    art, flat fills) are GRAPHIC when they hold strong edges and FLAT otherwise; all other
    blocks (rendered scenes, natural video) are TEXTURE. A strong edge is a luma step >= edge_thr
    between two horizontally or vertically adjacent pixels of the same block (counted once, at the
    first pixel), so a block's class depends on its own pixels only. valid: (H, W) bool mask of the
    original area of a padded frame (valid_pixels): padding pixels and steps touching them do not
    count, and blocks without valid pixels are TEXTURE -- zero padding (training crops) and edge
    padding (test frames) give the same classes.
    """
    cfg = cfg or RSMConfig()
    b = cfg.block
    h, w, _ = rgb.shape
    nbh, nbw = -(-h // b), -(-w // b)
    packed = (rgb[..., 0].astype(np.int32) << 16) | (rgb[..., 1].astype(np.int32) << 8) | rgb[..., 2]
    luma = rgb.astype(np.float32) @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    gx = np.zeros((h, w), np.float32)
    gy = np.zeros((h, w), np.float32)
    gx[:, :-1] = np.abs(np.diff(luma, axis=1))
    gy[:-1, :] = np.abs(np.diff(luma, axis=0))
    gx[:, b - 1::b] = 0.0                       # pairs across a block border
    gy[b - 1::b, :] = 0.0
    if valid is not None:
        gx[:, :-1] *= valid[:, :-1] & valid[:, 1:]
        gy[:-1, :] *= valid[:-1, :] & valid[1:, :]
    strong = np.maximum(gx, gy) >= cfg.edge_thr
    out = np.full((nbh, nbw), TEXTURE, np.uint8)
    for i in range(nbh):
        for j in range(nbw):
            rows, cols = slice(i * b, (i + 1) * b), slice(j * b, (j + 1) * b)
            if valid is None:
                px, st = packed[rows, cols].ravel(), strong[rows, cols].ravel()
            else:
                v = valid[rows, cols]
                if not v.any():
                    continue
                px, st = packed[rows, cols][v], strong[rows, cols][v]
            _, counts = np.unique(px, return_counts=True)
            if counts.size > cfg.topk:
                counts = np.partition(counts, -cfg.topk)[-cfg.topk:]
            if counts.sum() >= cfg.cov_thr * px.size:
                out[i, j] = GRAPHIC if st.mean() >= cfg.edge_frac else FLAT
    return out


def block_map_to_pixels(block_map, height, width, block=64):
    """Block map -> per-pixel map of the given size."""
    return np.repeat(np.repeat(block_map, block, axis=0), block, axis=1)[:height, :width]


def random_fmap(rng, true_f, p_reset=0.5, max_boxes=3):
    """Random rectangle map (training with probability rand_f, so that the class map acts as an input):
    starting from all-TEXTURE or from the true map, 1..max_boxes random block rectangles get a random class."""
    nbh, nbw = true_f.shape
    f = np.zeros_like(true_f) if rng.random() < p_reset else true_f.copy()
    for _ in range(int(rng.integers(1, max_boxes + 1))):
        h, w = int(rng.integers(1, nbh + 1)), int(rng.integers(1, nbw + 1))
        y, x = int(rng.integers(0, nbh - h + 1)), int(rng.integers(0, nbw - w + 1))
        r = rng.random()
        f[y:y + h, x:x + w] = GRAPHIC if r < 0.6 else (FLAT if r < 0.75 else TEXTURE)
    return f


# ----------------------------------------------------------------------------- window-state helpers

def window_change(prev, cur, tol=0, win=WIN, valid=None):
    """Per-window (changed pixel share, mean absolute difference, largest absolute difference) of two
    uint8 frames whose sides are multiples of `win`; flat arrays in raster order. valid: optional
    (H, W) bool mask of the original pixels -- the edge-replicated padding is not content."""
    d = np.abs(cur.astype(np.int16) - prev.astype(np.int16))
    dmax = d.max(axis=-1)
    h, w = dmax.shape
    nh, nw = h // win, w // win
    if valid is not None:
        dmax = dmax * valid
        d = d * valid[..., None]
    changed = dmax > tol
    n = (np.full((nh, nw), float(win * win)) if valid is None
         else np.maximum(valid.reshape(nh, win, nw, win).sum(axis=(1, 3)), 1).astype(np.float64))
    frac = changed.reshape(nh, win, nw, win).sum(axis=(1, 3)) / n
    mad = d.mean(axis=-1).reshape(nh, win, nw, win).sum(axis=(1, 3)) / n
    peak = dmax.reshape(nh, win, nw, win).max(axis=(1, 3))
    return frac.ravel(), mad.ravel(), peak.ravel()


def pad_to_multiple(img, multiple=WIN):
    """Edge-replicate padding on the bottom/right (GVC1D's test protocol)."""
    h, w = img.shape[:2]
    ph, pw = (-h) % multiple, (-w) % multiple
    if ph == 0 and pw == 0:
        return img
    return np.pad(img, ((0, ph), (0, pw)) + ((0, 0),) * (img.ndim - 2), mode="edge")


def age_bucket(age):
    """Display-duration bucket of the state embedding (frames since the window content last changed):
    1 | 2-4 | 5-15 | >15."""
    return np.digitize(age, [2, 5, 16])


def valid_box(valid_hw):
    """(y0, x0, y1, x1) of the original area inside a padded frame. valid_hw is either (h, w) -- the
    original frame in the top-left corner, padding at the bottom/right (test protocol) -- or a box
    (y0, x0, y1, x1), e.g. after GVC1D's symmetric zero padding of training crops."""
    if len(valid_hw) == 2:
        return 0, 0, int(valid_hw[0]), int(valid_hw[1])
    y0, x0, y1, x1 = (int(v) for v in valid_hw)
    return y0, x0, y1, x1


def valid_pixels(height, width, valid_hw=None):
    """(height, width) bool mask of the original area of a padded frame (valid_hw: see valid_box), or
    None when it covers the whole frame."""
    if valid_hw is None:
        return None
    y0, x0, y1, x1 = valid_box(valid_hw)
    if (y0, x0, y1, x1) == (0, 0, height, width):
        return None
    return ((np.arange(height) >= y0) & (np.arange(height) < y1))[:, None] & \
        ((np.arange(width) >= x0) & (np.arange(width) < x1))[None, :]


def valid_blocks(height, width, valid_hw=None, block=64):
    """Blocks of a padded (height, width) frame that overlap the original area (valid_hw: see
    valid_box); padding-only blocks are TEXTURE by definition, never sent and not part of a fidelity share."""
    nbh, nbw = height // block, width // block
    if valid_hw is None:
        return np.ones((nbh, nbw), bool)
    y0, x0, y1, x1 = valid_box(valid_hw)
    by, bx = np.arange(nbh) * block, np.arange(nbw) * block
    return ((by + block > y0) & (by < y1))[:, None] & ((bx + block > x0) & (bx < x1))[None, :]


# ----------------------------------------------------------------------------- side information

class AdaptiveCounter:
    """Context-adaptive frequency model; code() returns the ideal code length in bits (what an
    adaptive arithmetic coder spends)."""

    def __init__(self, num_symbols, num_contexts=1, alpha=0.5):
        self.counts = np.full((num_contexts, num_symbols), alpha, dtype=np.float64)

    def code(self, symbol, ctx=0):
        c = self.counts[ctx]
        bits = -np.log2(c[symbol] / c.sum())
        c[symbol] += 1.0
        return bits


class SideInfoCoder:
    """Cost of the side information, in the order a decoder parses it:

    frame header : frame type (1 bit) + qp_base (3 bits)
    I-frame F    : every block, 3-ary class, context = (left class, top class)
    window mode  : 4-ary, context = (previous mode of the window, mode of the left window); not sent
                   for the first P-frame, where every window is coded
    REVERT       : snapshot index, ceil(log2 K) bits (1 bit for K = 2)
    change flag  : 1 bit per UPDATE window -- did its content change since its last coded state (a
                   new state) or is it coded again unchanged (refinement)? context = first P-frame /
                   in a refresh period / otherwise. It lets the decoder restart refresh periods and
                   count display durations exactly
    P-frame F    : blocks of UPDATE/REFRESH windows only: change flag, context = (class of the block
                   in the previous frame, left class, top class); if changed, the new class as a
                   3-ary symbol, context = previous class
    Blocks that lie entirely in the padding are never sent (TEXTURE by definition).
    """

    def __init__(self, k_snap, n_wide=1, header_bits=4, valid=None):
        self.mode = AdaptiveCounter(4, 25)     # prev mode * 5 + (left mode | 4 = none)
        self.change = AdaptiveCounter(2, 3)    # first P-frame | in refresh period | otherwise
        self.cls_i = AdaptiveCounter(3, 16)    # (left | 3 = none) * 4 + (top | 3 = none)
        self.flag = AdaptiveCounter(2, 27)     # previous * 9 + left * 3 + top
        self.cls_new = AdaptiveCounter(3, 3)   # previous class
        self.slot_bits = math.ceil(math.log2(k_snap)) if k_snap > 1 else 0
        self.header_bits = header_bits
        self.nw = n_wide                       # windows per row (left neighbour)
        self.valid = valid
        self.last = {}                         # bits of the last frame per syntax element (statistics)

    def _ok(self, i, j):
        return self.valid is None or bool(self.valid[i, j])

    def iframe_bits(self, f0):
        self.last = {"header": float(self.header_bits), "modes": 0.0, "change": 0.0, "f": 0.0}
        nbh, nbw = f0.shape
        for i in range(nbh):
            for j in range(nbw):
                if not self._ok(i, j):
                    continue
                left = int(f0[i, j - 1]) if j > 0 else 3
                top = int(f0[i - 1, j]) if i > 0 else 3
                self.last["f"] += self.cls_i.code(int(f0[i, j]), left * 4 + top)
        return float(sum(self.last.values()))

    def frame_bits(self, modes, prev_modes, revert_slot, changed, change_ctx, f_new, f_prev, coded_blocks):
        """Order of the syntax (the decoder parses it in this order): header, window modes (+ revert
        slots), change flags of UPDATE windows, then F of the blocks of coded windows."""
        last = self.last = {"header": float(self.header_bits), "modes": 0.0, "change": 0.0, "f": 0.0}
        for w, m in enumerate(modes if prev_modes is not None else ()):   # first P-frame: implied
            left = int(modes[w - 1]) if w % self.nw else 4
            last["modes"] += self.mode.code(int(m), int(prev_modes[w]) * 5 + left)
            if m == REVERT:
                last["modes"] += self.slot_bits
        for w, m in enumerate(modes):
            if m == UPDATE:
                last["change"] += self.change.code(int(bool(changed[w])), int(change_ctx[w]))
        nbh, nbw = f_new.shape
        for i in range(nbh):
            for j in range(nbw):
                if not coded_blocks[i, j] or not self._ok(i, j):
                    continue
                p = int(f_prev[i, j])
                left = int(f_new[i, j - 1]) if j > 0 else p
                top = int(f_new[i - 1, j]) if i > 0 else p
                changed_cls = int(f_new[i, j] != p)
                last["f"] += self.flag.code(changed_cls, p * 9 + left * 3 + top)
                if changed_cls:
                    last["f"] += self.cls_new.code(int(f_new[i, j]), p)
        return float(sum(last.values()))


# ----------------------------------------------------------------------------- shared bookkeeping

class RSMBook:
    """Decoder-derivable window states of one sequence (windows in raster order).

    Per window: age = frames its current content has been shown (a new state starts at a content
    change: REFRESH, REVERT, or an UPDATE whose change flag is set); refresh_cnt = frames left in
    the refresh period; K snapshot slots with their block classes. Everything is a function of the
    transmitted symbols (modes, revert slots, change flags, F of coded windows) and of the frame size.
    """

    def __init__(self, cfg, height, width, valid_hw=None):
        assert height % WIN == 0 and width % WIN == 0, "pad frames to multiples of 256"
        self.cfg = cfg
        self.nh, self.nw = height // WIN, width // WIN
        self.n = self.nh * self.nw
        self.cpw = WIN // cfg.block                      # blocks per window side
        self.valid = valid_blocks(height, width, valid_hw, cfg.block)
        self.valid_per_win = self.valid.reshape(self.nh, self.cpw, self.nw, self.cpw).sum(axis=(1, 3)).ravel()
        k = max(cfg.k_snap, 1)
        self.age = np.ones(self.n, np.int64)
        self.refresh_cnt = np.zeros(self.n, np.int64)    # frames left in the refresh period
        self.slot_used = np.zeros((self.n, k), bool)
        self.slot_tick = np.zeros((self.n, k), np.int64)
        self.slot_f = np.zeros((self.n, k, self.cpw, self.cpw), np.uint8)
        self.f_eff = None
        self.prev_modes = None
        self.t = 0

    # windows <-> block maps
    def win_blocks(self, f, w):
        i, j = divmod(w, self.nw)
        c = self.cpw
        return f[i * c:(i + 1) * c, j * c:(j + 1) * c]

    def fid_frac(self, f):
        """Share of fidelity blocks among the valid (non-padding) blocks of every window."""
        c = self.cpw
        fid = ((f != TEXTURE) & self.valid).astype(np.float64)
        return fid.reshape(self.nh, c, self.nw, c).sum(axis=(1, 3)).ravel() / np.maximum(self.valid_per_win, 1)

    def start(self, f0):
        """After the I-frame: f0 is the transmitted block map of frame 0."""
        self.f_eff = np.where(self.valid, f0, TEXTURE).astype(np.uint8)
        self.t = 0
        self.age[:] = 1
        if self.cfg.iframe_refresh and self.cfg.use_skip:
            self.refresh_cnt[self.fid_frac(self.f_eff) >= self.cfg.fid_qp_frac] = self.cfg.n_refine

    def _slot_for(self, w, exclude):
        """Free slot, else the least recently used one; never the slot being reverted to
        (-1: no slot available, the snapshot is skipped)."""
        free = np.nonzero(~self.slot_used[w])[0]
        free = free[free != exclude]
        if free.size:
            return int(free[0])
        cand = [j for j in range(self.slot_tick.shape[1]) if j != exclude]
        if not cand:
            return -1
        return int(min(cand, key=lambda j: self.slot_tick[w, j]))

    def apply(self, modes, revert_slot, f_coded, changed):
        """Advance one P-frame. f_coded only needs to be valid inside UPDATE/REFRESH windows;
        changed (per window) is the transmitted change flag, read only for UPDATE windows."""
        cfg = self.cfg
        t = self.t + 1
        modes = np.asarray(modes, np.int64)
        changed = np.asarray(changed, bool)
        upd = modes == UPDATE
        refresh, revert = modes == REFRESH, modes == REVERT
        coded = upd | refresh
        in_period = self.refresh_cnt > 0
        event = refresh | revert | (upd & changed)        # the window shows a new state from now on
        refine = upd & ~changed                           # coded again with unchanged content
        mem_type = np.where(refresh, 0, np.where(refine, 1, 2))
        mem_age = age_bucket(self.age)
        snap_create = np.full(self.n, -1, np.int64)
        f_prev = self.f_eff.copy()
        if cfg.k_snap > 0:
            # snapshot: the content being left has been shown for >= a_min + 1 frames
            for w in np.nonzero(event & (self.age >= cfg.a_min + 1))[0]:
                excl = int(revert_slot[w]) if revert[w] else -1
                j = self._slot_for(w, excl)
                if j < 0:
                    continue
                snap_create[w] = j
                self.slot_used[w, j] = True
                self.slot_tick[w, j] = t
                self.slot_f[w, j] = self.win_blocks(f_prev, w)
        f_coded = np.where(self.valid, f_coded, TEXTURE)
        for w in range(self.n):
            if revert[w]:
                j = int(revert_slot[w])
                self.win_blocks(self.f_eff, w)[...] = self.slot_f[w, j]
                self.slot_tick[w, j] = t
            elif coded[w]:
                self.win_blocks(self.f_eff, w)[...] = self.win_blocks(f_coded, w)
        raised = (refine & in_period) | refresh            # refinement frames and REFRESH frames (r_k = 1)
        qp_off = cfg.delta_f * (self.fid_frac(self.f_eff) >= cfg.fid_qp_frac) + cfg.delta_r * raised
        if cfg.use_skip:                                   # refresh periods exist only with the window states
            start = refresh | (upd & changed & (~in_period | cfg.restart_on_change))
            self.refresh_cnt = np.where(start, cfg.n_refine,
                                        np.where(revert, 0, np.maximum(self.refresh_cnt - 1, 0)))
        else:
            self.refresh_cnt[:] = 0
        self.age = np.where(event, 1, self.age + 1)
        self.prev_modes = modes
        self.t = t
        return {
            "snap_create": snap_create,
            "qp_offset": qp_off.astype(np.int64),
            "mem_type": mem_type.astype(np.int64),
            "mem_age": mem_age.astype(np.int64),
            "in_period": raised.astype(bool),
            "f_eff": self.f_eff.copy(),
            "f_prev": f_prev,
        }

    def coded_block_mask(self, modes):
        coded = np.isin(np.asarray(modes), (UPDATE, REFRESH))
        return np.repeat(np.repeat(coded.reshape(self.nh, self.nw), self.cpw, 0), self.cpw, 1)


# ----------------------------------------------------------------------------- encoder

class RSMEncoder:
    """Encoder-side window state machine of one sequence: feed padded uint8 frames (multiples of 256); valid_hw is
    the original area, (h, w) or a box (see valid_box): blocks entirely in the padding are TEXTURE
    and never sent, padding pixels do not count as changes."""

    def __init__(self, cfg, height, width, valid_hw=None):
        self.cfg = cfg
        self.book = RSMBook(cfg, height, width, valid_hw)
        k = max(cfg.k_snap, 1)
        self.slot_src = np.zeros((self.book.n, k, WIN, WIN, 3), np.uint8)
        self.side = SideInfoCoder(cfg.k_snap, self.book.nw, cfg.header_bits, self.book.valid)
        self.ref = None   # per window: the source content of its last coded / restored state
        self.valid_px = valid_pixels(height, width, valid_hw)

    def _win(self, img, w):
        i, j = divmod(w, self.book.nw)
        return img[i * WIN:(i + 1) * WIN, j * WIN:(j + 1) * WIN]

    def classes(self, frame, f=None):
        """Block classes of a padded frame (f: a given map instead, e.g. a random training map);
        padding-only blocks are TEXTURE."""
        f = block_classes(frame, self.cfg, self.valid_px) if f is None else np.array(f, np.uint8, copy=True)
        f[~self.book.valid] = TEXTURE
        return f

    def start(self, frame0, f0=None):
        f0 = self.classes(frame0, f0)
        self.book.start(f0)
        self.ref = frame0.copy()
        return {"f_eff": self.book.f_eff.copy(), "side_bits": self.side.iframe_bits(self.book.f_eff)}

    def _match(self, w, cur):
        win = self._win(cur, w).astype(np.int16)
        used = np.nonzero(self.book.slot_used[w])[0]
        vp = None if self.valid_px is None else self._win(self.valid_px, w)
        for j in used[np.argsort(-self.book.slot_tick[w, used])]:
            d = np.abs(win - self.slot_src[w, j].astype(np.int16))
            if vp is not None:
                d = d[vp]
            if (self.cfg.revert_mad <= 0 and not d.any()) or (self.cfg.revert_mad > 0 and d.mean() <= self.cfg.revert_mad):
                return int(j)
        return -1

    def step(self, frame, f_src=None, force=None):
        """Decide the modes of one P-frame. force: windows the encoder chooses to code (UPDATE or
        REFRESH, never REVERT) even if they are still (used by training to keep a window active)."""
        cfg, book = self.cfg, self.book
        f_src = self.classes(frame, f_src)
        # compare with what each window currently shows (its last coded source), not with the
        # previous frame: changes below the tolerance cannot accumulate unseen
        frac, mad, peak = window_change(self.ref, frame, cfg.tol, valid=self.valid_px)
        static = (frac == 0) | ((mad <= cfg.static_mad) & (peak <= cfg.static_peak))
        forced = np.zeros(book.n, bool)
        if force is not None:
            forced[np.asarray(force, np.int64)] = True
        modes = np.full(book.n, UPDATE, np.int64)
        revert = np.full(book.n, -1, np.int64)
        first = book.t == 0          # first P-frame: no decoder features to copy yet, every window coded
        in_period = book.refresh_cnt > 0
        for w in range(book.n):
            if first:
                continue
            if static[w]:
                if not (in_period[w] or forced[w] or not cfg.use_skip):
                    modes[w] = SKIP
                continue
            j = self._match(w, frame) if (cfg.use_revert and cfg.k_snap > 0 and not forced[w]) else -1
            if j >= 0:
                modes[w], revert[w] = REVERT, j
            elif cfg.use_refresh and frac[w] >= cfg.refresh_frac and mad[w] >= cfg.refresh_mad:
                modes[w] = REFRESH
        changed = ~static
        change_ctx = np.where(first, 0, np.where(in_period, 1, 2))
        prev_modes = book.prev_modes
        info = book.apply(modes, revert, f_src, changed)
        for w in np.nonzero(info["snap_create"] >= 0)[0]:          # pre-change state of the window
            self.slot_src[w, info["snap_create"][w]] = self._win(self.ref, w)
        for w in range(book.n):
            if modes[w] in (UPDATE, REFRESH):
                self._win(self.ref, w)[...] = self._win(frame, w)
            elif modes[w] == REVERT:
                self._win(self.ref, w)[...] = self.slot_src[w, revert[w]]
        coded = book.coded_block_mask(modes)
        info["side_bits"] = self.side.frame_bits(modes, prev_modes, revert, changed, change_ctx,
                                                 info["f_eff"], info["f_prev"], coded)
        info.update(modes=modes, revert_slot=revert, f_src=f_src, static=static, changed=changed)
        return info

"""Frame metrics of the GSCV evaluation (scripts/test.py).

Every codec is scored on the same 8-bit RGB frames (reconstructions are rounded to uint8 first), cropped to the
original resolution.

Whole frame : PSNR (RGB), MS-SSIM, LPIPS-Alex and DISTS as GVC1D's evaluator/eval_video.py (VideoMetric_test);
              SIQM (Gu et al., ISCAS 2015), an exact port of the released SIQM.p
Per region  : PSNR on the fidelity-critical pixels (text, graphics and flat blocks of the source frame, psnr_fid)
FID / KID   : Inception features of overlapped 256x256 patches of every frame (GVC1D, following DLF), pooled per
              test set and rate point by scripts/fid_kid.py
"""
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

PSNR_CAP = 100.0
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _psnr(mse):
    return PSNR_CAP if mse <= 1e-10 else min(PSNR_CAP, 10.0 * math.log10(1.0 / mse))


def to_tensor(img, device):
    """uint8 (H, W, 3) -> float (1, 3, H, W) in [0, 1]."""
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)[None].float().div_(255.0).to(device)


# ----------------------------------------------------------------------------- SIQM

def _gauss2d(sigma, size, device, dtype=torch.float32):
    """MATLAB fspecial('gaussian', size, sigma)."""
    ax = torch.arange(size, dtype=torch.float64, device=device) - (size - 1) / 2.0
    g = torch.exp(-ax ** 2 / (2.0 * sigma ** 2))
    k = torch.outer(g, g)
    return (k / k.sum()).to(dtype)[None, None]


def _ssim_map(a, b, win, c1=(0.01 * 255) ** 2, c2=(0.03 * 255) ** 2):
    """SSIM map as Wang et al.'s ssim_index (2004): window `win`, 'valid' filtering, K = 0.01 / 0.03, L = 255."""
    f = lambda t: F.conv2d(t, win)  # noqa: E731
    mu1, mu2 = f(a), f(b)
    s11, s22, s12 = f(a * a) - mu1 ** 2, f(b * b) - mu2 ** 2, f(a * b) - mu1 * mu2
    return ((2 * mu1 * mu2 + c1) * (2 * s12 + c2)) / ((mu1 ** 2 + mu2 ** 2 + c1) * (s11 + s22 + c2))


RGB2GRAY = (0.298936021293775, 0.587043074451121, 0.114020904255103)     # MATLAB rgb2gray


def rgb2gray_u8(x):
    """MATLAB rgb2gray of an 8-bit RGB image, rounded to 8 bit like its uint8 output; x (1, 3, H, W) in [0, 1]
    -> (1, 1, H, W) in [0, 255]."""
    w = torch.tensor(RGB2GRAY, dtype=torch.float64, device=x.device)[None, :, None, None]
    return torch.floor((torch.round(x.double() * 255.0) * w).sum(1, keepdim=True) + 0.5).float()


def siqm(x, y):
    """Structure-induced quality metric (Gu, Wang, Zhai, Ma, Lin, ISCAS 2015), as the released SIQM.p:
      w = fspecial('gaussian', 7, 2.5)   -- the SSIM window and the low-pass alike
      r_f = filter2(w, r, 'same')        -- zero border
      SIQM = sum(SSIM_MAP(r, d) .* m) / sum(m),  m = (1 - SSIM_MAP(r, r_f)) .^ 0.4
    with Wang et al.'s ssim_index maps ('valid', K = 0.01 / 0.03, L = 255) on MATLAB's rgb2gray of the frames
    (rounded to 8 bit), in float64; 1 - SSIM slightly below 0 is set to 0. x, y: (1, 3, H, W) in [0, 1]
    (reference, distorted)."""
    r, d = rgb2gray_u8(x).double(), rgb2gray_u8(y).double()
    win = _gauss2d(2.5, 7, x.device, torch.float64)
    rf = F.conv2d(F.pad(r, (3, 3, 3, 3)), win)
    m = (1.0 - _ssim_map(r, rf, win)).clamp_min(0.0) ** 0.4
    s = _ssim_map(r, d, win)
    den = float(m.sum())
    return float((s * m).sum()) / den if den > 0 else float(s.mean())


# ----------------------------------------------------------------------------- FID / KID features (GVC1D)

def overlapped_patches(img, size=256, shift=128):
    """The 256x256 patches of a frame (H, W, 3) on the grid at offset 0 and on the grid shifted by `shift`
    pixels in both directions (GVC1D: "overlapped 256 x 256 patches, following DLF")."""
    h, w = img.shape[:2]
    out = []
    for off in (0, shift):
        for y0 in range(off, h - size + 1, size):
            for x0 in range(off, w - size + 1, size):
                out.append(img[y0:y0 + size, x0:x0 + size])
    return np.stack(out) if out else np.zeros((0, size, size, 3), np.uint8)


class PatchFeatures:
    """2048-d Inception-v3 pool features of the patches (GVC1D evaluator/inception.py: torch-fidelity's
    Inception-v3 with the pytorch-fid weights pt_inception-2015-12-05, uint8 input resized to 299 like
    TensorFlow), the features of GVC1D's rFID."""

    def __init__(self, device="cuda", size=256, shift=128, batch=64):
        if REPO_ROOT not in sys.path:
            sys.path.insert(0, REPO_ROOT)
        from evaluator.inception import get_inception_model
        self.net = get_inception_model().to(device).eval()      # downloads the weights on first use
        self.device, self.size, self.shift, self.batch = device, size, shift, batch

    @torch.no_grad()
    def __call__(self, img):
        p = overlapped_patches(img, self.size, self.shift)
        feats = []
        for i in range(0, len(p), self.batch):
            x = torch.from_numpy(p[i:i + self.batch]).permute(0, 3, 1, 2).contiguous().to(self.device)
            feats.append(self.net(x)["2048"].float().cpu())
        return torch.cat(feats).numpy() if feats else np.zeros((0, 2048), np.float32)


# ----------------------------------------------------------------------------- frame scorer

class FrameScorer(torch.nn.Module):
    def __init__(self, device="cuda", fid_kid=False):
        super().__init__()
        import lpips
        from DISTS_pytorch import DISTS
        self.device = device
        self.lpips = lpips.LPIPS(net="alex", verbose=False).eval().to(device)      # as GVC1D's test.py
        self.dists = DISTS().eval().to(device)
        self.patch_features = PatchFeatures(device) if fid_kid else None

    @torch.no_grad()
    def score(self, src, rec, fid_mask=None):
        """src, rec: uint8 (H, W, 3); fid_mask: optional bool (H, W) map of the fidelity-critical pixels."""
        from pytorch_msssim import ms_ssim
        x, y = to_tensor(src, self.device), to_tensor(rec, self.device)
        se = ((x - y) ** 2).mean(dim=1)[0]                      # (H, W)
        out = {
            "psnr": _psnr(float(se.mean())),
            "msssim": float(ms_ssim(x, y, data_range=1.0)),
            "lpips": float(self.lpips(x, y, normalize=True)),
            "dists": float(self.dists(x, y)),
            "siqm": siqm(x, y),
        }
        if fid_mask is not None:
            m = torch.from_numpy(np.ascontiguousarray(fid_mask)).to(self.device)
            out["psnr_fid"] = _psnr(float(se[m].mean())) if bool(m.any()) else None
        return out


def mean_metrics(rows):
    """Average per-frame dicts, skipping None entries."""
    keys = sorted({k for r in rows for k in r})
    out = {}
    for k in keys:
        vals = [r[k] for r in rows if r.get(k) is not None]
        if vals and all(isinstance(v, (int, float)) for v in vals):
            out[k] = float(np.mean(vals))
    return out

"""Region-adaptive rate-distortion-perception loss of GSCV (the loss function of the paper).

    L       = (1/T) sum_t [ R(y^_t) + R(z^_t) + lambda_fid * L_fid_t + lambda_gen * L_gen_t ]
    L_fid_t = eta_t * [ L1(x, x^ | O_fid) + alpha * L_grad(x, x^ | O_graphic) ] + beta * LPIPS(x, x^ | O_fid)
    L_gen_t = eta_t / 2 * [ L1 + MSE ](x, x^ | O_tex) + gamma * LPIPS(x, x^ | O_tex) + delta * L_adv(x^ | O_tex)

Configuration names (losses.*): lambda_fid = 2 and lambda_tex = 1 (lambda_fid = 2 lambda, lambda_gen = lambda, with
lambda the trade-off of each window's quality index qp_k), alpha_edge = alpha = 2, gamma_fid = beta = 1, gamma = 1,
gan_weight = delta = 0.1. eta_t is GVC1D's frame-type weight (train.py, DIST_WEIGHTS); it multiplies the pixel terms
only, as GVC1D's Codec_loss does. L_grad is the Sobel gradient error on text and graphics.

O_fid (text, graphics and flat blocks), O_graphic and O_tex (texture blocks) only cover the coded (Update /
Refresh) windows: Skip and Revert windows carry no new information. lambda (and the GAN scale delta(lambda) of GVC1D)
are applied per window. (. | O) is the sum over the pixels of O divided by the number of pixels of the frame (the
normalisation of the rate), so every window trades rate for distortion at its own lambda whatever else is coded.

The texture perceptual and adversarial terms see the texture composite x_tex = O_tex * x^ + (1 - O_tex) * x, so their
gradients cannot reach fidelity-critical pixels; the fidelity perceptual term sees the fidelity composite
x_fid = O_fid * x^ + (1 - O_fid) * x. LPIPS is GVC1D's mix 0.5 LPIPS-VGG + 0.5 LPIPS-Alex with spatial maps (GVC1D's
ConvNeXt-224 branch is dropped: it downsamples whole frames and cannot see text). GVC1D's conditional discriminator
sees (frame, previous frame) pairs: real (x_t, x_t-1), fake (x_tex_t, O_tex * x^_t-1 + (1 - O_tex) * x_t-1), so
outside the textures the two pairs are identical; the generator term weights every discriminator patch by its
lambda * delta(lambda)-weighted texture content, the hinge terms by its texture share.

With region=False (stage S0) the loss is GVC1D's own objective over the coded windows,
lambda * (eta_t * (0.5 L1 + 0.5 MSE) + LPIPS-mix + GAN) + R.
"""
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from modeling.modules.discriminator_video import VideoDiscriminator
from modeling.modules.lpips import LPIPS as LPIPS_vgg
from modeling.modules.lpips import normalize_tensor
from modeling.modules.lpips_alex.lpips import LPIPS as LPIPS_alex

_SOBEL = torch.tensor([[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]]) / 4.0


def sobel(x):
    with torch.autocast(device_type=x.device.type, enabled=False):
        x = x.float()
        c = x.shape[1]
        kx = _SOBEL.to(x)[None, None].repeat(c, 1, 1, 1)
        ky = _SOBEL.t().contiguous().to(x)[None, None].repeat(c, 1, 1, 1)
        xp = F.pad(x, (1, 1, 1, 1), mode="replicate")
        return F.conv2d(xp, kx, groups=c), F.conv2d(xp, ky, groups=c)


def window_map(values, grid, batch, size=256):
    """(B*nH*nW,) per-window values -> (B, 1, nH*size, nW*size) map."""
    nh, nw = grid
    m = values.view(batch, 1, nh, nw).to(torch.float32)
    return m.repeat_interleave(size, dim=2).repeat_interleave(size, dim=3)


def block_masks(fclass, height):
    """(B, H/b, W/b) classes -> (B, 1, H, W) float masks of GRAPHIC, FLAT, TEXTURE."""
    size = height // fclass.shape[1]
    f = fclass[:, None].repeat_interleave(size, dim=2).repeat_interleave(size, dim=3)
    return (f == 1).float(), (f == 2).float(), (f == 0).float()


def masked_mean(value, mask, weight=None):
    """Per-sample (weighted) sum of value (B, C, H, W) over the pixels of mask (B, 1, H, W) and the
    channels, divided by the number of frame pixels times C."""
    v = value * mask if weight is None else value * mask * weight
    den = torch.full_like(mask.sum(dim=(1, 2, 3)), float(mask.shape[-2] * mask.shape[-1]))
    return v.sum(dim=(1, 2, 3)) / (den * value.shape[1]).clamp_min(1.0)


class LPIPSMix(nn.Module):
    """0.5 * LPIPS-VGG + 0.5 * LPIPS-Alex on [0, 1] inputs (both correctly normalised); forward gives
    per-image values, spatial() per-pixel maps (B, 1, H, W)."""

    def __init__(self):
        super().__init__()
        self.vgg = LPIPS_vgg().eval()
        self.alex = LPIPS_alex(net="alex", version="0.1", verbose=False).eval()
        self.alex_spatial = LPIPS_alex(net="alex", version="0.1", spatial=True, verbose=False).eval()
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x, y):
        # lpips_alex forwards positional args only: (in0, in1, retPerLayer, normalize)
        return 0.5 * self.vgg(x, y).flatten() + 0.5 * self.alex(x, y, False, True).flatten()

    def _vgg_spatial(self, x, y):
        m = self.vgg
        a, b = m.net(m.scaling_layer(x * 2.0 - 1.0)), m.net(m.scaling_layer(y * 2.0 - 1.0))
        val = 0.0
        for k, lin in enumerate((m.lin0, m.lin1, m.lin2, m.lin3, m.lin4)):
            d = (normalize_tensor(a[k]) - normalize_tensor(b[k])) ** 2
            val = val + F.interpolate(lin.model(d), size=x.shape[-2:], mode="bilinear", align_corners=False)
        return val

    def spatial(self, x, y):
        return 0.5 * self._vgg_spatial(x, y) + 0.5 * self.alex_spatial(x, y, False, True)


class PairDiscriminator(nn.Module):
    """GVC1D's video discriminator run separately on the real and on the fake pairs inside one
    forward: one DDP call per step, and real / fake batches are normalised separately (BatchNorm),
    as in GVC1D and as in the generator pass (fake pairs only)."""

    def __init__(self):
        super().__init__()
        self.disc = VideoDiscriminator()

    def forward(self, real, prev_real, fake, prev_fake):
        return self.disc((real, prev_real)), self.disc((fake, prev_fake))


class GSCVLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.region = bool(cfg.get("region", True))
        self.lambda_fid = float(cfg.get("lambda_fid", 2.0))
        self.alpha_edge = float(cfg.get("alpha_edge", 2.0))
        self.lambda_tex = float(cfg.get("lambda_tex", 1.0))
        self.gamma = float(cfg.get("gamma", 1.0))
        self.gamma_fid = float(cfg.get("gamma_fid", 1.0))     # beta: LPIPS weight of the fidelity-critical regions
        self.gvc1d_lpips = float(cfg.get("gvc1d_lpips_weight", 1.0))
        self.gan_weight = float(cfg.get("gan_weight", 0.1))
        self.adaptive_gan = bool(cfg.get("adaptive_gan", True))
        self.lecam_weight = float(cfg.get("lecam_weight", 0.001))
        self.lecam_decay = 0.999
        self.perceptual = LPIPSMix()
        self.register_buffer("ema_real", torch.zeros(()))
        self.register_buffer("ema_fake", torch.zeros(()))

    def train(self, mode=True):
        super().train(mode)
        self.perceptual.eval()
        return self

    def frame_terms(self, x, x_hat, modes, fclass, lam_win, grid, delta_win=None):
        """Distortion terms of one P-frame (lambda_q-weighted; rd_t is applied by the caller).

        x, x_hat: (B, 3, H, W); modes, lam_win, delta_win: (B*nH*nW,); fclass: (B, H/b, W/b) or None.
        Returns per-sample terms, the texture composite and two pixel-weight maps for the GAN. The
        caller forms lambda * D_t as rd_t * pix + perc (rd_t on the pixel terms only, as GVC1D):
          region : pix = lambda-weighted fidelity terms + texture pixel term, perc = lambda-weighted
                   texture LPIPS + lambda_F * gamma_F * fidelity LPIPS; pix1 / perc1 = the texture pixel
                   / LPIPS terms without lambda (what the adversarial term competes with in the adaptive
                   GAN weight)
          GVC1D  : pix = lambda * mix, perc = lambda * LPIPS (weight 1.0); pix1 / perc1 without lambda
          both   : tex (B, 1, H, W) texture pixels of coded windows, adv_w = tex * lambda * delta
        """
        b, _, h, w = x.shape
        act = window_map((modes == 0) | (modes == 2), grid, b).to(x.device)
        lam = window_map(lam_win, grid, b).to(x.device)
        dlt = window_map(delta_win, grid, b).to(x.device) if delta_win is not None else torch.ones_like(lam)
        x, x_hat = x.float(), x_hat.float()
        e = x_hat - x
        if self.region and fclass is not None:
            gra, flat, tex = block_masks(fclass.to(x.device), h)
            fid, gra, tex = (gra + flat) * act, gra * act, tex * act
            gx_h, gy_h = sobel(x_hat)
            gx, gy = sobel(x)
            edge = 0.5 * ((gx_h - gx).abs() + (gy_h - gy).abs())
            comp = tex * x_hat + (1.0 - tex) * x
            lp = self.perceptual.spatial(x, comp).float()                    # (B, 1, H, W)
            lp_fid = None
            if self.gamma_fid > 0:                                           # fidelity composite
                lp_fid = self.perceptual.spatial(x, fid * x_hat + (1.0 - fid) * x).float()
            out = {}
            for tag, wgt in (("", lam), ("1", None)):
                fid_term = masked_mean(e.abs(), fid, wgt) + self.alpha_edge * masked_mean(edge, gra, wgt)
                tex_pix = masked_mean(0.5 * e.abs() + 0.5 * e ** 2, tex, wgt)
                tex_lp = masked_mean(lp, tex, wgt)
                if tag == "":
                    out["pix"] = self.lambda_fid * fid_term + self.lambda_tex * tex_pix
                    out["perc"] = self.lambda_tex * self.gamma * tex_lp
                    if lp_fid is not None:
                        out["perc"] = out["perc"] + self.lambda_fid * self.gamma_fid * masked_mean(lp_fid, fid, wgt)
                else:
                    out["pix1"] = self.lambda_tex * tex_pix
                    out["perc1"] = self.lambda_tex * self.gamma * tex_lp
        else:   # GVC1D objective on the coded windows
            tex = act
            comp = tex * x_hat + (1.0 - tex) * x
            mix = 0.5 * e.abs() + 0.5 * e ** 2
            perc1 = self.gvc1d_lpips * self.perceptual(x, comp).float()
            out = {"pix": masked_mean(mix, act, lam), "perc": perc1 * masked_mean(lam, act, None)
                   / masked_mean(torch.ones_like(lam), act, None).clamp_min(1e-12),
                   "pix1": masked_mean(mix, act, None), "perc1": perc1}
        out["tex"] = tex
        out["adv_w"] = tex * lam * dlt
        return out, comp, tex

    def generator_adv(self, pair, comp, prev_fake, adv_w, tex):
        """Hinge generator terms per sample (fake pairs only, all frames of a group in one batch):
        -sum_p w_p D_p with patch weights w = pooled lambda*delta*texture (g) or pooled texture
        (g1, lambda-free, for the adaptive weight), divided by the number of patches."""
        logits = pair.disc((comp, prev_fake)).float()
        shape = logits.shape[-2:]
        w, w1 = F.adaptive_avg_pool2d(adv_w, shape), F.adaptive_avg_pool2d(tex, shape)
        den = float(shape[0] * shape[1])
        return -(logits * w).sum(dim=(1, 2, 3)) / den, -(logits * w1).sum(dim=(1, 2, 3)) / den

    @staticmethod
    def adaptive_weight(rec, adv, last_layer, max_weight=1.0):
        """GVC1D's balance of the adversarial gradient against the reconstruction gradient at the
        last decoder layer (computed once per batch, against the texture terms only)."""
        g_rec = torch.autograd.grad(rec, last_layer, retain_graph=True, allow_unused=True)[0]
        g_adv = torch.autograd.grad(adv, last_layer, retain_graph=True, allow_unused=True)[0]
        if g_rec is None or g_adv is None:      # no texture pixel was coded in this group
            return torch.zeros((), device=last_layer.device)
        return (g_rec.norm() / (g_adv.norm() + 1e-4)).clamp(0.0, max_weight).detach()

    def discriminator_loss(self, pair, real, prev_real, fake, prev_fake, tex):
        """Texture-weighted hinge (real + fake, as GVC1D) + LeCam on real and fake pairs (one call of
        the DDP-wrapped PairDiscriminator). As GVC1D's compute_lecam_loss, LeCam compares the hinge
        terms (not the raw logits) with their EMAs; the EMAs follow the mean over the GPUs, so every
        GPU keeps the same values (GVC1D broadcasts rank 0's buffers)."""
        lr, lf = pair(real, prev_real, fake.detach(), prev_fake.detach())
        lr, lf = lr.float(), lf.float()
        wgt = F.adaptive_avg_pool2d(tex.float(), lr.shape[-2:])
        den = wgt.sum().clamp_min(1e-6)
        hr, hf = (wgt * F.relu(1.0 - lr)).sum() / den, (wgt * F.relu(1.0 + lf)).sum() / den
        loss = hr + hf
        lecam = F.relu(hr - self.ema_fake) ** 2 + F.relu(self.ema_real - hf) ** 2
        with torch.no_grad():
            stats = torch.stack([hr, hf, (wgt.sum() > 1e-6).float()]).detach()
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(stats)
                stats = stats / dist.get_world_size()
            if float(stats[2]) > 0:
                mr, mf = stats[0] / stats[2], stats[1] / stats[2]     # mean over the GPUs with texture
                d = 1 - self.lecam_decay
                self.ema_real.mul_(1 - d).add_(d * mr)
                self.ema_fake.mul_(1 - d).add_(d * mf)
        logs = {"d_real": ((wgt * lr).sum() / den).detach(), "d_fake": ((wgt * lf).sum() / den).detach()}
        return loss + self.lecam_weight * lecam, logs

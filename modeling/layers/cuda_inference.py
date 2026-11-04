# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn.functional as F


def clamp_reciprocal_with_quant(q_dec, y, min_val, force_pytorch=False):
    q_dec = torch.clamp_min(q_dec, min_val)
    q_enc = torch.reciprocal(q_dec)
    y = y * q_enc
    return q_dec, y


def process_with_mask(y, scales, means, mask, force_zero_thres, force_pytorch=False):
    scales_hat = scales * mask
    means_hat = means * mask

    y_res = (y - means_hat) * mask
    y_q = torch.round(y_res)
    if force_zero_thres is not None:
        cond = scales_hat > force_zero_thres
        y_q = y_q * cond
        scales_hat = scales_hat * cond
    y_q = torch.clamp(y_q, -128., 127.)
    y_hat = y_q + means_hat

    return y_res, y_q, y_hat, scales_hat


def build_index_dec(scales, scale_min, scale_max, log_scale_min, log_step_recip,
                    skip_thres=None, force_pytorch=False):
    skip_cond = None
    if skip_thres is not None:
        skip_cond = scales > skip_thres
    scales = scales.clamp_(scale_min, scale_max)
    indexes = (torch.log(scales) - log_scale_min) * log_step_recip
    indexes = indexes.to(dtype=torch.uint8)
    return indexes, skip_cond


def build_index_enc(symbols, scales, scale_min, scale_max, log_scale_min,
                    log_step_recip, skip_thres=None, force_pytorch=False):
    scales = scales.clamp_(scale_min, scale_max)
    indexes = (torch.log(scales) - log_scale_min) * log_step_recip
    indexes = indexes.to(dtype=torch.uint8)
    symbols = symbols.to(dtype=torch.int16)
    out = (symbols << 8) + indexes
    out = out.to(dtype=torch.int16)
    if skip_thres is not None:
        skip_cond = scales > skip_thres
        out = out[skip_cond]
    return out


def replicate_pad(x, pad_b, pad_r, force_pytorch=False):
    if pad_b == 0 and pad_r == 0:
        return x
    return F.pad(x, (0, pad_r, 0, pad_b), mode="replicate")


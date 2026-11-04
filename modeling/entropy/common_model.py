# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import math

import torch
from torch import nn
from pytorch_msssim import MS_SSIM

from ..layers.layers import LowerBound
from ..layers.cuda_inference import process_with_mask, replicate_pad, clamp_reciprocal_with_quant
from .entropy_models import BitEstimator, GaussianEncoder, EntropyCoder


class CompressionModel(nn.Module):
    def __init__(self, z_channel):
        super().__init__()

        self.z_channel = z_channel
        self.entropy_coder = None
        self.bit_estimator_z = BitEstimator(64, z_channel)
        # self.bit_estimator_z= None
        self.gaussian_encoder = GaussianEncoder()
        self.force_zero_thres = None

        self.mse = nn.MSELoss(reduction='none')
        self.ssim = MS_SSIM(data_range=1.0, size_average=False)

        self.masks = {}
        self.force_generate_mask = False

        self.cuda_streams = {}

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                torch.nn.init.xavier_normal_(m.weight, 1.)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.)
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
                m.weight.data = nn.init.trunc_normal_(m.weight.data, mean=0.0, std=0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                m.weight.data = nn.init.trunc_normal_(m.weight.data, mean=0.0, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                m.bias.data.zero_()
                m.weight.data.fill_(1.0)

    def set_force_zero_thres(self, scale_thres):
        self.force_zero_thres = scale_thres

    def set_force_generate_mask(self, force):
        self.force_generate_mask = force

    def get_force_generate_mask(self):
        return self.force_generate_mask

    def get_z_cdf_info(self):
        return self.bit_estimator_z.get_cdf_info()

    def get_y_cdf_info(self):
        return self.gaussian_encoder.get_cdf_info()

    def get_cuda_stream(self, device, idx=0, priority=0):
        key = f"{device}_{priority}_{idx}"
        if key not in self.cuda_streams:
            self.cuda_streams[key] = torch.cuda.Stream(device, priority=priority)
        return self.cuda_streams[key]

    def quant(self, x, force_detach=False):
        if self.training or force_detach:
            n = torch.round(x) - x
            return x + n.detach()

        return torch.round(x)

    @staticmethod
    def get_index_tensor(qp, device):
        if not isinstance(qp, list):
            qp = [qp]
        return torch.tensor(qp, dtype=torch.int32, device=device)

    @staticmethod
    def get_qp_num():
        return 8
        
    @staticmethod
    def get_padding_size(height, width, p=64):
        new_h = (height + p - 1) // p * p
        new_w = (width + p - 1) // p * p
        padding_right = new_w - width
        padding_bottom = new_h - height
        return padding_right, padding_bottom

    @staticmethod
    def get_downsampled_shape(height, width, p):
        new_h = (height + p - 1) // p * p
        new_w = (width + p - 1) // p * p
        return int(new_h / p + 0.5), int(new_w / p + 0.5)

    def add_noise(self, x):
        noise = torch.nn.init.uniform_(torch.empty_like(x), -0.5, 0.5)
        return x + noise.detach()

    @staticmethod
    def probs_to_bits(probs):
        factor = -1.0 / math.log(2.0)
        bits = torch.log(probs + 1e-5) * factor
        bits = LowerBound.apply(bits, 0)
        return bits

    def get_y_gaussian_bits(self, y, sigma):
        if self.training:
            probs = GaussianEncoder.get_gaussian_prob(y, sigma)
        else:
            sigma = sigma.clamp(1e-5, 1e10)
            gaussian = torch.distributions.normal.Normal(0., sigma)
            probs = gaussian.cdf(y + 0.5) - gaussian.cdf(y - 0.5)
            probs = probs.to(torch.float32)
        return CompressionModel.probs_to_bits(probs)

    def get_z_bits(self, z, bit_estimator, index):
        if self.training:
            probs = bit_estimator.get_prob(z, index)
        else:
            probs = bit_estimator.get_cdf(z + 0.5, index) - bit_estimator.get_cdf(z - 0.5, index)
            probs = probs.to(torch.float32)
        return CompressionModel.probs_to_bits(probs)

    def update(self, force=False):
        self.entropy_coder = EntropyCoder()
        self.gaussian_encoder.update(force=force, entropy_coder=self.entropy_coder)
        self.bit_estimator_z.update(force=force, entropy_coder=self.entropy_coder)

    def pad_for_y(self, y):
        if self.training:
            return y

        _, _, H, W = y.size()
        padding_r, padding_b = self.get_padding_size(H, W, 4)
        y_pad = replicate_pad(y, padding_b, padding_r)
        return y_pad

    def separate_prior(self, params, is_video=False):
        if is_video:
            quant_step, scales, means = params.chunk(3, 1)
            quant_step = LowerBound.apply(quant_step, 0.5)
            q_enc = 1. / quant_step
            q_dec = quant_step
        else:
            q = params[:, :2, :, :]
            q_enc, q_dec = (torch.sigmoid(q) * 1.5 + 0.5).chunk(2, 1)
            scales, means = params[:, 2:, :, :].chunk(2, 1)
        return q_enc, q_dec, scales, means

    @staticmethod
    def separate_prior_for_video_encoding(params, y):
        q_dec, scales, means = params.chunk(3, 1)
        q_dec, y = clamp_reciprocal_with_quant(q_dec, y, 0.5)
        return y, q_dec, scales, means

    @staticmethod
    def separate_prior_for_video_decoding(params):
        quant_step, scales, means = params.chunk(3, 1)
        quant_step = torch.clamp_min(quant_step, 0.5)
        return quant_step, scales, means

    def process_with_mask(self, y, scales, means, mask):
        if self.training:
            scales_hat = scales * mask
            means_hat = means * mask

            y_res = (y - means_hat) * mask
            y_q = self.quant(y_res)
            y_hat = y_q + means_hat

            return y_res, y_q, y_hat, scales_hat

        return process_with_mask(y, scales, means, mask, self.force_zero_thres)


    def forward_AR_prior(self, y, common_params,
                                y_spatial_prior_reduction,
                                y_spatial_prior,):
        q_enc, q_dec, scales, means = self.separate_prior(common_params,True)
        if y_spatial_prior_reduction is not None:
            # BCHW to BHWC
            common_params = y_spatial_prior_reduction(common_params.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        B, C, H, W = y.size()
        mask = torch.ones_like(y[..., :1], dtype=y.dtype, device=y.device)

        y = y * q_enc

        y_res_0, y_q_0, y_hat_0, s_hat_0 = self.process_with_mask(y[..., :1], scales[..., :1], means[..., :1], mask)
        y_hat_so_far = y_hat_0
        y_res= y_res_0
        y_q = y_q_0
        scales_hat = s_hat_0

        for i in range(1, W):
            params = torch.cat((y_hat_so_far, common_params), dim=-1)
            scales, means = y_spatial_prior(params, idx=i-1).chunk(2, 1)
            y_res_i, y_q_i, y_hat_i, s_hat_i = self.process_with_mask(y[..., i:i+1], scales, means, mask)
            y_hat_so_far = torch.cat((y_hat_so_far, y_hat_i), dim=-1)
            y_res = torch.cat((y_res, y_res_i), dim=-1)
            y_q = torch.cat((y_q, y_q_i), dim=-1)
            scales_hat = torch.cat((scales_hat, s_hat_i), dim=-1)

        y_hat = y_hat_so_far * q_dec

        return y_res, y_q, y_hat, scales_hat

    def compress_AR_prior(self, y, common_params,
                                y_spatial_prior_reduction,
                                y_spatial_prior,):
        y, q_dec, scales, means = self.separate_prior_for_video_encoding(common_params, y)
        common_params = y_spatial_prior_reduction(common_params.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        B, C, H, W = y.size()
        mask = torch.ones_like(y[..., :1], dtype=y.dtype, device=y.device)

        _, y_q_0, y_hat_0, s_hat_0 = self.process_with_mask(y[..., :1], scales[..., :1], means[..., :1], mask)
        y_hat_so_far = y_hat_0
        y_q = y_q_0
        scales_hat = s_hat_0

        for i in range(1, W):
            params = torch.cat((y_hat_so_far, common_params), dim=-1)
            scales, means = y_spatial_prior(params, idx=i-1).chunk(2, 1)
            _, y_q_i, y_hat_i, s_hat_i = self.process_with_mask(y[..., i:i+1], scales, means, mask)
            y_hat_so_far = torch.cat((y_hat_so_far, y_hat_i), dim=-1)
            y_q = torch.cat((y_q, y_q_i), dim=-1)
            scales_hat = torch.cat((scales_hat, s_hat_i), dim=-1)

        y_hat = y_hat_so_far * q_dec
        '''
        y_hat shape: B, C, 1,32
        y_q shape: B, C, 1,32
        scales_hat shape: B, C, 1,32
        '''

        return y_q, scales_hat, y_hat

    def decompress_AR_prior(self, common_params,
                                y_spatial_prior_reduction,
                                y_spatial_prior):
        q_dec, scales, means = self.separate_prior_for_video_decoding(common_params)
        common_params = y_spatial_prior_reduction(common_params.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        dtype = means.dtype
        device = means.device
        W= means.size(3)
        mask = torch.ones_like(means[..., :1], dtype=dtype, device=device)

        scales_0 = scales[..., :1]
        indexes, skip_cond = self.gaussian_encoder.build_indexes_decoder(scales_0,
                                                                         self.force_zero_thres)
        self.gaussian_encoder.decode_y(indexes)

        y_q_0 = self.gaussian_encoder.get_y(scales_0.shape,
                                            scales_0.numel(),
                                            dtype, device,
                                            skip_cond, indexes)
        y_hat_so_far = y_q_0 + means[..., :1]
        for i in range(1, W):
            params = torch.cat((y_hat_so_far, common_params), dim=-1)
            scales, means = y_spatial_prior(params, idx=i-1).chunk(2, 1)
            y_q_i = self.gaussian_encoder.decode_and_get_y(scales, dtype, device,
                                                       self.force_zero_thres)
            y_hat_i = y_q_i + means
            y_hat_so_far = torch.cat((y_hat_so_far, y_hat_i), dim=-1)
        y_hat = y_hat_so_far * q_dec

        return y_hat

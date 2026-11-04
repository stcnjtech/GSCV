# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from torch import nn

from .common_model import CompressionModel
from ..layers.transform_layers import ResidualAttentionBlock, ResidualAttentionUpDownsampleBlock, ResidualAttentionARBlock
from einops import rearrange

g_ch_enc_dec = 256
g_ch_recon = 320

def _expand_token(token, batch_size: int):
    return token.unsqueeze(0).expand(batch_size, -1, -1)

class WSiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sigmoid(4.0 * x) * x

class WSiLUChunkAdd(nn.Module):
    def __init__(self):
        super().__init__()
        self.silu = WSiLU()

    def forward(self, x):
        x1, x2 = self.silu(x).chunk(2, 1)
        return x1 + x2

class Feature_adaptor_i(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_size = 16
        self.patch_embed = nn.Conv2d(
            in_channels=3, out_channels=g_ch_enc_dec,
              kernel_size=self.patch_size, stride=self.patch_size, bias=True)
        self.dc = nn.Sequential(
            nn.Conv2d(g_ch_enc_dec, g_ch_enc_dec, 1),
            WSiLU(),
            nn.Conv2d(g_ch_enc_dec, g_ch_enc_dec, 3, padding=1, groups=g_ch_enc_dec),
            nn.Conv2d(g_ch_enc_dec, g_ch_enc_dec, 1),
        )
        self.ffn = nn.Sequential(
            nn.Conv2d(g_ch_enc_dec, g_ch_enc_dec * 4, 1),
            WSiLUChunkAdd(),
            nn.Conv2d(g_ch_enc_dec * 2, g_ch_enc_dec, 1),
        )
    def forward(self, x, stack_shape):
        x = self.patch_embed(x) # [B, C, H, W]
        out = self.dc(x) + x
        out = self.ffn(out) + out
        nH, nW = stack_shape
        out =  rearrange(out, 'B C (nH h) (nW w) -> (B nH nW) C h w', h=16, w=16, nH=nH, nW=nW).contiguous()
        return out

class FeatureExtractor(nn.Module):
    def __init__(self, y_tokens_num=32, use_long_term_feature=False, memory_num=32, memory_dim=512):
        super().__init__()
        self.patch_size = 16
        self.grid_size = 16
        self.num_latent_tokens = y_tokens_num
        self.use_long_term_feature = use_long_term_feature
        self.memory_num = 0
        self.width = g_ch_enc_dec
        scale = self.width ** -0.5
        self.conv1 = nn.Sequential(
            ResidualAttentionBlock(g_ch_enc_dec, g_ch_enc_dec),
            ResidualAttentionBlock(g_ch_enc_dec, g_ch_enc_dec),
        )
        self.conv2 = nn.Sequential(
            ResidualAttentionBlock(g_ch_enc_dec, g_ch_enc_dec),
            ResidualAttentionBlock(g_ch_enc_dec, g_ch_enc_dec),
            ResidualAttentionBlock(g_ch_enc_dec, g_ch_enc_dec),
            ResidualAttentionBlock(g_ch_enc_dec, g_ch_enc_dec),
        )
        self.latent_tokens = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens, self.width))
        self.class_embedding = nn.Parameter(scale * torch.randn(1, self.width))
        self.positional_embedding = nn.Parameter(
                scale * torch.randn(self.grid_size ** 2 + 1, self.width))
        self.latent_token_positional_embedding = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens, self.width))
        if use_long_term_feature:
            self.memory_num = memory_num
            self.memory_dim = memory_dim
            self.long_term_feature_projection = nn.Linear(memory_dim, self.width)
            self.long_term_feature_positional_embedding = nn.Parameter(
                scale * torch.randn(memory_num, self.width))


    def forward(self, x, long_term_feature, quant):
        x1, ctx_t = self.forward_part1(x, long_term_feature, quant)
        ctx = self.forward_part2(x1)
        return ctx, ctx_t

    def forward_part1(self, x, long_term_feature, quant):
        x = x.reshape(x.shape[0], x.shape[1], -1) # [B, C, L]
        x = x.permute(0, 2, 1) # shape = [*, grid ** 2, width]
        # class embeddings and positional embeddings
        x = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        x = x + self.positional_embedding.to(x.dtype) # shape = [*, grid ** 2 + 1, width]

        if self.use_long_term_feature:
            # add long term feature
            long_term_feature = self.long_term_feature_projection(long_term_feature) # (B,32,width)
            long_term_feature = long_term_feature + self.long_term_feature_positional_embedding.to(long_term_feature.dtype)
    
        latent_tokens = _expand_token(self.latent_tokens, x.shape[0]).to(x.dtype)
        latent_tokens = latent_tokens + self.latent_token_positional_embedding.to(x.dtype)
        x = torch.cat([x, latent_tokens], dim=1)
        if self.use_long_term_feature:
            x = torch.cat([long_term_feature, x], dim=1) # [num_memory+1+grid**2+num_latent_tokens]
        
        x1 = self.conv1(x)
        ctx_t = x1[:, self.memory_num+1+self.grid_size**2:].permute(0, 2, 1).unsqueeze(2).contiguous() # shape = [*, width, 1, num_latent_tokens]
        ctx_t = ctx_t * quant
        return x1, ctx_t

    def forward_part2(self, x1):
        ctx = self.conv2(x1)
        ctx = ctx[:, self.memory_num+1+self.grid_size**2:] # remove class embedding
        ctx = ctx.permute(0, 2, 1).unsqueeze(2).contiguous() # shape = [*, width, 1, num_latent_tokens]
        return ctx

class Encoder(nn.Module):
    def __init__(self, N, g_ch_src, y_tokens_num=32):
        super().__init__()

        self.enc_1 = ResidualAttentionBlock(g_ch_src, g_ch_enc_dec)
        self.enc_2 = nn.Sequential(
            ResidualAttentionBlock(g_ch_enc_dec*2, g_ch_enc_dec),
            ResidualAttentionBlock(g_ch_enc_dec, g_ch_enc_dec),
        )
        self.enc_3 = nn.Sequential(
            ResidualAttentionBlock(g_ch_enc_dec, g_ch_enc_dec),
        )
        self.down = nn.Linear(g_ch_enc_dec, N)
        # position embedding
        self.width = g_ch_src
        scale = self.width ** -0.5
        self.positional_embedding = nn.Parameter(scale * torch.randn(1,g_ch_src, 1, y_tokens_num), requires_grad=True)
    def forward(self, x, ctx, quant_step):
        '''
        x: [B, C, 1, N]
        ctx: [B, C, 1, N]
        '''
        feature = x + self.positional_embedding
        feature = self.enc_1(feature)
        feature = self.enc_2(torch.cat((feature, ctx), dim=1))
        feature = self.enc_3(feature)
        feature = feature * quant_step
        feature = feature.permute(0, 2, 3, 1)
        feature = self.down(feature)
        feature = feature.permute(0, 3, 1, 2)
        return feature

class HyperEncoder(nn.Module):
    def __init__(self, N, z_channel, y_tokens_num=32):
        super().__init__()
        self.conv = nn.Sequential(
            ResidualAttentionBlock(N, z_channel),
            ResidualAttentionUpDownsampleBlock(z_channel, z_channel, y_tokens_num//2),
            ResidualAttentionUpDownsampleBlock(z_channel, z_channel, y_tokens_num//4),
            ResidualAttentionUpDownsampleBlock(z_channel, z_channel, y_tokens_num//8),
        )
        # position embedding
        self.width = N
        scale = self.width ** -0.5
        self.positional_embedding = nn.Parameter(scale * torch.randn(1,N, 1, y_tokens_num), requires_grad=True)

    def forward(self, x):
        x = x + self.positional_embedding
        return self.conv(x)

class HyperDecoder(nn.Module):
    def __init__(self, N, z_channel, y_tokens_num=32):
        super().__init__()
        self.conv = nn.Sequential(
            ResidualAttentionUpDownsampleBlock(z_channel, z_channel, y_tokens_num//4),
            ResidualAttentionUpDownsampleBlock(z_channel, z_channel, y_tokens_num//2),
            ResidualAttentionUpDownsampleBlock(z_channel, z_channel, y_tokens_num),
            ResidualAttentionBlock(z_channel, N),
        )
        # position embedding
        self.width = z_channel
        scale = self.width ** -0.5
        self.positional_embedding = nn.Parameter(scale * torch.randn(1,z_channel, 1, y_tokens_num//8), requires_grad=True)

    def forward(self, x):
        x = x + self.positional_embedding
        return self.conv(x)

class PriorFusion(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.conv = nn.Sequential(
            ResidualAttentionBlock(N*3, N*3),
            ResidualAttentionBlock(N*3, N*3),
            ResidualAttentionBlock(N*3, N*3),
        )

    def forward(self, x):
        return self.conv(x)

class SpatialPrior(nn.Module):
    def __init__(self, N, y_tokens_num=32):
        super().__init__()
        self.y_spatial_prior = nn.Sequential(
            ResidualAttentionBlock(N , 2*N ),
            ResidualAttentionBlock(2*N , 2*N ),
        )
        self.y_spatial_prior_AR = ResidualAttentionARBlock(2*N, 2*N, y_tokens_num-1)

    def forward(self, x, idx):
        """
        x: [B, N*3, 1, 32]
        idx: index of the spatial prior
        """
        feature = self.y_spatial_prior(x)
        feature = self.y_spatial_prior_AR(feature, idx=idx)
        return feature

class Decoder(nn.Module):
    def __init__(self, N, y_tokens_num=32):
        super().__init__()

        self.dec_1 = nn.Sequential(
            ResidualAttentionBlock(N, g_ch_enc_dec),
        )
        self.dec_2 = nn.Sequential(
            ResidualAttentionBlock(g_ch_enc_dec*2, g_ch_enc_dec),
            ResidualAttentionBlock(g_ch_enc_dec, g_ch_enc_dec),
            ResidualAttentionBlock(g_ch_enc_dec, g_ch_enc_dec),
        )
        self.dec_3 = nn.Linear(g_ch_enc_dec, g_ch_enc_dec)
        # position embedding
        self.width = N
        scale = self.width ** -0.5
        self.positional_embedding = nn.Parameter(scale * torch.randn(1,N, 1, y_tokens_num), requires_grad=True)
    def forward(self, x, ctx, quant_step):
        feature = x + self.positional_embedding
        feature = self.dec_1(feature)
        feature = self.dec_2(torch.cat((feature, ctx), dim=1))
        feature = feature.permute(0, 2, 3, 1)
        feature = self.dec_3(feature)
        feature = feature.permute(0, 3, 1, 2)
        feature = feature * quant_step
        return feature

class ReconGeneration(nn.Module):
    def __init__(self, g_ch_src):
        super().__init__()
        self.conv = nn.Sequential(
            ResidualAttentionBlock(g_ch_enc_dec,     g_ch_recon),
            ResidualAttentionBlock(g_ch_recon, g_ch_recon),
            ResidualAttentionBlock(g_ch_recon, g_ch_recon),
            ResidualAttentionBlock(g_ch_recon, g_ch_recon),
        )
        self.head = nn.Linear(g_ch_recon, g_ch_src)

    def forward(self, x, quant_step):
        out = self.conv(x)
        out = out * quant_step
        out = out.permute(0, 2, 3, 1)
        out = self.head(out)
        out = out.permute(0, 3, 1, 2)
        return out

class Compressive_bottleneck(CompressionModel):
    def __init__(self, N=128, z_channel=128, y_tokens_num=32, g_ch_src=3, memory_num=0, memory_dim=512):
        super().__init__(z_channel=z_channel)
        self.feature_adaptor_i = Feature_adaptor_i() # input: B,3,256,256  out: B,g_ch_enc_dec,1,32
        self.feature_adaptor_p = nn.ModuleList([nn.Conv2d(1024, g_ch_enc_dec, 1) for _ in range(3)]) # input: B,g_ch_enc_dec,1,32  out: B,g_ch_enc_dec,1,32
        self.feature_extractor = FeatureExtractor(y_tokens_num=y_tokens_num, use_long_term_feature=(memory_num>0), memory_num=memory_num, memory_dim=memory_dim)

        self.encoder = Encoder(N, g_ch_src, y_tokens_num=y_tokens_num)

        self.hyper_encoder = HyperEncoder(N=N, z_channel=z_channel, y_tokens_num=y_tokens_num)
        self.hyper_decoder = HyperDecoder(N=N, z_channel=z_channel, y_tokens_num=y_tokens_num)

        self.temporal_prior_encoder = ResidualAttentionBlock(g_ch_enc_dec, N*2)

        self.y_prior_fusion = PriorFusion(N=N)
        self.y_spatial_prior = SpatialPrior(N=N, y_tokens_num=y_tokens_num)

        self.y_spatial_prior_reduction = nn.Linear(N * 3, N * 1)

        self.decoder = Decoder(N, y_tokens_num=y_tokens_num)
        self.recon_generation_net = ReconGeneration(g_ch_src)

        self.q_encoder = nn.Parameter(torch.ones((self.get_qp_num(), g_ch_enc_dec, 1, 1)))
        self.q_decoder = nn.Parameter(torch.ones((self.get_qp_num(), g_ch_enc_dec, 1, 1)))
        self.q_feature = nn.Parameter(torch.ones((self.get_qp_num(), g_ch_enc_dec, 1, 1)))
        self.q_recon = nn.Parameter(torch.ones((self.get_qp_num(), g_ch_recon, 1, 1)))
        self._initialize_weights()

    def apply_feature_adaptor(self, dpb, fa_idx, stack_shape):
        if dpb["ref_feature"] is None:
            return self.feature_adaptor_i(dpb["ref_frame"], stack_shape)
        if fa_idx == 3:
            ref_frame = dpb["ref_frame"]
            if not self.training:
                ref_frame = ref_frame.clamp_(0, 1)
            return self.feature_adaptor_i(ref_frame, stack_shape)
        feature = self.feature_adaptor_p[fa_idx](dpb["ref_feature"])
        feature = rearrange(feature, 'B C (nH h) (nW w) -> (B nH nW) C h w', h=16, w=16, nH=stack_shape[0], nW=stack_shape[1]).contiguous()
        return feature

    def res_prior_param_decoder(self, z_hat, ctx_t):
        hierarchical_params = self.hyper_decoder(z_hat)
        temporal_params = self.temporal_prior_encoder(ctx_t)
        params = self.y_prior_fusion(torch.cat((hierarchical_params, temporal_params), dim=1)) # [B, N*3, 1, 32]
        return params

    def get_recon_and_feature(self, y_hat, ctx, q_decoder, q_recon):
        feature = self.decoder(y_hat, ctx, q_decoder)
        x_hat = self.recon_generation_net(feature, q_recon)
        return x_hat, feature

    def forward_one_frame(self, x, dpb, qp, H, W, fa_idx, long_term_feature, stack_shape):
        assert x.shape[2]==1, "not fake 2d input, please check the input shape"
        pixel_num = H * W
        pixel_num_patch = 256 * 256
        nH, nW = stack_shape
        patch_num = nH * nW
        device = x.device
        qp = [x for x in qp for _ in range(patch_num)]
        index = self.get_index_tensor(qp, device)
        q_encoder = torch.index_select(self.q_encoder, 0, index)
        q_decoder = torch.index_select(self.q_decoder, 0, index)
        q_feature = torch.index_select(self.q_feature, 0, index)
        q_recon = torch.index_select(self.q_recon, 0, index)

        feature = self.apply_feature_adaptor(dpb, fa_idx, stack_shape)
        ctx, ctx_t = self.feature_extractor(feature, long_term_feature, q_feature)

        y = self.encoder(x, ctx, q_encoder)
        z = self.hyper_encoder(y)
        z_hat = self.quant(z)

        params = self.res_prior_param_decoder(z_hat, ctx_t)

        y_res, y_q, y_hat, scales_hat = self.forward_AR_prior(
            y, params, self.y_spatial_prior_reduction, self.y_spatial_prior)

        x_hat, feature = self.get_recon_and_feature(y_hat, ctx, q_decoder, q_recon)

        if self.training:
            y_for_bit = self.add_noise(y_res)
            z_for_bit = self.add_noise(z)
        else:
            y_for_bit = y_q
            z_for_bit = z_hat
        bits_y = self.get_y_gaussian_bits(y_for_bit, scales_hat)
        bits_z = self.get_z_bits(z_for_bit, self.bit_estimator_z, index)

        bpp_y = torch.sum(bits_y, dim=(1, 2, 3)) / pixel_num_patch
        bpp_z = torch.sum(bits_z, dim=(1, 2, 3)) / pixel_num_patch
        bpp_y = rearrange(bpp_y, '(B nH nW) -> B nH nW', nH=nH, nW=nW).contiguous().mean(dim=(1, 2))
        bpp_z = rearrange(bpp_z, '(B nH nW) -> B nH nW', nH=nH, nW=nW).contiguous().mean(dim=(1, 2))
        bpp = bpp_y + bpp_z
        bit = torch.sum(bpp) * pixel_num
        bit_y = torch.sum(bpp_y) * pixel_num
        bit_z = torch.sum(bpp_z) * pixel_num

        #get bppy_q bppz_q
        bits_y_q = self.get_y_gaussian_bits(y_q, scales_hat)
        bits_z_q = self.get_z_bits(z_hat, self.bit_estimator_z, index)
        bppy_q = torch.sum(bits_y_q, dim=(1, 2, 3)) / pixel_num_patch
        bpp_z_q = torch.sum(bits_z_q, dim=(1, 2, 3)) / pixel_num_patch
        bppy_q = rearrange(bppy_q, '(B nH nW) -> B nH nW', nH=nH, nW=nW).contiguous().mean(dim=(1, 2))
        bpp_z_q = rearrange(bpp_z_q, '(B nH nW) -> B nH nW', nH=nH, nW=nW).contiguous().mean(dim=(1, 2))
        bpp_q = bppy_q + bpp_z_q

        return x_hat, {"bpp_y": bpp_y,
                "bpp_z": bpp_z,
                "bpp": bpp,
                "bpp_q": bpp_q,
                "bpp_q_y": bppy_q,
                "bpp_q_z": bpp_z_q,
                "bit": bit,
                "bit_y": bit_y,
                "bit_z": bit_z,
                }

    def forward(self, x, dpb, qp, H, W, fa_idx=0, long_term_feature=None, stack_shape=None):
        return self.forward_one_frame(x, dpb, qp, H, W, fa_idx, long_term_feature, stack_shape)

if __name__ == "__main__":
    model = Compressive_bottleneck()
    #print model size
    params = sum(p.numel() for p in model.parameters())
    print("Model size: {:.2f}M".format(params / 1e6))
    x = torch.randn(5, 16, 1, 32)
    dpb = {
        'ref_frame': torch.randn(5, 3, 256, 256),
        'ref_feature': torch.randn(5, 256, 1, 32),
    }
    qp= 0
    H, W = 256, 256
    fa_idx = 0
    x_hat, feature, info = model.forward(x, dpb, qp, H, W, fa_idx)
    print("x_hat shape:", x_hat.shape)
    print("feature shape:", feature.shape)
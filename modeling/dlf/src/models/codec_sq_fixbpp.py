import torch
import pytorch_lightning as pl
from torch import nn
from einops import rearrange
import torch.utils.checkpoint as cp

from modeling.dlf.src.taming.models.vqgan import VQModel as VQGAN
from modeling.dlf.src.taming.models.vqgan_wo_attn import VQModel as VQGAN_wo_Attn
from modeling.dlf.src.blocks.swin_transformer import SwinBlock, Rearrange, rearrange
from modeling.dlf.src.blocks.conv_blocks import ConvNeXtBlock
from modeling.dlf.src.titok.quantizer import VectorQuantizer
from modeling.dlf.src.titok.titok import TiTokEncoder, TiTokDecoder

from modeling.dlf.src.models.utils import _expand_token
from .cross_blocks import Interactive_crossAttn_type4
from .sq_bottleneck import Compressive_bottleneck_varbpp_type2 as Compressive_bottleneck

from pytorch_lightning.utilities import rank_zero_only


@rank_zero_only
def print_rank0(information):
    print(information)


def get_swin(feat_width, num_layers, mlp_ratio=4.0, window_size=16, auto_bchw=True, inverse_shifted=False):
    assert feat_width % 64 == 0
    m_list = []
    if auto_bchw:
        m_list.append(Rearrange('b c h w -> b h w c'))
    for i in range(num_layers):
        shifted = not(bool(i % 2)) if inverse_shifted else bool(i % 2)
        relative_pos_embedding = False if inverse_shifted else bool(i==0)
        m_list.append(SwinBlock(feat_width, feat_width//64, 64, int(feat_width * mlp_ratio),
                                shifted=shifted, window_size=window_size, relative_pos_embedding=relative_pos_embedding))
    if auto_bchw:
        m_list.append(Rearrange('b h w c -> b c h w'))
    return nn.Sequential(*m_list)


class HybridEncoder(TiTokEncoder):
    def __init__(self, config, insert_pos, feat_width, num_attns=2, save_mem=False):
        super().__init__(config)
        # disable grad in TiTok's params
        self.requires_grad_(False)
        self.save_mem = save_mem
        if save_mem:
            print(f"[Hybrid Encoder] Enable checkpoint for memory saving!")

        # additional swin config
        self.feat_width = feat_width
        self.insert_pos = insert_pos
        self.cross_block_num = len(self.insert_pos)
        print(f"[Hybrid Encoder] current feat width: {feat_width}")
        print(f"[Hybrid Encoder] feat insert position: {insert_pos}")

        # pixel emb proj
        self.pix_emb_proj = nn.Conv2d(self.width, feat_width, kernel_size=1)

        inter_blocks = []
        feat_blocks = []
        for i in range(self.num_layers):
            if i in self.insert_pos:
                inter_blocks.append(Interactive_crossAttn_type4(self.width, feat_width, num_attns=num_attns,
                                                                titok_patch_size=self.patch_size,
                                                                feat_patch_size=self.patch_size,
                                                                extra_titok_tokens=self.num_latent_tokens+1))
                feat_blocks.append(nn.Sequential(
                    get_swin(feat_width, num_layers=2),
                    ConvNeXtBlock(feat_width, feat_width, mlp_ratio=2.0, kernel_size=5),
                    ConvNeXtBlock(feat_width, feat_width, mlp_ratio=2.0, kernel_size=5)
                ))
            else:
                inter_blocks.append(nn.Identity())
                feat_blocks.append(nn.Identity())
        self.inter_blocks = nn.ModuleList(inter_blocks)
        self.feat_blocks = nn.ModuleList(feat_blocks)
        print(f"[Hybrid Encoder] num attn blocks per cross: {num_attns}")

        self.feat_in = get_swin(feat_width, num_layers=4)
        self.feat_out = nn.Sequential(
            get_swin(feat_width, num_layers=2),
            nn.Conv2d(feat_width, feat_width, kernel_size=2, stride=2),
            Rearrange('b c h w -> b h w c'),
            nn.LayerNorm(feat_width),
            nn.Linear(feat_width, feat_width),
            Rearrange('b h w c -> b c h w'),
        )


    def set_to_zero(self):
        for m in self.inter_blocks:
            if hasattr(m, 'set_to_zero'):
                m.set_to_zero()


    def get_trainable_params(self, tune_titok=False):
        if tune_titok:
            p_list = list(self.parameters())
            print("[Hyper encoder] Tuning TiTok encoder's transformer!")
        else:
            p_list = list(self.pix_emb_proj.parameters()) +\
                    list(self.inter_blocks.parameters()) + list(self.feat_blocks.parameters()) +\
                    list(self.feat_in.parameters()) + list(self.feat_out.parameters())
        for p in p_list:
            p.requires_grad_(True)
        return p_list


    def forward(self, pixel_values, latent_tokens):
        # embedding
        x_emb = self.patch_embed(pixel_values)     # B C H W
        feat_emb = self.pix_emb_proj(x_emb)
        # print('pixel_values shape:', pixel_values.shape) # [B, 3, 512,768]
        # print('x_emb shape:', x_emb.shape) # [B, 1024, 32, 48]
        # print('feat_emb shape:', feat_emb.shape) # [B, 768, 32, 48]

        # make stack, to (B nH nW) C 16 16
        nH, nW = x_emb.shape[2] // 16, x_emb.shape[3] // 16
        stack_shape = (nH, nW)
        x = rearrange(x_emb, 'B C (nH h) (nW w) -> (B nH nW) C h w', h=16, w=16, nH=nH, nW=nW).contiguous()
        # print('x shape:', x.shape) # [B nH nW, 1024, 16, 16]

        # reshape to (B nH nW) 256 width, e.g. NLD
        B = x.shape[0]
        x = x.reshape(x.shape[0], x.shape[1], -1) # [B nH nW, 1024, 256]
        x = x.permute(0, 2, 1) # # [B nH nW, 256, 1024]

        # class embeddings and positional embeddings
        x = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        # print('x after class embedding shape:', x.shape) # [B nH nW, 1+256, 1024]
        x = x + self.positional_embedding.to(x.dtype) # shape = [*, grid ** 2 + 1, width]

        # concat latent tokens for quant in the sequence tail
        latent_tokens = _expand_token(latent_tokens, x.shape[0]).to(x.dtype)
        # print('latent_tokens shape:', latent_tokens.shape) # [B nH nW, 32, 1024]
        latent_tokens = latent_tokens + self.latent_token_positional_embedding.to(x.dtype)
        x = torch.cat([x, latent_tokens], dim=1)
        # print('x after latent token shape:', x.shape) # [B nH nW, 1+256 + 32, 1024]

        # feat init
        if self.save_mem:
            feat = cp.checkpoint(self.feat_in, feat_emb)
        else:
            feat = self.feat_in(feat_emb)
        # print('feat shape:', feat.shape) # [B, 768, 32, 48]

        # transform
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # print('num_layers:', self.num_layers) # 24
        for i in range(self.num_layers):
            if self.save_mem:
                # 1. transform in TiTok
                x = cp.checkpoint(self.transformer[i], x)

                # 2. extract information from titok
                if i in self.insert_pos:
                    feat, x = cp.checkpoint(self.inter_blocks[i], feat, x, stack_shape)
                    feat = cp.checkpoint(self.feat_blocks[i], feat)
            else:
                # 1. transform in TiTok
                x = self.transformer[i](x)

                # 2. extract information from titok
                if i in self.insert_pos:
                    feat, x = self.inter_blocks[i](feat, x, stack_shape)
                    feat = self.feat_blocks[i](feat)

        x = x.permute(1, 0, 2)  # LND -> NLD
        # print('x after transformer shape:', x.shape) # [B nH nW, 1+256 + 32, 1024]
        # print('feat after transformer shape:', feat.shape) # [B, 768, 32, 48]

        # output of titok
        latent_tokens = x[:, 1+self.grid_size**2:]
        latent_tokens = self.ln_post(latent_tokens)

        # fake 2D shape of titok
        latent_tokens = latent_tokens.reshape(B, self.width, self.num_latent_tokens, 1)
        latent_tokens = self.conv_out(latent_tokens)
        # print('latent_tokens after conv_out shape:', latent_tokens.shape) # [B nH nW, 12, 32, 1]
        latent_tokens = latent_tokens.reshape(B, self.token_size, 1, self.num_latent_tokens)
        # print('latent_tokens after reshape shape:', latent_tokens.shape) # [B nH nW, 12, 1, 32]

        if self.save_mem:
            feat = cp.checkpoint(self.feat_out, feat)
        else:
            feat = self.feat_out(feat)
        # print('feat after feat_out shape:', feat.shape) # [B, 768, 16, 24]
        return latent_tokens, feat, stack_shape


class HybridDecoder(TiTokDecoder):
    def __init__(self, config, insert_pos, feat_width, num_attns=2, save_mem=False):
        super().__init__(config)
        # disable grad in TiTok's params
        self.requires_grad_(False)
        self.save_mem = save_mem
        if save_mem:
            print(f"[Hybrid Decoder] Enable checkpoint for memory saving!")

        self.ffn = nn.Identity()    # remove original ffn

        # additional swin config
        self.insert_pos = insert_pos
        self.cross_block_num = len(self.insert_pos)
        print(f"[Hybrid Decoder] current feat width: {feat_width}")
        print(f"[Hybrid Decoder] feat insert position: {insert_pos}")

        self.init_feat_up = nn.Sequential(
            nn.Conv2d(feat_width, feat_width*4, kernel_size=1),
            nn.PixelShuffle(2),
            get_swin(feat_width, num_layers=4)
        )

        inter_blocks = []
        feat_blocks = []
        for i in range(self.num_layers):
            if i in self.insert_pos:
                inter_blocks.append(Interactive_crossAttn_type4(self.width, feat_width, num_attns=num_attns,
                                                                titok_patch_size=self.patch_size,
                                                                feat_patch_size=self.patch_size,
                                                                extra_titok_tokens=self.num_latent_tokens+1))
                feat_blocks.append(nn.Sequential(
                    get_swin(feat_width, num_layers=2),
                    ConvNeXtBlock(feat_width, feat_width, mlp_ratio=2.0, kernel_size=5),
                    ConvNeXtBlock(feat_width, feat_width, mlp_ratio=2.0, kernel_size=5)
                ))
            else:
                inter_blocks.append(nn.Identity())
                feat_blocks.append(nn.Identity())
        self.inter_blocks = nn.ModuleList(inter_blocks)
        self.feat_blocks = nn.ModuleList(feat_blocks)
        print(f"[Hybrid Decoder] num attn blocks per cross: {num_attns}")


    def set_to_zero(self):
        for m in self.inter_blocks:
            if hasattr(m, 'set_to_zero'):
                m.set_to_zero()


    def get_trainable_params(self, tune_titok=False):
        if tune_titok:
            p_list = list(self.parameters())
            print("[Hyper decoder] Tuning TiTok decoder's transformer!")
        else:
            p_list = list(self.init_feat_up.parameters()) + list(self.inter_blocks.parameters()) +\
                    list(self.feat_blocks.parameters())
        for p in p_list:
            p.requires_grad_(True)
        return p_list


    def forward(self, z_quantized, h_quantized, stack_shape, **kw_args):
        # get shape
        nH, nW = stack_shape
        N, C, H, W = z_quantized.shape
        assert H == 1 and W == self.num_latent_tokens, f"{H}, {W}, {self.num_latent_tokens}"
        x = z_quantized.reshape(N, C*H, W).permute(0, 2, 1) # NLD

        # decoder emb in titok
        x = self.decoder_embed(x)
        batchsize, seq_len, _ = x.shape

        # concat masked tokens
        mask_tokens = self.mask_token.repeat(batchsize, self.grid_size**2, 1).to(x.dtype)
        mask_tokens = torch.cat([_expand_token(self.class_embedding, mask_tokens.shape[0]).to(mask_tokens.dtype),
                                    mask_tokens], dim=1)
        mask_tokens = mask_tokens + self.positional_embedding.to(mask_tokens.dtype)
        x = x + self.latent_token_positional_embedding[:seq_len]
        x = torch.cat([mask_tokens, x], dim=1)
        # print('x after concat shape:', x.shape) # [B nH nW, 1+256 + 32, 1024]

        # for res feat branch
        if self.save_mem:
            feat = cp.checkpoint(self.init_feat_up, h_quantized)
        else:
            feat = self.init_feat_up(h_quantized)
        # print('feat after init_feat_up shape:', feat.shape) # [B, 768, 32, 48]

        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        for i in range(self.num_layers):
            if self.save_mem:
                # 1. titok transformer
                x = cp.checkpoint(self.transformer[i], x)

                # 2. extract information from titok
                if i in self.insert_pos:
                    feat, x = cp.checkpoint(self.inter_blocks[i], feat, x, stack_shape)
                    feat = cp.checkpoint(self.feat_blocks[i], feat)
            else:
                # 1. titok transformer
                x = self.transformer[i](x)

                # 2. extract information from titok
                if i in self.insert_pos:
                    feat, x = self.inter_blocks[i](feat, x, stack_shape)
                    feat = self.feat_blocks[i](feat)

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = x[:, 1:1+self.grid_size**2] # remove cls embed
        x = self.ln_post(x)
        # print('x after transformer shape:', x.shape) # [B nH nW, 256, 1024]
        # N L D -> N D H W
        x = x.permute(0, 2, 1).reshape(batchsize, self.width, self.grid_size, self.grid_size)
        # print('x after permute shape:', x.shape) # [B nH nW, 1024, 16, 16]
        x = rearrange(x, '(B nH nW) C h w -> B C (nH h) (nW w)',
                        nH=nH, nW=nW, h=self.patch_size, w=self.patch_size).contiguous()
        # print('x after rearrange shape:', x.shape) # [B, 1024, 32, 48]
        # print('feat after rearrange shape:', feat.shape) # [B, 768, 32, 48]
        return x, feat


class Hybrid_Codec(nn.Module):
    def __init__(self, config, in_pos_enc, in_pos_dec, feat_width, quant_dim, n_enc_attn, save_mem=False):
        super().__init__()
        self.config = config
        self.encoder = HybridEncoder(config, in_pos_enc, feat_width, num_attns=n_enc_attn, save_mem=save_mem)
        self.decoder = HybridDecoder(config, in_pos_dec, feat_width, num_attns=n_enc_attn, save_mem=save_mem)

        # TiTok's default part
        self.num_latent_tokens = config.model.vq_model.num_latent_tokens
        scale = self.encoder.width ** -0.5
        self.latent_tokens = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens, self.encoder.width))
        self.latent_tokens.requires_grad_(False)

        self.apply(self._init_weights)

        self.quantize = VectorQuantizer(
            codebook_size=config.model.vq_model.codebook_size,
            token_size=config.model.vq_model.token_size,
            commitment_cost=config.model.vq_model.commitment_cost,
            use_l2_norm=config.model.vq_model.use_l2_norm,)
        self.quantize.requires_grad_(False)

        # additional part
        self.quantize_feat = Compressive_bottleneck(feat_width, quant_dim, bpp_num=1) # feat width, quant_dim = 768,64
        self.quantize_feat.requires_grad_(True)

        # set zero conv
        self.encoder.set_to_zero()
        self.decoder.set_to_zero()


    def _init_weights(self, module):
        """ Initialize the weights.
            :param:
                module -> torch.nn.Module: module to initialize
        """
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.05)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.05)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


    def init_from_sd(self, sd):
        missing_keys, unexpected_keys = self.load_state_dict(sd, strict=False)
        print(f"[Hybrid Codec] init params from statedict...")
        print(f"missing_keys: {missing_keys}")
        print(f"unexpected_keys: {unexpected_keys}")


    def get_trainable_params(self, tune_titok=False):
        if tune_titok:
            p_list = list(self.parameters())
            print("[Codec] tunning titok's parameters!")
        else:
            p_list = self.encoder.get_trainable_params(tune_titok=False) +\
                     self.decoder.get_trainable_params(tune_titok=False) +\
                    list(self.quantize_feat.parameters())
        for p in p_list:
            p.requires_grad_(True)
        return p_list


    def encode(self, x):
        z, h, stack_shape = self.encoder(pixel_values=x, latent_tokens=self.latent_tokens)
        z_quantized, z_result_dict = self.quantize(z)
        h_quantized, h_result_dict = self.quantize_feat.forward(h, (x.shape[2], x.shape[3]), q_idx=0)
        return {
            'z_quantized': z_quantized, 'z_result_dict': z_result_dict,
            'h_quantized': h_quantized, 'h_result_dict': h_result_dict,
            'stack_shape': stack_shape
        }


    def decode(self, z_quantized, h_quantized, stack_shape, **kw_args):
        titok_hat, feat_hat = self.decoder(z_quantized, h_quantized, stack_shape)
        return titok_hat, feat_hat


    def forward(self, x):
        out = self.encode(x)
        titok_hat, feat_hat = self.decode(**out)
        out['titok_hat'] = titok_hat
        out['feat_hat'] = feat_hat
        return out


class FeatMerge(nn.Module):
    def __init__(self, titok_width, feat_width, n_embed=256, save_mem=False):
        super().__init__()
        self.save_mem = save_mem
        if save_mem:
            print(f"[FeatMerge] Enable checkpoint for memory saving!")

        self.feat_in = nn.Sequential(
            Rearrange('b c h w -> b h w c'),
            get_swin(feat_width, num_layers=2, auto_bchw=False)
        )
        self.titok_in = nn.Sequential(
            Rearrange('b c h w -> b h w c'),
            get_swin(titok_width, num_layers=2, auto_bchw=False)
        )
        inner_width = 1024
        self.merge = nn.Sequential(
            nn.Linear(titok_width + feat_width, titok_width*2),
            nn.LayerNorm(titok_width*2),
            nn.SiLU(),
            nn.Linear(titok_width*2, inner_width),
            get_swin(inner_width, num_layers=4, auto_bchw=False)
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(inner_width),
            nn.Linear(inner_width, 2*inner_width),
            nn.Tanh(),
            nn.Linear(2*inner_width, n_embed),
            Rearrange('b h w c -> b c h w')
        )


    def forward(self, titok, feat):
        # proj
        if self.save_mem:
            titok = cp.checkpoint(self.titok_in, titok)
            feat = cp.checkpoint(self.feat_in, feat)
            h = cp.checkpoint(self.merge, torch.cat([titok, feat], dim=-1))
            logits = cp.checkpoint(self.ffn, h)
        else:
            titok = self.titok_in(titok)
            feat = self.feat_in(feat)
            h = self.merge(torch.cat([titok, feat], dim=-1))
            logits = self.ffn(h)
        return logits


class Codec(pl.LightningModule):
    def __init__(self, embed_dim, feat_dim, in_pos_enc, in_pos_dec, n_attn,
                 config, vqganconfig, no_attn_vqgan=False, save_mem=False):
        super().__init__()
        # lightning configs
        self.image_key = "image"
        self.automatic_optimization = False

        # load titok and hybrid model
        self.config = config

        self.hybrid_codec = Hybrid_Codec(config, in_pos_enc, in_pos_dec, feat_dim, embed_dim, n_attn, save_mem=save_mem)


        # vqgan
        self.vqgan = VQGAN_wo_Attn(**vqganconfig) if no_attn_vqgan else VQGAN(**vqganconfig)
        self.vqgan.quantize.sane_index_shape = True
        self.vqgan.encoder.requires_grad_(False)
        self.vqgan.quant_conv.requires_grad_(False)

        # prior fusion
        self.prior_fusion = FeatMerge(self.hybrid_codec.decoder.width, feat_dim, self.vqgan.quantize.n_e)

        self.is_set_torchac = False

        print("[INFO] ControlTiTok codec v1-7 SQ fixed bpp, f16 VQGAN, FINAL model")

    def decode_to_latent(self, titok_hat, feat_hat, **kw_args):
        # print('titok_hat shape:', titok_hat.shape)  # [B, 1024, H/16, W/16]
        # print('feat_hat shape:', feat_hat.shape)    # [B, 768, H/16, W/16]
        logits = self.prior_fusion(titok_hat, feat_hat)
        # print('logits shape:', logits.shape)  # [B, 256, H/16, W/16]
        quantized_latent = torch.einsum(
            'nchw,cd->ndhw', logits.softmax(1),
            self.vqgan.quantize.embedding.weight)
        # print('quantized_latent shape:', quantized_latent.shape)  # [B, 256, H/16, W/16]
        return quantized_latent, logits


    def decode_to_image(self, quantized_latent):
        quantized_latent = self.vqgan.post_quant_conv(quantized_latent)
        x_hat = self.vqgan.decoder(quantized_latent)
        return x_hat


    def forward_one_frame(self, x,return_bpp=False):
        enc_info = self.hybrid_codec(x)   # send x with [0, 1] to titok
        vqgan_latent, logits = self.decode_to_latent(titok_hat=enc_info['titok_hat'],
                                                     feat_hat=enc_info['feat_hat'])

        x_hat = self.decode_to_image(vqgan_latent)
        x_hat=x_hat.clamp_(-1.0, 1.0)
        #to 0-1
        x_hat = (x_hat + 1.0) / 2.0
        if return_bpp:
            z_bpp= 32*12.0/256/256
            h_bpp= enc_info['h_result_dict']['bpp']
            bpp = z_bpp + h_bpp
            return x_hat, bpp

        return x_hat

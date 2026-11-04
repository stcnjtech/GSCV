import torch
from torch import nn
from einops import rearrange
from omegaconf import OmegaConf

from titok.titok import TiTok, TiTokEncoder, TiTokDecoder, VectorQuantizer, Pixel_Decoder, Pixel_Quantizer

def _expand_token(token, batch_size: int):
    return token.unsqueeze(0).expand(batch_size, -1, -1)

class TiTokEncoder_hacked(TiTokEncoder):
    def __init__(self, config, out_pos: list):
        super().__init__(config)
        self.out_pos = out_pos
    
    @torch.no_grad()
    def forward(self, pixel_values, latent_tokens):
        x = pixel_values
        x = self.patch_embed(x)

        # make stack, to (B nH nW) C 16 16
        nH, nW = x.shape[2] // 16, x.shape[3] // 16
        x = rearrange(x, 'B C (nH h) (nW w) -> (B nH nW) C h w', 
                      h=16, w=16, nH=nH, nW=nW)
        stacked_batch_size = x.shape[0]

        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1) # shape = [*, grid ** 2, width]
        # class embeddings and positional embeddings
        x = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        x = x + self.positional_embedding.to(x.dtype) # shape = [*, grid ** 2 + 1, width]

        latent_tokens = _expand_token(latent_tokens, x.shape[0]).to(x.dtype)
        latent_tokens = latent_tokens + self.latent_token_positional_embedding.to(x.dtype)
        x = torch.cat([x, latent_tokens], dim=1)

        # output feature container
        out_feat = []

        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        for i in range(self.num_layers):
            x = self.transformer[i](x)
            if i in self.out_pos:
                out_feat.append(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        
        latent_tokens = x[:, 1+self.grid_size**2:]
        latent_tokens = self.ln_post(latent_tokens)
        # fake 2D shape
        latent_tokens = latent_tokens.reshape(stacked_batch_size, self.width, self.num_latent_tokens, 1)
        latent_tokens = self.conv_out(latent_tokens)
        latent_tokens = latent_tokens.reshape(stacked_batch_size, self.token_size, 1, self.num_latent_tokens)
        return latent_tokens, out_feat, (nH, nW)


class TiTokDecoder_hacked(TiTokDecoder):
    def __init__(self, config, in_pos: list):
        super().__init__(config)
        self.in_pos = in_pos

    def forward(self, z_quantized, cond:dict, stack_shape):
        nH, nW = stack_shape
        N, C, H, W = z_quantized.shape
        assert H == 1 and W == self.num_latent_tokens, f"{H}, {W}, {self.num_latent_tokens}"
        x = z_quantized.reshape(N, C*H, W).permute(0, 2, 1) # NLD
        x = self.decoder_embed(x)

        batchsize, seq_len, _ = x.shape

        mask_tokens = self.mask_token.repeat(batchsize, self.grid_size**2, 1).to(x.dtype)
        mask_tokens = torch.cat([_expand_token(self.class_embedding, mask_tokens.shape[0]).to(mask_tokens.dtype),
                                    mask_tokens], dim=1)
        mask_tokens = mask_tokens + self.positional_embedding.to(mask_tokens.dtype)
        x = x + self.latent_token_positional_embedding[:seq_len]
        x = torch.cat([mask_tokens, x], dim=1)
        
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        for i in range(self.num_layers):
            x = self.transformer[i](x)

            # add condition
            if i in self.in_pos:
                x += cond[i]

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = x[:, 1:1+self.grid_size**2] # remove cls embed
        x = self.ln_post(x)
        # N L D -> N D H W
        x = x.permute(0, 2, 1).reshape(batchsize, self.width, self.grid_size, self.grid_size)
        x = rearrange(x, '(B nH nW) C h w -> B C (nH h) (nW w)', 
                        nH=nH, nW=nW, h=16, w=16)
        x = self.ffn(x.contiguous())
        x = self.conv_out(x)
        return x


class TiTok_hacked(nn.Module):
    def __init__(self, config, enc_out_pos, dec_in_pos):
        super().__init__()
        self.config = config
        self.encoder = TiTokEncoder_hacked(config, enc_out_pos)
        self.decoder = TiTokDecoder_hacked(config, dec_in_pos)
        self.encoder.requires_grad_(False)
        self.decoder.requires_grad_(False)

        # TiTok's default
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

        self.pixel_quantize = Pixel_Quantizer(
            num_embeddings=1024, embedding_dim=256, commitment_cost=0.25)
        self.pixel_decoder = Pixel_Decoder(OmegaConf.create(
            {"channel_mult": [1, 1, 2, 2, 4],
             "num_resolutions": 5,
             "dropout": 0.0,
             "hidden_channels": 128,
             "num_channels": 3,
             "num_res_blocks": 2,
             "resolution": 256,
             "z_channels": 256}))

    def _init_weights(self, module):
        """ Initialize the weights.
            :param:
                module -> torch.nn.Module: module to initialize
        """
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.2)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.2)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def init_from_sd(self, sd):
        missing_keys, unexpected_keys = self.load_state_dict(sd, strict=False)
        print(f"[In Hybrid Codec] init params from statedict...")
        print(f"missing_keys: {missing_keys}")
        print(f"unexpected_keys: {unexpected_keys}")

    def encode(self, x):
        z, out_feat, stack_shape = self.encoder(pixel_values=x, latent_tokens=self.latent_tokens)
        z_quantized, result_dict = self.quantize(z)
        return z_quantized, result_dict, out_feat, stack_shape
    
    def decode(self, z_quantized, cond, shape):
        decoded_latent = self.decoder(z_quantized, cond, shape)
        return decoded_latent

    def forward(self):
        pass
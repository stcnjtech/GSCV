"""Building blocks for TiTok.

Copyright (2024) Bytedance Ltd. and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

    http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License. 

Reference: 
    https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/transformer.py
    https://github.com/baofff/U-ViT/blob/main/libs/timm.py
"""


import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from collections import OrderedDict
from einops.layers.torch import Rearrange
from .global_attention import global_attention_Block
from einops import rearrange


def modulate(x, shift, scale):
    return x * (1 + scale) + shift

class ResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model,
            n_head,
            mlp_ratio = 4.0,
            act_layer = nn.GELU,
            norm_layer = nn.LayerNorm
        ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.mlp_ratio = mlp_ratio
        # optionally we can disable the FFN
        if mlp_ratio > 0:
            self.ln_2 = norm_layer(d_model)
            mlp_width = int(d_model * mlp_ratio)
            self.mlp = nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(d_model, mlp_width)),
                ("gelu", act_layer()),
                ("c_proj", nn.Linear(mlp_width, d_model))
            ]))

    def attention(
            self,
            x: torch.Tensor
    ):
        return self.attn(x, x, x, need_weights=False)[0]

    def forward(
            self,
            x: torch.Tensor,
    ):
        attn_output = self.attention(x=self.ln_1(x))
        x = x + attn_output
        if self.mlp_ratio > 0:
            x = x + self.mlp(self.ln_2(x))
        return x

def _expand_token(token, batch_size: int):
    return token.unsqueeze(0).expand(batch_size, -1, -1)

class GVC1DEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_size = config.dataset.preprocessing.crop_size 
        self.patch_size = config.model.vq_model.vit_enc_patch_size
        self.grid_size = self.image_size // self.patch_size
        self.model_size = config.model.vq_model.vit_enc_model_size
        self.num_latent_tokens = config.model.vq_model.num_latent_tokens
        self.token_size = config.model.vq_model.token_size

        if config.model.vq_model.get("quantize_mode", "vq") == "vae":
            self.token_size = self.token_size * 2 # needs to split into mean and std

        self.width = {
                "small": 512,
                "base": 768,
                "large": 1024,
            }[self.model_size]
        self.num_layers = {
                "small": 8,
                "base": 12,
                "large": 24,
            }[self.model_size]
        self.num_heads = {
                "small": 8,
                "base": 12,
                "large": 16,
            }[self.model_size]
        
        self.patch_embed = nn.Conv2d(
            in_channels=3, out_channels=self.width,
              kernel_size=self.patch_size, stride=self.patch_size, bias=True)
        
        scale = self.width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(1, self.width))
        self.positional_embedding = nn.Parameter(
                scale * torch.randn(self.grid_size ** 2 + 1, self.width))
        self.latent_token_positional_embedding = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens, self.width))
        self.ln_pre = nn.LayerNorm(self.width)
        self.transformer = nn.ModuleList()
        for i in range(self.num_layers):
            self.transformer.append(ResidualAttentionBlock(
                self.width, self.num_heads, mlp_ratio=4.0
            ))
        self.ln_post = nn.LayerNorm(self.width)
        self.conv_out = nn.Conv2d(self.width, self.token_size, kernel_size=1, bias=True)

        # conditional feature (I frame)
        scale = self.width ** -0.5
        self.cond_patch_embed = nn.Conv2d(
            in_channels=3, out_channels=self.width,
            kernel_size=self.patch_size, stride=self.patch_size, bias=True)
        self.cond_positional_embedding = nn.Parameter(
                scale * torch.randn(self.grid_size ** 2, self.width))
        # P frame emb
        if config.model.hyperprior.get('use_hyperprior', False):
            self.adapter_p = nn.ModuleList([nn.Conv2d(1024, self.width, 1) for _ in range(3)])
            self.cond_positional_embedding_p = nn.Parameter(
                    scale * torch.randn(self.grid_size ** 2, self.width))
        

        # long term feature
        if config.model.memory.get('use_memory', False):
            memory_dim = config.model.memory.memory_dim
            self.long_term_feature_adapter = nn.Linear(memory_dim, self.width)
            self.long_term_feature_positional_embedding = nn.Parameter(
                scale * torch.randn(1, config.model.memory.memory_num, self.width))
        
        self.global_attention_block_idx = []
        if config.model.global_atten.get('use_global_atten', False):
            # global attention
            self.global_attention_block_idx = config.model.global_atten.encoder_global_attention_block_idx
            self.global_attention_blocks = nn.ModuleList()
            use_memory=config.model.memory.get('use_memory', False)
            memory_token_num=config.model.memory.get('memory_num', 0)
            for i in self.global_attention_block_idx:
                self.global_attention_blocks.append(global_attention_Block(hidden_size=self.width,num_heads=self.num_heads,use_memory=use_memory,
                memory_token_num=memory_token_num,latent_token_num=self.num_latent_tokens))
        
        training_module=config.model.get('training_module', None)
        if training_module == 'frame_condition':
            for param in self.parameters():
                if param is not self.cond_patch_embed.weight and \
                   param is not self.cond_patch_embed.bias and \
                   param is not self.cond_positional_embedding:
                    param.requires_grad = False
        elif training_module == 'codec_module':
            # only training adapter_p and cond_positional_embedding_p
            for name, param in self.named_parameters():
                if not name.startswith('adapter_p') and not name.startswith('cond_positional_embedding_p'):
                    param.requires_grad = False
        elif training_module == 'global_atten_module':
            # only training global_attention_blocks
            self.requires_grad_(False)
            for name, param in self.named_parameters():
                if name.startswith('global_attention_blocks'):
                    param.requires_grad = True
        elif training_module == 'memory_module':
            for name, param in self.named_parameters():
                if not name.startswith('long_term_feature_adapter') and not name.startswith('long_term_feature_positional_embedding'):
                    param.requires_grad = False
    
    def apply_feature_adaptor(self, dpb, fa_idx, stack_shape):
        nH, nW = stack_shape
        if fa_idx == 3:
            cond_x = dpb["ref_frame"]
            if not self.training:
                cond_x = cond_x.clamp_(0, 1)
            cond_x = self.cond_patch_embed(cond_x)
            cond_x =  rearrange(cond_x, 'B C (nH h) (nW w) -> (B nH nW) C h w', h=16, w=16, nH=nH, nW=nW).contiguous()
            cond_x = cond_x.reshape(cond_x.shape[0], cond_x.shape[1], -1)
            cond_x = cond_x.permute(0, 2, 1) # [B,L,D]
            cond_x = cond_x + self.cond_positional_embedding.to(cond_x.dtype)
            return cond_x
        if dpb["ref_feature"] is None:
            # add conditional latent tokens
            cond_x = dpb["ref_frame"]
            cond_x = self.cond_patch_embed(cond_x)
            cond_x =  rearrange(cond_x, 'B C (nH h) (nW w) -> (B nH nW) C h w', h=16, w=16, nH=nH, nW=nW).contiguous()
            cond_x = cond_x.reshape(cond_x.shape[0], cond_x.shape[1], -1)
            cond_x = cond_x.permute(0, 2, 1) # [B,L,D]
            cond_x = cond_x + self.cond_positional_embedding.to(cond_x.dtype)
            return cond_x
        cond_x = dpb["ref_feature"] # [B, C, grid_size, grid_size]
        cond_x = self.adapter_p[fa_idx](cond_x)
        cond_x = rearrange(cond_x, 'B C (nH h) (nW w) -> (B nH nW) C h w', h=16, w=16, nH=nH, nW=nW).contiguous()
        cond_x = cond_x.reshape(cond_x.shape[0], cond_x.shape[1], -1)
        cond_x = cond_x.permute(0, 2, 1) # [B,L,D]
        cond_x = cond_x + self.cond_positional_embedding_p.to(cond_x.dtype)
        return cond_x

    def forward(self, pixel_values, latent_tokens,cond=None,fa_idx=0, long_term_feature=None):
        batch_size = pixel_values.shape[0]
        x = pixel_values
        x = self.patch_embed(x)
        nH, nW = x.shape[2] // 16, x.shape[3] // 16
        stack_shape = (nH, nW)
        x = rearrange(x, 'B C (nH h) (nW w) -> (B nH nW) C h w', h=16, w=16, nH=nH, nW=nW).contiguous()
        batch_size = x.shape[0]

        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1) # shape = [*, grid ** 2, width]
        # class embeddings and positional embeddings
        x = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        x = x + self.positional_embedding.to(x.dtype) # shape = [*, grid ** 2 + 1, width]

        cond_x = self.apply_feature_adaptor(cond, fa_idx=fa_idx, stack_shape=stack_shape)
        # add long term feature
        if long_term_feature is not None:
            long_term_feature = self.long_term_feature_adapter(long_term_feature) # B 32 C
            long_term_feature = long_term_feature + self.long_term_feature_positional_embedding.to(long_term_feature.dtype)
            cond_x = torch.cat([cond_x, long_term_feature], dim=1)

        latent_tokens = _expand_token(latent_tokens, x.shape[0]).to(x.dtype)
        latent_tokens = latent_tokens + self.latent_token_positional_embedding.to(x.dtype)
        x = torch.cat([x, latent_tokens], dim=1)
        x = torch.cat([x, cond_x], dim=1)

        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        if True: # use checkpoint to save memory
            for i in range(self.num_layers):
                # x = self.transformer[i](x)
                x= checkpoint(self.transformer[i], x, use_reentrant=False)
                if i in self.global_attention_block_idx:
                    # x: (L, B*nH*nW, D)
                    idx = self.global_attention_block_idx.index(i)
                    # x = self.global_attention_blocks[idx](x, stack_shape)
                    x = checkpoint(self.global_attention_blocks[idx], x, stack_shape, use_reentrant=False)
        else:
            for i in range(self.num_layers):
                x = self.transformer[i](x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        
        latent_tokens = x[:, 1+self.grid_size**2:1+self.grid_size**2+self.num_latent_tokens]

        latent_tokens = self.ln_post(latent_tokens)
        latent_tokens = latent_tokens.reshape(batch_size, self.num_latent_tokens, self.width, 1).permute(0, 2, 1, 3)

        latent_tokens = self.conv_out(latent_tokens)
        latent_tokens = latent_tokens.reshape(batch_size, self.token_size, 1, self.num_latent_tokens)
        return latent_tokens, stack_shape

class GVC1DDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_size = config.dataset.preprocessing.crop_size
        self.patch_size = config.model.vq_model.vit_dec_patch_size
        self.grid_size = self.image_size // self.patch_size
        self.model_size = config.model.vq_model.vit_dec_model_size
        self.num_latent_tokens = config.model.vq_model.num_latent_tokens
        self.token_size = config.model.vq_model.token_size
        self.width = {
                "small": 512,
                "base": 768,
                "large": 1024,
            }[self.model_size]
        self.num_layers = {
                "small": 8,
                "base": 12,
                "large": 24,
            }[self.model_size]
        self.num_heads = {
                "small": 8,
                "base": 12,
                "large": 16,
            }[self.model_size]

        self.decoder_embed = nn.Linear(
            self.token_size, self.width, bias=True)
        scale = self.width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(1, self.width))
        self.positional_embedding = nn.Parameter(
                scale * torch.randn(self.grid_size ** 2 + 1, self.width))
        # add mask token and query pos embed
        self.mask_token = nn.Parameter(scale * torch.randn(1, 1, self.width))
        self.latent_token_positional_embedding = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens, self.width))
        self.ln_pre = nn.LayerNorm(self.width)
        self.transformer = nn.ModuleList()
        for i in range(self.num_layers):
            self.transformer.append(ResidualAttentionBlock(
                self.width, self.num_heads, mlp_ratio=4.0
            ))
        self.ln_post = nn.LayerNorm(self.width)
        # Directly predicting RGB pixels
        self.ffn = nn.Sequential(
            nn.Conv2d(self.width, self.patch_size * self.patch_size * 3, 1, padding=0, bias=True),
            Rearrange('b (p1 p2 c) h w -> b c (h p1) (w p2)',
                p1 = self.patch_size, p2 = self.patch_size),)
        self.conv_out = nn.Conv2d(3, 3, 3, padding=1, bias=True)
        
        # conditional feature (I frame)
        self.cond_patch_embed = nn.Conv2d(
            in_channels=3, out_channels=self.width,
            kernel_size=self.patch_size, stride=self.patch_size, bias=True)
        self.cond_positional_embedding = nn.Parameter(
                scale * torch.randn(self.grid_size ** 2, self.width))
        # P frame emb
        if config.model.hyperprior.get('use_hyperprior', False):
            self.adapter_p = nn.ModuleList([nn.Conv2d(self.width, self.width, 1) for _ in range(3)])
            self.cond_positional_embedding_p = nn.Parameter(
                    scale * torch.randn(self.grid_size ** 2, self.width))
        
        # long-term memory feature
        if config.model.memory.get('use_memory', False):
            memory_dim = config.model.memory.memory_dim
            self.long_term_feature_adapter = nn.Linear(memory_dim, self.width)
            self.long_term_feature_positional_embedding = nn.Parameter(
                scale * torch.randn(1, config.model.memory.memory_num, self.width))
        
        self.global_attention_block_idx = []
        if config.model.global_atten.get('use_global_atten', False):
            # global attention
            self.global_attention_block_idx = config.model.global_atten.decoder_global_attention_block_idx
            self.global_attention_blocks = nn.ModuleList()
            use_memory=config.model.memory.get('use_memory', False)
            memory_token_num=config.model.memory.get('memory_num', 0)
            for i in self.global_attention_block_idx:
                self.global_attention_blocks.append(global_attention_Block(hidden_size=self.width,num_heads=self.num_heads,use_memory=use_memory,
                memory_token_num=memory_token_num,latent_token_num=self.num_latent_tokens))

        training_module=config.model.get('training_module', None)
        if training_module == 'frame_condition':
            for param in self.parameters():
                if param is not self.cond_patch_embed.weight and \
                   param is not self.cond_patch_embed.bias and \
                   param is not self.cond_positional_embedding:
                    param.requires_grad = False
        elif training_module == 'codec_module':
            # only training adapter_p and cond_positional_embedding_p
            for name, param in self.named_parameters():
                if not name.startswith('adapter_p') and not name.startswith('cond_positional_embedding_p'):
                    param.requires_grad = False
        elif training_module == 'global_atten_module':
            # only training global_attention_blocks
            self.requires_grad_(False)
            for name, param in self.named_parameters():
                if name.startswith('global_attention_blocks'):
                    param.requires_grad = True
        elif training_module == 'memory_module':
            for name, param in self.named_parameters():
                if not name.startswith('long_term_feature_adapter') and not name.startswith('long_term_feature_positional_embedding'):
                    param.requires_grad = False


    def apply_feature_adaptor(self, dpb, fa_idx, stack_shape):
        nH, nW = stack_shape
        if fa_idx == 3:
            cond_x = dpb["ref_frame"]
            if not self.training:
                cond_x = cond_x.clamp_(0, 1)
            cond_x = self.cond_patch_embed(cond_x)
            cond_x =  rearrange(cond_x, 'B C (nH h) (nW w) -> (B nH nW) C h w', h=16, w=16, nH=nH, nW=nW).contiguous()
            cond_x = cond_x.reshape(cond_x.shape[0], cond_x.shape[1], -1)
            cond_x = cond_x.permute(0, 2, 1) # [B,L,D]
            cond_x = cond_x + self.cond_positional_embedding.to(cond_x.dtype)
            return cond_x
        if dpb["ref_feature"] is None:
            # add conditional latent tokens
            cond_x = dpb["ref_frame"]
            cond_x = self.cond_patch_embed(cond_x)
            cond_x =  rearrange(cond_x, 'B C (nH h) (nW w) -> (B nH nW) C h w', h=16, w=16, nH=nH, nW=nW).contiguous()
            cond_x = cond_x.reshape(cond_x.shape[0], cond_x.shape[1], -1)
            cond_x = cond_x.permute(0, 2, 1) # [B,L,D]
            cond_x = cond_x + self.cond_positional_embedding.to(cond_x.dtype)
            return cond_x
        cond_x = dpb["ref_feature"] # [B, C, grid_size, grid_size]
        cond_x = self.adapter_p[fa_idx](cond_x)
        cond_x = rearrange(cond_x, 'B C (nH h) (nW w) -> (B nH nW) C h w', h=16, w=16, nH=nH, nW=nW).contiguous()
        cond_x = cond_x.reshape(cond_x.shape[0], cond_x.shape[1], -1)
        cond_x = cond_x.permute(0, 2, 1) # [B,L,D]
        cond_x = cond_x + self.cond_positional_embedding_p.to(cond_x.dtype)
        return cond_x

    def forward(self, z_quantized, cond, fa_idx=0, long_term_feature=None, stack_shape=(1,1)):
        N, C, H, W = z_quantized.shape
        nH, nW = stack_shape
        assert H == 1 and W == self.num_latent_tokens, f"{H}, {W}, {self.num_latent_tokens}"
        x = z_quantized.reshape(N, C*H, W).permute(0, 2, 1) # NLD
        x = self.decoder_embed(x)

        batchsize, seq_len, _ = x.shape

        mask_tokens = self.mask_token.repeat(batchsize, self.grid_size**2, 1).to(x.dtype)
        mask_tokens = torch.cat([_expand_token(self.class_embedding, mask_tokens.shape[0]).to(mask_tokens.dtype),
                                    mask_tokens], dim=1)
        mask_tokens = mask_tokens + self.positional_embedding.to(mask_tokens.dtype)
        # add conditional latent tokens
        cond_x = self.apply_feature_adaptor(cond, fa_idx, stack_shape)
        # add long term feature
        if long_term_feature is not None:
            long_term_feature = self.long_term_feature_adapter(long_term_feature) # B 32 C
            long_term_feature = long_term_feature + self.long_term_feature_positional_embedding.to(long_term_feature.dtype)
            cond_x = torch.cat([cond_x, long_term_feature], dim=1)

        x = x + self.latent_token_positional_embedding[:seq_len]
        x = torch.cat([mask_tokens, x], dim=1)

        x = torch.cat([x, cond_x], dim=1)
        
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        if True: # use checkpoint to save memory
            for i in range(self.num_layers):
                # x = self.transformer[i](x)
                x= checkpoint(self.transformer[i], x, use_reentrant=False)
                if i in self.global_attention_block_idx:
                    # x: (L, B*nH*nW, D)
                    idx = self.global_attention_block_idx.index(i)
                    # x = self.global_attention_blocks[idx](x, stack_shape)
                    x = checkpoint(self.global_attention_blocks[idx], x, stack_shape, use_reentrant=False)
        else:
            for i in range(self.num_layers):
                x = self.transformer[i](x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = x[:, 1:1+self.grid_size**2] # remove cls embed
        x = self.ln_post(x)
        # N L D -> N D H W
        x = x.permute(0, 2, 1).reshape(batchsize, self.width, self.grid_size, self.grid_size)
        feature = rearrange(x, '(B nH nW) C h w -> B C (nH h) (nW w)', 
                        nH=nH, nW=nW, h=self.patch_size, w=self.patch_size).contiguous()
        x = self.ffn(feature)
        x = self.conv_out(x)
        return x, feature

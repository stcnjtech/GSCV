import torch
import torch.nn as nn
from collections import OrderedDict

class ResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model,
            output_dim=None,
            n_head=8,
            mlp_ratio = 4.0,
            act_layer = nn.GELU,
            norm_layer = nn.LayerNorm,
        ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
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
            if output_dim is not None and output_dim != d_model:
                self.out = nn.Linear(d_model, output_dim)


    def attention(
            self,
            x: torch.Tensor
    ):
        # x = apply_rope(x)  # Apply RoPE if needed
        return self.attn(x, x, x, need_weights=False)[0]

    def forward(
            self,
            x: torch.Tensor,
    ):
        #if x shape [B,C,1,N]
        convert_dim= False
        if x.dim() == 4:
            # Convert to [B,N,C] for attention
            convert_dim = True
            x = x.squeeze(2)
            x = x.permute(0, 2, 1)     
        attn_output = self.attention(x=self.ln_1(x))
        x = x + attn_output
        if self.mlp_ratio > 0:
            x = x + self.mlp(self.ln_2(x))
        x= self.out(x) if hasattr(self, 'out') else x
        if convert_dim:
            # Convert back to [B,C,1,N]
            x = x.permute(0, 2, 1).unsqueeze(2)
        return x

class ResidualAttentionUpDownsampleBlock(nn.Module):
    def __init__(
        self,
        d_model,
        output_dim=None,
        num_latents=None,  # 降采样后的 token 数量
        n_head=8,
        mlp_ratio=4.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert num_latents is not None, "num_latents must be provided for ResidualAttentionUpDownsampleBlock"
        
        self.num_latents = num_latents
        self.latents = nn.Parameter(torch.randn(num_latents, d_model))  # [L, C]

        self.ln_1 = norm_layer(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.mlp_ratio = mlp_ratio

        if mlp_ratio > 0:
            self.ln_2 = norm_layer(d_model)
            mlp_width = int(d_model * mlp_ratio)
            self.mlp = nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(d_model, mlp_width)),
                ("gelu", act_layer()),
                ("c_proj", nn.Linear(mlp_width, d_model))
            ]))
            if output_dim is not None and output_dim != d_model:
                self.out = nn.Linear(d_model, output_dim)

    def forward(self, x: torch.Tensor):
        """
        x: [B, N, C]
        output: [B, L, C], L = num_latents = N / 2
        """
        convert_dim = False
        if x.dim() == 4:
            # Convert to [B,N,C] for attention
            convert_dim = True
            x = x.squeeze(2)
            x = x.permute(0, 2, 1)
        B, N, C = x.shape
        x_norm = self.ln_1(x)

        # Expand learnable tokens to batch: [B, L, C]
        latents = self.latents.unsqueeze(0).expand(B, -1, -1)  # [B, L, C]

        # Cross attention: Q=latents, K,V=x
        # apply RoPE to latents
        # latents = apply_rope(latents, dim=-1)
        # x_norm = apply_rope(x_norm, dim=-1)
        attn_output = self.cross_attn(latents, x_norm, x_norm, need_weights=False)[0]

        x_down = latents + attn_output  # residual

        if self.mlp_ratio > 0:
            x_down = x_down + self.mlp(self.ln_2(x_down))
        x_down = self.out(x_down) if hasattr(self, 'out') else x_down
        if convert_dim:
            # Convert back to [B,C,1,L]
            x_down = x_down.permute(0, 2, 1).unsqueeze(2)
        return x_down

class ResidualAttentionARBlock(nn.Module):
    def __init__(
        self,
        d_model,
        output_dim=None,
        num_latents=None, 
        n_head=8,
        mlp_ratio=4.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert num_latents is not None, "num_latents must be provided for ResidualAttentionUpDownsampleBlock"
        
        self.num_latents = num_latents
        self.latents = nn.Parameter(torch.randn(num_latents, d_model))  # [L, C]

        self.ln_1 = norm_layer(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.mlp_ratio = mlp_ratio

        if mlp_ratio > 0:
            self.ln_2 = norm_layer(d_model)
            mlp_width = int(d_model * mlp_ratio)
            self.mlp = nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(d_model, mlp_width)),
                ("gelu", act_layer()),
                ("c_proj", nn.Linear(mlp_width, d_model))
            ]))
            if output_dim is not None and output_dim != d_model:
                self.out = nn.Linear(d_model, output_dim)

    def forward(self, x: torch.Tensor, idx=None):
        """
        x: [B, N, C]
        output: [B, L, C], L = num_latents = N / 2
        """
        convert_dim = False
        if x.dim() == 4:
            # Convert to [B,N,C] for attention
            convert_dim = True
            x = x.squeeze(2)
            x = x.permute(0, 2, 1)
        B, N, C = x.shape
        x_norm = self.ln_1(x)
        #according to idx, we can select the latents
        if idx is not None:
            latents = self.latents[idx].unsqueeze(0).unsqueeze(0).expand(B, -1, -1)
        else:
            latents = self.latents.unsqueeze(0).expand(B, -1, -1)

        attn_output = self.cross_attn(latents, x_norm, x_norm, need_weights=False)[0]

        x_down = latents + attn_output  # residual

        if self.mlp_ratio > 0:
            x_down = x_down + self.mlp(self.ln_2(x_down))
        x_down = self.out(x_down) if hasattr(self, 'out') else x_down
        if convert_dim:
            # Convert back to [B,C,1,L]
            x_down = x_down.permute(0, 2, 1).unsqueeze(2)
        return x_down
import math

import torch
import torch.nn as nn
from functools import partial

from timm.models.layers import DropPath, trunc_normal_, to_2tuple
from einops import rearrange, repeat

class Rearrange(nn.Module):
    def __init__(self, action):
        super().__init__()
        assert action in [
            'b c h w -> b (h w) c',
            'b (h w) c -> b c h w'
        ]
        self.action = action
    
    def forward(self, x):
        return rearrange(x, self.action)


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return torch.nn.Sequential(
        nn.Conv2d(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
        ),
        nn.SyncBatchNorm(out_planes)
    )


class LPI(nn.Module):
    """
    Local Patch Interaction module that allows explicit communication between tokens in 3x3 windows
    to augment the implicit communcation performed by the block diagonal scatter attention.
    Implemented using 2 layers of separable 3x3 convolutions with GeLU and BatchNorm2d
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU,
                 drop=0., kernel_size=3):
        super().__init__()
        out_features = out_features or in_features

        padding = kernel_size // 2

        self.conv1 = torch.nn.Conv2d(in_features, out_features, kernel_size=kernel_size,
                                     padding=padding, groups=out_features)
        self.act = act_layer()
        self.bn = nn.SyncBatchNorm(in_features)
        self.conv2 = torch.nn.Conv2d(in_features, out_features, kernel_size=kernel_size,
                                     padding=padding, groups=out_features)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.conv1(x)
        x = self.act(x)
        x = self.bn(x)
        x = self.conv2(x)
        x = x.reshape(B, C, N).permute(0, 2, 1)

        return x


class XCA(nn.Module):
    """ Cross-Covariance Attention (XCA) operation where the channels are updated using a weighted
     sum. The weights are obtained from the (softmax normalized) Cross-covariance
    matrix (Q^T K \\in d_h \\times d_h)
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)    # [qkv, batch, heads, H*W, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature'}


class XCABlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 num_tokens=196, eta=1.0, kernel_size=3):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = XCA(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
            proj_drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer,
                       drop=drop)

        self.norm3 = norm_layer(dim)
        self.local_mp = LPI(in_features=dim, act_layer=act_layer, kernel_size=kernel_size)

        self.gamma1 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)
        self.gamma2 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)
        self.gamma3 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        return x
    

class SingleXCABlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., kernel_size=3,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.attn_0 = XCABlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, kernel_size=kernel_size)
    
    def forward(self, x: torch.Tensor):
        """ x for Q; y for K V
        """
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).permute(0, 2, 1).contiguous()

        h = self.attn_0(x, H, W)

        h = h.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        return h


class DualXCABlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., kernel_size=3,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.attn_0 = XCABlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, kernel_size=kernel_size)
        self.attn_1 = XCABlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, kernel_size=kernel_size)
    
    def forward(self, x: torch.Tensor):
        """ x for Q; y for K V
        """
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).permute(0, 2, 1).contiguous()

        h = self.attn_0(x, H, W)
        h = self.attn_1(h, H, W)

        h = h.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        return h


class XCA_Cross(nn.Module):
    """ Cross-Covariance Attention (XCA) operation where the channels are updated using a weighted
     sum. The weights are obtained from the (softmax normalized) Cross-covariance
    matrix (Q^T K \\in d_h \\times d_h)
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Linear(dim, dim * 1, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y):
        """ x for Q; y for K V
        """
        B, N, C = x.shape
        q_in = self.q(x)
        kv_in = self.kv(y)
        qkv = torch.cat([q_in, kv_in], dim=-1)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)    # [qkv, batch, heads, H*W, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class XCACrossBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 num_tokens=196, eta=1.0, kernel_size=3):
        super().__init__()
        self.norm1_x = norm_layer(dim)
        self.norm1_y = norm_layer(dim)
        self.attn = XCA_Cross(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
            proj_drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer,
                       drop=drop)

        self.norm3 = norm_layer(dim)
        self.local_mp = LPI(in_features=dim, act_layer=act_layer, kernel_size=kernel_size)

        self.gamma1 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)
        self.gamma2 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)
        self.gamma3 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)

    def forward(self, x, y, H, W):
        """ x for Q; y for K V
        """
        x = x + self.drop_path(self.gamma1 * self.attn(self.norm1_x(x), self.norm1_y(y)))
        x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        return x
    

class SingleXCACrossBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., kernel_size=5,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.attn_0 = XCACrossBlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, kernel_size=kernel_size)
    
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """ x for Q; y for K V
        """
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).permute(0, 2, 1).contiguous()
        y = y.reshape(B, C, H * W).permute(0, 2, 1).contiguous()

        h = self.attn_0(x, y, H, W)

        h = h.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        return h


class DualXCACrossBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., kernel_size=5,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.attn_0 = XCACrossBlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, kernel_size=kernel_size)
        self.attn_1 = XCACrossBlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, kernel_size=kernel_size)
    
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """ x for Q; y for K V
        """
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).permute(0, 2, 1).contiguous()
        y = y.reshape(B, C, H * W).permute(0, 2, 1).contiguous()

        h = self.attn_0(x, y, H, W)
        h = self.attn_1(x, h, H, W)

        h = h.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        return h


class DualXCACrossBlock_small(DualXCACrossBlock):
    def __init__(self, dim, num_heads, mlp_ratio=4, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__(dim, num_heads, mlp_ratio, act_layer, norm_layer)
        self.attn_0 = XCACrossBlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, kernel_size=3)
        self.attn_1 = XCACrossBlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, kernel_size=3)


class DualXCACrossBlock2(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., 
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.attn_0 = XCACrossBlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, kernel_size=5)
        self.attn_1 = XCACrossBlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, kernel_size=3)
    
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """ x for Q; y for K V
        """
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).permute(0, 2, 1).contiguous()
        y = y.reshape(B, C, H * W).permute(0, 2, 1).contiguous()

        h = self.attn_0(x, y, H, W)
        h = self.attn_1(h, y, H, W)

        h = h.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        return h


class DualXCACrossBlock3(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., 
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.attn_0 = XCACrossBlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, kernel_size=5)
        self.attn_1 = XCACrossBlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, kernel_size=5)
        self.attn_2 = XCABlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, kernel_size=5)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.feat_merge = Mlp(dim*2, hidden_features=mlp_hidden_dim, out_features=dim, norm_layer=nn.LayerNorm)
    
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """ x for Q; y for K V
        """
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).permute(0, 2, 1).contiguous()
        y = y.reshape(B, C, H * W).permute(0, 2, 1).contiguous()

        h_0 = self.attn_0(x, y, H, W)
        h_1 = self.attn_1(y, x, H, W)
        h = self.feat_merge(torch.cat([h_0, h_1], dim=-1))
        h = self.attn_2(h, H, W)

        h = h.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        return h

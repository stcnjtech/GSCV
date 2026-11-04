import torch
import torch.nn as nn
from itertools import repeat
import collections.abc
from torch.nn.functional import scaled_dot_product_attention
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


def drop_path(
    x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True
):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob,3):0.3f}"


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        bias=True,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        return self.drop2(self.fc2(self.drop1(self.act(self.fc1(x)))))


class Attention(nn.Module):

    def __init__(
        self, dim, rope=None, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope.float() if rope is not None else None

    def forward(self, x, xpos):
        B, N, C = x.shape

        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .transpose(1, 3)
        )
        q, k, v = [qkv[:, :, i] for i in range(3)]

        q_type = q.dtype
        k_type = k.dtype
        if self.rope is not None:
            q = q.float()
            k = k.float()
            with torch.autocast(device_type="cuda", enabled=False):
                q = self.rope(q, xpos)
                k = self.rope(k, xpos)
            q = q.to(q_type)
            k = k.to(k_type)

        x = (
            scaled_dot_product_attention(
                query=q, key=k, value=v, dropout_p=self.attn_drop.p, scale=self.scale
            )
            .transpose(1, 2)
            .reshape(B, N, C)
        )

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(
        self,
        dim,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        rope=None,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            rope=rope,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, xpos=None):
        x = x + self.drop_path(self.attn(self.norm1(x), xpos))
        x = x + self.drop_path(self.mlp(self.norm3(x)))
        return x
    

class long_term_feature_extractor(nn.Module):
    def __init__(self, input_size, output_size, n_embd, layer_nums, hidden_token_num=256, query_num=32):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.n_embd = n_embd
    
        self.in_head = nn.Linear(input_size, n_embd, bias=False)
        self.out_head = nn.Linear(n_embd, output_size, bias=False)
        self.query_num = query_num
        self.update_layers = nn.ModuleList([Block(n_embd) for layer_idx in range(layer_nums)])
        self.readout_layers = nn.ModuleList([Block(n_embd) for layer_idx in range(layer_nums)])
        self.norm = nn.LayerNorm(n_embd)
        self.query_tokens = nn.Parameter(torch.randn(1, query_num, n_embd))
        self.hidden_tokens = nn.Parameter(torch.zeros(1, hidden_token_num, n_embd)) # (1, 1, 32, n_embd)
        self.query_tokens_position_emb = nn.Parameter(torch.randn(1, query_num, n_embd))
        self.hidden_tokens_position_emb = nn.Parameter(torch.randn(1, hidden_token_num, n_embd))
        self.update_toknes_position_emb = nn.Parameter(torch.randn(1, query_num, n_embd))
        self.update_hidden_position_emb = nn.Parameter(torch.randn(1, hidden_token_num, n_embd))
        # init nn.parameters
        nn.init.normal_(self.hidden_tokens, std=0.02)
        nn.init.normal_(self.query_tokens, std=0.02)
        nn.init.normal_(self.query_tokens_position_emb, std=0.02)
        nn.init.normal_(self.hidden_tokens_position_emb, std=0.02)
        nn.init.normal_(self.update_toknes_position_emb, std=0.02)
        nn.init.normal_(self.update_hidden_position_emb, std=0.02)
        # init weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        pass
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv2d):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()

    def get_init_hidden_tokens(self, batch_size):
        return self.hidden_tokens.expand(batch_size, -1, -1).clone() # (B, hidden_token_num, n_embd)

    def get_feature(self,hidden_tokens):
        '''
        hidden_tokens: (B, hidden_token_num, n_embd)
        '''
        B, N, C = hidden_tokens.shape
        query_tokens = self.query_tokens.expand(B, -1, -1).clone()  # (B, 32, n_embd)
        # add position embedding
        query_tokens = query_tokens + self.query_tokens_position_emb.to(query_tokens.dtype)
        hidden_tokens = hidden_tokens + self.hidden_tokens_position_emb.to(hidden_tokens.dtype)
        x= torch.cat([query_tokens, hidden_tokens], dim=1) # (B, 32+hidden_token_num, n_embd)
        for layer in self.readout_layers:
            x = layer(
                x
            )
        query_tokens = x[:, :self.query_num, :] # (B, 32, n_embd)
        query_tokens = self.norm(query_tokens)
        feature = self.out_head(query_tokens)  # (B, 32, output_size)
        return feature

    def update_feature(self,x,hidden_tokens):
        '''
        x:(B,32,input_size)
        hidden_tokens: (B, hidden_token_num, n_embd)
        return: updated hidden_tokens: (B, hidden_token_num, n_embd)
        '''
        B, N, C = x.shape
        x = self.in_head(x)  # (B, 32, n_embd)
        # add position embedding
        x = x + self.update_toknes_position_emb.to(x.dtype) # (B, 32, n_embd)
        hidden_tokens = hidden_tokens + self.update_hidden_position_emb.to(hidden_tokens.dtype) # (B, hidden_token_num, n_embd)
        x = torch.cat([x, hidden_tokens], dim=1) # (B, 32+hidden_token_num, n_embd)
        for layer in self.update_layers:
            x = layer(
                x
            )
        hidden_tokens = x[:, N:, :] # (B, hidden_token_num, n_embd)
        return hidden_tokens

if __name__ == "__main__":
    model = long_term_feature_extractor(16,256,256,2,512).cuda()
    print(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6)

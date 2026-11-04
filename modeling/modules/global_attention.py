from typing import Callable, Optional
from types import SimpleNamespace
import torch
import torch.nn as nn
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from collections import OrderedDict
from einops import rearrange

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def rot_pos_emb(grid_thw, rotary_pos_emb, spatial_merge_size=1):
    pos_ids = []
    for t, h, w in grid_thw:
        hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w).to(device=grid_thw.device)
        hpos_ids = hpos_ids.reshape(
            h // spatial_merge_size,
            spatial_merge_size,
            w // spatial_merge_size,
            spatial_merge_size,
        )
        hpos_ids = hpos_ids.permute(0, 2, 1, 3)
        hpos_ids = hpos_ids.flatten()

        wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1).to(device=grid_thw.device)
        wpos_ids = wpos_ids.reshape(
            h // spatial_merge_size,
            spatial_merge_size,
            w // spatial_merge_size,
            spatial_merge_size,
        )
        wpos_ids = wpos_ids.permute(0, 2, 1, 3)
        wpos_ids = wpos_ids.flatten()
        wpos_ids = wpos_ids.repeat_interleave(t, dim=0)
        hpos_ids = hpos_ids.repeat_interleave(t, dim=0)
        pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1))
    pos_ids = torch.cat(pos_ids, dim=0)
    max_grid_size = grid_thw[:, 1:].max()
    rotary_pos_emb_full = rotary_pos_emb(max_grid_size)
    rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
    return rotary_pos_emb

class VisionRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs

def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    if len(q.shape) == 4:
        # add batch dim
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)
    return q_embed, k_embed


class VisionAttention(nn.Module):
    def __init__(self, hidden_size=3584, num_heads=16) -> None:
        super().__init__()
        self.dim = hidden_size
        self.num_heads = num_heads
        self.head_dim = self.dim // self.num_heads
        self.num_key_value_groups = 1  # needed for eager attention
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = 0.0
        self.is_causal = False
        self.config = SimpleNamespace(_attn_implementation="flash_attention_2")
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        # hidden_states shape: (B, seq_len, C)
        batch_size, seq_length, _ = hidden_states.shape

        query_states, key_states, value_states = (
            self.qkv(hidden_states).reshape(batch_size, seq_length, 3, self.num_heads, -1).permute(2, 0, 1, 3, 4).unbind(0)
        ) # each shape: (B, seq_len, num_heads, head_dim)
        cos, sin = position_embeddings  # each shape: (seq_len, head_dim)
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin) # each shape: (B, seq_len, num_heads, head_dim)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        # flash attention 2
        if self.training:
            attention_interface = ALL_ATTENTION_FUNCTIONS["flash_attention_2"]
        else:
            attention_interface: Callable = eager_attention_forward
        attn_output, _ = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask=None,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.attention_dropout,
            is_causal=False,
            **kwargs,
        )
        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self.proj(attn_output)
        return attn_output

class global_attention_Block(nn.Module):
    def __init__(self, hidden_size=3584, num_heads=16, mlp_ratio=4.0, norm_layer=nn.LayerNorm, act_layer=nn.GELU, use_memory=False, memory_token_num=0, latent_token_num=0) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.use_memory = use_memory
        self.memory_token_num = memory_token_num
        self.latent_token_num = latent_token_num
        self.head_dim = hidden_size // num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(self.head_dim // 2)
        self.norm1 = norm_layer(hidden_size)
        self.attn = VisionAttention(hidden_size=hidden_size, num_heads=num_heads)
        if mlp_ratio > 0:
            self.norm2 = norm_layer(hidden_size)
            mlp_width = int(hidden_size * mlp_ratio)
            self.mlp = nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(hidden_size, mlp_width)),
                ("gelu", act_layer()),
                ("c_proj", nn.Linear(mlp_width, hidden_size))
            ]))

    def local_to_global(self, x: torch.Tensor, stack_shape: tuple[int, int]):
        '''
        x: (L, B*nH*nW, D)
        L = 1 + grid_size*grid_size + num_latent_tokens + grid_size*grid_size
        '''
        nH, nW = stack_shape
        D=x.shape[2]
        x = x.permute(1, 0, 2) # (B*nH*nW, L, D)
        B = x.shape[0] // (nH * nW)
        cls_token, img_tokens, latent_tokens, cond_tokens = torch.split(
            x, [1, 16*16, self.latent_token_num, 16*16], dim=1
        ) # each: (B*nH*nW, L', D)
        cls_token = rearrange(cls_token, '(B nH nW) 1 D -> B (nH nW) D', B=B, nH=nH, nW=nW).contiguous()
        img_tokens = rearrange(img_tokens, '(B nH nW) (h w) D -> B (nH h) (nW w) D', B=B, nH=nH, nW=nW, h=16, w=16).reshape(B, -1, D).contiguous()
        latent_tokens = rearrange(latent_tokens, '(B nH nW) L D -> B (nH nW) L D', B=B, nH=nH, nW=nW).reshape(B, -1, D).contiguous()
        cond_tokens = rearrange(cond_tokens, '(B nH nW) (h w) D -> B (nH h) (nW w) D', B=B, nH=nH, nW=nW, h=16, w=16).reshape(B, -1, D).contiguous()
        global_latent = torch.cat([cls_token, img_tokens, latent_tokens, cond_tokens], dim=1) # (B, L, D)
        grid_thw = torch.tensor([[1, nH,nW],[1,16*nH,16*nW],[self.latent_token_num,nH,nW],[1,16*nH,16*nW]], device=x.device)
        pos_emb = rot_pos_emb(grid_thw, self.rotary_pos_emb) # (L, head_dim)
        emb = torch.cat((pos_emb, pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())
        return global_latent, position_embeddings

    def global_to_local(self, x: torch.Tensor, stack_shape: tuple[int, int]):
        '''
        x: (B, L, D)
        L = 1 + grid_size*grid_size + num_latent_tokens + grid_size*grid_size
        '''
        nH, nW = stack_shape
        B, L, D = x.shape
        
        cls_token, img_tokens, latent_tokens, cond_tokens = torch.split(
            x, [nH*nW, 16*nH*16*nW, self.latent_token_num*nH*nW, 16*nH*16*nW], dim=1
        ) # each: (B, L', D)
        
        cls_token = rearrange(cls_token, 'B (nH nW) D -> (B nH nW) 1 D', nH=nH, nW=nW).contiguous()
        
        # img_tokens: (B, 16*nH*16*nW, D) -> (B*nH*nW, 16*16, D)
        img_tokens = img_tokens.reshape(B, nH*16, nW*16, D)
        img_tokens = rearrange(img_tokens, 'B (nH h) (nW w) D -> (B nH nW) (h w) D', nH=nH, nW=nW, h=16, w=16).contiguous()
        
        # latent_tokens: (B, self.latent_token_num*nH*nW, D) -> (B*nH*nW, self.latent_token_num, D)
        latent_tokens = latent_tokens.reshape(B, nH*nW, self.latent_token_num, D)
        latent_tokens = rearrange(latent_tokens, 'B (nH nW) L D -> (B nH nW) L D', nH=nH, nW=nW).contiguous()
        
        # cond_tokens: (B, 16*nH*16*nW, D) -> (B*nH*nW, 16*16, D)
        cond_tokens = cond_tokens.reshape(B, nH*16, nW*16, D)
        cond_tokens = rearrange(cond_tokens, 'B (nH h) (nW w) D -> (B nH nW) (h w) D', nH=nH, nW=nW, h=16, w=16).contiguous()
        
        local_latent = torch.cat([cls_token, img_tokens, latent_tokens, cond_tokens], dim=1) # (B*nH*nW, L, D)
        
        local_latent = local_latent.permute(1, 0, 2) # (L, B*nH*nW, D)
        
        return local_latent

    def local_to_global_memory(self, x: torch.Tensor, stack_shape: tuple[int, int]):
        '''
        x: (L, B*nH*nW, D)
        L = 1 + grid_size*grid_size + num_latent_tokens + grid_size*grid_size
        '''
        nH, nW = stack_shape
        D=x.shape[2]
        x = x.permute(1, 0, 2) # (B*nH*nW, L, D)
        B = x.shape[0] // (nH * nW)
        cls_token, img_tokens, latent_tokens, cond_tokens, memory_tokens = torch.split(
            x, [1, 16*16, self.latent_token_num, 16*16, self.memory_token_num], dim=1
        ) # each: (B*nH*nW, L', D)
        cls_token = rearrange(cls_token, '(B nH nW) 1 D -> B (nH nW) D', B=B, nH=nH, nW=nW).contiguous()
        img_tokens = rearrange(img_tokens, '(B nH nW) (h w) D -> B (nH h) (nW w) D', B=B, nH=nH, nW=nW, h=16, w=16).reshape(B, -1, D).contiguous()
        latent_tokens = rearrange(latent_tokens, '(B nH nW) L D -> B (nH nW) L D', B=B, nH=nH, nW=nW).reshape(B, -1, D).contiguous()
        cond_tokens = rearrange(cond_tokens, '(B nH nW) (h w) D -> B (nH h) (nW w) D', B=B, nH=nH, nW=nW, h=16, w=16).reshape(B, -1, D).contiguous()
        memory_tokens = rearrange(memory_tokens, '(B nH nW) L D -> B (nH nW) L D', B=B, nH=nH, nW=nW).reshape(B, -1, D).contiguous()
        global_latent = torch.cat([cls_token, img_tokens, latent_tokens, cond_tokens, memory_tokens], dim=1) # (B, L, D)
        grid_thw = torch.tensor([[1, nH,nW],[1,16*nH,16*nW],[self.latent_token_num,nH,nW],[1,16*nH,16*nW],[self.memory_token_num,nH,nW]], device=x.device)
        pos_emb = rot_pos_emb(grid_thw, self.rotary_pos_emb) # (L, head_dim)
        emb = torch.cat((pos_emb, pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())
        return global_latent, position_embeddings

    def global_to_local_memory(self, x: torch.Tensor, stack_shape: tuple[int, int]):
        '''
        x: (B, L, D)
        L = 1 + grid_size*grid_size + num_latent_tokens + grid_size*grid_size
        '''
        nH, nW = stack_shape
        B, L, D = x.shape
        
        cls_token, img_tokens, latent_tokens, cond_tokens, memory_tokens = torch.split(
            x, [nH*nW, 16*nH*16*nW, self.latent_token_num*nH*nW, 16*nH*16*nW,self.memory_token_num*nH*nW], dim=1
        ) # each: (B, L', D)
        
        cls_token = rearrange(cls_token, 'B (nH nW) D -> (B nH nW) 1 D', nH=nH, nW=nW).contiguous()
        
        # img_tokens: (B, 16*nH*16*nW, D) -> (B*nH*nW, 16*16, D)
        img_tokens = img_tokens.reshape(B, nH*16, nW*16, D)
        img_tokens = rearrange(img_tokens, 'B (nH h) (nW w) D -> (B nH nW) (h w) D', nH=nH, nW=nW, h=16, w=16).contiguous()
        
        # latent_tokens: (B, self.latent_token_num*nH*nW, D) -> (B*nH*nW, self.latent_token_num, D)
        latent_tokens = latent_tokens.reshape(B, nH*nW, self.latent_token_num, D)
        latent_tokens = rearrange(latent_tokens, 'B (nH nW) L D -> (B nH nW) L D', nH=nH, nW=nW).contiguous()
        
        # cond_tokens: (B, 16*nH*16*nW, D) -> (B*nH*nW, 16*16, D)
        cond_tokens = cond_tokens.reshape(B, nH*16, nW*16, D)
        cond_tokens = rearrange(cond_tokens, 'B (nH h) (nW w) D -> (B nH nW) (h w) D', nH=nH, nW=nW, h=16, w=16).contiguous()
        
        # memory_tokens: (B, self.memory_token_num*nH*nW, D) -> (B*nH*nW, self.memory_token_num, D)
        memory_tokens = memory_tokens.reshape(B, nH*nW, self.memory_token_num, D)
        memory_tokens = rearrange(memory_tokens, 'B (nH nW) L D -> (B nH nW) L D', nH=nH, nW=nW).contiguous()
        
        local_latent = torch.cat([cls_token, img_tokens, latent_tokens, cond_tokens, memory_tokens], dim=1) # (B*nH*nW, L, D)
        
        local_latent = local_latent.permute(1, 0, 2) # (L, B*nH*nW, D)
        
        return local_latent

    def forward(
        self,
        hidden_states: torch.Tensor,
        stack_shape: Optional[tuple[int, int]] = None,
        **kwargs,
    ) -> torch.Tensor:
        if self.use_memory:
            hidden_states, position_embeddings = self.local_to_global_memory(hidden_states, stack_shape)
        else:
            hidden_states, position_embeddings = self.local_to_global(hidden_states, stack_shape)
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        if self.use_memory:
            hidden_states = self.global_to_local_memory(hidden_states, stack_shape)
        else:
            hidden_states = self.global_to_local(hidden_states, stack_shape)
        return hidden_states

if __name__ == "__main__":
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        model = global_attention_Block(hidden_size=768, num_heads=12).cuda()
        nH, nW = 2, 2
        stack_shape = (nH, nW)
        B = 3
        hidden_states =  torch.randn(1 + 256 + 32 + 256, B*nH*nW, 768).cuda()
        output = model(hidden_states,  stack_shape)
        print(output.shape)  # torch.Size([2, 64, 768])
    exit()

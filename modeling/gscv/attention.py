"""Cross-window (global) attention of GVC1D restricted to the coded windows of a frame.

Uses the parameters of an existing modeling.modules.global_attention.global_attention_Block
and reproduces its computation exactly when every window is active: tokens of all listed
windows of one sample attend to each other with 2D RoPE positions taken from the window's place
in the full frame (image/condition tokens in patch units, class/latent/memory tokens in window
units, as in GVC1D). Attention runs through PyTorch SDPA (no flash-attn, no O(L^2) eager peak).
"""
import torch
import torch.nn.functional as F

from modeling.modules.global_attention import apply_rotary_pos_emb_vision

PATCHES = 16  # patches per window side


def window_token_positions(win_hw, latent_num, memory_num):
    """(row, col) RoPE position of every token of each window -> LongTensor (Na, L, 2).

    Token order matches the local sequences of GVC1D:
    [cls | 256 image/mask | latent | 256 condition | memory].
    """
    h, w = win_hw[:, 1:2], win_hw[:, 2:3]
    dev = win_hw.device
    i = torch.arange(PATCHES, device=dev).repeat_interleave(PATCHES)[None]
    j = torch.arange(PATCHES, device=dev).repeat(PATCHES)[None]
    patch = torch.stack((PATCHES * h + i, PATCHES * w + j), dim=-1)          # (Na, 256, 2)
    win = torch.stack((h, w), dim=-1)                                         # (Na, 1, 2)
    parts = [win, patch, win.expand(-1, latent_num, -1), patch]
    if memory_num:
        parts.append(win.expand(-1, memory_num, -1))
    return torch.cat(parts, dim=1)


def _attention(attn, x, cos, sin):
    b, n, _ = x.shape
    q, k, v = attn.qkv(x).reshape(b, n, 3, attn.num_heads, -1).permute(2, 0, 1, 3, 4).unbind(0)
    q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
    o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                       scale=attn.scaling)
    return attn.proj(o.transpose(1, 2).reshape(b, n, -1))


def subset_global_attention(block, x, win_hw, grid_hw):
    """x: (L, Na, D) local layout of the active windows; win_hw: (Na, 3) [sample, row, col]
    sorted by sample; grid_hw: (nH, nW) windows of the full frame."""
    L, na, d = x.shape
    mem = block.memory_token_num if block.use_memory else 0
    pos = window_token_positions(win_hw, block.latent_token_num, mem)
    assert pos.shape[1] == L, (pos.shape, L)
    table = block.rotary_pos_emb(PATCHES * max(grid_hw))
    xs = x.permute(1, 0, 2)
    outs = []
    for b in torch.unique_consecutive(win_hw[:, 0]).tolist():
        idx = torch.nonzero(win_hw[:, 0] == b, as_tuple=True)[0]
        seq = xs[idx].reshape(1, -1, d)
        rot = table[pos[idx].reshape(-1, 2)].flatten(1)
        emb = torch.cat((rot, rot), dim=-1)
        h = seq + _attention(block.attn, block.norm1(seq), emb.cos(), emb.sin())
        h = h + block.mlp(block.norm2(h))
        outs.append(h.reshape(idx.numel(), L, d))
    return torch.cat(outs, dim=0).permute(1, 0, 2)

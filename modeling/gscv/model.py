"""GSCV model: GVC1D extended with Selective Fidelity Allocation (SFA) and the Interaction-Aware Context Model (IACM).

SFA (content side)
  - class embeddings e_F of the block classes F_t on the encoder patch tokens and on the decoder mask tokens
  - per-window quality index qp_k: fidelity-critical windows and refinement frames get a higher index
    (offsets derived by the window state machine, see rsm.py); the region losses are in losses.py
IACM (interaction side)
  - window states S_t in {Update, Skip, Refresh, Revert}, decided by rsm.RSMEncoder
  - Skip windows copy their pixels and decoder features, Revert windows restore a snapshot
    (pixels, decoder features, memory rows, tokens) taken before the window last changed
  - state-gated update of the long-term memory, with a state embedding (update type, display duration)
  - only Update / Refresh windows run through the transformers; cross-window attention runs among them

Parameter names of GVC1D are kept, so the released GVC1D weights load directly; the parameters GSCV adds are
initialised to the identity (zero embeddings, an open memory gate). With every switch of `model.gscv` off and every
window coded, the computation equals GVC1D.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from omegaconf import OmegaConf
from torch.utils.checkpoint import checkpoint

from modeling.GVC1D import GVC1D, DiagonalGaussianDistribution
from modeling.entropy.video_model import Compressive_bottleneck
from modeling.modules.base_model import BaseModel
from modeling.modules.blocks import GVC1DDecoder, GVC1DEncoder, _expand_token
from modeling.modules.long_term_memory import long_term_feature_extractor

from .attention import subset_global_attention
from .rsm import REFRESH, REVERT, SKIP, UPDATE

DEFAULT_GSCV = {
    "use_fmap": False,         # SFA: class embeddings e_F on the encoder and decoder tokens
    "use_null_ctx": False,     # learned null contexts of Refresh windows (trained in stage S2a)
    "use_gate": False,         # IACM: gated memory update
    "use_state_embed": False,  # IACM: state embedding of the memory update
    "gate_bias": 5.0,          # initial gate bias: sigmoid(5) = 0.993, close to GVC1D's plain update
    "k_snap": 2,               # snapshot slots per window (taken from rsm.k_snap)
}

# switches of the computation that hold no parameters (model.runtime of the configuration). Stage S2a trains the
# null contexts with both on; from S2b on, and at test time, Refresh windows keep their short-term context and
# memory rows (the defaults).
DEFAULT_RUNTIME = {
    "null_ctx": False,         # Refresh windows use the null contexts (if the model has them)
    "refresh_reset": False,    # Refresh windows reset their memory rows
}


def gscv_options(config):
    opts = dict(DEFAULT_GSCV)
    g = config.model.get("gscv", None)
    if g is not None:
        opts.update(OmegaConf.to_container(g, resolve=True) if OmegaConf.is_config(g) else dict(g))
    if "rsm" in config and "k_snap" in config.rsm:      # one source of truth for the snapshot slots
        opts["k_snap"] = max(int(config.rsm.k_snap), 1)
    return opts


def to_windows(t, size):
    return rearrange(t, "B C (nH h) (nW w) -> (B nH nW) C h w", h=size, w=size)


def from_windows(t, grid, batch):
    nh, nw = grid
    return rearrange(t, "(B nH nW) C h w -> B C (nH h) (nW w)", B=batch, nH=nh, nW=nw)


class _WindowTransformerMixin:
    """Local layers on the coded windows + cross-window attention among them."""

    def _layers(self, x, sel):
        # training: gradient checkpointing of every ckpt_stride-th layer (1 = every layer, as GVC1D; 2 = every
        # other one, faster but more memory; 0 = none); the computation is the same either way
        stride = getattr(self, "ckpt_stride", 1)
        for i in range(self.num_layers):
            blk = self.transformer[i]
            ck = self.training and stride > 0 and i % stride == 0
            x = checkpoint(blk, x, use_reentrant=False) if ck else blk(x)
            if i in self.global_attention_block_idx:
                g = self.global_attention_blocks[self.global_attention_block_idx.index(i)]
                if ck:
                    x = checkpoint(subset_global_attention, g, x, sel["win_hw"], sel["grid"], use_reentrant=False)
                else:
                    x = subset_global_attention(g, x, sel["win_hw"], sel["grid"])
        return x

    def _context(self, cond, fa_idx, sel, long_term_feature):
        cond_x = self.apply_feature_adaptor(cond, fa_idx, sel["grid"])[sel["active"]]
        if self.null_cond is not None and sel.get("refresh") is not None and bool(sel["refresh"].any()):
            cond_x = torch.where(sel["refresh"][:, None, None], self.null_cond.to(cond_x.dtype)[None], cond_x)
        if long_term_feature is not None:
            lt = self.long_term_feature_adapter(long_term_feature)
            lt = lt + self.long_term_feature_positional_embedding.to(lt.dtype)
            cond_x = torch.cat([cond_x, lt], dim=1)
        return cond_x

    def _init_gscv(self, config):
        opts = gscv_options(config)
        self.fmap_embed = nn.Embedding(3, self.width) if opts["use_fmap"] else None
        self.null_cond = nn.Parameter(torch.zeros(self.grid_size ** 2, self.width)) if opts["use_null_ctx"] else None

    def reset_gscv_parameters(self):
        if self.fmap_embed is not None:
            nn.init.zeros_(self.fmap_embed.weight)


class GSCVEncoder(_WindowTransformerMixin, GVC1DEncoder):
    def __init__(self, config):
        super().__init__(config)
        self._init_gscv(config)

    def forward(self, pixel_values, latent_tokens, cond, fa_idx, long_term_feature, sel):
        nh, nw = sel["grid"]
        x = self.patch_embed(pixel_values)
        x = rearrange(x, "B C (nH h) (nW w) -> (B nH nW) (h w) C", h=16, w=16, nH=nh, nW=nw)[sel["active"]]
        n = x.shape[0]
        x = torch.cat([_expand_token(self.class_embedding, n).to(x.dtype), x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        fp = sel.get("fpatch")
        if self.fmap_embed is not None and fp is not None:          # SFA: class embedding e_F^e
            x = torch.cat([x[:, :1], x[:, 1:] + self.fmap_embed(fp).to(x.dtype)], dim=1)
        cond_x = self._context(cond, fa_idx, sel, long_term_feature)
        latent = _expand_token(latent_tokens, n).to(x.dtype) + self.latent_token_positional_embedding.to(x.dtype)
        x = torch.cat([x, latent, cond_x.to(x.dtype)], dim=1)
        x = self.ln_pre(x).permute(1, 0, 2)
        x = self._layers(x, sel).permute(1, 0, 2)
        latent = x[:, 1 + self.grid_size ** 2:1 + self.grid_size ** 2 + self.num_latent_tokens]
        latent = self.ln_post(latent).reshape(n, self.num_latent_tokens, self.width, 1).permute(0, 2, 1, 3)
        return self.conv_out(latent).reshape(n, self.token_size, 1, self.num_latent_tokens)


class GSCVDecoder(_WindowTransformerMixin, GVC1DDecoder):
    def __init__(self, config):
        super().__init__(config)
        self._init_gscv(config)

    def forward(self, z, cond, fa_idx, long_term_feature, sel):
        """Decoder features (n, width, 16, 16) of the coded windows."""
        n, c, h, w = z.shape
        x = self.decoder_embed(z.reshape(n, c * h, w).permute(0, 2, 1))
        mask = self.mask_token.repeat(n, self.grid_size ** 2, 1).to(x.dtype)
        mask = torch.cat([_expand_token(self.class_embedding, n).to(x.dtype), mask], dim=1)
        mask = mask + self.positional_embedding.to(x.dtype)
        fp = sel.get("fpatch")
        if self.fmap_embed is not None and fp is not None:          # SFA: class embedding e_F^d
            mask = torch.cat([mask[:, :1], mask[:, 1:] + self.fmap_embed(fp).to(x.dtype)], dim=1)
        cond_x = self._context(cond, fa_idx, sel, long_term_feature)
        x = x + self.latent_token_positional_embedding[:x.shape[1]]      # as GVC1DDecoder (no cast)
        x = torch.cat([mask, x, cond_x], dim=1)
        x = self.ln_pre(x).permute(1, 0, 2)
        x = self._layers(x, sel).permute(1, 0, 2)
        x = self.ln_post(x[:, 1:1 + self.grid_size ** 2])
        return x.permute(0, 2, 1).reshape(n, self.width, self.grid_size, self.grid_size)

    def render(self, feature):
        """Full-frame decoder features (B, width, H/16, W/16) -> RGB (B, 3, H, W)."""
        return self.conv_out(self.ffn(feature))


class GSCVBottleneck(Compressive_bottleneck):
    def __init__(self, *args, use_null_ctx=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.null_ctx = nn.Parameter(torch.zeros(256, 16, 16)) if use_null_ctx else None

    def context_features(self, dpb, fa_idx, grid, active, refresh=None):
        feat = self.apply_feature_adaptor(dpb, fa_idx, grid)[active]
        if self.null_ctx is not None and refresh is not None and bool(refresh.any()):
            feat = torch.where(refresh[:, None, None, None], self.null_ctx.to(feat.dtype)[None], feat)
        return feat

    def code_windows(self, x, feat, long_term_feature, qp_index):
        """Code the 1D tokens of the given windows; returns decoded tokens and bits per window."""
        index = qp_index.to(torch.int32)
        q_encoder = torch.index_select(self.q_encoder, 0, index)
        q_decoder = torch.index_select(self.q_decoder, 0, index)
        q_feature = torch.index_select(self.q_feature, 0, index)
        q_recon = torch.index_select(self.q_recon, 0, index)
        ctx, ctx_t = self.feature_extractor(feat, long_term_feature, q_feature)
        y = self.encoder(x, ctx, q_encoder)
        z = self.hyper_encoder(y)
        z_hat = self.quant(z)
        params = self.res_prior_param_decoder(z_hat, ctx_t)
        y_res, y_q, y_hat, scales_hat = self.forward_AR_prior(y, params, self.y_spatial_prior_reduction,
                                                              self.y_spatial_prior)
        x_hat, _ = self.get_recon_and_feature(y_hat, ctx, q_decoder, q_recon)
        y_for_bit = self.add_noise(y_res) if self.training else y_q
        z_for_bit = self.add_noise(z) if self.training else z_hat
        bits_y = self.get_y_gaussian_bits(y_for_bit, scales_hat).sum(dim=(1, 2, 3))
        bits_z = self.get_z_bits(z_for_bit, self.bit_estimator_z, index).sum(dim=(1, 2, 3))
        return x_hat, bits_y, bits_z


class GSCVMemory(long_term_feature_extractor):
    """State-gated memory update (IACM): m_t = g * h + (1 - g) * m_{t-1},  h = T_update(y^_t + e_S ; m_{t-1}),
    g = sigmoid(W_g [h ; m_{t-1}] + b_g), e_S = Emb_type[update type] + Emb_age[display-duration bucket]."""

    def __init__(self, *args, use_gate=False, use_state_embed=False, gate_bias=5.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.gate_bias = gate_bias
        self.type_embed = nn.Embedding(3, self.n_embd) if use_state_embed else None
        self.age_embed = nn.Embedding(4, self.n_embd) if use_state_embed else None
        self.gate = nn.Linear(2 * self.n_embd, 1) if use_gate else None

    def reset_gscv_parameters(self):
        for e in (self.type_embed, self.age_embed):
            if e is not None:
                nn.init.zeros_(e.weight)
        if self.gate is not None:
            nn.init.zeros_(self.gate.weight)
            nn.init.constant_(self.gate.bias, self.gate_bias)

    def update_feature(self, x, hidden_tokens, mem_type=None, mem_age=None):
        b, n, _ = x.shape
        x = self.in_head(x)
        x = x + self.update_toknes_position_emb.to(x.dtype)
        hidden = hidden_tokens + self.update_hidden_position_emb.to(hidden_tokens.dtype)
        if self.type_embed is not None and mem_type is not None:      # state embedding e_S
            x = x + (self.type_embed(mem_type) + self.age_embed(mem_age))[:, None, :].to(x.dtype)
        seq = torch.cat([x, hidden], dim=1)     # as GVC1D: the memory's dtype (fp32 under autocast) wins
        for layer in self.update_layers:
            seq = layer(seq)
        h = seq[:, n:, :]
        self.last_gate = None
        if self.gate is not None:               # the gate in fp32 (bf16 cannot resolve sigmoid(5) = 0.9933)
            with torch.autocast(device_type=h.device.type, enabled=False):
                m = hidden_tokens.float()
                g = torch.sigmoid(F.linear(torch.cat([h.float(), m], dim=-1), self.gate.weight.float(),
                                           self.gate.bias.float()))
                h = (g * h.float() + (1 - g) * m).to(hidden_tokens.dtype)
            self.last_gate = g.detach().mean()  # training log
        return h


def runtime_from_config(config):
    """DEFAULT_RUNTIME with the configuration's model.runtime switches (set per training stage)."""
    rt = dict(DEFAULT_RUNTIME)
    over = OmegaConf.select(OmegaConf.create(config) if isinstance(config, dict) else config, "model.runtime",
                            default=None)
    if over:
        over = OmegaConf.to_container(over, resolve=True) if not isinstance(over, dict) else over
        unknown = set(over) - set(DEFAULT_RUNTIME)
        assert not unknown, f"unknown model.runtime switches {sorted(unknown)}"
        rt.update(over)
    return rt


class GSCV(GVC1D):
    def __init__(self, config):
        if isinstance(config, dict):
            config = OmegaConf.create(config)
        BaseModel.__init__(self)
        self.config = config
        self.opts = gscv_options(config)
        # GVC1D's training_module freezing is not used: scripts/train.py selects the trainable parameters
        assert str(config.model.get("training_module", None)) in ("None", "none", ""), \
            "GSCV ignores model.training_module (scripts/train.py: train.stages.<stage>.trainable)"
        assert config.model.hyperprior.get("use_hyperprior", False), "GSCV builds on the hyperprior codec"
        self.use_memory = config.model.memory.get("use_memory", False)
        self.num_latent_tokens = config.model.vq_model.num_latent_tokens
        self.encoder = GSCVEncoder(config)
        self.decoder = GSCVDecoder(config)
        scale = self.encoder.width ** -0.5
        self.latent_tokens = nn.Parameter(scale * torch.randn(self.num_latent_tokens, self.encoder.width))
        mem = config.model.memory
        self.codec_model = GSCVBottleneck(
            N=config.model.hyperprior.get("N", 16), z_channel=config.model.hyperprior.get("z_channel", 8),
            y_tokens_num=self.num_latent_tokens, g_ch_src=config.model.vq_model.get("token_size", 128),
            memory_num=mem.memory_num if self.use_memory else 0, memory_dim=mem.memory_dim if self.use_memory else 0,
            use_null_ctx=self.opts["use_null_ctx"])
        if self.use_memory:
            self.long_term_feature_extractor = GSCVMemory(
                input_size=config.model.vq_model.get("token_size", 128), output_size=mem.memory_dim,
                n_embd=mem.memory_dim, layer_nums=mem.get("memory_layers", 8),
                hidden_token_num=mem.total_memory_num, query_num=mem.memory_num,
                use_gate=self.opts["use_gate"], use_state_embed=self.opts["use_state_embed"],
                gate_bias=self.opts["gate_bias"])
        self.runtime = runtime_from_config(config)
        self.timing = False          # test: per-part CUDA times in forward_one_frame's "times_ms"
        self.dpb = {"ref_frame": None, "ref_feature": None, "long_term_mem": None}
        self.frame_idx = 1
        self.quantize = DiagonalGaussianDistribution
        self.apply(self._init_weights)
        self.encoder.reset_gscv_parameters()
        self.decoder.reset_gscv_parameters()
        if self.use_memory:
            self.long_term_feature_extractor.reset_gscv_parameters()

    def encode(self, *args, **kwargs):
        raise NotImplementedError("GSCV codes whole frames with forward_one_frame (window states, subsets)")

    def decode(self, *args, **kwargs):
        raise NotImplementedError("GSCV codes whole frames with forward_one_frame (window states, subsets)")

    # ------------------------------------------------------------------ weights
    def load_weights(self, path):
        """Load GVC1D or GSCV weights; parameters missing from the file keep their identity init
        and the null contexts are initialised from a mid-grey reference."""
        sd = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "model" in sd and not any(k.startswith("encoder.") for k in sd):
            sd = sd["model"]
        msg = self.load_state_dict(sd, strict=False)
        missing = [k for k in msg.missing_keys]
        unexpected = [k for k in msg.unexpected_keys]
        if any("null" in k for k in missing):
            self.init_null_context()
        return missing, unexpected

    @torch.no_grad()
    def init_null_context(self):
        p = next(self.parameters())
        gray = torch.full((1, 3, 256, 256), 0.5, device=p.device, dtype=p.dtype)
        for m in (self.encoder, self.decoder):
            if m.null_cond is not None:
                c = m.cond_patch_embed(gray).reshape(1, m.width, -1).permute(0, 2, 1)[0]
                m.null_cond.copy_(c + m.cond_positional_embedding)
        if self.codec_model.null_ctx is not None:
            self.codec_model.null_ctx.copy_(self.codec_model.feature_adaptor_i(gray, (1, 1))[0])

    # ------------------------------------------------------------------ helpers
    def init_memory(self, n, device, dtype):
        return self.long_term_feature_extractor.get_init_hidden_tokens(n).to(device=device, dtype=dtype)

    def _store_snapshots(self, dpb, create, n, mem):
        """Store the pre-change state (pixels, decoder features, memory rows, decoded tokens) of the windows in
        `create` (slot per window, -1 = none) in the snapshot buffer B -- the state before this frame."""
        idx = torch.nonzero(create >= 0, as_tuple=True)[0]
        snaps = dpb.get("snapshots")
        if idx.numel() == 0:
            return snaps
        k = self.opts["k_snap"]
        assert dpb.get("ref_feature") is not None, "a snapshot needs a decoded state (never at the first P-frame)"
        assert int(create.max()) < k, f"snapshot slot {int(create.max())} >= k_snap {k} (model.gscv / rsm k_snap)"
        # snapshots are detached: a Revert window copies them and carries no new information
        state = {"pix": to_windows(dpb["ref_frame"].float(), 256), "feat": to_windows(dpb["ref_feature"], 16)}
        if mem is not None:
            state["mem"] = mem
        if dpb.get("tokens") is not None:
            state["tok"] = dpb["tokens"]
        if snaps is None:
            snaps = {}
        snaps = {key: v.clone() for key, v in snaps.items()}
        slot = create[idx]
        for key, v in state.items():
            v = v.detach()
            if key not in snaps:
                snaps[key] = v.new_zeros((n, k) + v.shape[1:])
            snaps[key][idx, slot] = v[idx].to(snaps[key].dtype)
        return snaps

    @staticmethod
    def patch_classes(fclass, active, grid):
        """(B, H/b, W/b) block classes -> (n_active, 256) patch classes (raster in window)."""
        nh, nw = grid
        rep = 16 * nh // fclass.shape[1]   # 16-pixel patches per block side (4 for 64x64 blocks)
        assert rep >= 1 and rep * fclass.shape[1] == 16 * nh and rep * fclass.shape[2] == 16 * nw, \
            "block map does not match the frame (block size must be a multiple of 16)"
        fp = fclass.repeat_interleave(rep, dim=1).repeat_interleave(rep, dim=2)
        fp = rearrange(fp, "B (nH h) (nW w) -> (B nH nW) (h w)", h=16, w=16, nH=nh, nW=nw)
        return fp[active].long()

    # ------------------------------------------------------------------ coding
    def forward_one_frame(self, x, dpb, qp, fa_idx=0, rsm=None):
        """Code one P-frame.

        x: (B, 3, H, W) in [0, 1], H and W multiples of 256.
        qp: (B*nH*nW,) per-window quality index (int tensor) or list of per-sample indices.
        rsm: dict of per-window tensors (B*nH*nW,) -- modes, revert_slot, snap_create, mem_type, mem_age -- and
             fclass (B, H/64, W/64) from the window state machine (rsm.py); None codes every window (GVC1D).
        """
        b, _, h, w = x.shape
        grid = (h // 256, w // 256)
        n = b * grid[0] * grid[1]
        dev = x.device
        rt = self.runtime
        marks = [] if (self.timing and x.is_cuda) else None

        def mark(name):
            if marks is not None:             # on the stream of x's device (not the current device)
                with torch.cuda.device(dev):
                    e = torch.cuda.Event(enable_timing=True)
                    e.record(torch.cuda.current_stream(dev))
                marks.append((name, e))

        mark("start")
        if rsm is None:
            rsm = {}
        modes = rsm.get("modes")
        modes = torch.full((n,), UPDATE, dtype=torch.long, device=dev) if modes is None else modes.to(dev).long()
        if dpb.get("ref_feature") is None:   # nothing decoded yet to copy from: every window is coded
            modes = torch.where((modes == SKIP) | (modes == REVERT), torch.full_like(modes, UPDATE), modes)
        qp = torch.as_tensor(qp, device=dev).long().reshape(-1)
        if qp.numel() == b and b != n:                 # one quality index per sample
            qp = qp.repeat_interleave(n // b)
        assert qp.numel() == n, f"qp: {qp.numel()} values for {n} windows"
        qp = qp.to(dev)
        refresh, skip, revert = modes == REFRESH, modes == SKIP, modes == REVERT
        coded = (modes == UPDATE) | refresh
        active = torch.nonzero(coded, as_tuple=True)[0]                     # windows whose tokens are sent

        mem = None
        if self.use_memory:
            mem = dpb.get("long_term_mem")
            if mem is None:
                mem = self.init_memory(n, dev, next(self.long_term_feature_extractor.parameters()).dtype)
        snaps = self._store_snapshots(dpb, rsm["snap_create"].to(dev), n, mem) if "snap_create" in rsm \
            else dpb.get("snapshots")
        # reconstructed pixels are kept in fp32 (bit-exact Skip / Revert copies); the networks see
        # a copy of the reference frame in their own dtype (the adaptors may clamp it in place)
        ctx = dict(dpb, ref_frame=dpb["ref_frame"].to(self.latent_tokens.dtype, copy=True))
        rev_idx = torch.nonzero(revert, as_tuple=True)[0]
        rev_slot = rsm["revert_slot"].to(dev).long()[rev_idx] if rev_idx.numel() else None
        tok_cache = dpb.get("tokens")              # last decoded tokens of every window (decoder side)
        if tok_cache is not None and rev_idx.numel() and snaps is not None and "tok" in snaps:
            tok_cache = tok_cache.index_copy(0, rev_idx, snaps["tok"][rev_idx, rev_slot].to(tok_cache.dtype))

        if self.use_memory:
            if rt["refresh_reset"] and bool(refresh.any()):
                mem = torch.where(refresh[:, None, None], self.init_memory(n, dev, mem.dtype), mem)
            if rev_idx.numel() and snaps is not None and "mem" in snaps:
                mem = mem.index_copy(0, rev_idx, snaps["mem"][rev_idx, rev_slot].to(mem.dtype))

        bits = x.new_zeros(n, dtype=torch.float32)
        bits_y = bits.clone()
        bits_z = bits.clone()
        feat_new = None
        if active.numel():
            win_hw = torch.stack((active // (grid[0] * grid[1]), (active % (grid[0] * grid[1])) // grid[1],
                                  active % grid[1]), dim=1)
            fcls = rsm.get("fclass")
            want_f = self.encoder.fmap_embed is not None or self.decoder.fmap_embed is not None
            fp = self.patch_classes(fcls.to(dev), active, grid) if (fcls is not None and want_f) else None
            sel = {"active": active, "grid": grid, "win_hw": win_hw,
                   "refresh": refresh[active] if (self.opts["use_null_ctx"] and rt["null_ctx"]) else None,
                   "fpatch": fp}
            lt = None
            if self.use_memory:
                lt = self.long_term_feature_extractor.get_feature(mem[active])
                if mem.dtype != self.latent_tokens.dtype:
                    lt = lt.to(self.latent_tokens.dtype)
            mark("memory_read")
            z = self.encoder(x, self.latent_tokens, ctx, fa_idx, lt, sel)
            z, _ = torch.chunk(z, 2, dim=1)
            mark("encoder")
            # the bottleneck (entropy model, token reconstruction) works per coded window
            feat = self.codec_model.context_features(ctx, fa_idx, grid, active, sel["refresh"])
            tok_c, by, bz = self.codec_model.code_windows(z, feat, lt, qp[active])
            mark("bottleneck")
            bits_y = bits_y.index_copy(0, active, by.float())
            bits_z = bits_z.index_copy(0, active, bz.float())
            bits = bits_y + bits_z
            if self.use_memory:
                upd = self.long_term_feature_extractor.update_feature(
                    tok_c.squeeze(2).permute(0, 2, 1).to(mem.dtype), mem[active],
                    rsm["mem_type"].to(dev).long()[active] if "mem_type" in rsm else None,
                    rsm["mem_age"].to(dev).long()[active] if "mem_age" in rsm else None)
                mem = mem.index_copy(0, active, upd.to(mem.dtype))
            mark("memory_update")
            feat_new = self.decoder(tok_c, ctx, fa_idx, lt, sel)
            mark("decoder")
            if tok_cache is None:
                tok_cache = tok_c.new_zeros((n,) + tok_c.shape[1:])
            tok_cache = tok_cache.index_copy(0, active, tok_c.to(tok_cache.dtype))

        # compose decoder features (next short-term context) and pixels
        if dpb.get("ref_feature") is not None:
            feat_w = to_windows(dpb["ref_feature"], 16)
        else:
            feat_w = feat_new.new_zeros((n,) + feat_new.shape[1:])
        if feat_new is not None:
            feat_w = feat_w.index_copy(0, active, feat_new.to(feat_w.dtype))
        if rev_idx.numel():
            feat_w = feat_w.index_copy(0, rev_idx, snaps["feat"][rev_idx, rev_slot].to(feat_w.dtype))
        feature = from_windows(feat_w, grid, b)
        if active.numel():
            pix_w = to_windows(self.decoder.render(feature).float(), 256)
            hold = skip | revert
            if bool(hold.any()):
                prev = to_windows(dpb["ref_frame"].float(), 256)
                pix_w = torch.where(hold[:, None, None, None], prev, pix_w)
        else:
            pix_w = to_windows(dpb["ref_frame"].float(), 256)
        if rev_idx.numel():
            pix_w = pix_w.index_copy(0, rev_idx, snaps["pix"][rev_idx, rev_slot].to(pix_w.dtype))
        x_hat = from_windows(pix_w, grid, b)
        bpp = bits.view(b, -1).sum(dim=1) / (h * w)
        out = {
            "x_hat": x_hat, "bpp": bpp, "bits_win": bits, "bits_y_win": bits_y, "bits_z_win": bits_z,
            "modes": modes, "active": active, "qp_win": qp,
            "dpb": {"ref_frame": x_hat, "ref_feature": feature, "long_term_mem": mem, "snapshots": snaps,
                    "tokens": tok_cache},
        }
        if feat_new is not None and self.use_memory:   # mean gate of this frame's memory update (if gated)
            out["gate_mean"] = getattr(self.long_term_feature_extractor, "last_gate", None)
        if marks is not None:
            mark("compose")                   # output head + Skip / Revert copies
            torch.cuda.synchronize(dev)
            times = {name: prev.elapsed_time(e) for (_, prev), (name, e) in zip(marks[:-1], marks[1:])}
            times["total"] = marks[0][1].elapsed_time(marks[-1][1])
            # the decoder side runs everything but the encoder transformer
            times["decode"] = times["total"] - times.get("encoder", 0.0)
            out["times_ms"] = times
        return out

    def forward(self, frames, dpb, qps, fa_idx, rsms, anchor=False):
        """Code consecutive P-frames in one call (single DDP forward per optimisation step).

        frames: (B, G, 3, H, W); qps / fa_idx / rsms: lists with one entry per frame. anchor=True also
        returns outs[-1]["anchor"], 0 x every trainable parameter, to be added to the loss: when only a
        few new parameters train (stage S2a), a group can reach none of them (no Refresh window, no window
        coded twice) and would have no gradient at all. It has to be computed here, inside the DDP
        forward: a zero term added after the call leaves DDP waiting.
        """
        outs = []
        for g in range(frames.shape[1]):
            out = self.forward_one_frame(frames[:, g], dpb, qps[g], fa_idx[g], rsms[g])
            dpb = out["dpb"]
            outs.append(out)
        if anchor:
            outs[-1]["anchor"] = sum(p.reshape(-1)[0] for p in self.parameters() if p.requires_grad) * 0.0
        return outs


def detach_dpb(dpb):
    out = {}
    for k, v in dpb.items():
        if torch.is_tensor(v):
            out[k] = v.detach()
        elif isinstance(v, dict):
            out[k] = {kk: (vv.detach() if torch.is_tensor(vv) else vv) for kk, vv in v.items()}
        else:
            out[k] = v
    return out

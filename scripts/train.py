"""Train GSCV, one stage per call, in passes over the training entries.

    S0    GVC1D fine-tuned on SCVCD-5K with the GSCV data pipeline, every GSCV part off, full model
    S1a   SFA: only the class embeddings trained, region losses and quality allocation on
    S1b   SFA: full model
    S2a   IACM: only its new parameters trained (null contexts, state embedding, memory gate), window states on
    S2b   IACM: full model
    S3_disc  a GAN run from S2b (cascade groups of 3) whose discriminator initialises S3
    S3    full model with the texture discriminator (GAN), from S2b

The stages, their passes and switches are in train.stages of the configuration (configs/gscv.yaml); scripts/train.sh
runs them in order, each one starting from the final weights of the previous one (S3_disc and S3 both from S2b).

A pass uses every training entry once (data.train_entries), sharded over the GPUs (the same whole number of batches
on every GPU; the few entries left over change from pass to pass), with GVC1D's frame selection and crop padding
(modeling/gscv/data.py) and a learning rate that is constant for the pass (no warm-up, no decay, the same rate for
the new parameters and for the discriminator). A pass lists its frames T, its rate, the P-frames per optimiser step
(group: 1 = GVC1D's non-cascaded short sequences, otherwise its cascade group) and the batch per GPU. Before the
first pass and after every pass the validation clips (data.test_set) are evaluated as GVC1D's eval_reconstruction:
the first T frames, centre crop with symmetric zero padding, quality indices train.eval_qps, bpp / LPIPS / DISTS /
PSNR of the P-frames (eval.jsonl). The weights are saved after every pass (pass<k>_f<T>.pt); final.pt, a hard link to
the last pass, initialises the next stage (no checkpoint is picked by validation scores). ckpt.pt (model,
optimisers, position) resumes a stage and is removed once final.pt exists; done.json records the finished passes.
Passes appended to a finished stage are trained from its final.pt with a fresh generator optimiser; a GAN stage also
keeps its discriminator (disc_final.pt). The samples (frames, crop, events, random class map) and their quality
indices depend on (seed, stage, pass, entry), so a resumed or extended run draws the same samples as an
uninterrupted one. start.json records the weights a stage started from. --status prints new / resume / extend /
done / stale (the init weights changed since the stage started) / mismatch (other passes than the finished ones).

Launch (4 GPUs, from the repository root):
    accelerate launch --num_processes 4 --mixed_precision bf16 scripts/train.py --config configs/gscv.yaml --stage S1a
Options after the flags are OmegaConf dot-list overrides, e.g. data.num_workers=4.
"""
import argparse
import json
import math
import os
import shutil
import sys
import time
import zlib
from collections import defaultdict
from datetime import timedelta

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from accelerate import Accelerator  # noqa: E402
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs, set_seed  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from modeling.gscv.data import EntryDataset, TestClips, collate, load_description  # noqa: E402
from modeling.gscv.iframe import load_iframe_codec  # noqa: E402
from modeling.gscv.losses import GSCVLoss  # noqa: E402
from modeling.gscv.model import GSCV, detach_dpb  # noqa: E402
from modeling.gscv.rsm import REFRESH, REVERT, SKIP, UPDATE, RSMConfig  # noqa: E402

INDEX_MAP = [0, 1, 0, 2, 0, 2, 0, 2]          # frame-type cycle of test.py (rate_gop_size = 8)
GVC1D_PERC_SCALE = 1.1                         # GVC1D's PerceptualLoss returns (LPIPS + 0.1 ConvNeXt) / 1.1: its
                                               # adaptive GAN weight sees that value (the loss multiplies by 1.1)
DIST_WEIGHTS = [1.0, 2.0, 1.5]                 # GVC1D get_rd_info (rd_t) per frame type: 2.0 for type 1,
                                               # i.e. P-frames 1, 9, 17, 25 (no extra 2.0 per cascade group)
RSM_KEYS = ("modes", "revert_slot", "snap_create", "mem_type", "mem_age")
# parameters trained alone in the module-initialisation stages (as GVC1D's training_module):
TRAINABLE = {
    "content_new": ("fmap_embed",),                                              # S1a: class embeddings
    "change_new": ("null_cond", "null_ctx", "type_embed", "age_embed",           # S2a: null contexts, state
                   "long_term_feature_extractor.gate."),                         # embedding, memory gate
}
CKPT_FORMAT = "passes-v1"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/gscv.yaml")
    p.add_argument("--stage", required=True, help="S0, S1a, S1b, S2a, S2b, S3_disc, S3 or any stage of train.stages")
    p.add_argument("--out", default="", help="default: <train.output_root>/<stage>")
    p.add_argument("--init", default="", help="override the stage's init weights")
    p.add_argument("--no_resume", action="store_true")
    p.add_argument("--stop_after", type=int, default=0,
                   help="tests: stop after this many batches of this run (resume point written), 0 = never")
    p.add_argument("--status", action="store_true", help="print the state of the stage's directory and exit")
    p.add_argument("overrides", nargs="*")
    return p.parse_args()


def load_config(args):
    """The file, then the command-line overrides, then the stage's own model.gscv switches and run-time
    switches (model.runtime) -- except switches given on the command line (model.gscv.*, model.runtime.*), which
    win."""
    cfg = OmegaConf.load(args.config)
    cli = OmegaConf.from_dotlist(args.overrides) if args.overrides else OmegaConf.create()
    cfg = OmegaConf.merge(cfg, cli)
    assert args.stage in cfg.train.stages, f"unknown stage {args.stage}: {list(cfg.train.stages)}"
    st = cfg.train.stages[args.stage]
    cli_gscv = OmegaConf.select(cli, "model.gscv", default=None)
    cfg.model.gscv = OmegaConf.merge(cfg.model.gscv, st.get("gscv", {}), cli_gscv if cli_gscv is not None else {})
    cli_rt = OmegaConf.select(cli, "model.runtime", default=None)
    cfg.model.runtime = OmegaConf.merge(cfg.model.get("runtime", None) or {}, st.get("runtime", None) or {},
                                        cli_rt if cli_rt is not None else {})
    return cfg, st


def stage_passes(st, default_batch=1):
    """[{frames, lr, group, batch}] of the stage; batch: the pass's, else the stage's batch_per_gpu, else
    default_batch (train.batch_per_gpu)."""
    out = []
    for p in st.passes:
        p = OmegaConf.to_container(p, resolve=True) if OmegaConf.is_config(p) else p
        if isinstance(p, (list, tuple)):
            p = {"frames": p[0], "lr": p[1]}
        if not isinstance(p, dict) or not {"frames", "lr"} <= set(p):   # YAML reads {frames:2} as key "frames:2"
            raise ValueError(f"pass {p!r}: write [frames, lr] or {{frames: T, lr: x, group: g, batch: b}} "
                             "with a space after every colon (also in command-line overrides)")
        out.append({"frames": int(p["frames"]), "lr": float(p["lr"]),
                    "group": int(p.get("group", st.get("cascade_group", 4))),
                    "batch": int(p.get("batch", st.get("batch_per_gpu", default_batch)))})
    assert out and all(q["frames"] >= 2 and q["group"] >= 1 and q["batch"] >= 1 for q in out), out
    return out


# ----------------------------------------------------------------------------- model / optimiser

def select_trainable(model, mode):
    """all, or only the parameters GSCV adds for one axis (GVC1D trains a new module alone first)."""
    if mode == "all":
        model.requires_grad_(True)
        return
    keys = TRAINABLE[mode]
    for n, p in model.named_parameters():
        p.requires_grad = any(k in n for k in keys)
    assert any(p.requires_grad for p in model.parameters()), f"trainable={mode}: the stage has no such parameters"


def param_groups(model, wd):
    """AdamW groups as GVC1D: no weight decay for gains, biases, embeddings and tokens."""
    no_wd = (lambda n, p: p.ndim < 2 or "ln" in n or "bias" in n or "latent_tokens" in n or "mask_token" in n
             or "embedding" in n or "norm" in n or "gamma" in n or "embed" in n or "null" in n)
    decay, plain = [], []
    for n, p in model.named_parameters():
        if p.requires_grad:
            (plain if no_wd(n, p) else decay).append(p)
    return [g for g in ({"params": decay, "weight_decay": wd}, {"params": plain, "weight_decay": 0.0}) if g["params"]]


# ----------------------------------------------------------------------------- data

def stage_rsm(cfg, st):
    rcfg = RSMConfig.from_dict(OmegaConf.to_container(cfg.rsm, resolve=True))
    if not st.states:                          # S0 / S1a / S1b: every window coded (as GVC1D)
        rcfg.use_skip = rcfg.use_refresh = rcfg.use_revert = False
        rcfg.k_snap = 0
    return rcfg


def stage_id(stage):
    """A stable number per stage name: the stages draw different samples and quality indices."""
    return zlib.crc32(stage.encode()) & 0x7FFFFFFF


def pass_order(n, seed, p, world, rank, batch, sid=0):
    """Entries of pass p for one GPU: one permutation shared by all GPUs, the same whole number of
    batches per GPU (DDP needs equal step counts); the remainder is left out of this pass."""
    perm = np.random.default_rng((int(seed), int(sid), int(p), 1)).permutation(n)   # 1: not an item key (data.py)
    per = n // (world * batch) * batch
    return perm[:per * world][rank::world]


def make_loader(cfg, st, entries, order, ps, seed, p, sid=0):
    aug = OmegaConf.to_container(cfg.data.aug, resolve=True)
    if not st.states:                          # no interaction events before IACM
        aug = {k: 0.0 for k in aug}
    ds = EntryDataset(cfg.data.root, entries, order, frames=ps["frames"], crop=int(st.get("crop", cfg.data.crop)),
                      rsm_cfg=stage_rsm(cfg, st), aug=aug, bias_crop=float(cfg.data.bias_crop),
                      rand_f=float(st.get("rand_f", 0.0)), cascade=ps["group"], seed=seed, stage_id=sid,
                      pass_index=p, max_gap=int(cfg.data.get("max_gap", 6)))
    nw = int(cfg.data.num_workers)
    return DataLoader(ds, batch_size=ps["batch"], shuffle=False, num_workers=nw, collate_fn=collate,
                      pin_memory=True, drop_last=True, prefetch_factor=4 if nw > 0 else None)


def ensure_coded(batch, ts):
    """Every sample codes at least one window in the group (the dataset already forces one; this is
    a safety net: a sample without a coded window has no gradient, which DDP cannot handle)."""
    m = batch["modes"]
    sub_ = m[:, ts[0] - 1:ts[-1]]
    none = ~((sub_ == UPDATE) | (sub_ == REFRESH)).flatten(1).any(dim=1)
    if bool(none.any()):
        m[none, ts[-1] - 1, 0] = UPDATE
    return int(none.sum())


def group_inputs(batch, ts, qp_all, device):
    rsms, qps, fas = [], [], []
    for t in ts:
        r = {k: batch[k][:, t - 1].reshape(-1) for k in RSM_KEYS}
        r["fclass"] = batch["fclass"][:, t]
        rsms.append({k: v.to(device, non_blocking=True) for k, v in r.items()})
        qps.append(qp_all[:, t - 1].reshape(-1).to(device))
        fas.append(INDEX_MAP[t % 8])
    return rsms, qps, fas


# ----------------------------------------------------------------------------- evaluation

def gvc1d_video_metric():
    """VideoMetric of GVC1D's evaluator/eval_video.py (used by eval_reconstruction), loaded from its file:
    the evaluator package's __init__ also imports the FID code (torch_fidelity), which is not needed."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("gvc1d_eval_video", os.path.join(REPO, "evaluator", "eval_video.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.VideoMetric()


@torch.no_grad()
def evaluate(core, icodec, metric, test_dir, cfg, st, frames, accelerator, qps):
    """GVC1D's eval_reconstruction on the validation clips: the first `frames` frames, centre crop with
    symmetric zero padding (TestClips), every P-frame coded as at test time (the stage's window states, per-window
    quality offsets if qp_alloc), quality indices `qps`; means over all P-frames of bpp (tokens), bpp_side (side
    information), LPIPS (Alex), DISTS and PSNR. The clips are shared out over the GPUs and the sums reduced."""
    device = accelerator.device
    ds = TestClips(test_dir, frames, crop=int(cfg.train.get("eval_crop", 512)), rsm_cfg=stage_rsm(cfg, st),
                   states=bool(st.states))
    core.eval()
    keys = ("bpp", "bpp_side", "lpips", "dists", "psnr")
    acc = torch.zeros(len(qps), len(keys) + 1, device=device, dtype=torch.float64)
    for i in range(accelerator.process_index, len(ds), accelerator.num_processes):
        it = ds[i]
        x = it["frames"][None].to(device).float().div_(255.0)              # (1, T, 3, c, c)
        ref = icodec.forward_one_frame(x[:, 0]).float()
        b1 = {k: v[None] for k, v in it.items() if torch.is_tensor(v)}
        npx = float(x.shape[-1] * x.shape[-2])
        for qi, qp in enumerate(qps):
            qp_off = b1["qp_offset"] if st.qp_alloc else torch.zeros_like(b1["qp_offset"])
            qp_all = (qp + qp_off).clamp(0, 7)
            dpb = {"ref_frame": ref, "ref_feature": None, "long_term_mem": None, "snapshots": None}
            recs, bpp = [], 0.0
            for t in range(1, x.shape[1]):
                rsms, qw, fas = group_inputs(b1, [t], qp_all, device)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
                    o = core.forward_one_frame(x[:, t], dpb, qw[0], fas[0], rsms[0])
                recs.append(o["x_hat"].float().clamp(0.0, 1.0))
                bpp += float(o["bpp"].float().mean())
                dpb = o["dpb"]
            m = metric(x[:, 1:], torch.stack(recs, dim=1), is_01=True)
            n = x.shape[1] - 1
            acc[qi] += torch.tensor([bpp, float(b1["side_bits"].sum()) / npx, float(m["lpips"].sum()),
                                     float(m["dists"].sum()), float(m["psnr"].clamp(max=100).sum()), n],
                                    device=device, dtype=torch.float64)
    acc = accelerator.reduce(acc, reduction="sum")
    core.train()
    out = {}
    for qi, qp in enumerate(qps):
        n = max(float(acc[qi, -1]), 1.0)
        out[f"qp{qp}"] = {k: float(acc[qi, j]) / n for j, k in enumerate(keys)}
    return out


# ----------------------------------------------------------------------------- checkpoints

def save_model(path, core):
    tmp = path + ".tmp"
    torch.save(core.state_dict(), tmp)
    os.replace(tmp, path)


def link_or_copy(src, dst):
    """dst becomes a second name of src (a hard link, no extra space; a copy where links fail)."""
    tmp = dst + ".tmp"
    if os.path.lexists(tmp):
        os.remove(tmp)
    try:
        os.link(src, tmp)
    except OSError:
        shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def save_resume(path, accelerator, model, optimizer, disc, disc_opt, loss_fn, cfg, info):
    """ckpt.pt: everything needed to continue the stage at the given position (info)."""
    if not accelerator.is_main_process:
        return
    state = {"format": CKPT_FORMAT, "model": accelerator.unwrap_model(model).state_dict(),
             "optimizer": optimizer.state_dict(), "config": OmegaConf.to_container(cfg, resolve=True),
             "world": accelerator.num_processes, "gscv": OmegaConf.to_container(cfg.model.gscv, resolve=True),
             "runtime": OmegaConf.to_container(cfg.model.runtime, resolve=True), **info}
    if disc is not None:
        state["disc"] = accelerator.unwrap_model(disc).state_dict()
        state["disc_opt"] = disc_opt.state_dict()
        state["lecam"] = (loss_fn.ema_real.detach().cpu(), loss_fn.ema_fake.detach().cpu())
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def read_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def read_done(out_dir):
    return read_json(os.path.join(out_dir, "done.json"))


def file_id(path):
    """Size and modification time of a weights file: the final.pt of an upstream stage is a hard
    link that moves on when that stage is trained again or extended."""
    info = os.stat(path)
    return f"{info.st_size}:{int(info.st_mtime)}"


def other_init(start, init):
    """The init weights path recorded in start.json, if it is not the current one (another --init or
    stage default than the stage started with), else None."""
    rec = (start or {}).get("init")
    if rec and init and os.path.realpath(rec) != os.path.realpath(init):
        return rec
    return None


def stage_status(out_dir, passes, init, use_ckpt=True):
    """resume (ckpt.pt), new, missing_init (new, but its init weights do not exist yet: train the
    previous stage first), done, extend (passes appended to the finished ones), stale (the init
    weights are not the ones the stage started from -- start.json records their path and id; without
    it: older than final.pt -- because an upstream stage was trained again or extended, or another
    init path is given; checked before resuming too) or mismatch."""
    final = os.path.join(out_dir, "final.pt")
    start = read_json(os.path.join(out_dir, "start.json"))
    have_init = bool(init) and os.path.exists(init)
    if start is not None and start.get("init_id"):
        changed = other_init(start, init) is not None or (have_init and start["init_id"] != file_id(init))
    else:
        changed = have_init and os.path.exists(final) and os.path.getmtime(init) > os.path.getmtime(final)
    if use_ckpt and os.path.exists(os.path.join(out_dir, "ckpt.pt")):
        return "stale" if changed else "resume"
    if not os.path.exists(final):
        return "new" if have_init else "missing_init"
    if changed:
        return "stale"
    done = read_done(out_dir)
    if done is None or done["passes"] == passes:
        return "done"
    if len(passes) > len(done["passes"]) and passes[:len(done["passes"])] == done["passes"]:
        return "extend"
    return "mismatch"


# ----------------------------------------------------------------------------- training

def main():
    args = parse_args()
    os.chdir(REPO)
    cfg, st = load_config(args)
    tc = cfg.train
    passes = stage_passes(st, int(tc.get("batch_per_gpu", 1)))
    out_dir = args.out or os.path.join(tc.output_root, args.stage)
    init = args.init or str(st.init)
    status = stage_status(out_dir, passes, init, use_ckpt=not args.no_resume)
    if args.status:
        print(status)
        return
    if status == "missing_init":
        raise SystemExit(f"{args.stage}: its init weights {init} do not exist: train the previous stage first "
                         f"(or give --init)")
    if status in ("stale", "mismatch"):
        if status == "stale":
            rec = other_init(read_json(os.path.join(out_dir, "start.json")), init)
            why = (f"it started from {rec}, now the init is {init}" if rec else
                   f"{init} changed since the stage started (an upstream stage was trained again or extended)")
        else:
            why = "it finished other passes (done.json)"
        raise RuntimeError(f"{out_dir} is {status}: {why}; move the directory away first")
    if status == "done":
        print(f"{out_dir} is finished (final.pt, done.json): nothing to do")
        return
    os.makedirs(out_dir, exist_ok=True)
    sid = stage_id(args.stage)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    accelerator = Accelerator(mixed_precision="bf16",
                              kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True),
                                               InitProcessGroupKwargs(timeout=timedelta(hours=2))])
    device = accelerator.device
    world, rank = accelerator.num_processes, accelerator.process_index
    seed = int(tc.seed)
    log_file = open(os.path.join(out_dir, "log.txt"), "a") if accelerator.is_main_process else None

    def log(msg):
        if accelerator.is_main_process:
            print(msg, flush=True)
            log_file.write(msg + "\n")
            log_file.flush()

    if status == "new" and accelerator.is_main_process:
        stamp = time.strftime(".old_%Y%m%d_%H%M%S")
        for name in ("eval.jsonl", "ckpt.pt"):         # a start that stopped before its first resume point, or
            old = os.path.join(out_dir, name)          # a --no_resume restart: kept aside, never resumed later
            if os.path.exists(old):
                os.replace(old, old + stamp)
                log(f"moved {old} to {old + stamp}")
        with open(os.path.join(out_dir, "start.json"), "w") as f:
            json.dump({"stage": args.stage, "init": init, "init_id": file_id(init) if os.path.exists(init) else None,
                       "time": time.time()}, f, indent=1)

    if accelerator.is_main_process:
        OmegaConf.save(cfg, os.path.join(out_dir, "config.yaml"))

    # ---------------- model
    set_seed((seed + sid) % 2 ** 32)                   # new parameters and the discriminator: reproducible
    model = GSCV(cfg)
    ckpt_stride = int(tc.get("ckpt_stride", 1))      # gradient checkpointing: every n-th layer (speed vs memory)
    model.encoder.ckpt_stride = model.decoder.ckpt_stride = ckpt_stride
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    resume = None
    start_pass = skip = updates = 0
    if status == "resume":
        resume = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if resume.get("format") != CKPT_FORMAT:
            raise RuntimeError(f"{ckpt_path} is not a {CKPT_FORMAT} checkpoint (an older training scheme): "
                               f"move it away or use another --out")
        if resume["passes"] != passes:
            raise RuntimeError(f"{ckpt_path} was written for other passes: {resume['passes']} vs {passes}")
        for key, now in (("stage", args.stage), ("world", world),
                         ("gscv", OmegaConf.to_container(cfg.model.gscv, resolve=True)),
                         ("runtime", OmegaConf.to_container(cfg.model.runtime, resolve=True))):
            if key in resume and resume[key] != now:      # the pass order depends on the number of GPUs
                raise RuntimeError(f"{ckpt_path} was written with {key}={resume[key]}, now {now}")
        old_cfg, now_cfg = resume.get("config") or {}, OmegaConf.to_container(cfg, resolve=True)
        skip_keys = ("passes", "init")
        for name, a, b in ((f"train.stages.{args.stage}",
                            {k: v for k, v in old_cfg.get("train", {}).get("stages", {}).get(args.stage, {}).items()
                             if k not in skip_keys} or None,
                            {k: v for k, v in now_cfg["train"]["stages"][args.stage].items() if k not in skip_keys}),
                           ("losses", old_cfg.get("losses"), now_cfg.get("losses")),
                           ("rsm", old_cfg.get("rsm"), now_cfg.get("rsm")),
                           ("data.aug", old_cfg.get("data", {}).get("aug"), now_cfg["data"].get("aug"))):
            if a is not None and a != b:               # the rest of the stage must not change under a resume
                raise RuntimeError(f"{ckpt_path} was written with other {name} settings: {a} vs {b}")
        model.load_state_dict(resume["model"], strict=True)
        start_pass, skip, updates = int(resume["pass"]), int(resume["batches_done"]), int(resume["updates"])
        log(f"resumed {ckpt_path}: pass {start_pass + 1}, batch {skip}, {updates} parameter updates")
    elif status == "extend":
        done_info = read_done(out_dir)
        final_path = os.path.join(out_dir, "final.pt")
        missing, unexpected = model.load_weights(final_path)
        assert not missing and not unexpected, (missing, unexpected)
        start_pass, updates = len(done_info["passes"]), int(done_info["updates"])
        log(f"extending {out_dir}: passes {start_pass + 1}-{len(passes)} appended, from {final_path} "
            f"with a fresh optimiser (the resume checkpoint is removed when a stage ends)")
    else:
        missing, unexpected = model.load_weights(init)
        log(f"init from {init}: {len(missing)} missing {missing[:12]}, {len(unexpected)} unexpected {unexpected[:6]}")
    trainable = str(st.get("trainable", "all"))
    select_trainable(model, trainable)
    wd = float(tc.weight_decay)
    optimizer = torch.optim.AdamW(param_groups(model, wd), lr=passes[0]["lr"], betas=(0.9, 0.999))
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"stage {args.stage}: trainable={trainable} {n_train / 1e6:.3f}M / "
        f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params, gscv={OmegaConf.to_container(cfg.model.gscv)}")
    log(f"run-time switches: {model.runtime}")
    log(f"gradient checkpointing of every {ckpt_stride}. encoder / decoder layer" if ckpt_stride > 0
        else "no gradient checkpointing")
    log("passes: " + ", ".join(f"{q['frames']}f@{q['lr']:g}/g{q['group']}/b{q['batch']}" for q in passes))

    loss_cfg = OmegaConf.to_container(cfg.losses, resolve=True)
    if st.get("losses"):                               # the stage's own loss weights (S1a)
        loss_cfg.update(OmegaConf.to_container(st.losses, resolve=True))
    loss_cfg["region"] = bool(st.region)
    loss_fn = GSCVLoss(loss_cfg).to(device).eval()
    disc, disc_opt, disc_module = None, None, None
    if st.gan:
        from modeling.gscv.losses import PairDiscriminator
        disc = PairDiscriminator()
        disc_opt = torch.optim.AdamW(param_groups(disc, wd), lr=passes[0]["lr"], betas=(0.9, 0.999))
        if resume is not None and "disc" in resume:
            disc.load_state_dict(resume["disc"])
        elif status == "extend":                       # appended passes go on with the stage's discriminator
            dpath = os.path.join(out_dir, "disc_final.pt")
            if os.path.exists(dpath):
                dstate = torch.load(dpath, map_location="cpu", weights_only=False)
                disc.load_state_dict(dstate["disc"])
                disc_opt.load_state_dict(dstate["disc_opt"])
                loss_fn.ema_real.copy_(dstate["lecam"][0])
                loss_fn.ema_fake.copy_(dstate["lecam"][1])
                log(f"discriminator, its optimiser and the LeCam averages from {dpath}")
            else:
                log(f"WARNING: no {dpath}: the discriminator starts afresh")
        elif st.get("disc_init"):                      # a new GAN stage with a trained discriminator: the
            dpath = str(st.disc_init)                  # generator never follows an untrained one
            if not os.path.exists(dpath):
                raise FileNotFoundError(f"train.stages.{args.stage}.disc_init: no {dpath} (empty: start afresh)")
            dstate = torch.load(dpath, map_location="cpu", weights_only=False)
            disc.load_state_dict(dstate["disc"])
            disc_opt.load_state_dict(dstate["disc_opt"])
            loss_fn.ema_real.copy_(dstate["lecam"][0])
            loss_fn.ema_fake.copy_(dstate["lecam"][1])
            log(f"discriminator, its optimiser and the LeCam averages initialised from {dpath}")
    if resume is not None:
        optimizer.load_state_dict(resume["optimizer"])
        if disc is not None and "disc_opt" in resume:
            disc_opt.load_state_dict(resume["disc_opt"])
        if "lecam" in resume:
            loss_fn.ema_real.copy_(resume["lecam"][0])
            loss_fn.ema_fake.copy_(resume["lecam"][1])
        resume = None                                  # ~6 GB of host memory per process
    if disc is not None:
        model, optimizer, disc, disc_opt = accelerator.prepare(model, optimizer, disc, disc_opt)
        disc_module = accelerator.unwrap_model(disc)
    else:
        model, optimizer = accelerator.prepare(model, optimizer)
    core = accelerator.unwrap_model(model)
    has_gate = getattr(getattr(core, "long_term_feature_extractor", None), "gate", None) is not None
    train_params = [q for q in model.parameters() if q.requires_grad]
    anchor = trainable != "all"                      # module-only stages: see GSCV.forward
    icodec = load_iframe_codec(cfg.I_frame_codec.ckpt, device, cfg.I_frame_codec.config)
    metric = gvc1d_video_metric().to(device).eval()  # GVC1D's own LPIPS (Alex) / DISTS evaluator

    l0, l1 = (float(v) for v in cfg.losses.lmbdas)
    lam_table = torch.tensor(np.exp(np.linspace(math.log(l0), math.log(l1), 8)), dtype=torch.float32, device=device)
    gan_scalers = (lam_table - l0) / l1                    # delta(lambda_q), as GVC1D
    lambda_tex = float(cfg.losses.get("lambda_tex", 1.0))

    # ---------------- data
    with open(cfg.data.train_entries) as f:
        entries = json.load(f)
    test_dir = cfg.data.test_set
    test_paths = {s["path"] for s in load_description(test_dir)[0]}
    leak = sorted({e["clip"] for e in entries} & test_paths)
    assert not leak, f"validation clips among the training entries: {leak[:5]}"
    log(f"{len(entries)} training entries from {len({e['clip'] for e in entries})} clips; "
        f"{len(test_paths)} validation clips in {test_dir}")

    eval_qps = [int(q) for q in tc.get("eval_qps", [0, 7])]
    eval_file = os.path.join(out_dir, "eval.jsonl")

    def run_eval(p_done, frames, part=None):
        """part: "k/n" for an evaluation within pass p_done + 1 (train.evals_per_pass), else after pass p_done."""
        res = evaluate(core, icodec, metric, test_dir, cfg, st, frames, accelerator, eval_qps)
        msg = " | ".join(f"{q}: " + " ".join(f"{k}={v:.5g}" for k, v in r.items()) for q, r in res.items())
        where = f"after pass {p_done}/{len(passes)}" if part is None else f"at {part} of pass {p_done + 1}/{len(passes)}"
        log(f"[{args.stage}] eval {where} ({frames} frames): {msg}")
        if accelerator.is_main_process:
            with open(eval_file, "a") as f:
                rec = {"stage": args.stage, "pass": p_done, "frames": frames, "time": time.time(), "results": res}
                if part is not None:
                    rec["part"] = part
                f.write(json.dumps(rec) + "\n")

    set_seed((seed + sid + updates) % 2 ** 32, device_specific=True)   # torch RNG (e.g. quantisation noise)
    parts = int(tc.get("saves_per_pass", 1)) if trainable == "all" else 1
    evals = max(1, int(tc.get("evals_per_pass", 1)))      # > 1: also evaluate within a pass (an early signal)
    resume_every = int(tc.get("resume_every", 1000))
    log_every = int(tc.get("log_every", 50))
    if status == "new" and bool(tc.get("eval_before", True)):
        run_eval(0, passes[0]["frames"])
    model.train()
    loss_fn.eval()
    ran = 0                                            # batches of this run (--stop_after)

    for p in range(start_pass, len(passes)):
        ps = passes[p]
        T, lr, group, bsz = ps["frames"], ps["lr"], ps["group"], ps["batch"]
        for opt in (optimizer, disc_opt):            # the discriminator uses the generator's rate
            if opt is not None:
                for grp in opt.param_groups:
                    grp["lr"] = lr
        order = pass_order(len(entries), seed, p, world, rank, bsz, sid)
        n_batches = len(order) // bsz
        done = skip if p == start_pass else 0
        loader = make_loader(cfg, st, entries, order[done * bsz:], ps, seed, p, sid)
        marks = {round(n_batches * k / parts): k for k in range(1, parts)}
        eval_marks = {round(n_batches * k / evals): k for k in range(1, evals)}
        log(f"[{args.stage}] pass {p + 1}/{len(passes)}: {T} frames, lr {lr:g}, up to {min(group, T - 1)} P-frame(s) "
            f"per update, batch {bsz}/GPU x {world} GPUs, {n_batches} batches per GPU"
            + (f", from batch {done}" if done else ""))
        meter, n_meter, t_last = defaultdict(float), 0, time.time()
        for batch in loader:
            frames = batch["frames"].to(device, non_blocking=True).float().div_(255.0)    # (B, T, 3, H, W)
            b, t_all, _, h, w = frames.shape
            grid = (h // 256, w // 256)
            with torch.no_grad():
                ref = icodec.forward_one_frame(frames[:, 0]).float()
            dpb = {"ref_frame": ref, "ref_feature": None, "long_term_mem": None, "snapshots": None}
            qp_base = batch["qp_base"].long() if "qp_base" in batch else torch.randint(0, 8, (b,))
            qp_off = batch["qp_offset"] if st.qp_alloc else torch.zeros_like(batch["qp_offset"])
            qp_all = (qp_base[:, None, None] + qp_off).clamp(0, 7)                         # (B, T-1, nWin)
            p_idx = list(range(1, t_all))
            for g0 in range(0, len(p_idx), group):
                ts = p_idx[g0:g0 + group]
                if ensure_coded(batch, ts):
                    log(f"pass {p + 1} batch {done}: a sample without a coded window in frames {ts}; one window coded")
                rsms, qps, fas = group_inputs(batch, ts, qp_all, device)
                if disc is not None:
                    disc_module.requires_grad_(False)
                outs = model(frames[:, ts[0]:ts[-1] + 1], dpb, qps, fas, rsms, anchor=anchor)

                rate, rec, rec1 = 0.0, 0.0, 0.0
                gate_sum, gate_cnt = 0.0, 0.0
                comps, prevs_fake, prevs_real, texs, advws = [], [], [], [], []
                prev_rec = dpb["ref_frame"].detach().float()      # previous group: detached (as GVC1D)
                stats = defaultdict(float)
                with accelerator.autocast():
                    for k, (t, o) in enumerate(zip(ts, outs)):
                        x = frames[:, t]
                        lam_win = lam_table[qps[k]]
                        fcl = rsms[k]["fclass"] if st.region else None
                        terms, comp, tex = loss_fn.frame_terms(x, o["x_hat"], o["modes"], fcl, lam_win, grid,
                                                               delta_win=gan_scalers[qps[k]])
                        wd_t = DIST_WEIGHTS[fas[k]]       # rd_t of the frame type (test-time 8-frame cycle)
                        # lambda_q * D_t with rd_t on the pixel terms only (GVC1D's Codec_loss); r1: the
                        # terms the adversarial term competes with in the adaptive weight
                        rec = rec + (wd_t * terms["pix"] + terms["perc"]).mean()
                        r1 = (wd_t * terms["pix1"] + terms["perc1"] / GVC1D_PERC_SCALE).mean()
                        rate = rate + o["bpp"].float().mean()
                        if disc is not None:
                            x_prev = frames[:, t - 1]
                            rec1 = rec1 + r1
                            comps.append(comp)
                            prevs_fake.append(tex * prev_rec + (1.0 - tex) * x_prev)
                            prevs_real.append(x_prev)
                            texs.append(tex)
                            advws.append(terms["adv_w"])
                        prev_rec = o["x_hat"].float()   # within the group the pair keeps its gradient (GVC1D)
                        m = o["modes"]
                        stats["active"] += ((m == UPDATE) | (m == REFRESH)).float().mean()
                        stats["skip"] += (m == SKIP).float().mean()
                        stats["refresh"] += (m == REFRESH).float().mean()
                        stats["revert"] += (m == REVERT).float().mean()
                        stats["pix"] += terms["pix"].detach().mean()
                        stats["perc"] += terms["perc"].detach().mean()
                        if o.get("gate_mean") is not None:      # mean gate of the memory update
                            gate_sum += o["gate_mean"]
                            gate_cnt += 1.0
                    kf = float(len(ts))
                    loss = (rate + rec) / kf
                    d_weight = adv = torch.zeros((), device=device)
                    if disc is not None:
                        comp_all, prev_fake_all, tex_all = torch.cat(comps), torch.cat(prevs_fake), torch.cat(texs)
                        g, g1 = loss_fn.generator_adv(disc_module, comp_all, prev_fake_all, torch.cat(advws), tex_all)
                        g, g1 = g.view(len(ts), b), g1.view(len(ts), b)
                        scale = lambda_tex if st.region else 1.0      # no rd_t on the adversarial term (GVC1D)
                        adv = scale * g.mean(dim=1).sum()            # lambda*delta inside g (per window)
                        adv1 = scale * g1.mean(dim=1).sum()
                        d_weight = loss_fn.adaptive_weight(rec1, adv1, core.decoder.conv_out.weight) \
                            if loss_fn.adaptive_gan else torch.ones((), device=device)
                        loss = loss + loss_fn.gan_weight * d_weight * adv / kf
                if anchor:
                    loss = loss + outs[-1]["anchor"]

                accelerator.backward(loss)
                gn = accelerator.clip_grad_norm_(train_params, float(tc.max_grad_norm))
                g_ok = bool(torch.isfinite(torch.as_tensor(gn)))
                if anchor:      # parameters only the anchor reached (zero on every GPU) are not updated
                    for q in train_params:
                        if q.grad is not None and not bool(q.grad.any()):
                            q.grad = None
                    meter["nograd"] += float(all(q.grad is None for q in train_params))
                if g_ok:
                    optimizer.step()
                else:
                    log(f"pass {p + 1} batch {done}: non-finite grad norm, update skipped")
                optimizer.zero_grad(set_to_none=True)
                if disc is not None:
                    disc_module.requires_grad_(True)
                    real = torch.cat([frames[:, t] for t in ts])
                    with accelerator.autocast():
                        d_loss, dlog = loss_fn.discriminator_loss(disc, real, torch.cat(prevs_real), comp_all.detach(),
                                                                  prev_fake_all.detach(), tex_all)
                    accelerator.backward(d_loss)
                    dn = accelerator.clip_grad_norm_(disc.parameters(), float(tc.max_grad_norm))
                    if g_ok and bool(torch.isfinite(torch.as_tensor(dn))):   # as GVC1D: a bad group trains nothing
                        disc_opt.step()
                    else:
                        log(f"pass {p + 1} batch {done}: discriminator update skipped (non-finite gradients)")
                    disc_opt.zero_grad(set_to_none=True)
                    stats["d_loss"] += d_loss.detach() * kf
                    stats["d_real"] += dlog["d_real"] * kf
                    stats["d_fake"] += dlog["d_fake"] * kf
                    stats["d_weight"] += d_weight * kf
                    stats["adv"] += adv.detach()
                dpb = detach_dpb(outs[-1]["dpb"])
                del outs
                updates += 1
                n_meter += 1
                meter["loss"] += loss.detach()
                meter["bpp"] += rate.detach() / kf
                meter["gnorm"] += torch.as_tensor(gn, device=device).float()
                for k2, v in stats.items():
                    meter[k2] += v / kf
                if has_gate:              # every GPU logs the same keys (reduce); averaged over gated frames
                    meter["gate_sum"] += gate_sum
                    meter["gate_cnt"] += gate_cnt
            done += 1
            ran += 1
            if done % log_every == 0 or done == n_batches:
                keys = sorted(meter)
                vals = accelerator.reduce(torch.stack([torch.as_tensor(meter[k2], device=device).float()
                                                       for k2 in keys]) / max(n_meter, 1), reduction="mean")
                shown = dict(zip(keys, vals.tolist()))
                if "gate_sum" in shown:                         # memory gate: mean value, bias b_g, |W_g|
                    shown["gate"] = shown.pop("gate_sum") / max(shown.pop("gate_cnt"), 1e-12)
                if has_gate:
                    gate_mod = core.long_term_feature_extractor.gate
                    shown["gate_b"] = float(gate_mod.bias.detach().float().mean())
                    shown["gate_w"] = float(gate_mod.weight.detach().float().norm())
                dt = (time.time() - t_last) / max(n_meter, 1)
                mem = torch.cuda.max_memory_allocated(device) / 2 ** 30 if torch.cuda.is_available() else 0.0
                log(f"[{args.stage}] pass {p + 1}/{len(passes)} batch {done}/{n_batches} updates {updates} lr={lr:g} "
                    f"frames={T} " + " ".join(f"{k2}={v:.4g}" for k2, v in sorted(shown.items()))
                    + f" {dt:.2f}s/update mem={mem:.1f}G")
                meter.clear()
                n_meter, t_last = 0, time.time()
            if done in marks:                        # full-model stages: also within the pass
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    save_model(os.path.join(out_dir, f"pass{p + 1:02d}_f{T}_{marks[done]}of{parts}.pt"), core)
                    log(f"saved pass{p + 1:02d}_f{T}_{marks[done]}of{parts}.pt (batch {done})")
            if done in eval_marks:                   # evaluation within the pass (train.evals_per_pass)
                accelerator.wait_for_everyone()
                run_eval(p, T, part=f"{eval_marks[done]}/{evals}")
                accelerator.wait_for_everyone()
                model.train()
            if updates and done < n_batches and done % max(1, resume_every // max(1, math.ceil((T - 1) / group))) == 0:
                accelerator.wait_for_everyone()
                save_resume(ckpt_path, accelerator, model, optimizer, disc, disc_opt, loss_fn, cfg,
                            {"stage": args.stage, "passes": passes, "pass": p, "batches_done": done, "updates": updates})
            if done >= n_batches:
                break
            if args.stop_after and ran >= args.stop_after:
                accelerator.wait_for_everyone()
                save_resume(ckpt_path, accelerator, model, optimizer, disc, disc_opt, loss_fn, cfg,
                            {"stage": args.stage, "passes": passes, "pass": p, "batches_done": done, "updates": updates})
                log(f"stopped after {ran} batches (--stop_after): pass {p + 1}, batch {done}")
                accelerator.wait_for_everyone()
                accelerator.end_training()
                return
        # ---- end of the pass: weights (kept), evaluation, resume point at the next pass
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            save_model(os.path.join(out_dir, f"pass{p + 1:02d}_f{T}.pt"), core)
            log(f"saved pass{p + 1:02d}_f{T}.pt")
        run_eval(p + 1, T)
        accelerator.wait_for_everyone()
        save_resume(ckpt_path, accelerator, model, optimizer, disc, disc_opt, loss_fn, cfg,
                    {"stage": args.stage, "passes": passes, "pass": p + 1, "batches_done": 0, "updates": updates})
        accelerator.wait_for_everyone()
        if args.stop_after and ran >= args.stop_after and p + 1 < len(passes):
            log(f"stopped after {ran} batches (--stop_after): before pass {p + 2}")
            accelerator.end_training()
            return

    if accelerator.is_main_process:
        last = os.path.join(out_dir, f"pass{len(passes):02d}_f{passes[-1]['frames']}.pt")
        final = os.path.join(out_dir, "final.pt")
        if os.path.exists(last):
            link_or_copy(last, final)
            what, final_name = f"a hard link to {os.path.basename(last)}", os.path.basename(last)
        else:                                          # the last pass file was removed by hand
            save_model(final, core)
            what, final_name = "the weights after the last pass", "final.pt"
        if disc is not None:                           # GAN stage: appended passes keep the discriminator
            dpath = os.path.join(out_dir, "disc_final.pt")
            torch.save({"disc": disc_module.state_dict(), "disc_opt": disc_opt.state_dict(),
                        "lecam": (loss_fn.ema_real.detach().cpu(), loss_fn.ema_fake.detach().cpu())}, dpath + ".tmp")
            os.replace(dpath + ".tmp", dpath)
        with open(os.path.join(out_dir, "done.json.tmp"), "w") as f:
            json.dump({"stage": args.stage, "passes": passes, "updates": updates, "world": world,
                       "final": final_name, "time": time.time()}, f, indent=1)
        os.replace(os.path.join(out_dir, "done.json.tmp"), os.path.join(out_dir, "done.json"))
        if os.path.exists(ckpt_path):                  # the stage is done: one resume checkpoint per run
            os.remove(ckpt_path)
        log(f"saved {final}, {what} ({len(passes)} passes, {updates} parameter updates per GPU); "
            f"ckpt.pt removed")
    accelerator.end_training()


if __name__ == "__main__":
    main()

# GSCV

### Generative Screen Content Video Compression with Selective Fidelity and Interaction-Aware Context

Official code and model weights of GSCV (IEEE ISCAS 2027 submission).

## Overview

Screen content video (SCV) mixes sharp text and graphics with natural textures, and user interaction changes its
content abruptly while large areas stay static. GSCV is a generative codec for SCV at ultra-low bitrates, built on
[GVC1D](https://gvc1d.github.io/), with two SCV-specific modules:

- **Selective Fidelity Allocation (SFA)** classifies every 64x64 block as text and graphics, flat region, or
  natural / rendered texture. The class map conditions the encoder and decoder, raises the quality index of
  fidelity-critical windows, and selects the loss: a fidelity loss for text, graphics and flat regions, a generation
  loss (with an adversarial term) for textures.
- **Interaction-Aware Context Model (IACM)** assigns every 256x256 window a state (*Update*, *Skip*, *Refresh*,
  *Revert*). Unchanged windows are copied, revisited windows are restored from snapshots, only new content is coded,
  and the long-term memory is updated through a gate guided by the window state.

On four SCV test sets, GSCV saves 69.5% and 72.8% bitrate over VTM SCC on average under LPIPS and DISTS.

## Project Structure

```
GSCV/
├── configs/gscv.yaml           # model, window state machine, losses, data and training stages
├── modeling/
│   ├── gscv/                   # GSCV
│   │   ├── model.py            #   GSCV model (subclasses of the GVC1D encoder, decoder, bottleneck, memory)
│   │   ├── rsm.py              #   window state machine: class map, window states, quality offsets, side information
│   │   ├── attention.py        #   cross-window attention among the coded windows
│   │   ├── losses.py           #   region-adaptive rate-distortion-perception loss
│   │   ├── data.py             #   training clips with synthetic interaction events
│   │   └── iframe.py           #   I-frame codec (DLF) loader
│   ├── GVC1D.py, modules/, entropy/, layers/, dlf/   # GVC1D and its I-frame codec DLF (unchanged)
├── evaluator/                  # GVC1D's evaluator (unchanged)
├── scripts/
│   ├── test.py, test.sh        # testing
│   ├── metrics.py              # PSNR, MS-SSIM, LPIPS, DISTS, SIQM, PSNR of fidelity-critical regions, FID/KID features
│   ├── fid_kid.py              # FID / KID per test set and rate point
│   ├── bd_rate.py              # BD-rates on the average RD curve of each test set
│   ├── prepare_testsets.py     # test sequences -> PNG + description.json
│   ├── train.py, train.sh      # training
│   └── slim_dlf.py             # optional: a codec-only copy of the DLF checkpoint
├── data/scvcd5k/               # SCVCD-5K training entries and the SCVCD5K-TEST64 description
├── checkpoints/                # GSCV weights
├── ckpts/                      # GVC1D and DLF weights (downloaded, see below)
└── LICENSE                     # MIT
```

## Installation

Linux x86-64 and Python 3.10 (the bundled entropy-coder extensions of DLF are built for CPython 3.10 and 3.8);
4 GPUs for training. The versions below produced the paper's results on RTX 5090 GPUs (CUDA 12.8).

```bash
conda create -n gscv python=3.10
conda activate gscv
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install pytorch-lightning==2.6.6 --no-deps
```

LPIPS, DISTS and the FID / KID features download their backbone weights (AlexNet, VGG-16, the Inception-v3 of
pytorch-fid) into the torch hub cache on first use; training also downloads the LPIPS-VGG weights of GVC1D's loss
(7 KB) into `models/vgg_lpips.pth/`.

## Model Weights

| Weights | Path | Source |
|---|---|---|
| GSCV (final model, fp32) | `checkpoints/gscv.pt` | this repository's release |
| GSCV discriminator (optional, for further GAN fine-tuning) | `checkpoints/gscv_disc.pt` | this repository's release |
| GVC1D 32x16 (initialisation of training) | `ckpts/32x16/pytorch_model.bin` | [GVC1D on Hugging Face](https://huggingface.co/jacob-zzh/GVC1D/tree/main/ckpts/32x16) |
| DLF I-frame codec | `ckpts/I_frame/DLF.ckpt` | [GVC1D on Hugging Face](https://huggingface.co/jacob-zzh/GVC1D/tree/main/ckpts/I_frame) |

```bash
hf download jacob-zzh/GVC1D --include "ckpts/**" --local-dir .
```

`DLF.ckpt` (15 GB) also holds the training state of DLF; `python scripts/slim_dlf.py ckpts/I_frame/DLF.ckpt
ckpts/I_frame/DLF_slim.ckpt` writes the same codec in 5 GB, which loads faster in every training process.

## Test Data

Every test set is a folder of PNG sequences with a `description.json` (the layout of GVC1D); the paths in it are
relative to the parent of the set folder. The tests use the first 96 frames of every sequence.

| Test set | Sequences | Source |
|---|---|---|
| AOM CTC Class B (`AOM_B`) | 19: B1 (10) and B2 (9) of the AV2 CTC v8.0 | `https://media.xiph.org/video/aomctc/test_set/b1_syn/` and `b2_scc/` (y4m) |
| HEVC Class F (`HEVC_F`) | 4: BasketballDrillText, ChinaSpeed, SlideEditing, SlideShow (4:2:0, 8 bit) | JCT-VC common test conditions (JCTVC-L1100) |
| HEVC-SCC (`HEVC_SCC`) | 14 JCT-VC SCC sequences (YCbCr 4:4:4; 10 bit for EBURainFruits and Kimono1) | JCT-VC SCC common test conditions (JCTVC-Z1015) |
| SCVCD5K-TEST64 | 64 clips held out from SCVCD-5K | [SCVCD-5K](https://doi.org/10.1109/LSP.2026.3722151); list in `data/scvcd5k/SCVCD5K_TEST64/description.json` |

AOM CTC Class B sequences: B1 `CosmosTreeTrunk_sdr_2048x858_25_8bit`, `EuroTruckSimulator2_1920x1080p60_v2`,
`GlassHalf_1920x1080p_24p_8bit_420`, `Life_1080p30_v2`, `MINECRAFT_1080p_60_8bit`, `STARCRAFT_1080p60`,
`Sniper_1920x1080P_30fps_8bit`, `SolLevanteDragons_sdr_1920x1080_24_10bit`, `SolLevanteFace_sdr_1920x1080_24_10bit`,
`WITCHER3_1920x1080_60_8bit_420`; B2 `BigBuckBunnyStudio1_1920x1080_60fps_10bit_420_020_0149`,
`Debugging_1920x1080_30fps_8bit_420`, `MissionControlClip1_1920x1080_60fps_10bit_420_0450_0579`,
`MobileDeviceScreenSharing`, `SceneComposition_2`, `Slides1_1920x1080_30fps_8bit_420`,
`Slides2r_1920x1080_30fps_8bit_420`, `Spreadsheet_1920x1080_30fps_8bit_420_130f`, `Wikipedia_1920x1080p30`.

HEVC-SCC sequences (frame rate): BasketballScreen 2560x1440 (60), MissionControlClip2 2560x1440 (60),
MissionControlClip3 1920x1080 (60), ChineseEditing 1920x1080 (60), Console 1920x1080 (60), Desktop 1920x1080 (60),
FlyingGraphics 1920x1080 (60), Map 1280x720 (60), Programming 1280x720 (60), SlideShow 1280x720 (20),
WebBrowsing 1280x720 (30), Robot 1280x720 (30), EBURainFruits 1920x1080 (50), Kimono1 1920x1080 (24).

YUV sources are converted to RGB with BT.709, limited range:

```bash
# AOM CTC Class B: one set from both folders
python scripts/prepare_testsets.py y4m --inputs <b1_syn>/*.y4m <b2_scc>/*.y4m --out_root data/testsets --set AOM_B
# HEVC Class F: one call per size and frame rate
python scripts/prepare_testsets.py raw --inputs BasketballDrillText_832x480_50.yuv --width 832 --height 480 \
    --fps 50 --out_root data/testsets --set HEVC_F
# HEVC-SCC: YCbCr 4:4:4 (--bitdepth 10 for EBURainFruits and Kimono1)
python scripts/prepare_testsets.py raw --inputs Console_1920x1080.yuv --width 1920 --height 1080 --chroma 444 \
    --fps 60 --out_root data/testsets --set HEVC_SCC
```

SCVCD5K-TEST64 needs only the SCVCD-5K frames (see Training Data): its `description.json` is in
`data/scvcd5k/SCVCD5K_TEST64/`.

## Testing

```bash
bash scripts/test.sh            # the four test sets on GPUs 0-3, then FID / KID
```

or, for one set:

```bash
python scripts/test.py --sets data/testsets/HEVC_F --frames 96 --intra_period 32 --qps 0 1 2 3 4 5 6 7 \
    --fid_kid --out results/gscv
python scripts/fid_kid.py --sets data/testsets/HEVC_F --results GSCV=results/gscv --out results/fid_kid
```

`scripts/test.py` codes every sequence with a DLF I-frame every 32 frames and GSCV P-frames in between (FP16
weights, as GVC1D), at the base quality indices 0-7, and writes `results.json` with the rate (bits per pixel,
including the side information) and the metrics of every frame and sequence. The rate is that of the entropy model.

BD-rates against an anchor on the average RD curve of each test set (the protocol of the paper):

```bash
python scripts/bd_rate.py --anchor VTM=results/vtm --test GSCV=results/gscv \
    --fid_kid results/fid_kid/fid_kid.json --metrics lpips dists kid psnr siqm
```

The anchor needs a `results.json` of the same layout, scored with `scripts/metrics.py` on the same PNG frames.

## Training

### Training Data

GSCV is trained on [SCVCD-5K](https://doi.org/10.1109/LSP.2026.3722151). Convert its clips to PNG frames
`data/scvcd5k/<class>/<sub-class>/<clip>/im00001.png, im00002.png, ...` (spaces in folder names replaced by
underscores). `data/scvcd5k/train_entries.json` lists the training entries `{clip, start, end}` (frame numbers,
1-based and inclusive; 4982 clips, those longer than 193 frames split into near-equal pieces); the 64 test clips
of SCVCD5K-TEST64 are the validation clips. Defective clips are left out; five clips keep only part of their frames,
renumbered from `im00001.png`:

| Clip | Frames kept (original numbering) |
|---|---|
| `Basic_SC_videos/personal_computer/1_1_00023` | 9-76 |
| `Basic_SC_videos/personal_computer/1_1_00144` | 32-190 |
| `Basic_SC_videos/personal_computer/1_1_00534` | 1-180 |
| `Basic_SC_videos/mobile_phone/1_2_00341` | 44-180 |
| `Computer-rendered_videos/Animation+movie/2_2_00081` | 1-140 |

### Stages

Training starts from the released GVC1D weights and runs in stages, each a few passes over the training entries
(4 GPUs, bf16, 512x512 crops); every stage starts from the final weights of the previous one, S3_disc and S3 both
from S2b.

| Stage | Trained | Description |
|---|---|---|
| S0 | full model | GVC1D fine-tuned on SCVCD-5K with GVC1D's objective (GSCV off) |
| S1a | class embeddings | SFA: class embeddings, quality allocation, region losses |
| S1b | full model | SFA |
| S2a | null contexts, state embedding, memory gate | IACM: window states, snapshots, gated memory |
| S2b | full model | IACM |
| S3_disc | full model | a GAN run from S2b; only its discriminator is kept and initialises S3 |
| S3 | full model | final model: texture-region GAN, from S2b |

```bash
bash scripts/train.sh                         # all stages into runs/gscv (GPUs 0-3)
GPUS=4,5,6,7 STAGES="S2b S3_disc S3" bash scripts/train.sh
```

The passes, learning rates and switches of every stage are in `configs/gscv.yaml` (`train.stages`). The final
weights are `runs/gscv/S3/final.pt`; test them with `python scripts/test.py --ckpt runs/gscv/S3/final.pt ...`.

## Citation

```bibtex
```

## License

The GSCV code (`modeling/gscv/`, `scripts/`, `configs/`) is released under the [MIT License](LICENSE). The GVC1D code
in `modeling/` (outside `modeling/gscv/`) and `evaluator/`, including the DLF I-frame codec, is included unchanged and
remains under the terms of its original authors.

## Acknowledgements

GSCV builds on [GVC1D](https://gvc1d.github.io/) (its model code, entropy model and evaluator are used unchanged,
and training starts from its released weights) and its I-frame codec
[DLF](https://github.com/dlfcodec/Dual-generative-Latent-Fusion). It is
trained on [SCVCD-5K](https://doi.org/10.1109/LSP.2026.3722151) and evaluated with
[LPIPS](https://github.com/richzhang/PerceptualSimilarity), [DISTS](https://github.com/dingkeyan93/DISTS),
SIQM and [torch-fidelity](https://github.com/toshas/torch-fidelity).

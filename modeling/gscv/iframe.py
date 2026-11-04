"""I-frame codec of GSCV and GVC1D: DLF (modeling/dlf), frozen."""
import torch
from omegaconf import OmegaConf

DLF_CONFIG = "modeling/dlf/codec_sq_fixbpp_for_test.yaml"
# entries of the released DLF checkpoint that the codec does not use (training losses and their bookkeeping)
_UNUSED_PREFIXES = ("img_loss", "loss.perceptual_loss")
_UNUSED_KEYS = ("epoch_for_strategy", "lmbda_idx", "lmbda_list")


def codec_state_dict(path):
    """The codec weights of a DLF checkpoint (the full release or a slim copy, scripts/slim_dlf.py)."""
    sd = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
    return {k: v for k, v in sd.items() if not k.startswith(_UNUSED_PREFIXES) and k not in _UNUSED_KEYS}


def load_iframe_codec(ckpt, device, config=DLF_CONFIG):
    from modeling.dlf.src.models.codec_sq_fixbpp import Codec
    codec = Codec(**OmegaConf.load(config).model.params)
    codec.load_state_dict(codec_state_dict(ckpt), strict=True)
    codec.requires_grad_(False)
    return codec.to(device).eval()

"""Write a copy of GVC1D's DLF I-frame checkpoint (15 GB, with its training state) that holds only the codec weights
(5 GB). The codec is the same; training loads the slim copy faster and with less host memory per GPU process.

    python scripts/slim_dlf.py ckpts/I_frame/DLF.ckpt ckpts/I_frame/DLF_slim.ckpt
and set I_frame_codec.ckpt (configs/gscv.yaml) and --iframe_ckpt (scripts/test.py) to the copy.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modeling.gscv.iframe import codec_state_dict  # noqa: E402


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    src, dst = sys.argv[1:]
    sd = codec_state_dict(src)
    torch.save({"state_dict": sd}, dst + ".tmp")
    os.replace(dst + ".tmp", dst)
    print(f"{dst}: {len(sd)} codec tensors, {os.path.getsize(dst) / 2 ** 30:.2f} GiB")


if __name__ == "__main__":
    main()

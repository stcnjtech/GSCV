import pandas as pd
import torch
import torch.nn as nn

import lpips
from DISTS_pytorch import DISTS
from pytorch_msssim import ms_ssim
from einops import rearrange

import argparse
import os, sys
sys.path.append(os.getcwd())

# enable print all dataframe
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


class VideoMetric(nn.Module):
    def __init__(self):
        super(VideoMetric, self).__init__()
        self.lpips_metric = lpips.LPIPS(net='alex').eval()
        self.dists_metric = DISTS().eval()


    def forward(self, x, x_hat, is_01=False):
        """
        Calculate quality for a video.

        Args:
            x: Reference video tensor of shape (B, C, H, W).
            x_hat: Reconstructed video tensor of shape (B, C, H, W).
            is_01: If True, the input videos are in [0.0, 1.0] range.
                   If False, they are in [-1.0, 1.0] range.

        Returns:
            A dictionary with 'lpips', 'dists' and 'psnr' metrics.
        """
        device = next(self.parameters()).device
        x = x.to(device)
        x_hat = x_hat.to(device)

        if not is_01:
            x = (x + 1.0) / 2.0
            x_hat = (x_hat + 1.0) / 2.0
        if x.dim() == 5:
            x = rearrange(x, 'b t c h w -> (b t) c h w')
        if x_hat.dim() == 5:
            x_hat = rearrange(x_hat, 'b t c h w -> (b t) c h w')
        # calculate metrics
        mse = torch.mean((x - x_hat) ** 2, dim=(1, 2, 3)) # (B*T,)
        psnr = -10 * torch.log10(mse) # (B*T,)

        lpips_this = self.lpips_metric(x, x_hat, normalize=True).mean(dim=(1, 2, 3))  # (B*T,)
        dists_this = self.dists_metric(x, x_hat)

        return {'lpips': lpips_this, 'dists': dists_this, 'psnr': psnr,}


class VideoMetric_test(nn.Module):
    def __init__(self):
        super(VideoMetric_test, self).__init__()
        self.lpips_metric = lpips.LPIPS(net='alex').eval()
        self.dists_metric = DISTS().eval()


    def forward(self, x, x_hat, is_01=False):
        """
        Calculate quality for a video.

        Args:
            x: Reference video tensor of shape (B, C, H, W).
            x_hat: Reconstructed video tensor of shape (B, C, H, W).
            is_01: If True, the input videos are in [0.0, 1.0] range.
                   If False, they are in [-1.0, 1.0] range.

        Returns:
            A dictionary with 'lpips', 'dists', 'psnr' and 'msssim' metrics.
        """
        device = next(self.parameters()).device
        x = x.to(device)
        x_hat = x_hat.to(device)

        if not is_01:
            x = (x + 1.0) / 2.0
            x_hat = (x_hat + 1.0) / 2.0
        if x.dim() == 5:
            x = rearrange(x, 'b t c h w -> (b t) c h w')
        if x_hat.dim() == 5:
            x_hat = rearrange(x_hat, 'b t c h w -> (b t) c h w')
        # calculate metrics
        mse = torch.mean((x - x_hat) ** 2, dim=(1, 2, 3)) # (B*T,)
        psnr = -10 * torch.log10(mse) # (B*T,)
        msssim = ms_ssim(x, x_hat, data_range=1., size_average=False)

        lpips_this = self.lpips_metric(x, x_hat, normalize=True).mean(dim=(1, 2, 3))  # (B*T,)
        dists_this = self.dists_metric(x, x_hat)

        return {'lpips': lpips_this, 'dists': dists_this, 'psnr': psnr, 'msssim': msssim}


if __name__ == "__main__":
    # Example usage
    parser = argparse.ArgumentParser(description="Video Metric Example")
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for testing')
    args = parser.parse_args()

    # Create dummy data
    x = torch.rand(args.batch_size, 3, 256, 256)  # Reference video
    x_hat = torch.rand(args.batch_size, 3, 256, 256)  # Reconstructed video

    # Initialize and run the metric
    metric = VideoMetric()
    result = metric(x, x_hat)
    # print(result)

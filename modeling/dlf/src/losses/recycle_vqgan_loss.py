from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F


class VQGANRecycleLoss(nn.Module):
    def __init__(self, vqgan_encoder, vqgan_quant_conv, recycle_weight) -> None:
        super().__init__()
        self.encoder = deepcopy(vqgan_encoder)
        self.quant_conv = deepcopy(vqgan_quant_conv)
        self.recycle_weight = recycle_weight
        self.mse_loss = nn.MSELoss(reduce='mean')
        self.requires_grad_(False)
    

    def forward(self, input, reconstruction, split=None):
        """ Input image should be [-1,1]
        """
        if not split:
            split = "train" if self.training else "val"
        
        # encode original image
        with torch.no_grad():
            h_label = self.encoder(input)
            h_label = self.quant_conv(h_label).detach()
        
        # encode recon image
        h_recon = self.encoder(reconstruction)
        h_recon = self.quant_conv(h_recon)

        recycle_loss = self.mse_loss(h_recon, h_label)
        weighted_loss = recycle_loss * self.recycle_weight

        return weighted_loss, {
            f"{split}/recycle_mse_loss": recycle_loss,
            f"{split}/recycle_weight": self.recycle_weight,
            f"{split}/recycle_mse_loss_weighted": weighted_loss,
        }


        
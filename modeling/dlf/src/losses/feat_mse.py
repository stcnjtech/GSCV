import torch
from torch import nn
from taming.modules.distribution.distributions import DiagonalGaussianDistribution

class FeatLoss(nn.Module):
    def __init__(self, mse_weight, ce_weight, vq_weight) -> None:
        super().__init__()
        self.mse_weight = mse_weight
        self.ce_weight = ce_weight
        self.vq_weight = vq_weight
    
    def forward(self, feat_in, logits_in, feat_target, label_target, vq_loss, split=None):
        mse_loss = nn.functional.mse_loss(feat_in, feat_target)
        ce_loss = nn.functional.cross_entropy(logits_in, label_target)
        total_loss = self.mse_weight * mse_loss + self.ce_weight * ce_loss + self.vq_weight * vq_loss
        if not split:
            split = "train" if self.training else "val"
        return total_loss, {
            f"{split}/mse_loss": mse_loss,
            f"{split}/ce_loss": ce_loss,
            f"{split}/vq_loss": vq_loss
        }


class FeatLoss_sq_vq(nn.Module):
    def __init__(self, mse_weight, ce_weight, sq_weight, vq_weight) -> None:
        super().__init__()
        self.mse_weight = mse_weight
        self.ce_weight = ce_weight
        self.sq_weight = sq_weight
        self.vq_weight = vq_weight
    
    def forward(self, feat_in, logits_in, feat_target, label_target, vq_loss, sq_loss, split=None):
        mse_loss = nn.functional.mse_loss(feat_in, feat_target)
        ce_loss = nn.functional.cross_entropy(logits_in, label_target)
        total_loss = self.mse_weight * mse_loss + self.ce_weight * ce_loss
        total_loss += self.vq_weight * vq_loss + self.sq_weight * sq_loss
        if not split:
            split = "train" if self.training else "val"
        return total_loss, {
            f"{split}/mse_loss": mse_loss,
            f"{split}/ce_loss": ce_loss,
            f"{split}/sq_loss": sq_loss,
            f"{split}/vq_loss": vq_loss,
            f"{split}/sq_lambda": self.sq_weight,
        }
    

class FeatLoss_KL(nn.Module):
    def __init__(self, kl_weight, norm_kl_weight, feat_weight, vq_weight) -> None:
        super().__init__()
        self.kl_weight = kl_weight
        self.norm_kl_weight = norm_kl_weight
        assert norm_kl_weight >= 0.0 and norm_kl_weight <= 1.0
        self.vq_weight = vq_weight
        self.feat_weight = feat_weight
    
    def forward(self,
                posterior_in: DiagonalGaussianDistribution, 
                posterior_target: DiagonalGaussianDistribution, 
                vq_loss, split=None):
        kl_dist_with_target = posterior_in.kl(other=posterior_target).mean()
        kl_dist_with_norm = posterior_in.kl(other=None).mean()
        kl_dist_weighted = self.norm_kl_weight * kl_dist_with_norm + (1. - self.norm_kl_weight) * kl_dist_with_target
        
        feat_loss = nn.functional.mse_loss(posterior_in.sample(), posterior_target.sample())

        total_loss = self.vq_weight * vq_loss + self.kl_weight * kl_dist_weighted + self.feat_weight * feat_loss

        if not split:
            split = "train" if self.training else "val"
        return total_loss, {
            f"{split}/kl_dist_with_target": kl_dist_with_target,
            f"{split}/kl_dist_with_norm": kl_dist_with_norm,
            f"{split}/kl_dist_weighted": kl_dist_weighted,
            f"{split}/feat_loss": feat_loss,
            f"{split}/vq_loss": vq_loss
        }


class FeatLoss_KL_type2(nn.Module):
    def __init__(self, enc_kl_scale_weight, norm_kl_scale_weight, kl_enc_balance_weight, 
                 kl_weight, feat_weight, vq_weight) -> None:
        super().__init__()
        self.enc_kl_scale_weight = enc_kl_scale_weight
        self.norm_kl_scale_weight = norm_kl_scale_weight
        self.kl_enc_balance_weight = kl_enc_balance_weight
        assert kl_enc_balance_weight >= 0.0 and kl_enc_balance_weight <= 1.0
        self.feat_weight = feat_weight
        self.kl_weight = kl_weight
        self.vq_weight = vq_weight
    
    def forward(self,
                posterior_in: DiagonalGaussianDistribution, 
                posterior_target: DiagonalGaussianDistribution, 
                vq_loss, split=None):
        # 1. kl part
        kl_dist_with_target = posterior_in.kl(other=posterior_target).mean()
        kl_dist_with_norm = posterior_in.kl(other=None).mean()
        kl_dist_weighted = self.kl_enc_balance_weight * kl_dist_with_target * self.enc_kl_scale_weight +\
                           (1. - self.kl_enc_balance_weight) * kl_dist_with_norm * self.norm_kl_scale_weight

        # 2. feat MSE part
        feat_loss = nn.functional.mse_loss(posterior_in.sample(), posterior_target.sample())

        # 3. total loss
        total_loss = self.vq_weight * vq_loss + self.kl_weight * kl_dist_weighted + self.feat_weight * feat_loss

        if not split:
            split = "train" if self.training else "val"
        return total_loss, {
            f"{split}/kl_dist_with_target": kl_dist_with_target,
            f"{split}/kl_dist_with_norm": kl_dist_with_norm,
            f"{split}/kl_dist_weighted": kl_dist_weighted,
            f"{split}/feat_loss": feat_loss,
            f"{split}/vq_loss": vq_loss
        }


class FeatLoss_KL_type3(nn.Module):
    def __init__(self, feat_weight, vq_weight) -> None:
        super().__init__()
        self.feat_weight = feat_weight
        self.vq_weight = vq_weight
    
    def forward(self,
                latent_in: torch.Tensor, 
                posterior_target: DiagonalGaussianDistribution, 
                vq_loss, split=None):
        feat_loss = nn.functional.mse_loss(latent_in, posterior_target.mode())

        # 3. total loss
        total_loss = self.vq_weight * vq_loss + self.feat_weight * feat_loss

        if not split:
            split = "train" if self.training else "val"
        return total_loss, {
            f"{split}/feat_loss": feat_loss,
            f"{split}/vq_loss": vq_loss
        }
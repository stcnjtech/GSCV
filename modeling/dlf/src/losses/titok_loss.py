import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from taming.modules.losses.lpips import LPIPS
from taming.modules.discriminator.model import NLayerDiscriminator, weights_init
from losses.titok_disc import NLayerDiscriminator as TiTokDiscriminator


def hinge_d_loss_fn(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def lecam_loss_fn(
    logits_real_mean: torch.Tensor,
    logits_fake_mean: torch.Tensor,
    ema_logits_real_mean: torch.Tensor,
    ema_logits_fake_mean: torch.Tensor
) -> torch.Tensor:
    """Computes the LeCam loss for the given average real and fake logits.

    Args:
        logits_real_mean -> torch.Tensor: The average real logits.
        logits_fake_mean -> torch.Tensor: The average fake logits.
        ema_logits_real_mean -> torch.Tensor: The EMA of the average real logits.
        ema_logits_fake_mean -> torch.Tensor: The EMA of the average fake logits.

    Returns:
        lecam_loss -> torch.Tensor: The LeCam loss.
    """
    lecam_loss = torch.mean(torch.pow(F.relu(logits_real_mean - ema_logits_fake_mean), 2))
    lecam_loss += torch.mean(torch.pow(F.relu(ema_logits_real_mean - logits_fake_mean), 2))
    return lecam_loss


class ConvNextPerceptualLoss(torch.nn.Module):
    def __init__(self, model_name: str = "convnext_s"):
        """Initializes the PerceptualLoss class.

        Args:
            model_name: A string, the name of the perceptual loss model to use.

        Raise:
            ValueError: If the model_name does not contain "convnext_s".
        """
        super().__init__()
        if "convnext_s" not in model_name:
            raise ValueError(f"Unsupported Perceptual Loss model name {model_name}")
        _IMAGENET_MEAN = [0.485, 0.456, 0.406]
        _IMAGENET_STD = [0.229, 0.224, 0.225]

        self.convnext = models.convnext_small(weights=models.ConvNeXt_Small_Weights.IMAGENET1K_V1).eval()
        self.register_buffer("imagenet_mean", torch.Tensor(_IMAGENET_MEAN)[None, :, None, None])
        self.register_buffer("imagenet_std", torch.Tensor(_IMAGENET_STD)[None, :, None, None])

        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, input: torch.Tensor, target: torch.Tensor, range_to_01:bool):
        """Computes the perceptual loss.

        Args:
            input: A tensor of shape (B, C, H, W), the input image. Normalized to [0, 1].
            target: A tensor of shape (B, C, H, W), the target image. Normalized to [0, 1].
            range_to_01: normalize [-1, 1] to [0, 1]

        Returns:
            A scalar tensor, the perceptual loss.
        """
        # Always in eval mode.
        self.eval()

        if range_to_01:
            input = input * 0.5 + 0.5
            target = target * 0.5 + 0.5

        input = torch.nn.functional.interpolate(input, size=224, mode="bilinear", align_corners=False, antialias=True)
        target = torch.nn.functional.interpolate(target, size=224, mode="bilinear", align_corners=False, antialias=True)
        pred_input = self.convnext((input - self.imagenet_mean) / self.imagenet_std)
        pred_target = self.convnext((target - self.imagenet_mean) / self.imagenet_std)
        loss = torch.nn.functional.mse_loss(
            pred_input,
            pred_target,
            reduction="mean")
    
        return loss


class PerceptualConvNext_DiscVQGAN(nn.Module):
    def __init__(self, codebook_weight=1.0, pixelloss_weight=1.0, perceptual_weight=1.0, disc_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, 
                 use_actnorm=False, disc_ndf=64, adaptive_disc_max=1.0e4):
        super().__init__()
        # weight define
        self.codebook_weight = codebook_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_weight = perceptual_weight
        self.discriminator_weight = disc_weight

        # loss define
        self.pixel_loss_fn = nn.L1Loss(reduce="mean")
        self.perceptual_loss_fn = ConvNextPerceptualLoss().eval()
        self.lpips_loss_fn = LPIPS().eval()
        self.disc_loss = hinge_d_loss_fn

        # disc part
        self.adaptive_disc_max = adaptive_disc_max
        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm,
                                                 ndf=disc_ndf
                                                 ).apply(weights_init)

        print("[INFO] ConvNext perceptual loss + VQGAN's discriminator")


    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, self.adaptive_disc_max).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight


    def forward(self, codebook_loss, inputs, reconstructions, optimizer_idx,
                global_step, last_layer=None, split="train"):
        inputs = inputs.contiguous()
        reconstructions = reconstructions.contiguous()

        # now the GAN part
        if optimizer_idx == 0:
            codebook_loss = codebook_loss.mean()
            pixel_loss = self.pixel_loss_fn(inputs, reconstructions).mean()
            perceptual_loss = self.perceptual_loss_fn(inputs, reconstructions, range_to_01=True).mean()
            nll_loss = pixel_loss * self.pixel_weight + perceptual_loss * self.perceptual_weight

            # generator update
            logits_fake = self.discriminator(reconstructions)
            g_loss = -torch.mean(logits_fake)

            # adaptive weight
            try:
                d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
            except RuntimeError:
                assert not self.training
                d_weight = torch.tensor(0.0, device=inputs.device)

            loss = nll_loss + d_weight * g_loss + self.codebook_weight * codebook_loss

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                   "{}/quant_loss".format(split): codebook_loss.detach(),
                   "{}/nll_loss".format(split): nll_loss.detach(),
                   "{}/pixel_loss".format(split): pixel_loss.detach(),
                   "{}/perceptual_loss".format(split): perceptual_loss.detach(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   }

            # calculate lpips if not training
            if not self.training:
                lpips_loss = self.lpips_loss_fn(inputs.detach(), reconstructions.detach())
                log[f"{split}/lpips_loss"] = lpips_loss.detach().mean()

            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            logits_real = self.discriminator(inputs.detach())
            logits_fake = self.discriminator(reconstructions.detach())

            d_loss = self.disc_loss(logits_real, logits_fake).mean()

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return d_loss, log


class PerceptualConvNext_DiscVQGAN_SQ(PerceptualConvNext_DiscVQGAN):
    def __init__(self, 
                 codebook_weight=1, sq_weight=1.0,
                 pixelloss_weight=1, perceptual_weight=1, disc_weight=1, 
                 disc_num_layers=3, disc_in_channels=3, use_actnorm=False, 
                 disc_ndf=64, adaptive_disc_max=10000):
        super().__init__(codebook_weight, pixelloss_weight, perceptual_weight, 
                         disc_weight, disc_num_layers, disc_in_channels, 
                         use_actnorm, disc_ndf, adaptive_disc_max)
        self.sq_weight = sq_weight
        print("[INFO] ConvNext perceptual loss + VQGAN's discriminator + SQ")
    
    def forward(self, codebook_loss, sq_loss, inputs, reconstructions, optimizer_idx,
                global_step, last_layer=None, split="train"):
        loss, log = super().forward(codebook_loss, inputs, reconstructions, optimizer_idx,
                        global_step, last_layer, split)
        if optimizer_idx == 0:
            sq_loss = sq_loss.mean()
            loss += self.sq_weight * sq_loss
            log["{}/sq_loss".format(split)] = sq_loss.detach()
            log["{}/sq_lambda".format(split)] = self.sq_weight
        return loss, log


class PerceptualLPIPS_DiscTiTok(nn.Module):
    def __init__(self, codebook_weight=1.0, pixelloss_weight=1.0, perceptual_weight=1.0, disc_weight=1.0,
                 lecam_regularization_weight=1., lecam_ema_decay=0.999):
        super().__init__()
        # weight define
        self.codebook_weight = codebook_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_weight = perceptual_weight
        self.discriminator_weight = disc_weight

        # loss define
        self.pixel_loss_fn = nn.L1Loss(reduce="mean")
        self.perceptual_loss_fn = LPIPS().eval()
        self.disc_loss = hinge_d_loss_fn

        # disc part
        self.discriminator = TiTokDiscriminator()
        self.lecam_regularization_weight = lecam_regularization_weight
        self.lecam_ema_decay = lecam_ema_decay
        if self.lecam_regularization_weight > 0.0:
            self.register_buffer("ema_real_logits_mean", torch.zeros((1)))
            self.register_buffer("ema_fake_logits_mean", torch.zeros((1)))
        
        print("[INFO] LPIPS perceptual loss + TiTok's discriminator")


    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, self.adaptive_disc_max).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight


    def forward(self, codebook_loss, inputs, reconstructions, optimizer_idx,
                global_step=None, last_layer=None, split="train"):
        inputs = inputs.contiguous()
        reconstructions = reconstructions.contiguous()

        # now the GAN part
        if optimizer_idx == 0:
            self.discriminator.requires_grad_(False)

            # pix and codebook part
            codebook_loss = codebook_loss.mean()
            pixel_loss = self.pixel_loss_fn(inputs, reconstructions).mean()
            perceptual_loss = self.perceptual_loss_fn(inputs, reconstructions).mean()
            nll_loss = pixel_loss * self.pixel_weight + perceptual_loss * self.perceptual_weight

            # generator update
            logits_fake = self.discriminator(reconstructions)
            g_loss = -torch.mean(logits_fake)
            g_loss = g_loss * self.discriminator_weight

            loss = nll_loss + g_loss + self.codebook_weight * codebook_loss

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                   "{}/quant_loss".format(split): codebook_loss.detach(),
                   "{}/nll_loss".format(split): nll_loss.detach(),
                   "{}/pixel_loss".format(split): pixel_loss.detach(),
                   "{}/perceptual_loss".format(split): perceptual_loss.detach(),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   }

            # calculate lpips if not training
            if not self.training:
                log[f"{split}/lpips_loss"] = perceptual_loss.clone().detach()

            return loss, log

        if optimizer_idx == 1:
            self.discriminator.requires_grad_(True)
            inputs = inputs.detach().requires_grad_(True)
            reconstructions = reconstructions.detach()

            # second pass for discriminator update
            logits_real = self.discriminator(inputs)
            logits_fake = self.discriminator(reconstructions)

            # hindge loss part
            d_loss = self.disc_loss(logits_real, logits_fake).mean()

            if self.lecam_regularization_weight > 0.0:
                lecam_loss = lecam_loss_fn(
                    torch.mean(logits_real),
                    torch.mean(logits_fake),
                    self.ema_real_logits_mean,
                    self.ema_fake_logits_mean
                ) * self.lecam_regularization_weight

                if self.training:
                    self.ema_real_logits_mean = self.ema_real_logits_mean * self.lecam_ema_decay + torch.mean(logits_real).detach()  * (1 - self.lecam_ema_decay)
                    self.ema_fake_logits_mean = self.ema_fake_logits_mean * self.lecam_ema_decay + torch.mean(logits_fake).detach()  * (1 - self.lecam_ema_decay)
                
            else:
                lecam_loss = torch.zeros_like(inputs).mean()

            loss = d_loss + lecam_loss

            log = {"{}/disc_loss".format(split): loss.detach().mean(),
                   "{}/disc_d_loss".format(split): d_loss.detach().mean(),
                   "{}/disc_lecam_loss".format(split): lecam_loss.detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return loss, log


class PerceptualLPIPS_DiscTiTok_SQ(PerceptualLPIPS_DiscTiTok):
    def __init__(self, 
                 codebook_weight=1, sq_weight=1,
                 pixelloss_weight=1, perceptual_weight=1, disc_weight=1, 
                 lecam_regularization_weight=1, lecam_ema_decay=0.999):
        super().__init__(codebook_weight, pixelloss_weight, perceptual_weight, disc_weight, 
                         lecam_regularization_weight, lecam_ema_decay)
        self.sq_weight = sq_weight
        print("[INFO] LPIPS perceptual loss + TiTok's discriminator + SQ")
    
    def forward(self, codebook_loss, sq_loss, inputs, reconstructions, optimizer_idx,
                global_step, last_layer=None, split="train"):
        loss, log = super().forward(codebook_loss, inputs, reconstructions, optimizer_idx,
                        global_step, last_layer, split)
        if optimizer_idx == 0:
            sq_loss = sq_loss.mean()
            loss += self.sq_weight * sq_loss
            log["{}/sq_loss".format(split)] = sq_loss.detach()
            log["{}/sq_lambda".format(split)] = self.sq_weight
        return loss, log


class PerceptualConvNext_DiscTiTok(nn.Module):
    def __init__(self, codebook_weight=1.0, pixelloss_weight=1.0, perceptual_weight=1.0, disc_weight=1.0,
                 lecam_regularization_weight=1., lecam_ema_decay=0.999, img_range_01=False):
        super().__init__()
        # weight define
        self.codebook_weight = codebook_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_weight = perceptual_weight
        self.discriminator_weight = disc_weight

        # loss define
        self.pixel_loss_fn = nn.MSELoss(reduce="mean")
        self.perceptual_loss_fn = ConvNextPerceptualLoss().eval()
        self.lpips_loss_fn = LPIPS().eval()
        self.disc_loss = hinge_d_loss_fn
        self.img_range_01 = img_range_01

        # disc part
        self.discriminator = TiTokDiscriminator()
        self.lecam_regularization_weight = lecam_regularization_weight
        self.lecam_ema_decay = lecam_ema_decay
        if self.lecam_regularization_weight > 0.0:
            self.register_buffer("ema_real_logits_mean", torch.zeros((1)))
            self.register_buffer("ema_fake_logits_mean", torch.zeros((1)))

        print("[INFO] ConvNext perceptual loss + TiTok's discriminator")


    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, self.adaptive_disc_max).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight


    def forward(self, codebook_loss, inputs, reconstructions, optimizer_idx,
                global_step=None, last_layer=None, split="train"):
        inputs = inputs.contiguous()
        reconstructions = reconstructions.contiguous()

        # now the GAN part
        if optimizer_idx == 0:
            self.discriminator.requires_grad_(False)

            # pix and codebook part
            codebook_loss = codebook_loss.mean()
            pixel_loss = self.pixel_loss_fn(inputs, reconstructions).mean()
            perceptual_loss = self.perceptual_loss_fn(inputs, reconstructions, range_to_01=(not self.img_range_01)).mean()
            nll_loss = pixel_loss * self.pixel_weight + perceptual_loss * self.perceptual_weight

            # generator update
            logits_fake = self.discriminator(reconstructions)
            g_loss = -torch.mean(logits_fake)
            weighted_g_loss = g_loss * self.discriminator_weight

            loss = nll_loss + weighted_g_loss + self.codebook_weight * codebook_loss

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                   "{}/quant_loss".format(split): codebook_loss.detach(),
                   "{}/nll_loss".format(split): nll_loss.detach(),
                   "{}/pixel_loss".format(split): pixel_loss.detach(),
                   "{}/perceptual_loss".format(split): perceptual_loss.detach(),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   "{}/weighted_g_loss".format(split): weighted_g_loss.detach().mean(),
                   }
            
            # calculate lpips if not training
            if not self.training:
                if self.img_range_01:
                    inputs = inputs * 2.0 - 1.0
                    reconstructions = reconstructions * 2.0 - 1.0
                lpips_loss = self.lpips_loss_fn(inputs.detach(), reconstructions.detach())
                log[f"{split}/lpips_loss"] = lpips_loss.detach().mean()

            return loss, log

        if optimizer_idx == 1:
            self.discriminator.requires_grad_(True)
            inputs = inputs.detach().requires_grad_(True)
            reconstructions = reconstructions.detach()

            # second pass for discriminator update
            logits_real = self.discriminator(inputs)
            logits_fake = self.discriminator(reconstructions)

            # hindge loss part
            d_loss = self.disc_loss(logits_real, logits_fake).mean()

            if self.lecam_regularization_weight > 0.0:
                lecam_loss = lecam_loss_fn(
                    torch.mean(logits_real),
                    torch.mean(logits_fake),
                    self.ema_real_logits_mean,
                    self.ema_fake_logits_mean
                ) * self.lecam_regularization_weight

                if self.training:
                    self.ema_real_logits_mean = self.ema_real_logits_mean * self.lecam_ema_decay + torch.mean(logits_real).detach()  * (1 - self.lecam_ema_decay)
                    self.ema_fake_logits_mean = self.ema_fake_logits_mean * self.lecam_ema_decay + torch.mean(logits_fake).detach()  * (1 - self.lecam_ema_decay)
                    
            else:
                lecam_loss = torch.zeros_like(inputs).mean()

            loss = d_loss + lecam_loss

            log = {"{}/disc_loss".format(split): loss.detach().mean(),
                   "{}/disc_d_loss".format(split): d_loss.detach().mean(),
                   "{}/disc_lecam_loss".format(split): lecam_loss.detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return loss, log


class PerceptualConvNext_DiscTiTok_SQ(PerceptualConvNext_DiscTiTok):
    def __init__(self, 
                 codebook_weight=1, sq_weight=1,
                 pixelloss_weight=1, perceptual_weight=1, disc_weight=1, 
                 lecam_regularization_weight=1, lecam_ema_decay=0.999, 
                 img_range_01=False):
        super().__init__(codebook_weight, pixelloss_weight, 
                         perceptual_weight, disc_weight, lecam_regularization_weight, lecam_ema_decay,
                         img_range_01)
        self.sq_weight = sq_weight
        print("[INFO] ConvNext perceptual loss + TiTok's discriminator + SQ")
    
    def forward(self, codebook_loss, sq_loss, inputs, reconstructions, optimizer_idx,
                global_step, last_layer=None, split="train"):
        loss, log = super().forward(codebook_loss, inputs, reconstructions, optimizer_idx,
                        global_step, last_layer, split)
        if optimizer_idx == 0:
            sq_loss = sq_loss.mean()
            loss += self.sq_weight * sq_loss
            log["{}/sq_loss".format(split)] = sq_loss.detach()
            log["{}/sq_lambda".format(split)] = self.sq_weight
        return loss, log


class Improve_VQGAN_loss(nn.Module):
    def __init__(self, codebook_weight=1.0, pixelloss_weight=1.0, perceptual_weight=1.0, 
                 convnext_weight=1.0, lecam_regularization_weight=1., lecam_ema_decay=0.999,
                 disc_weight=1.0, adaptive_disc_max=1.0e4, img_range_01=False):
        super().__init__()
        # weights
        self.codebook_weight = codebook_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_weight = perceptual_weight
        self.convnext_weight = convnext_weight
        self.discriminator_weight = disc_weight
        self.lecam_regularization_weight = lecam_regularization_weight
        self.lecam_ema_decay = lecam_ema_decay

        # perceptual loss
        self.pixel_loss_fn = nn.MSELoss()
        self.perceptual_loss_fn = LPIPS().eval()
        self.convnext_loss_fn = ConvNextPerceptualLoss().eval()
        self.img_range_01 = img_range_01

        # discriminator
        self.discriminator = TiTokDiscriminator()
        self.disc_loss = hinge_d_loss_fn
        self.disc_factor = 1.0
        self.adaptive_disc_max = adaptive_disc_max
        if self.lecam_regularization_weight > 0.0:
            self.register_buffer("ema_real_logits_mean", torch.zeros((1)))
            self.register_buffer("ema_fake_logits_mean", torch.zeros((1)))


    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, self.adaptive_disc_max).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight


    def forward(self, codebook_loss, inputs, reconstructions, optimizer_idx,
                global_step, last_layer=None, split="train"):
        inputs = inputs.contiguous()
        reconstructions = reconstructions.contiguous()

        # now the GAN part
        if optimizer_idx == 0:
            self.discriminator.requires_grad_(False)

            if self.img_range_01:
                inputs = inputs * 2.0 - 1.0
                reconstructions = reconstructions * 2.0 - 1.0

            # pix and codebook part
            codebook_loss = codebook_loss.mean()
            pixel_loss = self.pixel_loss_fn(inputs, reconstructions).mean()
            perceptual_loss = self.perceptual_loss_fn(inputs, reconstructions).mean()
            convnext_loss = self.convnext_loss_fn(inputs, reconstructions, range_to_01=True).mean()
            nll_loss = pixel_loss * self.pixel_weight + perceptual_loss * self.perceptual_weight + convnext_loss * self.convnext_weight
            nll_for_weight = pixel_loss * self.pixel_weight + perceptual_loss * self.perceptual_weight

            # generator update
            logits_fake = self.discriminator(reconstructions)
            g_loss = -torch.mean(logits_fake)
            if self.training:
                ada_d_weight = self.calculate_adaptive_weight(nll_for_weight, g_loss, last_layer=last_layer)
            else:
                ada_d_weight = torch.tensor(0.0, device=inputs.device)

            loss = nll_loss + g_loss * ada_d_weight + self.codebook_weight * codebook_loss

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                   "{}/quant_loss".format(split): codebook_loss.detach(),
                   "{}/nll_loss".format(split): nll_loss.detach(),
                   "{}/pixel_loss".format(split): pixel_loss.detach(),
                   "{}/lpips_loss".format(split): perceptual_loss.detach(),
                   "{}/convnext_loss".format(split): convnext_loss.detach(),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   "{}/adapt_d_weight".format(split): ada_d_weight.detach().mean(),
                   }

            return loss, log

        if optimizer_idx == 1:
            self.discriminator.requires_grad_(True)
            inputs = inputs.detach().requires_grad_(True)
            reconstructions = reconstructions.detach()

            # second pass for discriminator update
            logits_real = self.discriminator(inputs)
            logits_fake = self.discriminator(reconstructions)

            # hindge loss part
            d_loss = self.disc_loss(logits_real, logits_fake).mean()

            if self.lecam_regularization_weight > 0.0:
                lecam_loss = lecam_loss_fn(
                    torch.mean(logits_real),
                    torch.mean(logits_fake),
                    self.ema_real_logits_mean,
                    self.ema_fake_logits_mean
                ) * self.lecam_regularization_weight

                if self.training:
                    self.ema_real_logits_mean = self.ema_real_logits_mean * self.lecam_ema_decay + torch.mean(logits_real).detach()  * (1 - self.lecam_ema_decay)
                    self.ema_fake_logits_mean = self.ema_fake_logits_mean * self.lecam_ema_decay + torch.mean(logits_fake).detach()  * (1 - self.lecam_ema_decay)
                    
            else:
                lecam_loss = torch.zeros_like(inputs).mean()

            loss = d_loss + lecam_loss

            log = {"{}/disc_loss".format(split): loss.detach().mean(),
                   "{}/disc_d_loss".format(split): d_loss.detach().mean(),
                   "{}/disc_lecam_loss".format(split): lecam_loss.detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return loss, log


class Improve_VQGAN_loss_SQ(Improve_VQGAN_loss):
    def __init__(self, codebook_weight=1, sq_weight=1.0, pixelloss_weight=1, 
                 perceptual_weight=1, convnext_weight=1, 
                 lecam_regularization_weight=1, lecam_ema_decay=0.999, 
                 disc_weight=1, adaptive_disc_max=10000, img_range_01=False):
        super().__init__(codebook_weight, pixelloss_weight, perceptual_weight, convnext_weight, 
                         lecam_regularization_weight, lecam_ema_decay, disc_weight, adaptive_disc_max,
                         img_range_01)
        self.sq_weight = sq_weight
    
    def forward(self, codebook_loss, sq_loss, inputs, reconstructions, optimizer_idx,
                global_step, last_layer=None, split="train"):
        loss, log = super().forward(codebook_loss, inputs, reconstructions, optimizer_idx,
                        global_step, last_layer, split)
        if optimizer_idx == 0:
            sq_loss = sq_loss.mean()
            loss += self.sq_weight * sq_loss
            log["{}/sq_loss".format(split)] = sq_loss.detach()
            log["{}/sq_lambda".format(split)] = self.sq_weight
        return loss, log
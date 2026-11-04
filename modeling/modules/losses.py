"""This files contains training loss implementation.

Copyright (2024) Bytedance Ltd. and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

    http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License. 

Ref:
    https://github.com/CompVis/taming-transformers/blob/master/taming/modules/losses/vqperceptual.py
"""
from typing import Mapping, Text, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.cuda.amp import autocast
from .perceptual_loss import PerceptualLoss
from .discriminator import NLayerDiscriminator
from .discriminator_video import VideoDiscriminator

class GANLoss(nn.Module):
    def __init__(self):
        super(GANLoss, self).__init__()
        self.loss = nn.ReLU()
        
    def forward(self, input, target_is_real, is_disc=False):
        if is_disc:  # for discriminators in hinge-gan
            input = -input if target_is_real else input
            loss = self.loss(1 + input)
        else:  # for generators in hinge-gan
            loss = -input
        loss = torch.mean(loss, axis=(1, 2, 3))
        return loss

def calculate_adaptive_weight(net_g, l1_loss, lpips_loss, g_gan, disc_weight_max=1.0, g_gan_adapt_layer="codec_dec"):
    if next(net_g.named_parameters())[0][:len("module.")] == "module.":
        net = net_g.module
    else:
        net = net_g

    if g_gan_adapt_layer == "codec_recon":
        last_layer = net.decoder.conv_out.weight
        d_weights = torch.zeros_like(l1_loss)
        for i in range(l1_loss.shape[0]):
            g_grads = torch.autograd.grad(g_gan[i], last_layer, retain_graph=True)[0]
            recon_grads = torch.autograd.grad(l1_loss[i] + lpips_loss[i], last_layer, retain_graph=True, allow_unused=True)[0]

            d_weight = torch.norm(recon_grads) / (torch.norm(g_grads) + 1e-4)
            d_weight = torch.clamp(d_weight, 0.0, disc_weight_max).detach()

            d_weights[i] = d_weight
            
    elif g_gan_adapt_layer == "codec_dec":
        last_layer = net.decoder.conv_out.weight
        d_weights = torch.zeros_like(l1_loss)
        for i in range(l1_loss.shape[0]):
            g_grads = torch.autograd.grad(g_gan[i], last_layer, retain_graph=True)[0]
            recon_grads = torch.autograd.grad(l1_loss[i] + lpips_loss[i], last_layer, retain_graph=True, allow_unused=True)[0]

            d_weight = torch.norm(recon_grads) / (torch.norm(g_grads) + 1e-4)
            d_weight = torch.clamp(d_weight, 0.0, disc_weight_max).detach()

            d_weights[i] = d_weight

    return d_weights

def cal_mix_Loss(x,x_hat,mse_l1_weight=0.5):
    N = x.shape[1]
    pixel_num = x.shape[2] * x.shape[3]

    l1_loss = F.l1_loss(x, x_hat, reduction='none')
    l1_loss = torch.sum(l1_loss, dim=(1, 2, 3)) / pixel_num / N
    mse_loss = F.mse_loss(x, x_hat, reduction='none')
    mse_loss = torch.sum(mse_loss, dim=(1, 2, 3)) / pixel_num / N
    mix_loss = mse_l1_weight * l1_loss + (1 - mse_l1_weight) * mse_loss
    return mix_loss

def hinge_d_loss(logits_real: torch.Tensor, logits_fake: torch.Tensor) -> torch.Tensor:
    """Hinge loss for discrminator.

    This function is borrowed from
    https://github.com/CompVis/taming-transformers/blob/master/taming/modules/losses/vqperceptual.py#L20
    """
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def compute_lecam_loss(
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

class Codec_loss(torch.nn.Module):
    def __init__(
        self,
        config
    ):
        super().__init__()
        loss_config = config.losses
        self.discriminator =  VideoDiscriminator()

        self.perceptual_loss = PerceptualLoss(
            loss_config.perceptual_loss).eval()

        self.discriminator_factor = loss_config.discriminator_factor
        self.discriminator_weight = loss_config.discriminator_weight
        self.lecam_regularization_weight = loss_config.lecam_regularization_weight
        self.lecam_ema_decay = loss_config.get("lecam_ema_decay", 0.999)
        if self.lecam_regularization_weight > 0.0:
            self.register_buffer("ema_real_logits_mean", torch.zeros((1)))
            self.register_buffer("ema_fake_logits_mean", torch.zeros((1)))
        self.should_discriminator_be_trained = False
        self.config = config

        logvar_init = loss_config.get("logvar_init", 0.0)
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init, requires_grad=False)
        self.g_gan_adapt_layer = None
        self.gan_loss = GANLoss()

    # @autocast(enabled=False)
    def forward(self,
                output,
                curr_lmbdas,
                curr_gan_scalers,
                inputs,
                p_frame_num,
                net_g=None,
                prev_frame=None,
                mode="generator",
                ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        # Both inputs and reconstructions are in range [0, 1].
        B = inputs.shape[0]
        # inputs = inputs.float()
        # reconstructions = output['x_hat'].float()
        inputs = inputs
        reconstructions = output['x_hat']
        if reconstructions.dim() == 5:
            # If reconstructions are in shape (B, T, C, H, W), we need to rearrange them.
            reconstructions = rearrange(reconstructions, 'b t c h w -> (b t) c h w')
        assert inputs.dim() == 5, "Inputs should be 5D tensor (B, T, C, H, W)"
        inputs = rearrange(inputs, 'b t c h w -> (b t) c h w')
        if prev_frame != None:
            assert prev_frame.dim() == 5, "Inputs should be 5D tensor (B, T, C, H, W)"
            prev_frame = rearrange(prev_frame, 'b t c h w -> (b t) c h w')

        if mode == "generator":
            return self._forward_generator(inputs, reconstructions, output,lmbdas=curr_lmbdas, p_frame_num=p_frame_num,B=B,curr_gan_scalers=curr_gan_scalers,net_g=net_g)
        elif mode == "discriminator":
            return self._forward_discriminator(inputs, output, B, p_frame_num, prev_frame)
        else:
            raise ValueError(f"Unsupported mode {mode}")

    def _forward_generator(self,
                           inputs,
                           reconstructions,
                           output,
                           lmbdas,
                           p_frame_num,
                           B,
                           curr_gan_scalers,
                           net_g,
                           ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        """Generator training step."""
        bpp = output["bpp"].view(B, p_frame_num)
        inputs = inputs.contiguous()
        reconstructions = reconstructions.contiguous()
        reconstruction_loss = cal_mix_Loss(inputs, reconstructions)
        # Compute perceptual loss.
        perceptual_loss = self.perceptual_loss(inputs, reconstructions).mean(dim=(1, 2, 3))

        reconstruction_loss = (reconstruction_loss.view(B, p_frame_num)*output['rd']).mean(dim=1)  # B
        perceptual_loss = perceptual_loss.view(B, p_frame_num).mean(dim=1) # B


        # Compute discriminator loss.
        generator_loss = torch.zeros((), device=inputs.device)
        discriminator_factor = self.discriminator_factor if self.should_discriminator_be_trained else 0
        d_weight = 1.0
        if discriminator_factor > 0.0 and self.discriminator_weight > 0.0:
             # Disable discriminator gradients.
            for param in self.discriminator.parameters():
                param.requires_grad = False
            inp_disc = (reconstructions, output["prev_x_hat"].view(B * p_frame_num, 3, inputs.shape[2], inputs.shape[3]))
            fake_g_pred = self.discriminator(inp_disc)
            generator_loss = self.gan_loss(fake_g_pred, True, is_disc=False)
            generator_loss = generator_loss.view(B, p_frame_num).mean(dim=1) # B
            if self.g_gan_adapt_layer is not None:
                d_weight = calculate_adaptive_weight(net_g, reconstruction_loss, perceptual_loss, generator_loss, g_gan_adapt_layer=self.g_gan_adapt_layer) # B
            else:
                d_weight = 0.01

        d_weight *= self.discriminator_weight
        distortion_loss = reconstruction_loss*self.reconstruction_weight/ torch.exp(self.logvar) + perceptual_loss * self.perceptual_weight + \
            d_weight * discriminator_factor * generator_loss * curr_gan_scalers
        bpp_loss = bpp.mean(dim=1) * self.bpp_weight
        losses = lmbdas * distortion_loss + bpp_loss
        loss = torch.mean(losses)
        bpp_q =output["bpp_q"].view(B)
        bpp_q_y = output["bpp_q_y"].view(B)
        bpp_q_z = output["bpp_q_z"].view(B)
        loss_dict = dict(
            total_loss=loss.clone().detach(),
            reconstruction_loss=reconstruction_loss.detach(),
            perceptual_loss=(self.perceptual_weight * perceptual_loss).detach(),
            weighted_gan_loss=(d_weight * discriminator_factor * generator_loss).detach(),
            discriminator_factor=torch.tensor(discriminator_factor),
            d_weight=d_weight,
            bpp_loss=bpp_loss.detach(),
            gan_loss=generator_loss.detach(),
            bpp_q=bpp_q.detach(),
            bpp_q_y=bpp_q_y.detach(),
            bpp_q_z=bpp_q_z.detach(),
        )

        return loss, loss_dict
    
    def update_loss_scheme(self, loss_scheme):
        if loss_scheme == "mse":
            # latent space RD loss (feature MSE + Code Pred.)
            self.reconstruction_weight = 1.0
            self.perceptual_weight = 1.1
            self.should_discriminator_be_trained = False
            self.bpp_weight = 0.0
        elif loss_scheme == "rdc_mse":
            # latent space RD loss (feature MSE only)
            self.reconstruction_weight = 1.0
            self.perceptual_weight = 1.1
            self.should_discriminator_be_trained = False
            self.bpp_weight = 1.0
        elif loss_scheme == "rdc_mse_gan":
            self.reconstruction_weight = 1.0
            self.perceptual_weight = 1.1
            self.should_discriminator_be_trained = True
            self.g_gan_adapt_layer = "codec_recon"
            self.bpp_weight = 1.0
        else:
            assert False

    def _forward_discriminator(self,
                               inputs,
                               output,
                               B,
                               p_frame_num,
                               prev_frame,
                               ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        """Discrminator training step."""
        discriminator_factor = self.discriminator_factor if self.should_discriminator_be_trained else 0
        loss_dict = {}
        # Turn the gradients on.
        for param in self.discriminator.parameters():
            param.requires_grad = True
        inp_disc = (inputs, prev_frame)
        losses_d_real, loss_d_real = self.discriminator_loss(inp_disc, True)
        
        inp_disc = (output["x_hat"].view(B * p_frame_num, 3, inputs.shape[2], inputs.shape[3]).detach(), output["prev_x_hat"].view(B * p_frame_num, 3, inputs.shape[2], inputs.shape[3]).detach())

        losses_d_fake, loss_d_fake = self.discriminator_loss(inp_disc, False)
        
        loss_d = (loss_d_real + loss_d_fake)* discriminator_factor
        # optional lecam regularization
        lecam_loss = torch.zeros((), device=inputs.device)
        if self.lecam_regularization_weight > 0.0:
            lecam_loss = compute_lecam_loss(
                torch.mean(loss_d_real),
                torch.mean(loss_d_fake),
                self.ema_real_logits_mean,
                self.ema_fake_logits_mean
            ) * self.lecam_regularization_weight

            self.ema_real_logits_mean = self.ema_real_logits_mean * self.lecam_ema_decay + torch.mean(loss_d_real).detach()  * (1 - self.lecam_ema_decay)
            self.ema_fake_logits_mean = self.ema_fake_logits_mean * self.lecam_ema_decay + torch.mean(loss_d_fake).detach()  * (1 - self.lecam_ema_decay)

        loss_d += lecam_loss

        loss_dict = dict(
            discriminator_loss=loss_d.detach(),
            logits_real=loss_d_real.detach().mean(),
            logits_fake=loss_d_fake.detach().mean(),
            lecam_loss=lecam_loss.detach(),
        )
        return loss_d, loss_dict

    def discriminator_loss(self,data, target_is_real):
        if target_is_real:
            losses = self.gan_loss(self.discriminator(data), True, is_disc=True)
        else:
            losses = self.gan_loss(self.discriminator(data), False, is_disc=True)
        loss = torch.sum(losses) / losses.shape[0]
        return losses, loss



class Reconstruction_loss(torch.nn.Module):
    def __init__(
        self,
        config
    ):
        super().__init__()
        loss_config = config.losses
        self.discriminator = NLayerDiscriminator()

        self.reconstruction_loss = loss_config.reconstruction_loss
        self.reconstruction_weight = loss_config.reconstruction_weight
        self.perceptual_loss = PerceptualLoss(
            loss_config.perceptual_loss).eval()
        self.perceptual_weight = loss_config.perceptual_weight
        self.discriminator_iter_start = loss_config.discriminator_start

        self.discriminator_factor = loss_config.discriminator_factor
        self.discriminator_weight = loss_config.discriminator_weight
        self.lecam_regularization_weight = loss_config.lecam_regularization_weight
        self.lecam_ema_decay = loss_config.get("lecam_ema_decay", 0.999)
        if self.lecam_regularization_weight > 0.0:
            self.register_buffer("ema_real_logits_mean", torch.zeros((1)))
            self.register_buffer("ema_fake_logits_mean", torch.zeros((1)))

        self.config = config
        self.quantize_mode = config.model.vq_model.get("quantize_mode", "vae")
        
        if self.quantize_mode == "vae":
            self.kl_weight = loss_config.get("kl_weight", 1e-6)
            logvar_init = loss_config.get("logvar_init", 0.0)
            self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init, requires_grad=False)

    @autocast(enabled=False)
    def forward(self,
                inputs: torch.Tensor,
                reconstructions: torch.Tensor,
                extra_result_dict: Mapping[Text, torch.Tensor],
                global_step: int,
                mode: str = "generator",
                ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        # Both inputs and reconstructions are in range [0, 1].
        inputs = inputs.float()
        reconstructions = reconstructions.float()

        if mode == "generator":
            return self._forward_generator(inputs, reconstructions, extra_result_dict, global_step)
        elif mode == "discriminator":
            return self._forward_discriminator(inputs, reconstructions, global_step)
        else:
            raise ValueError(f"Unsupported mode {mode}")
   
    def should_discriminator_be_trained(self, global_step : int):
        return global_step >= self.discriminator_iter_start
    
    def _forward_generator(self,
                           inputs: torch.Tensor,
                           reconstructions: torch.Tensor,
                           extra_result_dict: Mapping[Text, torch.Tensor],
                           global_step: int
                           ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        """Generator training step."""
        inputs = inputs.contiguous()
        reconstructions = reconstructions.contiguous()
        if self.reconstruction_loss == "l1":
            reconstruction_loss = F.l1_loss(inputs, reconstructions, reduction="mean")
        elif self.reconstruction_loss == "l2":
            reconstruction_loss = F.mse_loss(inputs, reconstructions, reduction="mean")
        else:
            raise ValueError(f"Unsuppored reconstruction_loss {self.reconstruction_loss}")
        reconstruction_loss *= self.reconstruction_weight

        # Compute perceptual loss.
        perceptual_loss = self.perceptual_loss(inputs, reconstructions).mean()

        # Compute discriminator loss.
        generator_loss = torch.zeros((), device=inputs.device)
        discriminator_factor = self.discriminator_factor if self.should_discriminator_be_trained(global_step) else 0
        d_weight = 1.0
        if discriminator_factor > 0.0 and self.discriminator_weight > 0.0:
            # Disable discriminator gradients.
            for param in self.discriminator.parameters():
                param.requires_grad = False
            logits_fake = self.discriminator(reconstructions)
            generator_loss = -torch.mean(logits_fake)

        d_weight *= self.discriminator_weight


        assert self.quantize_mode == "vae", "Only VAE quantize mode supports KL loss."
        # Compute kl loss.
        reconstruction_loss = reconstruction_loss / torch.exp(self.logvar)
        posteriors = extra_result_dict
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        total_loss = (
            reconstruction_loss
            + self.perceptual_weight * perceptual_loss
            + self.kl_weight * kl_loss
            + d_weight * discriminator_factor * generator_loss
        )
        loss_dict = dict(
            total_loss=total_loss.clone().detach(),
            reconstruction_loss=reconstruction_loss.detach(),
            perceptual_loss=(self.perceptual_weight * perceptual_loss).detach(),
            kl_loss=(self.kl_weight * kl_loss).detach(),
            weighted_gan_loss=(d_weight * discriminator_factor * generator_loss).detach(),
            discriminator_factor=torch.tensor(discriminator_factor),
            d_weight=d_weight,
            gan_loss=generator_loss.detach(),
        )


        return total_loss, loss_dict

    def _forward_discriminator(self,
                               inputs: torch.Tensor,
                               reconstructions: torch.Tensor,
                               global_step: int,
                               ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        """Discrminator training step."""
        discriminator_factor = self.discriminator_factor if self.should_discriminator_be_trained(global_step) else 0
        loss_dict = {}
        # Turn the gradients on.
        for param in self.discriminator.parameters():
            param.requires_grad = True

        real_images = inputs.detach().requires_grad_(True)
        logits_real = self.discriminator(real_images)
        logits_fake = self.discriminator(reconstructions.detach())

        discriminator_loss = discriminator_factor * hinge_d_loss(logits_real=logits_real, logits_fake=logits_fake)

        # optional lecam regularization
        lecam_loss = torch.zeros((), device=inputs.device)
        if self.lecam_regularization_weight > 0.0:
            lecam_loss = compute_lecam_loss(
                torch.mean(logits_real),
                torch.mean(logits_fake),
                self.ema_real_logits_mean,
                self.ema_fake_logits_mean
            ) * self.lecam_regularization_weight

            self.ema_real_logits_mean = self.ema_real_logits_mean * self.lecam_ema_decay + torch.mean(logits_real).detach()  * (1 - self.lecam_ema_decay)
            self.ema_fake_logits_mean = self.ema_fake_logits_mean * self.lecam_ema_decay + torch.mean(logits_fake).detach()  * (1 - self.lecam_ema_decay)
        
        discriminator_loss += lecam_loss

        loss_dict = dict(
            discriminator_loss=discriminator_loss.detach(),
            logits_real=logits_real.detach().mean(),
            logits_fake=logits_fake.detach().mean(),
            lecam_loss=lecam_loss.detach(),
        )
        return discriminator_loss, loss_dict
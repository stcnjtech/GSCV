import torch
import torch.nn as nn
import torch.nn.functional as F
from lpips import LPIPS
from taming.modules.discriminator.model import NLayerDiscriminator, weights_init


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss


class VQLPIPSWithDiscriminator(nn.Module):
    def __init__(self, codebook_weight=1.0, pixelloss_weight=1.0, perceptual_weight=1.0, 
                 disc_num_layers=3, disc_in_channels=3, disc_weight=1.0,
                 use_actnorm=False, disc_conditional=False,
                 disc_ndf=64, disc_loss="hinge", adaptive_disc_max=1.0e4, data_range_01=False):
        super().__init__()
        # codebook part
        self.codebook_weight = codebook_weight

        # pixel and perceptual part
        self.data_range_01 = data_range_01
        self.pixel_weight = pixelloss_weight
        self.perceptual_weight = perceptual_weight
        self.pixel_loss_fn = nn.L1Loss(reduce="mean")
        self.perceptual_loss_fn = LPIPS(net='vgg').eval()

        # disc part
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional
        self.adaptive_disc_max = adaptive_disc_max
        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm,
                                                 ndf=disc_ndf
                                                 ).apply(weights_init)
        assert disc_loss in ["hinge", "vanilla"]
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
        print(f"VQLPIPSWithDiscriminator running with {disc_loss} loss.")


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
                global_step, last_layer=None, cond=None, split="train"):
        inputs = inputs.contiguous()
        reconstructions = reconstructions.contiguous()
        codebook_loss = codebook_loss.mean()
        pixel_loss = self.pixel_loss_fn(inputs, reconstructions).mean()
        perceptual_loss = self.perceptual_loss_fn(inputs, reconstructions, normalize=self.data_range_01).mean()
        nll_loss = pixel_loss * self.pixel_weight + perceptual_loss * self.perceptual_weight

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            g_loss = -torch.mean(logits_fake)

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
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                logits_real = self.discriminator(inputs.detach())
                logits_fake = self.discriminator(reconstructions.detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs.detach(), cond.detach()), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.detach(), cond.detach()), dim=1))

            d_loss = self.disc_loss(logits_real, logits_fake).mean()

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return d_loss, log


class VQLPIPSWithDiscriminator_sq_vq(VQLPIPSWithDiscriminator):
    def __init__(self, sq_weight=1.0, codebook_weight=1., pixelloss_weight=1., perceptual_weight=1., 
                 disc_num_layers=3, disc_in_channels=3, disc_weight=1., use_actnorm=False, 
                 disc_conditional=False, disc_ndf=64, disc_loss="hinge",
                 adaptive_disc_max=10000, data_range_01=False):
        super().__init__(codebook_weight, pixelloss_weight, perceptual_weight, 
                         disc_num_layers, disc_in_channels, disc_weight, use_actnorm, 
                         disc_conditional, disc_ndf, disc_loss, 
                         adaptive_disc_max, data_range_01)
        self.sq_weight = sq_weight
    
    def forward(self, codebook_loss, sq_loss, inputs, reconstructions, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train"):
        loss, log = super().forward(codebook_loss, inputs, reconstructions, optimizer_idx,
                        global_step, last_layer, cond, split)
        if optimizer_idx == 0:
            sq_loss = sq_loss.mean()
            loss += self.sq_weight * sq_loss
            log["{}/sq_loss".format(split)] = sq_loss.detach()
            log["{}/sq_lambda".format(split)] = self.sq_weight
        return loss, log




def pixshuffle_up2x(dim):
    return nn.Sequential(
        nn.Conv2d(dim, dim*4, kernel_size=1),
        nn.PixelShuffle(2),
    )


class NLayerDiscriminator_vqgan_cond(NLayerDiscriminator):
    def __init__(self, input_nc=3, cond_ch=128, cond_down='16x', 
                 ndf=64, n_layers=3, use_actnorm=False):
        super().__init__(input_nc+(cond_ch//4), ndf, n_layers, use_actnorm)
        assert cond_down in ['16x', '8x']
        cond_layer = [
            nn.Conv2d(cond_ch, cond_ch, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            pixshuffle_up2x(cond_ch),
            nn.Conv2d(cond_ch, cond_ch//2, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            pixshuffle_up2x(cond_ch//2),
            nn.Conv2d(cond_ch//2, cond_ch//4, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            pixshuffle_up2x(cond_ch//4),
            nn.Conv2d(cond_ch//4, cond_ch//4, kernel_size=3, padding=1),
        ]

        if cond_down=='16x':
            cond_layer = [
                nn.Conv2d(cond_ch, cond_ch, kernel_size=3, padding=1),
                nn.LeakyReLU(),
                pixshuffle_up2x(cond_ch),
            ] + cond_layer
        
        self.cond_transform = nn.Sequential(*cond_layer)
    
    def forward(self, input, cond):
        cond = self.cond_transform(cond)
        return self.main(torch.concat([input, cond], dim=1))


class VQLPIPSWithDiscriminator_vqgan_cond(nn.Module):
    def __init__(self, codebook_weight=1.0, pixelloss_weight=1.0, perceptual_weight=1.0, 
                 disc_num_layers=3, disc_in_channels=3, disc_weight=1.0,
                 use_actnorm=False, disc_ndf=64, disc_loss="hinge", 
                 cond_ch=128, vqgan_down="16x",
                 adaptive_disc_max=1.0e4, data_range_01=False):
        super().__init__()
        # codebook part
        self.codebook_weight = codebook_weight

        # pixel and perceptual part
        self.data_range_01 = data_range_01
        self.pixel_weight = pixelloss_weight
        self.perceptual_weight = perceptual_weight
        self.pixel_loss_fn = nn.L1Loss(reduce="mean")
        self.perceptual_loss_fn = LPIPS(net='vgg').eval()

        # disc part
        self.discriminator_weight = disc_weight
        self.adaptive_disc_max = adaptive_disc_max
        self.discriminator = NLayerDiscriminator_vqgan_cond(input_nc=disc_in_channels,
                                                n_layers=disc_num_layers,
                                                use_actnorm=use_actnorm,
                                                ndf=disc_ndf,
                                                cond_ch=cond_ch,
                                                cond_down=vqgan_down
                                                ).apply(weights_init)
        assert disc_loss in ["hinge", "vanilla"]
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
        print(f"VQLPIPSWithDiscriminator running with {disc_loss} loss.")


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


    def forward(self, codebook_loss, inputs, reconstructions, cond, optimizer_idx,
                global_step, last_layer=None, split="train"):
        inputs = inputs.contiguous()
        reconstructions = reconstructions.contiguous()
        codebook_loss = codebook_loss.mean()
        pixel_loss = self.pixel_loss_fn(inputs, reconstructions).mean()
        perceptual_loss = self.perceptual_loss_fn(inputs, reconstructions, normalize=self.data_range_01).mean()
        nll_loss = pixel_loss * self.pixel_weight + perceptual_loss * self.perceptual_weight

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            logits_fake = self.discriminator(reconstructions, cond)
            g_loss = -torch.mean(logits_fake)

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
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            logits_real = self.discriminator(inputs.detach(), cond.detach())
            logits_fake = self.discriminator(reconstructions.detach(), cond.detach())
            d_loss = self.disc_loss(logits_real, logits_fake).mean()

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return d_loss, log


class VQLPIPSWithDiscriminator_sq_vq_vqgan_cond(VQLPIPSWithDiscriminator_vqgan_cond):
    def __init__(self, sq_weight=1., codebook_weight=1., pixelloss_weight=1., perceptual_weight=1., 
                 disc_num_layers=3, disc_in_channels=3, disc_weight=1, use_actnorm=False, 
                 disc_ndf=64, disc_loss="hinge", cond_ch=128, vqgan_down="16x", adaptive_disc_max=10000, data_range_01=False):
        super().__init__(codebook_weight, pixelloss_weight, perceptual_weight, 
                         disc_num_layers, disc_in_channels, disc_weight, use_actnorm, 
                         disc_ndf, disc_loss, cond_ch, vqgan_down, adaptive_disc_max, data_range_01)
        self.sq_weight = sq_weight

    def forward(self, codebook_loss, sq_loss, inputs, reconstructions, cond, optimizer_idx,
                global_step, last_layer=None, split="train"):
        loss, log = super().forward(codebook_loss, inputs, reconstructions, cond, optimizer_idx,
                        global_step, last_layer, split)
        if optimizer_idx == 0:
            sq_loss = sq_loss.mean()
            loss += self.sq_weight * sq_loss
            log["{}/sq_loss".format(split)] = sq_loss.detach()
            log["{}/sq_lambda".format(split)] = self.sq_weight
        return loss, log
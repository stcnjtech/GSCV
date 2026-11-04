import torch
import torch.nn as nn

from torch.nn.utils import  spectral_norm

class CkptModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.use_ckpt = False

    def set_use_ckpt(self, use_ckpt=True):
        self.use_ckpt = use_ckpt

    def internal_forward(self, *args):
        raise NotImplementedError

    def forward(self, *args):
        if self.use_ckpt:
            return torch.utils.checkpoint.checkpoint(self.custom_forward(self.internal_forward),
                                                     *args, preserve_rng_state=False,
                                                     use_reentrant=False)
        return self.internal_forward(*args)

    @staticmethod
    def custom_forward(f):
        def custom_func(*args):
            return f(*args)
        return custom_func


def convert_to_syncbatchnorm(model, process_group=None):
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            sync_bn_network = nn.SyncBatchNorm.convert_sync_batchnorm(module, process_group)
            sync_bn_network.track_running_stats = False
            setattr(model, name, sync_bn_network)
        else:
            convert_to_syncbatchnorm(module, process_group)


class Conv2d(nn.Module):
    def __init__(self, nch_in, nch_out, kernel_size=4, stride=1, padding=1, bias=True, snorm=False):
        super(Conv2d, self).__init__()
        if snorm:
            # self.conv = SpectralNorm(nn.Conv2d(nch_in, nch_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
            self.conv = spectral_norm(nn.Conv2d(nch_in, nch_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        else:
            self.conv = nn.Conv2d(nch_in, nch_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        return self.conv(x)

class Norm2d(nn.Module):
    def __init__(self, nch, norm_mode):
        super(Norm2d, self).__init__()
        if norm_mode == 'bnorm':
            self.norm = nn.BatchNorm2d(nch)
        elif norm_mode == 'inorm':
            self.norm = nn.InstanceNorm2d(nch)

    def forward(self, x):
        return self.norm(x)

class ReLU(nn.Module):
    def __init__(self, relu):
        super(ReLU, self).__init__()
        if relu > 0:
            self.relu = nn.LeakyReLU(relu, True)
        elif relu == 0:
            self.relu = nn.ReLU(True)

    def forward(self, x):
        return self.relu(x)

class CNR2d(CkptModule):
    def __init__(self, nch_in, nch_out, kernel_size=4, stride=1, padding=1, norm='bnorm', relu=0.0, drop=[], bias=[], snorm=False):
        super().__init__()

        if bias == []:
            if norm == 'bnorm':
                bias = False
            else:
                bias = True

        layers = []
        layers += [Conv2d(nch_in, nch_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, snorm=snorm)]

        # if snorm:
        #     layers += [SpectralNorm(layers[-1].conv)]

        if norm != []:
            layers += [Norm2d(nch_out, norm)]

        if relu != []:
            layers += [ReLU(relu)]

        if drop != []:
            layers += [nn.Dropout2d(drop)]

        self.cbr = nn.Sequential(*layers)

    def internal_forward(self, x):
        return self.cbr(x)



class VideoDiscriminator(CkptModule):
    def __init__(self, nch_in=3, nch_ker=64, norm='bnorm', latent_dim=256):
        super(VideoDiscriminator, self).__init__()
        self.nch_in = nch_in
        self.nch_ker = nch_ker
        self.norm = norm

        self.latent_dim = latent_dim

        self.activation = nn.LeakyReLU(negative_slope=0.2)
        
        self.dsc1 = CNR2d(2 * self.nch_in,  1 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.2, snorm=True)
        self.dsc2 = CNR2d(1 * self.nch_ker, 2 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.2, snorm=True)
        self.dsc3 = CNR2d(2 * self.nch_ker, 4 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.2, snorm=True)
        self.dsc4 = CNR2d(4 * self.nch_ker, 8 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.2, snorm=True)
        self.dsc5 = CNR2d(8 * self.nch_ker, 1,                kernel_size=4, stride=1, padding=1, norm=[],        relu=[], bias=False)

        # self.set_use_ckpt(True)

    def set_use_ckpt(self, use_ckpt=True):
        self.dsc1.set_use_ckpt(use_ckpt)
        self.dsc2.set_use_ckpt(use_ckpt)
        self.dsc3.set_use_ckpt(use_ckpt)
        self.dsc4.set_use_ckpt(use_ckpt)
        self.dsc5.set_use_ckpt(use_ckpt)

    def convert_to_syncbatchnorm(self):
        convert_to_syncbatchnorm(self)

    def internal_forward(self, inp_disc):
        """
        x: Concatenated real/gen images
        y: Quantized latents
        """

        x, prev_x = inp_disc

        # Concatenate upscaled encoder output y as contextual information
        x = torch.cat((prev_x, x), dim=1)

        x = self.dsc1(x)
        x = self.dsc2(x)
        x = self.dsc3(x)
        x = self.dsc4(x)
        x = self.dsc5(x)

        # x = torch.sigmoid(x)
        
        return x
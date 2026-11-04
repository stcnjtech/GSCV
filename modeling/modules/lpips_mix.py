import torch.nn as nn
from .lpips import LPIPS as LPIPS_vgg
from .lpips_alex.lpips import LPIPS as LPIPS_alex
class LPIPS_mix(nn.Module):
    # Learned perceptual metric.
    def __init__(self):
        super(LPIPS_mix, self).__init__()
        self.Lpips_vgg = LPIPS_vgg().eval()
        self.Lpips_alex = LPIPS_alex(net='alex', version='0.1').eval()

    def forward(self, input, target):
        loss_vgg = self.Lpips_vgg(input, target)
        loss_alex = self.Lpips_alex(input, target)
        loss_mix = loss_vgg * 0.5+ loss_alex * 0.5
        return loss_mix
if __name__ == '__main__':
    import torch
    input = torch.randn(2, 3, 64, 64)
    target = torch.randn(2, 3, 64, 64)
    model = LPIPS_mix()
    output = model(input, target)
    print(output.shape)
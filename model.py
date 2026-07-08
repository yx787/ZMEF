import torch
import torch.nn as nn
from guided_filter_pytorch.guided_filter import FastGuidedFilter
import torch.nn.functional as F


class Res_block(nn.Module):
    def __init__(self, nFeat, kernel_size=3):
        super(Res_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(nFeat, nFeat, kernel_size=kernel_size, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(nFeat, nFeat, kernel_size=kernel_size, padding=1, bias=True),
        )

    def forward(self, x):
        return self.conv(x) + x


class Res_blocks(nn.Module):
    def __init__(self, nFeat, nReslayer):
        super(Res_blocks, self).__init__()
        modules = []
        for i in range(nReslayer):
            modules.append(Res_block(nFeat))
        self.dense_layers = nn.Sequential(*modules)

    def forward(self, x):
        out = self.dense_layers(x)
        return out


class MEF(nn.Module):
    def __init__(self, filters_in=3, filters_out=3, nFeat=32, radius=1, eps=1e-4, scale=16):
        super(MEF, self).__init__()
        self.gf = FastGuidedFilter(radius, eps)
        self.scale = scale

        # encoder
        self.conv = nn.Sequential(
            nn.Conv2d(filters_in, nFeat, 3, 1, 1, bias=True),
            nn.ReLU(inplace=True),
            Res_blocks(nFeat, 3),
            nn.Conv2d(nFeat, filters_out, 3, 1, 1, bias=True)
        )

    def forward(self, img):
        _, _, H, W = img.shape
        img = self.padding(img, self.scale)
        img_lr = F.interpolate(img, scale_factor=1 / self.scale, mode='bilinear')
        out = self.conv(img_lr)
        out = self.gf(img_lr, out, img) + img
        out = out[:, :, :H, :W]
        return torch.clamp(out, 0, 1)

    def padding(self, img, scale=16):
        _, _, H, W = img.shape
        h_padding = ((H // scale + 1) * scale - H) % scale
        w_padding = ((W // scale + 1) * scale - W) % scale
        padding = nn.ReflectionPad2d((0, w_padding, 0, h_padding))
        return padding(img)

import torch
import torch.nn as nn
import torchvision.models as py_models
import numpy as np
from pytorch_msssim import ssim
from math import exp
import torch.nn.functional as F
from util.mefssim import *
torch.cuda.set_device(1)

laplace = torch.tensor([[1 / 8, 1 / 8, 1 / 8],
                        [1 / 8,    -1, 1 / 8],
                        [1 / 8, 1 / 8, 1 / 8]]).view(1, 1, 3, 3).cuda().type(torch.cuda.FloatTensor)

mefssim_loss = MEF_MSSSIM(is_lum=False)
# gamma = [0.6, 0.8, 1.0, 1.3, 1.5]
gamma = [0.8, 0.9, 1.0, 1.3, 1.5]


def mef_loss(output, imgs):
    K, _, _, _ = imgs.size()
    imgs_g = []
    for i in range(K):
        for j in range(len(gamma)):
            imgs_g.append(imgs[i, :, :, :] ** gamma[j])
    imgs_g = torch.stack(imgs_g, dim=0)
    return mefssim_loss(output, imgs_g)



def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss / (gauss.sum())


def create_window(window_size, channel, gauss=False):
    if gauss:
        _1D_window = gaussian(window_size, window_size/6.).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    else:
        _2D_window = torch.ones([window_size, window_size]).float().unsqueeze(0).unsqueeze(0) / (window_size ** 2)
    window = torch.Tensor(_2D_window.expand(1, channel, window_size, window_size).contiguous()) / channel
    return window


class GradientLoss():
    def __init__(self, loss=nn.L1Loss(), n_scale=3):
        super(GradientLoss, self).__init__()
        self.downsample = nn.AvgPool2d(2, stride=2)
        self.criterion = loss
        self.n_scale = n_scale
        self.window = create_window(window_size=11, channel=3, gauss=False)

    def grad_xy(self, img):
        gradient_x = img[:, :, :, :-1] - img[:, :, :, 1:]
        gradient_y = img[:, :, :-1, :] - img[:, :, 1:, :]
        return gradient_x, gradient_y

    def grad_loss(self, fakeIm, realIm):
        loss = 0
        for i in range(self.n_scale):
            fakeIm_ = F.conv2d(fakeIm, self.window, stride=11)
            realIm_ = F.conv2d(realIm, self.window, stride=11)
            grad_fx, grad_fy = self.grad_xy(fakeIm_)
            grad_rx, grad_ry = self.grad_xy(realIm_)
            loss += pow(4, i) * (self.criterion(grad_fx, grad_rx) + self.criterion(grad_fy, grad_ry))
            fakeIm = self.downsample(fakeIm)
            realIm = self.downsample(realIm)
        return loss

    def getloss(self, img, imgs):
        self.window = self.window.cuda(img.get_device()).type_as(img)
        num = imgs.shape[1]
        loss = 0
        for i in range(num):
            loss += self.grad_loss(img, imgs[:, i, :, :, :]) / num
        return loss


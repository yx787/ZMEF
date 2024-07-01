from util.mefssim import MEF_MSSSIM
import os
import glob
import cv2
import torch
import numpy as np
from torchvision import transforms

# transform = transforms.ToTensor()
# mefssimc = MEF_MSSSIM(is_lum=False)
# dataset_path = 'D:/MEF/Dynamic_RGB/IF2021_benchmark_MEF/MEFB-main/input/'
# results_path = 'D:/实验室/论文/多曝光融合/MESPD_TCSVT-2021-main/results_MEFB/'
# dirpath = sorted(os.listdir(dataset_path))
# mef_vals = 0.0
#
# for scene in dirpath:
#     print(results_path + scene + '.png')
#     fused_img = transform(cv2.imread(results_path + scene + '.png')).type(torch.FloatTensor).unsqueeze(0)
#     imgs_path = dataset_path + scene
#     path1 = glob.glob(imgs_path + '/*_A*')[0]
#     path2 = glob.glob(imgs_path + '/*_B*')[0]
#     img1 = transform(cv2.imread(path1)).type(torch.FloatTensor)
#     img2 = transform(cv2.imread(path2)).type(torch.FloatTensor)
#     imgs = [img1, img2]
#     imgs = torch.stack(imgs, 0).contiguous()
#     cur = mefssimc(fused_img, imgs).data.numpy()
#     mef_vals += cur / 100.0
#
# print('Average MEFSSIM_C: {}'.format(mef_vals))



transform = transforms.ToTensor()
mefssimc = MEF_MSSSIM(is_lum=False)
dataset_path = '/data5/tanxiao/dataset_SICE/Test_2E/'
results_path = '/data5/tanxiao/results_SICE/DSIFT/'
dirpath = sorted(os.listdir(dataset_path))
mef_vals = 0.0
idx = 1

for scene in dirpath:
    print(results_path + scene + '.jpg')
    fused_img = transform(cv2.imread(results_path + '{:03d}.jpg'.format(idx))).cuda().type(torch.cuda.FloatTensor).unsqueeze(0)
    imgs_path = dataset_path + scene
    img1 = transform(cv2.imread(imgs_path + '/1.JPG')).cuda().type(torch.cuda.FloatTensor)
    img2 = transform(cv2.imread(imgs_path + '/2.JPG')).cuda().type(torch.cuda.FloatTensor)
    imgs = [img1, img2]
    imgs = torch.stack(imgs, 0).contiguous()
    cur = mefssimc(fused_img, imgs).data.cpu().numpy()
    mef_vals += cur / 20.0
    print('Current MEFSSIM_C: {}'.format(cur))
    idx += 1

print('Average MEFSSIM_C: {}'.format(mef_vals))






from image_utils import psnr
from loss_utils import ssim
import lpips

from PIL import Image
import cv2
import torch
import os
import torchvision.transforms.functional as tf

base_path = "/your/render/img/folder/path"

lpips_fn = lpips.LPIPS(net='vgg')
lpips_fn.cuda()

_psnr = 0
_ssim = 0
_lpips = 0

img_list = os.listdir(os.path.join(base_path, 'renders'))

for img_name in img_list:
    img_gt = tf.to_tensor(Image.open(os.path.join(base_path, 'gt', img_name))).unsqueeze(0)[:,:3,:,:].cuda()
    img_render = tf.to_tensor(Image.open(os.path.join(base_path, 'renders', img_name))).unsqueeze(0)[:,:3,:,:].cuda()
    _ssim += ssim(img_render, img_gt).item()
    _psnr += psnr(img1=img_render,img2=img_gt).item()
    _lpips += lpips_fn(img_gt, img_render).detach().item()
_psnr /= len(img_list)
_ssim /= len(img_list)
_lpips /= len(img_list)

print(f"psnr:\t\t", _psnr)
print(f"ssim:\t\t", _ssim)
print(f"lpips:\t\t", _lpips)
    
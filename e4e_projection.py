# import os
# import sys
# import numpy as np
# from PIL import Image
# import torch
# import torchvision.transforms as transforms
from argparse import Namespace
from e4e.models.psp import pSp
from util import *


@ torch.no_grad()
def projection(img, name, device='cuda'):
    model_path = '../../models/multistyle/e4e_ffhq_encode.pt'
    ckpt = torch.load(model_path, map_location='cpu')
    opts = ckpt['opts']
    opts['checkpoint_path'] = model_path
    opts= Namespace(**opts)
    net = pSp(opts, device).eval().to(device)

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    img = transform(img).unsqueeze(0).to(device)
    images, w_plus = net(img, randomize_noise=False, return_latents=True)
    result_file = {}
    result_file['latent'] = w_plus[0]
    if name is not None:
        torch.save(result_file, name)

    net.cpu()
    del net
    torch.cuda.empty_cache()

    return w_plus[0]


def projection_with_image(img, device='cuda'):
    '''
    assuming the input image is already a torch tensor
    Args:
        img:
        device:

    Returns:

    '''
    model_path = '../../models/multistyle/e4e_ffhq_encode.pt'
    ckpt = torch.load(model_path, map_location='cpu')
    opts = ckpt['opts']
    opts['checkpoint_path'] = model_path
    opts= Namespace(**opts)
    net = pSp(opts, device).eval().to(device)

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(256),
            #transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    img = transform(img)
    images, w_plus = net(img, randomize_noise=False, return_latents=True)
    result_file = {}
    result_file['latent'] = w_plus[0]
    net.cpu()
    del net
    torch.cuda.empty_cache()
    return w_plus[0], images
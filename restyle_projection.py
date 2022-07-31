import os
import sys
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from argparse import Namespace
from restyle_encoder.models.psp import pSp
from util import *

def get_average_image(net):
    avg_image = net(net.latent_avg.unsqueeze(0),
                    input_code=True,
                    randomize_noise=False,
                    return_latents=False,
                    average_code=True)[0]
    avg_image = avg_image.to('cuda').float().detach()
    return avg_image

@ torch.no_grad()
def restyle_projection(img, name, device='cuda'):
    model_path = '../../models/multistyle/restyle_psp_ffhq_encode.pt'
    ckpt = torch.load(model_path, map_location='cpu')
    opts = ckpt['opts']
    opts['checkpoint_path'] = model_path
    opts= Namespace(**opts)
    net = pSp(opts).eval().to(device)
    results = []
    results_latent = []
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    y_hat, latent = None, None
    opts.n_iters_per_batch = 5
    opts.resize_outputs = False  # generate outputs at full resolution
    avg_image = get_average_image(net)
    img = transform(img).unsqueeze(0).to(device)
    avg_image = avg_image.unsqueeze(0)
    for iter in range(opts.n_iters_per_batch):
        if iter == 0:
            x_input = torch.cat([img, avg_image], dim=1)
        else:
            x_input = torch.cat([img, y_hat], dim=1)

        y_hat, latent = net.forward(x_input,
                                    latent=latent,
                                    randomize_noise=False,
                                    return_latents=True,
                                    resize=opts.resize_outputs)


        # store intermediate outputs

        results.append(y_hat)
        results_latent.append(latent.cpu().numpy())

        # resize input to 256 before feeding into next iteration
        y_hat = net.face_pool(y_hat)

    result_file = {}
    result_file['latent'] = latent[0]
    torch.save(result_file, name)

    net.cpu()
    del net
    torch.cuda.empty_cache()

    return latent[0]

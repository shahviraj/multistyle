import scipy.stats
import torch
torch.backends.cudnn.benchmark = True
from torchvision import transforms, utils
from util import *
from PIL import Image
import math
import random
import os
import wandb

from model_utils import *
import numpy as np
from torch import nn, autograd, optim
from torch.nn import functional as F
from tqdm import tqdm
#import wandb
from model import *

from e4e_projection import projection as e4e_projection
from restyle_projection import restyle_projection
from copy import deepcopy

from id_loss import IDLoss
import contextual_loss.functional as FCX

use_wandb = True
run_name = 'check inversion'
run_desc = 'check the output of inversion after performing the inversion mixing for various layers'


hyperparam_defaults = dict(
    names = [
            #'arcane_caitlyn.png',
            #'art.png',
            'arcane_jinx.png',
            'jojo.png',
            'jojo_yasuho.png',
            'sketch.png',
            #'arcane_viktor.png',
            #'jojo_yasuho.png',
            #'sketch.png',
            #'arcane_jayce.png',
            #'arcane_caitlyn.png',
            #'titan.jpeg',
            #'audrey.jpg',
            #'arcane_texture.jpeg',
            #'cheryl.jpg',
            #'flower.jpeg',
            #'elliee.jpeg',
            #'yukako.jpeg',
            #'marilyn.jpg',
            #'water.jpeg',
            #'matisse.jpeg',
             ],#, 'jojo.png'],
    inv_mix = [0],
    preserve_color = False,
    inv_method = 'e4e',
    seed = 9,  # @param {type:"number"},
)

if use_wandb:
    wandb.init(config=hyperparam_defaults,
               project="multistyle_inv_check",
               entity='shahviraj',
               name=run_name,
               dir = '../../outputs/multistyle/wandb/',
               notes=run_desc)
    config = wandb.config
else:
    config = hyperparam_defaults

 #@param {type:"string"}
filepathlist = []
for filename in config['filenamelist']:
    filepathlist.append(f'test_input/{filename}')

latent_dim = 512
device = 'cuda'

# Load original generator
original_generator = Generator(1024, latent_dim, 8, 2).to(device)
ckpt = torch.load('../../models/multistyle/stylegan2-ffhq-config-f.pt', map_location=lambda storage, loc: storage)
original_generator.load_state_dict(ckpt["g_ema"], strict=False)
with torch.no_grad():
    mean_latent = original_generator.mean_latent(10000)

transform = transforms.Compose(
    [
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

def inversion_mixing(latent, config):
    if config['preserve_shape']:
        return latent
    else:
        for j in range(len(latent)):
            if j in config['inv_mix']: # corresp. to 8 and 11 in S space
                 latent[j] = mean_latent
            # elif j in (3, 4,): # corresp. to 3 and 5 in S Space
            #     if j == 3:
            #         alpha = .1
            #     else:
            #         alpha = .3
            #     mouth_latent = (alpha) * mean_latent + (1.0 - alpha)  * latent[j]
            #
            #     latent[j] = mouth_latent
            else:
                latent[j] = latent[j]
        return latent

def prepare_targets(config):

    targets = []
    latents = []

    for name in config['names']:
        name = strip_path_extension(name)
        style_aligned_path = os.path.join('style_images_aligned', f'{name}.png')

        if not os.path.exists(style_aligned_path):
            try:
                style_path = os.path.join('style_images', name+ '.jpg')
                assert os.path.exists(style_path), f"{style_path} does not exist!"
            except:
                style_path = os.path.join('style_images', name + '.jpeg')
                assert os.path.exists(style_path), f"{style_path} does not exist!"

            style_aligned = align_face(style_path)
            style_aligned.save(style_aligned_path)
        else:
            style_aligned = Image.open(style_aligned_path).convert('RGB')

        # GAN invert
        style_code_path = os.path.join(f'../../models/multistyle/{config["inv_method"]}_inversion_codes', f'{name}.pt')
        if not os.path.exists(style_code_path):
            if config['inv_method'] == 'e4e':
                latent = e4e_projection(style_aligned, style_code_path, device)
            elif config['inv_method'] == 'restyle':
                latent = restyle_projection(style_aligned, style_code_path, device)
            else:
                raise NotImplementedError
        else:
            latent = torch.load(style_code_path)['latent']

        latent = inversion_mixing(latent, config)

        targets.append(transform(style_aligned).to(device))
        latents.append(latent.to(device))

    targets = torch.stack(targets, 0)
    latents = torch.stack(latents, 0)

    return targets, latents

targets, latents = prepare_targets(config)

with torch.no_grad():
    inv_styles = original_generator(latents, input_is_latent=True)

    display_image(utils.make_grid(inv_styles, normalize=True, range=(-1, 1)), title='Reference Inversions after mixing',
                  save=False, use_wandb=use_wandb)

    inv_styles = inv_styles.detach()
    del inv_styles
    torch.cuda.empty_cache()


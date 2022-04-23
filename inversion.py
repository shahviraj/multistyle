import torch
torch.backends.cudnn.benchmark = True
from torchvision import transforms, utils
from util import *
from PIL import Image
import math
import random
import os
import wandb

import numpy as np
from torch import nn, autograd, optim
from torch.nn import functional as F
from tqdm import tqdm
from model import *
from e4e_projection import projection as e4e_projection
from restyle_projection import restyle_projection

from copy import deepcopy


run_name = 'two_styles_fixed_bug_learnt_dirs_tanh_identity_init'
run_desc = 'learning two styles at once, using learnable directions using single layer FCN, with tanh activations and identity matrix as initialization'
names = ['arcane_jinx.png', 'jojo.png']
num_iter = 1000
use_wandb = False

learning_rate = 2e-3

n_sample = 5  # @param {type:"number"}
seed = 3000  # @param {type:"number"}


#filename = 'iu.jpeg' #@param {type:"string"}
#filepath = f'test_input/{filename}'

latent_dim = 512
device = 'cuda'



# Inversion using e4e


transform = transforms.Compose(
    [
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

targets = []
latents = []

for name in names:
    name = strip_path_extension(name)
    style_aligned_path = os.path.join('style_images_aligned', f'{name}.png')

    if not os.path.exists(style_aligned_path):
        style_path = os.path.join('style_images', name)
        assert os.path.exists(style_path), f"{style_path} does not exist!"
        style_aligned = align_face(style_path)
        style_aligned.save(style_aligned_path)
    else:
        style_aligned = Image.open(style_aligned_path).convert('RGB')

    # GAN invert
    style_code_path = os.path.join('../../models/multistyle/inversion_codes', f'{name}.pt')
    if not os.path.exists(style_code_path):
        latent = e4e_projection(style_aligned, style_code_path, device)
        latent_restyle = restyle_projection(style_aligned, style_code_path, device)
    else:
        latent_restyle = restyle_projection(style_aligned, style_code_path, device)
        latent = torch.load(style_code_path)['latent']

    targets.append(transform(style_aligned).to(device))
    latents.append(latent.to(device))

targets = torch.stack(targets, 0)
latents = torch.stack(latents, 0)



target_im = utils.make_grid(targets, normalize=True, range=(-1, 1))
display_image(target_im, title='Style References', save=True)


original_generator = Generator(1024, latent_dim, 8, 2).to(device)
ckpt = torch.load('../../models/multistyle/stylegan2-ffhq-config-f.pt', map_location=lambda storage, loc: storage)
original_generator.load_state_dict(ckpt["g_ema"], strict=False)
# @param {type:"slider", min:0, max:1, step:0.1}
#alpha = 1 - alpha

# @markdown Tries to preserve color of original image by limiting family of allowable transformations. Set to false if you want to transfer color from reference image. This also leads to heavier stylization

torch.manual_seed(seed)
z = torch.randn(n_sample, latent_dim, device=device)
original_sample = original_generator([z], truncation=0.7, truncation_latent=mean_latent)
original_my_sample = original_generator(my_w, input_is_latent=True)


generator = deepcopy(original_generator)
del original_generator


# Which layers to swap for generating a family of plausible real images -> fake image
if preserve_color:
    id_swap = [9, 11, 15, 16, 17]
else:
    id_swap = list(range(7, generator.n_latent))

if fake_splatting:
    n_styles = len(names) + 1
else:
    n_styles = len(names)

dirnet = DirNet(latent_dim, latent_dim, n_styles, id_swap, activation=dir_act, device=device).to(device)

if learning:
    g_optim = optim.Adam(list(generator.parameters()) + list(dirnet.parameters()), lr=learning_rate, betas=(0, 0.99))
else:
    g_optim = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0, 0.99))

if use_wandb:
    wandbrun = wandb.init(project="multistyle", entity='shahviraj', name=run_name, dir = '../../outputs/multistyle/wandb/',
               notes='{}. Alpha is {}. Splatting is {}. Stlyes used: {}. Ids swapped: {}.'.format(run_desc, alpha, splatting, names, id_swap))
    config = wandb.config
    config.num_iter = num_iter
    config.preserve_color = preserve_color
    config.learning_rate = learning_rate
    config.alpha = alpha
    config.seed = seed
    config.splatting = splatting
    config.fake_splatting = fake_splatting
    config.n_sample = n_sample
    wandb.log(
        {"Style reference": [wandb.Image(transforms.ToPILImage()(target_im))]},
        step=0)


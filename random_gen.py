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
import glob
use_wandb=False
if use_wandb:
    wandb.init(
               project="multistyle_ctx",
               entity='shahviraj',
               dir='../../outputs/multistyle/wandb/',
                )
latent_dim = 512
device = 'cuda'
n_styles = 4
model_dir = '/home/vjshah3/research/outputs/multistyle/results/'

model_path = glob.glob(model_dir + f'{n_styles}styles_*/*.pt')

# Load original generator
#
generator = Generator(1024, latent_dim, 8, 2).to(device)
ckpt = torch.load(model_path[1], map_location=lambda storage, loc: storage)
generator.load_state_dict(ckpt["g"], strict=False)

config = ckpt['config']

id_swap = config['sty_mix'] + list(range(7, generator.n_latent))
dirnet = DirNet(latent_dim, latent_dim, n_styles, id_swap, init=config['init'], activation=config['dir_act'], device=device).to(device)
dirnet.load_state_dict(ckpt['dirnet'], strict=True)

def prepare_inputs(config):
    my_wlist = []
    aligned_facelist = []
    for i, file in enumerate(config['filenamelist']):
        imgname = strip_path_extension(file)
        unaligned_img_path = f'test_input/{strip_path_extension(file)}'
        aligned_img_path = f'test_input_aligned/{imgname}_aligned.png'

        if not os.path.exists(aligned_img_path):
            try:
                face_path = unaligned_img_path +'.jpg'
                assert os.path.exists(face_path), f"{face_path} does not exist!"
            except:
                face_path = unaligned_img_path + '.jpeg'
                assert os.path.exists(face_path), f"{face_path} does not exist!"

            aligned_face = align_face(face_path)
            aligned_face.save(aligned_img_path)
        else:
            aligned_face = Image.open(aligned_img_path).convert('RGB')

        aligned_facelist.append(aligned_face)

        inv_code_path = os.path.join(f'../../models/multistyle/{config["inv_method"]}_inversion_codes', f'{imgname}_aligned.pt')
        if os.path.exists(inv_code_path):
            my_w = torch.load(inv_code_path)['latent']
        else:

            if config['inv_method'] == 'e4e':
                my_w = e4e_projection(aligned_face, inv_code_path, device)
            elif config['inv_method'] == 'restyle':
                my_w = restyle_projection(aligned_face, inv_code_path, device)
            else:
                raise NotImplementedError
        my_wlist.append(my_w.to(device))
    my_ws = torch.stack(my_wlist, 0)
    return my_ws, aligned_facelist

my_ws, aligned_facelist = prepare_inputs(config)

stylized_my_w = dirnet(my_ws.unsqueeze(1).repeat([1, n_styles , 1, 1]))
print(model_path)
torch.manual_seed(655445)
for i in range(2,7):
    for j in range(1,5):
        myw = stylized_my_w[0,1].unsqueeze(0)
        z = torch.randn(1, latent_dim, device=device)
        #w = generator.get_latent(z).unsqueeze(1).repeat(1, generator.n_latent, 1)
        # keep z completely random (not necessarily from w+ space)
        w = (z).unsqueeze(1).repeat(1, generator.n_latent, 1)
        myw[0,i:,:] = (1-0.1*j)*myw[0,i:,:] + (0.1*j)*w[0,i:,:]
        img = generator(myw, input_is_latent=True)
        display_image(utils.make_grid(img, normalize=True, range=(-1, 1)), title=f'random gen, i : {i}, j: {j}',
                  save=False, use_wandb=use_wandb)
#
# import matplotlib as mpl
# mpl.rcParams['figure.dpi'] = 100
# for i in range(20):
#     z = torch.randn(1, latent_dim, device=device)
#     w = generator.get_latent(z).unsqueeze(1).repeat(1, generator.n_latent, 1)
#     img = generator(w, input_is_latent=True)
#     display_image(utils.make_grid(img, normalize=True, range=(-1, 1)), title='random gen',
#                   save=False, use_wandb=use_wandb)
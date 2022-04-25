from util import *
from PIL import Image
import math
import random
import os
import wandb
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
inv_dirname = '/home/vjshah3/research/outputs/multistyle/inversion/'

names = sorted(list(glob.glob('/home/vjshah3/research/remote-gan-codes/multistyle/style_images_aligned/*')))

for i, name in enumerate(names):
    imgname = name.split('/')[-1].split('.')[-2]
    orgimg = Image.open(name)
    hfgiimg = Image.open(inv_dirname + 'hfgi/inversion/{:05d}.jpg'.format(i)).resize((1024,1024), Image.LANCZOS)
    restyleimg = Image.open(inv_dirname + f'restyle_encoder/inference_results/4/{imgname}.png')
    projimg = Image.open(inv_dirname + f'stylegan2_projection/proj_{imgname}.png')
    e4eimg = Image.open(inv_dirname + f'e4e/inversions/{i+1:05d}.jpg')

    hfgiimg.save(f'/home/vjshah3/research/models/multistyle/hfgi_inversion_images/{imgname}.png')
    restyleimg.save(f'/home/vjshah3/research/models/multistyle/restyle_inversion_images/{imgname}.png')
    projimg.save(f'/home/vjshah3/research/models/multistyle/opt_inversion_images/{imgname}.png')
    e4eimg.save(f'/home/vjshah3/research/models/multistyle/e4e_inversion_images/{imgname}.png')

    hfgiz = torch.load(f'/home/vjshah3/research/outputs/multistyle/inversion/hfgi/inversion/code_{i:05d}.pt').squeeze(0)
    result_file = {}
    result_file['latent'] = hfgiz
    torch.save(result_file, f'/home/vjshah3/research/models/multistyle/hfgi_inversion_codes/{imgname}.pt')


    optz = np.load(f'/home/vjshah3/research/outputs/multistyle/inversion/stylegan2_projection/projected_w_{imgname}.npz')['w']
    result_file = {}
    result_file['latent'] = torch.tensor(optz).squeeze(0)

    torch.save(result_file, f'/home/vjshah3/research/models/multistyle/opt_inversion_codes/{imgname}.pt')





#wandb.log({title: [wandb.Image(image)]})
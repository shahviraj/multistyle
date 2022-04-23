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


run_name = 'inversion_comparisons'
run_desc = 'comparing various inversion methods for inverting stylized images'
wandb.init(project='multistyle', entity='shahviraj', name = run_name, notes=run_desc, dir = '/home/vjshah3/research/outputs/multistyle/inversion/' )

names = sorted(list(glob.glob('/home/vjshah3/research/remote-gan-codes/multistyle/style_images_aligned/*')))

for i, name in enumerate(names):
    fig = plt.figure(figsize=(8., 4.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(1, 5),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )

    imgname = name.split('/')[-1].split('.')[-2]
    orgimg = Image.open(name)
    hfgiimg = Image.open(inv_dirname + 'hfgi/inversion/{:05d}.jpg'.format(i)).resize((1024,1024), Image.LANCZOS)
    restyleimg = Image.open(inv_dirname + f'restyle_encoder/inference_results/4/{imgname}.png')
    projimg = Image.open(inv_dirname + f'stylegan2_projection/proj_{imgname}.png')
    e4eimg = Image.open(inv_dirname + f'e4e/inversions/{i+1:05d}.jpg')
    titles = ['Original', 'Optimization', 'e4e', 'Restyle', 'HFGI']
    img = [orgimg, projimg, e4eimg, restyleimg, hfgiimg]
    reconsts = []
    for k in range(len(titles)):
        image = wandb.Image(img[k], caption=f"{titles[k]}")
        reconsts.append(image)
    wandb.log({"Inversion Comparison": reconsts}, step=i)
    # for ax, im in zip(grid, [orgimg, projimg, e4eimg, restyleimg, hfgiimg]):
    #     # Iterating over the grid returns the Axes.
    #     ax.imshow(im)
    #     ax.axis('off')
    #     ax.set_title("")
    # wandb.log({"Inversion Comparison": fig}, step=i)






#wandb.log({title: [wandb.Image(image)]})
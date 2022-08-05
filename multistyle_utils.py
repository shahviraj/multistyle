# how to save the images in suitable fashion?

# keep 0 reserved for pure input image or pure ref image.
# naming is done as: for image of input_code i in the ref style of ref_code r --> {input_code}_{ref_code}.png
# if either of the i or r are 0, then it is pure ref image or pure input image (0_12.png, 2_0.png etc)
# folder name is ref_codes separated by `_`, i.e. if 4 ref styles are used, name would be 4styles_2_5_12_8

import os

import torch.cuda
from torchvision import transforms, utils
import numpy as np
import wandb

input_code = {
    "chris" : 1,
    "gal" : 2,
    "iu" : 3,
    "tom" : 4,
    "arnold" : 5,
    'deep' : 6,
    'deep2' : 7,
    'cruz' : 8,
    'cruz2' : 9,
    'priy1' : 10,
    'priy4' : 11,
    'ritik2' : 12,
    'ran2' : 13,
    'robert2' : 14,
    'brad' : 15
    }

ref_code = {
            'art': 1,
            'jojo': 2,
            'arcane_jinx': 3,
            'jojo_yasuho': 4,
            'sketch': 5,
            'arcane_viktor': 6,
            'arcane_jayce': 7,
            'titan': 8,
            'audrey': 9,
            'arcane_texture': 10,
            'cheryl': 11,
            'flower': 12,
            'elliee': 13,
            'yukako': 14,
            'marilyn': 15,
            'water': 16,
            'matisse': 17,
            'arcane_caitlyn': 18,
            'star': 19,
            'audrey2': 20,
            'greeneye' : 21,
            'mark': 22
        }

def strip_path_extension(path):
   return  os.path.splitext(path)[0]

def get_savepath(config):
    dirname = f"{len(config['names'])}styles"
    for name in config['names']:
        dirname = dirname + f'_{ref_code[strip_path_extension(name)]}'

    dirname = dirname + f'_ctx_wt_{config["ctx_loss_w"]}_n_iter_{config["num_iter"]}'
    dirpath = os.path.join(config['savepath'], dirname)

    if not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)

    savepath = dirpath + f"/model_{dirname}.pt"

    return savepath

def save_single_image(mode,i,j, img, config, use_wandb):
    '''
    save format: {input_code}_{ref_code}
    '''
    dirname = f"{len(config['names'])}styles"
    for name in config['names']:
        dirname = dirname + f'_{ref_code[strip_path_extension(name)]}'

    dirname = dirname + f'_ctx_wt_{config["ctx_loss_w"]}_n_iter_{config["num_iter"]}'

    dirpath = os.path.join(config['savepath'], dirname, mode)

    if not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)

    ref = ref_code[strip_path_extension(config['names'][i])]

    if mode == 'test':
        input = input_code[strip_path_extension(config['filenamelist'][j])]
    elif mode == 'random':
        input = j
    elif mode == 'cross':
        input = ref_code[strip_path_extension(config['names'][j])]
    elif mode == 'cross_extra':
        input = ref_code[strip_path_extension(config['cross_names'][j])]
    else:
        raise NotImplementedError

    fname1 = f'{input}_{ref}.png'
    fname2 = f'{input}_{ref}.jpeg'
    savepath1 = dirpath + '/' + fname1
    savepath2 = dirpath + '/' + fname2

    utils.save_image(img, savepath1, normalize=True, range=(-1, 1))
    utils.save_image(img, savepath2, normalize=True, range=(-1, 1))

    if use_wandb:
        # wandbpath1 = wandb.run.dir + '/' + mode + '_' + fname1
        wandbpath2 = wandb.run.dir + '/' + mode + '_' + fname2
        utils.save_image(img, wandbpath2, normalize=True, range=(-1, 1))

    img.cpu()
    del img
    torch.cuda.empty_cache()


def save_batch_images(batch_num, images, config, use_wandb):
    dirname = f"{len(config['names'])}styles"
    for name in config['names']:
        dirname = dirname + f'_{ref_code[strip_path_extension(name)]}'

    dirname = dirname + f'_ctx_wt_{config["ctx_loss_w"]}_n_iter_{config["num_iter"]}'
    dirpath = os.path.join(config['savepath'], dirname, 'full_random')

    if not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)

    for i in range(len(images)):
        fname2 = f'{config["batchsize"]*batch_num + i}.jpeg'
        savepath2 = dirpath + '/' + fname2
        utils.save_image(images[i,...], savepath2, normalize=True, range=(-1, 1))

        if use_wandb:
            wandbpath2 = wandb.run.dir + '/' + 'full_random' + '_' + fname2
            utils.save_image(images[i,...], wandbpath2, normalize=True, range=(-1, 1))
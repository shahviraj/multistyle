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
from model_styledir import *

from e4e_projection import projection as e4e_projection
from restyle_projection import restyle_projection
from copy import deepcopy

use_wandb = False

hyperparam_defaults = dict(
    splatting = False,
    learning = True,
    names = ['jojo.png', 'arcane_jinx.png'],#, 'jojo.png'],
    filenamelist = ['iu.jpeg'],
    fake_splatting = False,
    preserve_color = False,
    per_style_iter = None,
    num_iter = 500,
    dir_act = 'tanh',
    init = 'identity',
    weight_type = 'styledir', # dont change
    inv_method = 'e4e',
    log_interval = 100,
    learning_rate = 2e-3,
    alpha = 0.7,
    n_sample = 5,  # @param {type:"number"},
    seed = 9,  # @param {type:"number"},
)

if use_wandb:
    wandb.init(config=hyperparam_defaults)
    config = wandb.config
    if config['per_style_iter'] == 'None':
        config['per_style_iter'] = None
    if config['preserve_color'] == 'True':
        config['preserve_color'] = True
    if config['preserve_color'] == 'False':
        config['preserve_color'] = False
else:
    config = hyperparam_defaults



if config['splatting'] and config['learning']:
    raise ValueError("both splatting and learning can't be True!")

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
original_generator = modify_generator(original_generator, n_styles=len(config['names']))
mean_latent = original_generator.mean_latent(10000)

# to be finetuned generator
# generator = deepcopy(original_generator)


transform = transforms.Compose(
    [
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


# uploaded = files.upload()
# filepath = list(uploaded.keys())[0]
testimglist = []
my_wlist = []
aligned_facelist = []
for i, file in enumerate(config['filenamelist']):
    imgname = strip_path_extension(file)
    unaligned_img_path = f'test_input/{file}'
    aligned_img_path = f'test_input_aligned/{imgname}_aligned.png'
    if os.path.exists(aligned_img_path):
        aligned_face = Image.open(aligned_img_path).convert('RGB')
    else:
        aligned_face = align_face(unaligned_img_path)
    aligned_facelist.append(aligned_face)

    inv_code_path = os.path.join(f'../../models/multistyle/{config["inv_method"]}_inversion_codes', f'{imgname}_aligned.pt')
    if os.path.exists(inv_code_path):
        my_w = torch.load(inv_code_path)['latent']
    else:
        testimglist.append(inv_code_path)
        if config['inv_method'] == 'e4e':
            my_w = e4e_projection(aligned_face, testimglist[i], device)
        elif config['inv_method'] == 'restyle':
            my_w = restyle_projection(aligned_face, testimglist[i], device)
        else:
            raise NotImplementedError
    my_wlist.append(my_w.to(device))
# my_w = restyle_projection(aligned_face, name, device, n_iters=1).unsqueeze(0)

targets = []
latents = []

for name in config['names']:
    name = strip_path_extension(name)
    style_aligned_path = os.path.join('style_images_aligned', f'{name}.png')

    if not os.path.exists(style_aligned_path):
        style_path = os.path.join('style_images', name+ '.jpg')
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

    targets.append(transform(style_aligned).to(device))
    latents.append(latent.to(device))

targets = torch.stack(targets, 0)
latents = torch.stack(latents, 0)
my_ws = torch.stack(my_wlist, 0)

target_im = utils.make_grid(targets, normalize=True, range=(-1, 1))
display_image(target_im, title='Style References', save=False)

with torch.no_grad():
    original_generator.eval()
    inv_styles = original_generator(latents, input_is_latent=True)
    inv_tests = original_generator(my_ws, input_is_latent=True)
    display_image(utils.make_grid(inv_styles, normalize=True, range=(-1, 1)), title='Reference Inversions',
                  save=False, use_wandb=use_wandb)
    display_image(utils.make_grid(inv_tests, normalize=True, range=(-1, 1)), title='Input Inversions',
                  save=False, use_wandb=use_wandb)
    del inv_styles
    del inv_tests
    original_generator.train()
# @param {type:"slider", min:0, max:1, step:0.1}
#alpha = 1 - alpha

# @markdown Tries to preserve color of original image by limiting family of allowable transformations. Set to false if you want to transfer color from reference image. This also leads to heavier stylization


# load discriminator for perceptual loss
discriminator = Discriminator(1024, 2).eval().to(device)
ckpt = torch.load('../../models/multistyle/stylegan2-ffhq-config-f.pt', map_location=lambda storage, loc: storage)
discriminator.load_state_dict(ckpt["d"], strict=False)

torch.manual_seed(config['seed'])
z = torch.randn(config['n_sample'], latent_dim, device=device)
original_sample = original_generator([z], truncation=0.7, truncation_latent=mean_latent)
#original_my_sample = original_generator(my_w, input_is_latent=True)

generator = deepcopy(original_generator)
del original_generator


# Which layers to swap for generating a family of plausible real images -> fake image
if config['preserve_color']:
    id_swap = [9, 11, 15, 16, 17]
else:
    id_swap = list(range(3, generator.n_latent))

if config['fake_splatting']:
    n_styles = len(config['names']) + 1
else:
    n_styles = len(config['names'])



g_optim = optim.Adam(generator.parameters(), lr=config['learning_rate'], betas=(0, 0.99))

if use_wandb:
    wandb.log(
            {"Style reference": [wandb.Image(transforms.ToPILImage()(target_im))]},
            step=0)

for idx in tqdm(range(config['num_iter'])):
    mean_w = generator.get_latent(torch.randn([latents.size(0), latent_dim]).to(device)).unsqueeze(1).repeat(1,
                                                                                                             generator.n_latent,
                                                                                                             1)
    in_latent = latents.clone()
    in_latent[:, id_swap] = config['alpha'] * latents[:, id_swap] + (1 - config['alpha']) * mean_w[:, id_swap]

    #in_latent = dirnet(in_latent)
    img = []
    for i in range(len(in_latent)):
        img.append(generator(in_latent[i].unsqueeze(0), style_indx= i+1 , input_is_latent=True))
    img = torch.vstack(img)
    #img = generator(in_latent, input_is_latent=True)

    # k will determine which style image will be used to calculate loss, k varies from 0 to n_styles-1
    if config['per_style_iter'] is not None:
        k = (idx // config['per_style_iter']) % n_styles
        with torch.no_grad():
            real_feat = discriminator(targets[k].unsqueeze(0))
        fake_feat = discriminator(img[k].unsqueeze(0))

    else:
        with torch.no_grad():
            real_feat = discriminator(targets)
        fake_feat = discriminator(img)

    loss = sum([F.l1_loss(a, b) for a, b in zip(fake_feat, real_feat)]) / len(fake_feat)

    if use_wandb:
        wandb.log({"loss": loss}, step=idx)
        # if idx == 10 or idx % config['log_interval'] == (config['log_interval']-1):
        #     generator.eval()
        #     if config['splatting']:
        #         my_samples_eval = []
        #         for i in range(len(config['names'])):
        #             stylized_my_ws_eval = my_ws.clone()
        #             stylized_my_ws_eval[:, id_swap,
        #             i * (int(latent_dim / n_styles)):(i + 1) * (int(latent_dim / n_styles))] = 0.0
        #             my_samples_eval.append(generator(stylized_my_ws_eval, input_is_latent=True))
        #     elif config['learning']:
        #         my_samples_eval = []
        #         stylized_my_ws_eval = dirnet(my_ws.unsqueeze(1).repeat([1, n_styles , 1, 1])) # input and output are n_image x n_Style x 18 x 512
        #         for i in range(len(config['names'])):
        #             my_samples_eval.append(generator(stylized_my_ws_eval[:, i,...], input_is_latent=True))
        #     else:
        #         my_sample = generator(my_ws, input_is_latent=True)
        #     generator.train()
        #     if config['splatting'] or config['learning']:
        #         for i in range(len(config['names'])):
        #             my_sample = transforms.ToPILImage()(utils.make_grid(my_samples_eval[i], nrow= my_samples_eval[i].shape[0], normalize=True, range=(-1, 1)))
        #             wandb.log(
        #                 {"Current stylization {}".format(i): [wandb.Image(my_sample)]},
        #                 step=idx)
        #         del my_samples_eval, stylized_my_ws_eval, my_sample
        #     else:
        #         my_sample = transforms.ToPILImage()(utils.make_grid(my_sample, normalize=True, range=(-1, 1)))
        #         wandb.log(
        #             {"Current stylization": [wandb.Image(my_sample)]},
        #             step=idx)

    g_optim.zero_grad()
    loss.backward()
    g_optim.step()

# Try on new images

with torch.no_grad():
    generator.eval()
    my_samples = []
    stylized_my_w = my_ws.unsqueeze(1).repeat([1, n_styles, 1, 1])
    for i in range(len(config['names'])):
        my_samples.append(generator(stylized_my_w[:, i, ...], style_indx=i + 1, input_is_latent=True))
    facelist = []
    for aligned_face in aligned_facelist:
        facelist.append(transform(aligned_face).to(device))
    faces = torch.stack(facelist, 0)
    for i in range(len(config['names'])):
        my_output = torch.cat([faces, my_samples[i]], 0)
        display_image(utils.make_grid(my_output, nrow= my_samples[i].shape[0],normalize=True, range=(-1, 1)), title='Test Samples'.format(i,config['num_iter'],config['alpha']),
                      save=False, use_wandb=use_wandb)

    w = generator.get_latent(z).unsqueeze(1).unsqueeze(1).repeat(1, n_styles, generator.n_latent, 1) # output becomes n_images x n_styles x 18 x 512

    samples = []
    for i in range(n_styles):
        samples.append(generator(w[:, i,...], style_indx = i+1, input_is_latent=True))




# display reference images
style_images = []
for name in config['names']:
    style_path = f'style_images_aligned/{strip_path_extension(name)}.png'
    style_image = transform(Image.open(style_path))
    style_images.append(style_image)

facelist = []
for aligned_face in aligned_facelist:
    facelist.append(transform(aligned_face).to(device))
faces = torch.stack(facelist, 0)
style_images = torch.stack(style_images, 0).to(device)
display_image(utils.make_grid(style_images, normalize=True, range=(-1, 1)), title='References',
              save=True, use_wandb=use_wandb)

if config['splatting'] or config['learning']:
    for i in range(len(config['names'])):
        my_output = torch.cat([faces, my_samples[i]], 0)
        display_image(utils.make_grid(my_output, nrow= my_samples[i].shape[0],normalize=True, range=(-1, 1)), title='Test Samples'.format(i,config['num_iter'],config['alpha']),
                      save=False, use_wandb=use_wandb)

if config['splatting'] or config['learning']:
    for i in range(len(config['names'])):
        output = torch.cat([original_sample, samples[i]], 0)
        display_image(utils.make_grid(output, normalize=True, range=(-1, 1), nrow=config['n_sample']),
                      title='Random samples'.format(i,config['num_iter'],config['alpha']),
                      save=False, use_wandb=use_wandb)

print("Done!")
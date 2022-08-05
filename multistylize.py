# Author: Viraj Shah, vjshah3@illinois.edu
# July 2022
import torch
torch.backends.cudnn.benchmark = True
from torchvision import transforms, utils
from util import *
from PIL import Image
import math
import random
import os
import wandb
import pickle
from model_utils import *
import numpy as np
from torch import nn, autograd, optim
from torch.nn import functional as F
from tqdm import tqdm
from model import *

from e4e_projection import projection as e4e_projection
from restyle_projection import restyle_projection
from copy import deepcopy

from id_loss import IDLoss
import contextual_loss.functional as FCX

from multistyle_utils import *

use_wandb = True
run_name = 'jojogan_sweep'
run_desc = 'baseline multistyle + use sep dirnet + for style mixing, use 7 onwards (just like before) + use low contextual loss with wt 0.002  + use original generator to generate w codes for style mixing'

hyperparam_defaults = dict(
    learning = True,
    mse_loss = False,
    l1_loss = False,
    id_loss = False,
    ctx_loss= True,
    mixed_inv = True,
    preserve_shape = False,
    ctx_loss_w = 0.002,
    id_loss_w = 2e-3,
    verbose = 'less', # set to `full` to save all the progress to wandb
    inv_mix = [],
    sty_mix = [],
    savepath = '../../outputs/multistyle/results/',
    names = [
            #'arcane_caitlyn.png',
            #'art.png',
            'arcane_jinx.png',
            #'jojo.png',
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
            #'audrey2.jpg',
            #'star.jpg',
            #'cute1.jpg',
            #'trump1.jpg',
            #'cute2.jpg',
            #'greeneye.jpg',
            #'paint1.jpg',
            #'mark.jpg'
             ],#, 'jojo.png'],
    filenamelist = [
                    'iu.jpeg',
                    'arnold.jpeg',
                    'chris.jpeg',
                    'gal.jpeg',
                    'tom.jpeg',
                    'deep.jpg',
                    'deep2.jpg',
                    'arnold.jpeg',
                    'brad.jpg',
                    'cruz.jpg',
                    'cruz2.jpg',
                    'priy1.jpg',
                    'priy4.jpg',
                    'ran2.jpg',
                    'ritik2.jpg',
                    'robert2.jpg',
                    ], #, ],
    cross_names = [
            'art.png',
            'arcane_jinx.png',
            'jojo.png',
            'sketch.png',
            'jojo_yasuho.png',
            'arcane_jayce.png',
            'arcane_viktor.png',
            'arcane_caitlyn.png',
            'audrey.jpg',
            'arcane_texture.jpeg',
            'cheryl.jpg',
            'flower.jpeg',
            'elliee.jpeg',
            'yukako.jpeg',
            'marilyn.jpg',
            'water.jpeg',
            'matisse.jpeg',
            'audrey2.jpg',
            'star.jpg',
            'greeneye.jpg',
            'mark.jpg'
                ],
    preserve_color = False,
    per_style_iter = None,
    num_iter = 500,
    dir_act = 'tanh', # options ['tanh', 'relu' , None (for identity)]
    init = 'identity',
    weight_type = 'std',
    inv_method = 'e4e',
    log_interval = 100,
    learning_rate = 2e-3,
    n_batches = 20,
    batchsize = 1,
    alpha = 0.7,
    n_sample = 5,  # @param {type:"number"},
    seed = 9,  # @param {type:"number"},
)

if use_wandb:
    wandb.init(config=hyperparam_defaults,
               project="multistyle_final",
               entity='shahviraj',
               name=run_name,
               dir = '../../outputs/multistyle/wandb/',
               notes=run_desc)
    config = wandb.config
    if config['per_style_iter'] == 'None':
        config['per_style_iter'] = None
    if config['preserve_color'] == 'True':
        config['preserve_color'] = True
    if config['preserve_color'] == 'False':
        config['preserve_color'] = False
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

def inversion_mixing(latent, config):
    if config['preserve_shape']:
        return latent
    else:
        for j in range(len(latent)):
            if j in config['inv_mix']:
                 latent[j] = mean_latent
            else:
                latent[j] = latent[j]
        return latent

def prepare_targets(config, cross=False):

    targets = []
    latents = []

    if cross:
        namelist = config['cross_names']
    else:
        namelist = config['names']

    for name in namelist:
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

my_ws, aligned_facelist = prepare_inputs(config)

targets, latents = prepare_targets(config)

if config['verbose'] == 'half':
    if config['mixed_inv']:
        with torch.no_grad():
            inv_styles = original_generator(latents, input_is_latent=True)

            display_image(utils.make_grid(inv_styles, normalize=True, range=(-1, 1)), title='Reference Inversions after mixing',
                          save=False, use_wandb=use_wandb)

            inv_styles = inv_styles.detach()
            del inv_styles
            torch.cuda.empty_cache()

if config['verbose'] == 'half':
    target_im = utils.make_grid(targets, normalize=True, range=(-1, 1))
    display_image(target_im, title='Style References', save=False)

# load discriminator for perceptual loss
discriminator = Discriminator(1024, 2).eval().to(device)
ckpt = torch.load('../../models/multistyle/stylegan2-ffhq-config-f.pt', map_location=lambda storage, loc: storage)
discriminator.load_state_dict(ckpt["d"], strict=False)

for p in discriminator.parameters():
    p.requires_grad = False

torch.manual_seed(config['seed'])
z = torch.randn(config['n_sample'], latent_dim, device=device)
with torch.no_grad():
    original_sample = original_generator([z], truncation=0.7, truncation_latent=mean_latent)

generator = deepcopy(original_generator)

if not config['id_loss'] or not config['ctx_loss']:
    original_generator.eval()
else:
    id_encoder = IDLoss().to(device).eval()
    original_generator.eval()

if config['ctx_loss']:
    vgg = VGG19().to(device).eval()

# Which layers to swap for generating a family of plausible real images -> fake image
if config['preserve_color']:
    id_swap = [9, 11, 15, 16, 17]
else:
    id_swap = config['sty_mix'] + list(range(7, generator.n_latent))

n_styles = len(config['names'])

if n_styles > 1:
    if config['weight_type'] == 'std':
        dirnet = DirNet(latent_dim, latent_dim, n_styles, id_swap, init=config['init'], activation=config['dir_act'], device=device).to(device)
    elif config['weight_type'] == 'sep': # separate Transformation applied for every row for each style (total Tx = n_rows to be modified x n_styles)
        dirnet = DirNetSep(latent_dim, latent_dim, n_styles, id_swap, init=config['init'], activation=config['dir_act'], device=device).to(device)
    elif config['weight_type'] == 'dummy':
        dirnet = DirNet_Id().to(device)
    else:
        raise NotImplementedError
else:
    dirnet = DirNet_Id().to(device)

g_optim = optim.Adam(list(generator.parameters()) + list(dirnet.parameters()), lr=config['learning_rate'], betas=(0, 0.99))

if config['verbose'] == 'half':
    if use_wandb:
        wandb.log(
                {"Style reference": [wandb.Image(transforms.ToPILImage()(target_im))]},
                step=0)

for idx in tqdm(range(config['num_iter'])):
    with torch.no_grad():
        mean_w = original_generator.get_latent(torch.randn([latents.size(0), latent_dim]).to(device)).unsqueeze(1).repeat(1,
                                                                                                             generator.n_latent,
                                                                                                             1)
    in_latent = latents.clone()
    in_latent[:, id_swap] = config['alpha'] * latents[:, id_swap] + (1 - config['alpha']) * mean_w[:, id_swap]


    in_latent = dirnet(in_latent)

    img = generator(in_latent, input_is_latent=True)

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

    if config['mse_loss']:
        loss = sum([F.mse_loss(a, b) for a, b in zip(fake_feat, real_feat)]) / len(fake_feat)

    else:
        loss = sum([F.l1_loss(a, b) for a, b in zip(fake_feat, real_feat)]) / len(fake_feat)

    if config['l1_loss']:
        loss = loss + F.l1_loss(img, targets)

    if config['id_loss'] or config['ctx_loss']:
        # id loss
        rand_img = generator(dirnet(mean_w), input_is_latent=True)
        if config['id_loss']:
            with torch.no_grad():
                o_rand_img = original_generator(mean_w, input_is_latent=True)
            id_loss = config['id_loss_w'] * id_encoder(rand_img.mean(1, keepdim=True).repeat(1, 3, 1, 1),
                                        o_rand_img.mean(1, keepdim=True).repeat(1, 3, 1, 1))
            loss = loss + id_loss
        if config['ctx_loss']:
            fake_feats = vgg(rand_img)
            with torch.no_grad():
                target_feats = vgg(targets)

            ctx_loss = config['ctx_loss_w'] * FCX.contextual_loss(fake_feats[2], target_feats[2].detach(), band_width=0.5,
                                                 loss_type='cosine')

            loss = loss + ctx_loss

    if use_wandb:
        wandb.log({"loss": loss}, step=idx)
        if config['verbose'] == 'full':
            if idx == 10 or idx % config['log_interval'] == (config['log_interval']-1):
                generator.eval()

                my_samples_eval = []
                stylized_my_ws_eval = dirnet(my_ws.unsqueeze(1).repeat([1, n_styles , 1, 1])) # input and output are n_image x n_Style x 18 x 512
                for i in range(len(config['names'])):
                    my_samples_eval.append(generator(stylized_my_ws_eval[:, i,...], input_is_latent=True))

                generator.train()

                for i in range(len(config['names'])):
                    my_sample = transforms.ToPILImage()(utils.make_grid(my_samples_eval[i], nrow= my_samples_eval[i].shape[0], normalize=True, range=(-1, 1)))
                    wandb.log(
                        {"Current stylization {}".format(i): [wandb.Image(my_sample)]},
                        step=idx)
                stylized_my_ws_eval.detach().cpu()
                my_samples_eval = torch.stack(my_samples_eval,0).detach().cpu()
                del my_samples_eval, stylized_my_ws_eval, my_sample
                torch.cuda.empty_cache()

    g_optim.zero_grad()
    loss.backward()
    g_optim.step()


# delete original generator to save memory
original_generator.cpu()
del original_generator
torch.cuda.empty_cache()

# samples for cross stylizations
_, extra_latents = prepare_targets(config, cross=True)

# Save the fine-tuned generator and dirnet
savepath = get_savepath(config)
torch.save(
    {"g": generator.state_dict(), "dirnet": dirnet.state_dict(), "config": dict(config)},
    savepath)

# Try on new images
with torch.no_grad():
    generator.eval()

    # Save stylizations of randomly generated images
    w = generator.get_latent(z).unsqueeze(1).unsqueeze(1).repeat(1, n_styles, generator.n_latent, 1) # output becomes n_images x n_styles x 18 x 512
    stylized_w = dirnet(w)  # output becomes n_images x n_styles x 18 x 512 -- but contains stylized directions

    if config['verbose'] == 'half':
        samples = []
        for i in range(n_styles):
            samples.append(generator(stylized_w[:, i,...], input_is_latent=True))

        for i in range(len(config['names'])):
            output = torch.cat([original_sample, samples[i]], 0)
            display_image(utils.make_grid(output, normalize=True, range=(-1, 1), nrow=config['n_sample']),
                          title='Random samples'.format(i, config['num_iter'], config['alpha']),
                          save=False, use_wandb=use_wandb)
            output.detach().cpu()
            del output
            torch.cuda.empty_cache()
    else:
        for i in range(n_styles):
            for j in range(config['n_sample']):
                save_single_image('random', i, j, generator(stylized_w[j, i,...].unsqueeze(0), input_is_latent=True), config, use_wandb)

    # Save stylizations of test input images
    stylized_my_w = dirnet(my_ws.unsqueeze(1).repeat([1, n_styles , 1, 1]))

    if config['verbose'] == 'half':
        my_samples = []
        for i in range(len(config['names'])):
            my_samples.append(generator(stylized_my_w[:, i,...], input_is_latent=True))

        facelist = []
        for aligned_face in aligned_facelist:
            facelist.append(transform(aligned_face).to(device))
        faces = torch.stack(facelist, 0)

        for i in range(len(config['names'])):
            my_output = torch.cat([faces, my_samples[i]], 0)
            # my_output = my_samples[i]
            display_image(utils.make_grid(my_output, nrow=my_samples[i].shape[0], normalize=True, range=(-1, 1)),
                          title='Test Samples'.format(i, config['num_iter'], config['alpha']),
                          save=False, use_wandb=use_wandb)
            my_output.cpu()
            stylized_my_w.cpu()
            del my_output, stylized_my_w
            torch.cuda.empty_cache()
    else:
        for i in range(len(config['names'])):
            for j in range(len(config['filenamelist'])):
                save_single_image('test', i, j, generator(stylized_my_w[j, i,...].unsqueeze(0), input_is_latent=True), config, use_wandb)

    # Save stylizations of other stylized images (cross stylization)
    stylized_target_w = dirnet(latents.unsqueeze(1).repeat([1, n_styles, 1, 1]))

    if config['verbose'] == 'half':
        target_samples = []
        for i in range(len(config['names'])):
            target_samples.append(generator(stylized_target_w[:, i, ...], input_is_latent=True))

        style_images = []
        for name in config['names']:
            style_path = f'style_images_aligned/{strip_path_extension(name)}.png'
            style_image = transform(Image.open(style_path))
            style_images.append(style_image)
        style_images = torch.stack(style_images, 0).to(device)
        display_image(utils.make_grid(style_images, normalize=True, range=(-1, 1)), title='References',
                      save=False, use_wandb=use_wandb)

        target_images = torch.cat(target_samples, 0).to(device)
        all_t = torch.cat([style_images, target_images], 0)
        display_image(utils.make_grid(all_t, nrow=2, normalize=True, range=(-1, 1)),
                      title='Target Stylizations (with ref images on top)',
                      save=False, use_wandb=use_wandb)

        all_t.cpu()
        del all_t
        torch.cuda.empty_cache()
    else:
        for i in range(len(config['names'])):
            for j in range(len(config['names'])):
                save_single_image('cross',i,j, generator(stylized_target_w[j, i, ...].unsqueeze(0), input_is_latent=True), config, use_wandb)

    # Save stylizations of other stylized images (cross stylization) but for outside the "names" set
    stylized_target_w = dirnet(extra_latents.unsqueeze(1).repeat([1, len(config['cross_names']), 1, 1]))

    if config['verbose'] == 'half':
        target_samples = []
        for i in range(len(config['names'])):
            target_samples.append(generator(stylized_target_w[:, i, ...], input_is_latent=True))

        style_images = []
        for name in config['cross_names']:
            style_path = f'style_images_aligned/{strip_path_extension(name)}.png'
            style_image = transform(Image.open(style_path))
            style_images.append(style_image)
        style_images = torch.stack(style_images, 0).to(device)

        target_images = torch.cat(target_samples, 0).to(device)
        all_t = torch.cat([style_images, target_images], 0)
        display_image(utils.make_grid(all_t, nrow=2, normalize=True, range=(-1, 1)),
                      title='Target Stylizations (with ref images on top)',
                      save=False, use_wandb=use_wandb)
        all_t.cpu()
        del all_t
        torch.cuda.empty_cache()
    else:
        for i in range(len(config['names'])):
            for j in range(len(config['cross_names'])):
                save_single_image('cross_extra', i,j, generator(stylized_target_w[j, i, ...].unsqueeze(0), input_is_latent=True), config,
                              use_wandb)

    if config['n_batches'] != 0:
        for i in range(config['n_batches']):
            z = torch.randn(config['batchsize'], latent_dim, device=device)
            w = generator.get_latent(z).unsqueeze(1).repeat(1, generator.n_latent, 1)
            img = generator(w, input_is_latent=True)
            save_batch_images(i, img, config, use_wandb)

print("Done!")

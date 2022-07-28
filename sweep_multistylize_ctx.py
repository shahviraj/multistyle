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
run_name = 'multistyle_baseline_og_ctx_inv_mix_sty_mix'
run_desc = 'baseline multistyle +  use inversion code mixing + for style mixing, use 1,3,5,7 rows as well + use contextual loss with wt 0.01  + use original generator to generate w codes for style mixing'


hyperparam_defaults = dict(
    learning = True,
    mse_loss = False,
    l1_loss = False,
    id_loss = False,
    ctx_loss= True,
    mixed_inv = True,
    preserve_shape = False,
    ctx_loss_w = 0.01,
    id_loss_w = 2e-3,
    inv_mix = [2],
    names = [
            #'arcane_caitlyn.png',
            #'art.png',
            'arcane_jinx.png',
            'jojo.png',
            #'jojo_yasuho.png',
            #'sketch.png',
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
    filenamelist = ['iu.jpeg', 'arnold.jpeg', 'chris.jpeg', 'gal.jpeg', 'tom.jpeg'],
    preserve_color = False,
    per_style_iter = None,
    num_iter = 500,
    dir_act = 'tanh',
    init = 'identity',
    weight_type = 'std',
    inv_method = 'e4e',
    log_interval = 100,
    learning_rate = 2e-3,
    alpha = 0.7,
    n_sample = 5,  # @param {type:"number"},
    seed = 9,  # @param {type:"number"},
)

if use_wandb:
    wandb.init(config=hyperparam_defaults,
               project="multistyle_ctx",
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
    my_ws = torch.stack(my_wlist, 0)
    return my_ws, aligned_facelist

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

my_ws, aligned_facelist = prepare_inputs(config)

targets, latents = prepare_targets(config)

if config['mixed_inv']:
    with torch.no_grad():
        inv_styles = original_generator(latents, input_is_latent=True)

        display_image(utils.make_grid(inv_styles, normalize=True, range=(-1, 1)), title='Reference Inversions after mixing',
                      save=False, use_wandb=use_wandb)

        inv_styles = inv_styles.detach()
        del inv_styles
        torch.cuda.empty_cache()


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
    # original_generator.cpu()
    # del original_generator
    # torch.cuda.empty_cache()
else:
    id_encoder = IDLoss().to(device).eval()
    original_generator.eval()
if config['ctx_loss']:
    vgg = VGG19().to(device).eval()
# Which layers to swap for generating a family of plausible real images -> fake image
if config['preserve_color']:
    id_swap = [9, 11, 15, 16, 17]
else:
    #id_swap = list(range(7, generator.n_latent))
    id_swap = list(range(7, generator.n_latent))

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
            #fake_styles = [F.adaptive_avg_pool2d(fake_feat, output_size=1) for fake_feat in fake_feats]
            with torch.no_grad():
                target_feats = vgg(targets)
                # target_styles = [F.adaptive_avg_pool2d(target_feat, output_size=1).detach() for target_feat in target_feats]
                # real_feats = vgg(o_rand_img)

            # sty_loss = .25*(F.mse_loss(fake_styles[1], target_styles[1]) + F.mse_loss(fake_styles[2], target_styles[2]))
            ctx_loss = config['ctx_loss_w'] * FCX.contextual_loss(fake_feats[2], target_feats[2].detach(), band_width=0.5,
                                                 loss_type='cosine')
            # ctx_loss2 =  .05*FCX.contextual_loss(fake_feats[4], real_feats[4].detach(), band_width=0.2, loss_type='cosine')
            loss = loss + ctx_loss

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


    w = generator.get_latent(z).unsqueeze(1).unsqueeze(1).repeat(1, n_styles, generator.n_latent, 1) # output becomes n_images x n_styles x 18 x 512
    stylized_w = dirnet(w)  # output becomes n_images x n_styles x 18 x 512 -- but contains stylized directions
    samples = []
    for i in range(n_styles):
        samples.append(generator(stylized_w[:, i,...], input_is_latent=True))


    my_samples = []
    stylized_my_w = dirnet(my_ws.unsqueeze(1).repeat([1, n_styles , 1, 1]))
    for i in range(len(config['names'])):
        my_samples.append(generator(stylized_my_w[:, i,...], input_is_latent=True))
    #
    target_samples = []
    stylized_target_w = dirnet(latents.unsqueeze(1).repeat([1, n_styles, 1, 1]))
    for i in range(len(config['names'])):
        target_samples.append(generator(stylized_target_w[:, i, ...], input_is_latent=True))

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
              save=False, use_wandb=use_wandb)

for i in range(len(config['names'])):
    my_output = torch.cat([faces, my_samples[i]], 0)
    #my_output = my_samples[i]
    display_image(utils.make_grid(my_output, nrow= my_samples[i].shape[0],normalize=True, range=(-1, 1)), title='Test Samples'.format(i,config['num_iter'],config['alpha']),
                  save=False, use_wandb=use_wandb)

my_output.cpu()
del my_output
torch.cuda.empty_cache()


target_images = torch.cat(target_samples, 0).to(device)
all_t = torch.cat([style_images, target_images], 0)
display_image(utils.make_grid(all_t, nrow=2, normalize=True, range=(-1, 1)), title='Target Stylizations (with ref images on top)',
              save=False, use_wandb=use_wandb)


all_t.cpu()
del all_t
torch.cuda.empty_cache()

for i in range(len(config['names'])):
    output = torch.cat([original_sample, samples[i]], 0)
    display_image(utils.make_grid(output, normalize=True, range=(-1, 1), nrow=config['n_sample']),
                  title='Random samples'.format(i,config['num_iter'],config['alpha']),
                  save=False, use_wandb=use_wandb)

# t = ''
# for v in config['names']:
#     t = v + '_' + t
# torch.save(
#             {
#                 "g": generator.state_dict(),
#             },
#             #f"checkpoint_{args.dataset}/{names[0]}_preserve_color.pt",
#             #f"checkpoint_{args.dataset}/restyle_{names[0]}.pt",
#
#             f"../checkpoint_{t}_{config['num_iter']}.pt",
#             #f"checkpoint_{args.dataset}/latest.pt",
#         )
print("Done!")

# interpolate between two styles
# with torch.no_grad():
#     my_samples = []
#     for i in range(5):
#         blended_w = (0.1)*i*stylized_my_w[:,0,:,:] + (1.0 - 0.1*i)*stylized_my_w[:,1,:,:]
#         my_samples.append(generator(blended_w, input_is_latent=True))
#
#     my_samples = torch.cat(my_samples,0)
#
#     display_image(utils.make_grid(my_samples, nrow=5,normalize=True, range=(-1, 1)), title='Blended samples',
#                   save=False, use_wandb=use_wandb)

#
# with torch.no_grad():
#     my_samples = []
#     for i in range(10):
#         blended_w = stylized_my_w[:,0,:,:]
#         blended_w[:,7:,:] = (0.1)*i*stylized_my_w[:,0,7:,:] + (1.0 - 0.1*i)*stylized_my_w[:,1,7:,:]
#         my_samples.append(generator(blended_w, input_is_latent=True))
#
#     my_samples = torch.cat(my_samples,0)
#
#     display_image(utils.make_grid(my_samples, nrow= 5,normalize=True, range=(-1, 1)), title='Blended samples',
#                   save=False, use_wandb=use_wandb)
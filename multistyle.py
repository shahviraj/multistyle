

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
#import wandb
from model import *
from e4e_projection import projection as e4e_projection
from copy import deepcopy

splatting = False
learning = True
if splatting and learning:
    raise ValueError("both splatting and learning can't be True!")
run_name = 'four_styles_fixed_bug_learnt_dirs_tanh_identity_init'
run_desc = 'learning four styles at once, using learnable directions using single layer FCN, with tanh activations and identity matrix as initialization'
names = ['arcane_jinx.png', 'jojo.png', 'sketch.png', 'arcane_caitlyn.png']
fake_splatting = False
preserve_color = False
init = 'identity'
per_style_iter = None
num_iter = 1000
use_wandb = True
dir_act = 'tanh'
log_interval = 100
learning_rate = 2e-3
alpha = 0.7
n_sample = 5  # @param {type:"number"}
seed = 3000  # @param {type:"number"}


filename = 'iu.jpeg' #@param {type:"string"}
filepath = f'test_input/{filename}'

latent_dim = 512
device = 'cuda'

def perform_splat(latent, id_swap):
    n_styles = latent.shape[0]
    latent_dim = latent.shape[-1]
    n_style_codes = latent.shape[1]
    blocksize = int(latent_dim/n_styles)
    splat_latent = latent.clone()
    for i in range(n_styles):
        #splat_latent[i, id_swap, i*blocksize:(i+1)*blocksize] = (1e-3)*torch.rand_like(splat_latent)[i, id_swap, i*blocksize:(i+1)*blocksize]
        splat_latent[i, id_swap, i*blocksize:(i+1)*blocksize] = 0.0
    return splat_latent

def perform_splat_only_on_one(latent, id_swap):

    blocksize = int(latent_dim/2)
    splat_latent = latent.clone()

    splat_latent[0, id_swap, 0:blocksize] = 0.0

    return splat_latent

class DirNet(nn.Module):
    def __init__(
            self, in_dim, out_dim, n_out, n_indx, init, bias=True, bias_init=0, lr_mul=1, activation=None, device='cpu'
    ):
        super(DirNet, self).__init__()
        self.n_indx = n_indx # index of the style code that is to be used in style mixing
        self.n_out = n_out
        eqlinlayers = []
        for i in range(n_out):
            eqlinlayers.append(EqualLinearAct(in_dim, out_dim, init, bias, bias_init, lr_mul, activation).to(device))
        self.layers = nn.ModuleList(eqlinlayers) # crucial in order to register every layer in the list properly

        self.conv = nn.Conv2d(1,1,3)

    def forward(self, input):
        if len(input.shape) == 4:
            outs = input.clone()
            for k in range(input.shape[0]):
                for i in range(self.n_out):
                    for j in self.n_indx:
                        outs[k, i, j, :] = self.layers[i](input[k, i, j, :])
        elif len(input.shape) == 3:
            outs = input.clone()
            for i in range(self.n_out):
                for j in self.n_indx:
                    outs[i, j, :] = self.layers[i](input[i, j, :])
        else:
            raise NotImplementedError("not implemented when the input tensor is 2-dimensional")

        return outs


# Load original generator
original_generator = Generator(1024, latent_dim, 8, 2).to(device)
ckpt = torch.load('../../models/multistyle/stylegan2-ffhq-config-f.pt', map_location=lambda storage, loc: storage)
original_generator.load_state_dict(ckpt["g_ema"], strict=False)
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
name = strip_path_extension(filepath)+'.pt'

# aligns and crops face
aligned_face = align_face(filepath)

# my_w = restyle_projection(aligned_face, name, device, n_iters=1).unsqueeze(0)
my_w = e4e_projection(aligned_face, name, device).unsqueeze(0)

display_image(aligned_face, title='Aligned face', save=True)


#@param {type:"raw"}

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
    else:
        latent = torch.load(style_code_path)['latent']

    targets.append(transform(style_aligned).to(device))
    latents.append(latent.to(device))

targets = torch.stack(targets, 0)
latents = torch.stack(latents, 0)



target_im = utils.make_grid(targets, normalize=True, range=(-1, 1))
display_image(target_im, title='Style References', save=True)

# @param {type:"slider", min:0, max:1, step:0.1}
#alpha = 1 - alpha

# @markdown Tries to preserve color of original image by limiting family of allowable transformations. Set to false if you want to transfer color from reference image. This also leads to heavier stylization


# load discriminator for perceptual loss
discriminator = Discriminator(1024, 2).eval().to(device)
ckpt = torch.load('../../models/multistyle/stylegan2-ffhq-config-f.pt', map_location=lambda storage, loc: storage)
discriminator.load_state_dict(ckpt["d"], strict=False)


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

dirnet = DirNet(latent_dim, latent_dim, n_styles, id_swap, init=init, activation=dir_act, device=device).to(device)

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


for idx in tqdm(range(num_iter)):
    mean_w = generator.get_latent(torch.randn([latents.size(0), latent_dim]).to(device)).unsqueeze(1).repeat(1,
                                                                                                             generator.n_latent,
                                                                                                             1)
    in_latent = latents.clone()
    in_latent[:, id_swap] = alpha * latents[:, id_swap] + (1 - alpha) * mean_w[:, id_swap]

    if splatting:
        if fake_splatting:
            in_latent = perform_splat_only_on_one(in_latent, id_swap)
        else:
            in_latent = perform_splat(in_latent, id_swap)
    elif learning:
            in_latent = dirnet(in_latent)

    img = generator(in_latent, input_is_latent=True)

    # k will determine which style image will be used to calculate loss, k varies from 0 to n_styles-1
    if per_style_iter is not None:
        k = (idx // per_style_iter) % n_styles
        with torch.no_grad():
            real_feat = discriminator(targets[k])
        fake_feat = discriminator(img[k])

    else:
        with torch.no_grad():
            real_feat = discriminator(targets)
        fake_feat = discriminator(img)

    loss = sum([F.l1_loss(a, b) for a, b in zip(fake_feat, real_feat)]) / len(fake_feat)

    if use_wandb:
        wandb.log({"loss": loss}, step=idx)
        if idx == 10 or idx % log_interval == (log_interval-1):
            generator.eval()
            if splatting:
                my_samples_eval = []
                for i in range(len(names)):
                    stylized_my_w_eval = my_w.clone()
                    stylized_my_w_eval[:, id_swap,
                    i * (int(latent_dim / n_styles)):(i + 1) * (int(latent_dim / n_styles))] = 0.0
                    my_samples_eval.append(generator(stylized_my_w_eval, input_is_latent=True))
            elif learning:
                my_samples_eval = []
                stylized_my_w_eval = dirnet(my_w.repeat([n_styles, 1, 1]))
                for i in range(len(names)):
                    my_samples_eval.append(generator(stylized_my_w_eval[i,...].unsqueeze(0), input_is_latent=True))
            else:
                my_sample = generator(my_w, input_is_latent=True)
            generator.train()
            if splatting or learning:
                for i in range(len(names)):
                    my_sample = transforms.ToPILImage()(utils.make_grid(my_samples_eval[i], normalize=True, range=(-1, 1)))
                    wandb.log(
                        {"Current stylization {}".format(i): [wandb.Image(my_sample)]},
                        step=idx)
                del my_samples_eval, stylized_my_w_eval, my_sample
            else:
                my_sample = transforms.ToPILImage()(utils.make_grid(my_sample, normalize=True, range=(-1, 1)))
                wandb.log(
                    {"Current stylization": [wandb.Image(my_sample)]},
                    step=idx)

    g_optim.zero_grad()
    loss.backward()
    g_optim.step()


# Try on new images


with torch.no_grad():
    generator.eval()

    if splatting:

        w = generator.get_latent(z).unsqueeze(1).repeat(1,generator.n_latent,1)

        samples = []
        for i in range(len(names)):
            stylized_w = w.clone()
            stylized_w[:,id_swap, i*(int(latent_dim/n_styles)):(i+1)*(int(latent_dim/n_styles))] = 0.0
            samples.append(generator(stylized_w, input_is_latent=True))
    elif learning:
        w = generator.get_latent(z).unsqueeze(1).unsqueeze(1).repeat(1, n_styles, generator.n_latent, 1) # output becomes n_images x n_styles x 18 x 512
        stylized_w = dirnet(w)  # output becomes n_images x n_styles x 18 x 512 -- but contains stylized directions
        samples = []
        for i in range(n_styles):
            samples.append(generator(stylized_w[:, i,...], input_is_latent=True))
    else:
        sample = generator([z], truncation=0.7, truncation_latent=mean_latent)


    if splatting:
        my_samples = []
        for i in range(len(names)):
            stylized_my_w = my_w.clone()
            stylized_my_w[:,id_swap, i*(int(latent_dim/n_styles)):(i+1)*(int(latent_dim/n_styles))] = 0.0
            my_samples.append(generator(stylized_my_w, input_is_latent=True))
    elif learning:
        my_samples = []
        stylized_my_w = dirnet(my_w.repeat([n_styles, 1, 1]))
        for i in range(len(names)):
            my_samples.append(generator(stylized_my_w[i,...].unsqueeze(0), input_is_latent=True))
    else:
        my_sample = generator(my_w, input_is_latent=True)

# display reference images
style_images = []
for name in names:
    style_path = f'style_images_aligned/{strip_path_extension(name)}.png'
    style_image = transform(Image.open(style_path))
    style_images.append(style_image)

face = transform(aligned_face).to(device).unsqueeze(0)
style_images = torch.stack(style_images, 0).to(device)
display_image(utils.make_grid(style_images, normalize=True, range=(-1, 1)), title='References',
              save=True, use_wandb=use_wandb)

if splatting or learning:
    for i in range(len(names)):
        my_output = torch.cat([face, my_samples[i]], 0)
        display_image(utils.make_grid(my_output, normalize=True, range=(-1, 1)), title='Training sample'.format(i,num_iter,alpha),
                      save=True, use_wandb=use_wandb)
else:
    my_output = torch.cat([face, my_sample], 0)
    display_image(utils.make_grid(my_output, normalize=True, range=(-1, 1)), title='My sample')

if splatting or learning:
    for i in range(len(names)):
        output = torch.cat([original_sample, samples[i]], 0)
        display_image(utils.make_grid(output, normalize=True, range=(-1, 1), nrow=n_sample),
                      title='Random samples'.format(i,num_iter,alpha),
                      save=True, use_wandb=use_wandb)
else:
    output = torch.cat([original_sample, sample], 0)
    display_image(utils.make_grid(output, normalize=True, range=(-1, 1), nrow=n_sample), title='Random samples')

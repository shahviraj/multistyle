import argparse

import torch
from torchvision import utils

from model import Generator
from model_utils import DirNet
import os

if __name__ == "__main__":
    torch.set_grad_enabled(False)

    parser = argparse.ArgumentParser(description="Apply closed form factorization")

    parser.add_argument(
        "-i", "--index", type=int, default=0, help="index of eigenvector"
    )
    parser.add_argument(
        "-d",
        "--degree",
        type=float,
        default=5,
        help="scalar factors for moving latent vectors along eigenvector",
    )
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help='channel multiplier factor. config-f = 2, else = 1',
    )
    parser.add_argument("--ckpt", type=str, required=True, help="stylegan2 checkpoints")
    parser.add_argument(
        "--size", type=int, default=1024, help="output image size of the generator"
    )
    parser.add_argument(
        "-n", "--n_sample", type=int, default=7, help="number of samples created"
    )
    parser.add_argument(
        "--truncation", type=float, default=0.7, help="truncation factor"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="device to run the model"
    )
    parser.add_argument(
        "--out_prefix",
        type=str,
        default="factor",
        help="filename prefix to result samples",
    )
    parser.add_argument(
        "factor",
        type=str,
        help="name of the closed form factorization result factor file",
    )

    args = parser.parse_args()
    for i in range(20):
        args.index = i
        eigvec = torch.load(args.factor)["eigvec"].to(args.device)
        ckpt = torch.load(args.ckpt)
        g = Generator(args.size, 512, 8, channel_multiplier=args.channel_multiplier).to(args.device)
        g.load_state_dict(ckpt["g"], strict=False)

        config = ckpt['config']

        id_swap = config['sty_mix'] + list(range(7, g.n_latent))
        dirnet = DirNet(512, 512, len(config['names']), id_swap, init=config['init'], activation=config['dir_act'],
                        device=args.device).to(args.device)
        dirnet.load_state_dict(ckpt['dirnet'], strict=True)

        trunc = g.mean_latent(4096)

        #latent = torch.randn(args.n_sample, 512, device=args.device)
        #latent = g.get_latent(latent)
        inv_code_path = os.path.join(f'../../models/multistyle/e4e_inversion_codes',
                                     f'iu_aligned.pt')
        my_ws =torch.stack([torch.load(inv_code_path)['latent']],0)

        stylized_my_w = dirnet(my_ws.unsqueeze(1).repeat([1, len(config['names']), 1, 1]))
        latent = stylized_my_w[0,0]
        direction = args.degree * eigvec[:, args.index].unsqueeze(0)

        img = g(
            latent.unsqueeze(0),
            truncation=args.truncation,
            truncation_latent=trunc,
            input_is_latent=True,

        )
        img1 = g(
            (latent + direction).unsqueeze(0),
            truncation=args.truncation,
            truncation_latent=trunc,
            input_is_latent=True,
        )
        img2 = g(
            (latent - direction).unsqueeze(0),
            truncation=args.truncation,
            truncation_latent=trunc,
            input_is_latent=True,
        )

        grid = utils.save_image(
            torch.cat([img1, img, img2], 0),
            f"/home/vjshah3/research/outputs/multistyle/results/sefa/{args.out_prefix}_index-{args.index}_degree-{args.degree}.jpeg",
            normalize=True,
            range=(-1, 1),
            nrow=args.n_sample,
        )

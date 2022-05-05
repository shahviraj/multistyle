from model import *
import scipy


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

    blocksize = int(latent.shape[-1]/2)
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

class DirNetDiag(nn.Module):
    def __init__(
            self, in_dim, out_dim, n_out, n_indx, init, bias=True, bias_init=0, lr_mul=1, activation=None, device='cpu'
    ):
        super(DirNetDiag, self).__init__()
        self.n_indx = n_indx # index of the style code that is to be used in style mixing
        self.n_out = n_out
        eqlinlayers = []
        for i in range(n_out):
            eqlinlayers.append(EqualLinearActDiag(in_dim, out_dim, init, bias, bias_init, lr_mul, activation).to(device))
        self.layers = nn.ModuleList(eqlinlayers) # crucial in order to register every layer in the list properly

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


class DirNetOrtho(nn.Module):
    def __init__(
            self, in_dim, out_dim, n_out, n_indx, init, bias=True, bias_init=0, lr_mul=1, activation=None, device='cpu'
    ):
        super(DirNetOrtho, self).__init__()
        self.n_indx = n_indx # index of the style code that is to be used in style mixing
        self.n_out = n_out
        self.activation = activation
        eqlinlayers = []
        for i in range(n_out):
            eqlinlayers.append(nn.utils.parametrizations.orthogonal(nn.Linear(in_dim, out_dim, bias)).to(device))
        self.layers = nn.ModuleList(eqlinlayers) # crucial in order to register every layer in the list properly

    def forward(self, input):
        if self.activation == 'relu':
            act = nn.ReLU()
        elif self.activation == 'tanh':
            act = nn.Tanh()
        else:
            act = nn.Identity
        if len(input.shape) == 4:
            outs = input.clone()
            for k in range(input.shape[0]):
                for i in range(self.n_out):
                    for j in self.n_indx:
                        outs[k, i, j, :] = act(self.layers[i](input[k, i, j, :]))
        elif len(input.shape) == 3:
            outs = input.clone()
            for i in range(self.n_out):
                for j in self.n_indx:
                    outs[i, j, :] = act(self.layers[i](input[i, j, :]))
        else:
            raise NotImplementedError("not implemented when the input tensor is 2-dimensional")

        return outs

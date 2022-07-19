from model import *
import scipy
import copy
import torchvision

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

class DirNet_Id(nn.Module):
    def forward(self, x):
        return x

class DirNet(nn.Module):
    def __init__(
            self, in_dim, out_dim, n_out, n_indx, init, bias=True, bias_init=0, lr_mul=1, activation=None, device='cpu'
    ):
        super(DirNet, self).__init__()
        self.n_indx = n_indx # list of indices of the style code that is to be used in style mixing
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

class DirNetSep(nn.Module):
    def __init__(
            self, in_dim, out_dim, n_out, n_indx, init, bias=True, bias_init=0, lr_mul=1, activation=None, device='cpu'
    ):
        super(DirNetSep, self).__init__()
        self.n_indx = n_indx # list of indices of the style code that is to be used in style mixing
        self.n_out = n_out
        self.layers = []
        for i in range(n_out):
            eqlinlayers = []
            for j in range(len(self.n_indx)):
                eqlinlayers.append(
                    EqualLinearAct(in_dim, out_dim, init, bias, bias_init, lr_mul, activation).to(device))
            self.layers.append((eqlinlayers))  # crucial in order to register every layer in the list properly

        for i in range(n_out):
            self.layers[i] = nn.ModuleList(self.layers[i])
        self.layers = nn.ModuleList(self.layers)

    def forward(self, input):
        if len(input.shape) == 4:
            outs = input.clone()
            for k in range(input.shape[0]):
                for i in range(self.n_out):
                    for j in range(len(self.n_indx)):
                        outs[k, i, self.n_indx[j], :] = self.layers[i][j](input[k, i, self.n_indx[j], :])
        elif len(input.shape) == 3:
            outs = input.clone()
            for i in range(self.n_out):
                for j in range(len(self.n_indx)):
                    outs[i, self.n_indx[j], :] = self.layers[i][j](input[i, self.n_indx[j], :])
        else:
            raise NotImplementedError("not implemented when the input tensor is 2-dimensional")

        return outs

class DirNetMod(nn.Module):
    def __init__(
            self, in_dim, out_dim, n_out, n_indx, init, bias=True, bias_init=0, lr_mul=1, activation=None, device='cpu'
    ):
        super(DirNetMod, self).__init__()
        self.out_dim_list = [512]*15 + [256, 256, 256, 128, 128, 128, 64, 64, 64, 32, 32]
        self.n_indx = n_indx # index of the style code that is to be used in style mixing
        self.n_out = n_out
        self.layers = []
        for i in range(n_out):
            eqlinlayers = []
            for j in range(len(self.n_indx)):
                eqlinlayers.append(
                    EqualLinearAct(in_dim, self.out_dim_list[self.n_indx[j]], init, bias, bias_init, lr_mul, activation).to(device))
            self.layers.append((eqlinlayers))  # crucial in order to register every layer in the list properly
        for i in range(n_out):
            self.layers[i] = nn.ModuleList(self.layers[i])
        self.layers = nn.ModuleList(self.layers)

    def forward(self, input):
            if len(input.shape) == 4:
                outs = input.clone()
                for k in range(input.shape[0]):
                    for i in range(self.n_out):
                        for j in range(len(self.n_indx)):
                            outs[k, i, self.n_indx[j], :self.out_dim_list[self.n_indx[j]]] = self.layers[i][j](input[k, i, self.n_indx[j], :])
            elif len(input.shape) == 3:
                outs = input.clone()
                for i in range(self.n_out):
                    for j in range(len(self.n_indx)):
                        outs[i, self.n_indx[j], :self.out_dim_list[self.n_indx[j]]] = self.layers[i][j](input[i, self.n_indx[j], :])
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
            act = nn.Identity()
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


class DirNetSingleOrtho(nn.Module):
    def __init__(
            self, in_dim, out_dim, n_out, n_indx, init, bias=True, bias_init=0, lr_mul=1, activation=None, device='cpu'
    ):
        super(DirNetSingleOrtho, self).__init__()
        self.n_indx = n_indx # index of the style code that is to be used in style mixing
        self.n_out = n_out
        self.activation = activation
        eqlinlayers = []
        for i in range(n_out):
            if i == 0:
                eqlinlayers.append(nn.Identity().to(device))
            else:
                #eqlinlayers.append(nn.utils.parametrizations.orthogonal(nn.Linear(in_dim, out_dim, bias)).to(device))
                eqlinlayers.append((nn.Linear(in_dim, out_dim, bias)).to(device))
        self.layers = nn.ModuleList(eqlinlayers) # crucial in order to register every layer in the list properly

    def forward(self, input):
        if self.activation == 'relu':
            act = nn.ReLU()
        elif self.activation == 'tanh':
            act = nn.Tanh()
        else:
            act = nn.Identity()
        if len(input.shape) == 4:
            outs = input.clone()
            for k in range(input.shape[0]):
                for i in range(self.n_out):
                    if i == 0:
                        for j in self.n_indx:
                            outs[k, i, j, :] = (self.layers[i](input[k, i, j, :]))
                    else:
                        for j in self.n_indx:
                            outs[k, i, j, :] = act(self.layers[i](input[k, i, j, :]))
        elif len(input.shape) == 3:
            outs = input.clone()
            for i in range(self.n_out):
                if i == 0:
                    for j in self.n_indx:
                        outs[i, j, :] = (self.layers[i](input[i, j, :]))
                else:
                    for j in self.n_indx:
                        outs[i, j, :] = act(self.layers[i](input[i, j, :]))
        else:
            raise NotImplementedError("not implemented when the input tensor is 2-dimensional")

        return outs

def modify_generator_old(generator, n_styles):
    generator.conv1.conv.modulation = nn.ModuleList([copy.deepcopy(generator.conv1.conv.modulation)]*(n_styles+1))
    generator.to_rgb1.conv.modulation = nn.ModuleList([copy.deepcopy(generator.to_rgb1.conv.modulation)] * (n_styles + 1))

    for l in generator.convs:
        l.conv.modulation = nn.ModuleList([copy.deepcopy(l.conv.modulation)] * (n_styles + 1))
    for l in generator.to_rgbs:
        l.conv.modulation = nn.ModuleList([copy.deepcopy(l.conv.modulation)]*(n_styles+1))

    return generator

def modify_generator(generator, n_styles):
    state_dict = generator.state_dict()
    for i in range(n_styles+1):
        generator.conv1.conv.stylemodulation[i].weight = torch.nn.Parameter(state_dict['conv1.conv.modulation.weight'].detach().clone())
        generator.to_rgb1.conv.stylemodulation[i].weight = torch.nn.Parameter(state_dict['to_rgb1.conv.modulation.weight'].detach().clone())
        generator.conv1.conv.stylemodulation[i].bias = torch.nn.Parameter(state_dict['conv1.conv.modulation.bias'].detach().clone())
        generator.to_rgb1.conv.stylemodulation[i].bias = torch.nn.Parameter(state_dict['to_rgb1.conv.modulation.bias'].detach().clone())

    for i, l in enumerate(generator.convs):
        for j in range(n_styles+1):
            l.conv.stylemodulation[j].weight = torch.nn.Parameter(state_dict['convs.{}.conv.modulation.weight'.format(i)].detach().clone())
            l.conv.stylemodulation[j].bias = torch.nn.Parameter(state_dict['convs.{}.conv.modulation.bias'.format(i)].detach().clone())
    for i,l in enumerate(generator.to_rgbs):
        for j in range(3):
            l.conv.stylemodulation[j].weight = torch.nn.Parameter(state_dict['to_rgbs.{}.conv.modulation.weight'.format(i)].detach().clone())
            l.conv.stylemodulation[j].bias = torch.nn.Parameter(state_dict['to_rgbs.{}.conv.modulation.bias'.format(i)].detach().clone())

    return generator

def modify_state_dict(generator, ckpt, n_styles):

    for i in range(n_styles+1):
        if i == 0:
            ckpt['g_ema'][f'conv1.conv.stylemodulation.{i}.weight'] = ckpt['g_ema']['conv1.conv.modulation.weight'].data
            ckpt['g_ema'][f'conv1.conv.stylemodulation.{i}.bias'] =  ckpt['g_ema']['conv1.conv.modulation.bias'].data
            ckpt['g_ema'][f'to_rgb1.conv.stylemodulation.{i}.weight'] =  ckpt['g_ema']['to_rgb1.conv.modulation.weight'].data
            ckpt['g_ema'][f'to_rgb1.conv.stylemodulation.{i}.bias'] = ckpt['g_ema']['to_rgb1.conv.modulation.bias'].data
        else:
            ckpt['g_ema'][f'conv1.conv.stylemodulation.{i}.weight'] = torch.randn_like(ckpt['g_ema']['conv1.conv.modulation.weight'].data)
            ckpt['g_ema'][f'conv1.conv.stylemodulation.{i}.bias'] =  torch.zeros_like(ckpt['g_ema']['conv1.conv.modulation.bias'].data)
            ckpt['g_ema'][f'to_rgb1.conv.stylemodulation.{i}.weight'] =  torch.randn_like(ckpt['g_ema']['to_rgb1.conv.modulation.weight'].data)
            ckpt['g_ema'][f'to_rgb1.conv.stylemodulation.{i}.bias'] =  torch.zeros_like(ckpt['g_ema']['to_rgb1.conv.modulation.bias'].data)

    for i, l in enumerate(generator.convs):
        for j in range(n_styles+1):
            if j == 0:
                ckpt['g_ema'][f'convs.{i}.conv.stylemodulation.{j}.weight']=  ckpt['g_ema'][f'convs.{i}.conv.modulation.weight'].data
                ckpt['g_ema'][f'convs.{i}.conv.stylemodulation.{j}.bias'] = ckpt['g_ema'][
                    f'convs.{i}.conv.modulation.bias'].data
            else:
                ckpt['g_ema'][f'convs.{i}.conv.stylemodulation.{j}.weight'] = torch.randn_like(ckpt['g_ema'][
                    f'convs.{i}.conv.modulation.weight'].data)
                ckpt['g_ema'][f'convs.{i}.conv.stylemodulation.{j}.bias'] = torch.zeros_like(ckpt['g_ema'][
                    f'convs.{i}.conv.modulation.bias'].data)

    for i,l in enumerate(generator.to_rgbs):
        for j in range(n_styles+1):
            if j == 0:
                ckpt['g_ema'][f'to_rgbs.{i}.conv.stylemodulation.{j}.weight'] = ckpt['g_ema'][
                    f'to_rgbs.{i}.conv.modulation.weight'].data
                ckpt['g_ema'][f'to_rgbs.{i}.conv.stylemodulation.{j}.bias'] = ckpt['g_ema'][
                    f'to_rgbs.{i}.conv.modulation.bias'].data
            else:
                ckpt['g_ema'][f'to_rgbs.{i}.conv.stylemodulation.{j}.weight'] = torch.randn_like(ckpt['g_ema'][
                    f'to_rgbs.{i}.conv.modulation.weight'].data)
                ckpt['g_ema'][f'to_rgbs.{i}.conv.stylemodulation.{j}.bias'] = torch.zeros_like(ckpt['g_ema'][
                    f'to_rgbs.{i}.conv.modulation.bias'].data)
    return ckpt

def modify_state_dict_old(generator, ckpt, n_styles):

    for i in range(n_styles+1):
        ckpt['g_ema'][f'conv1.conv.stylemodulation.{i}.weight'] = ckpt['g_ema']['conv1.conv.modulation.weight'].data
        ckpt['g_ema'][f'conv1.conv.stylemodulation.{i}.bias'] =  ckpt['g_ema']['conv1.conv.modulation.bias'].data
        ckpt['g_ema'][f'to_rgb1.conv.stylemodulation.{i}.weight'] =  ckpt['g_ema']['to_rgb1.conv.modulation.weight'].data
        ckpt['g_ema'][f'to_rgb1.conv.stylemodulation.{i}.bias'] = ckpt['g_ema']['to_rgb1.conv.modulation.bias'].data

    for i, l in enumerate(generator.convs):
        for j in range(n_styles+1):
            ckpt['g_ema'][f'convs.{i}.conv.stylemodulation.{j}.weight']=  ckpt['g_ema'][f'convs.{i}.conv.modulation.weight'].data
            ckpt['g_ema'][f'convs.{i}.conv.stylemodulation.{j}.bias'] = ckpt['g_ema'][
                f'convs.{i}.conv.modulation.bias'].data
    for i,l in enumerate(generator.to_rgbs):
        for j in range(3):
            ckpt['g_ema'][f'to_rgbs.{i}.conv.stylemodulation.{j}.weight'] = ckpt['g_ema'][
                f'to_rgbs.{i}.conv.modulation.weight'].data
            ckpt['g_ema'][f'to_rgbs.{i}.conv.stylemodulation.{j}.bias'] = ckpt['g_ema'][
                f'to_rgbs.{i}.conv.modulation.bias'].data

    return ckpt


class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.slice6 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 32):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        for x in range(32, 36):
            self.slice6.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

        self.pool = nn.AdaptiveAvgPool2d(output_size=1)

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1).cuda() * 2 - 1
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1).cuda() * 2

    def forward(self, X):  # relui_1
        X = F.interpolate(X, size=(256, 256), mode='area')
        X = (X - self.mean) / self.std
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5[:-2](h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


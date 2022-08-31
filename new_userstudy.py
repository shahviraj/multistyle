import os.path
import os
import random

from PIL import Image
from matplotlib import pyplot as plt
plt.rcParams['figure.dpi'] = 300
plt.rcParams["font.family"] = "Serif"
plt.rcParams["font.size"] = 10
import json
from matplotlib.gridspec import GridSpec

n_qsts = 8

all_ref_codes = [3, 5, 4, 9, 12, 13, 14, 15, 16, 20, 19, 22]
# codes of the style images that are to be included
ref_codes = random.sample(all_ref_codes, n_qsts)
all_input_codes = list(range(1,16))
# codes of the input images that are to be included
input_codes = random.sample(all_input_codes, n_qsts)


# location of jojogan folder
jojogan_path = '/home/vjshah3/research/outputs/multistyle/results/1styles_'

# location of multistyle folder
multistyle_path = '/home/vjshah3/research/outputs/multistyle/results/12styles_3_5_4_9_12_13_14_15_16_20_19_22_ctx_wt_0.005_n_iter_1000_type_std_inv_method_e4e_init_identity/test/'
mtg_path = '/home/vjshah3/research/remote-gan-codes/MindTheGap/output/inference_all_ii2s/'
#inputs
input_path = '/home/vjshah3/research/outputs/multistyle/results/inputs/'
ref_path = '/home/vjshah3/research/outputs/multistyle/results/refs/'

# save the user study images at:
save_path = f'/home/vjshah3/research/outputs/multistyle/userstudies/new_userstudy_{input_codes}_{ref_codes}/'
if not os.path.exists(save_path):
    os.makedirs(save_path, exist_ok=True)

from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np

gt_dict = {}
assert len(ref_codes) == len(input_codes)




def load_image(method):
    if method == 'jojogan':
        return Image.open(jojogan_path+f'{ref_codes[i]}/test/{input_codes[i]}_{ref_codes[i]}.jpeg').convert('RGB')
    elif method == 'multistyle':
        return Image.open(multistyle_path + f'{input_codes[i]}_{ref_codes[i]}.jpeg').convert('RGB')
    elif method == 'mtg':
        return Image.open(mtg_path + f'{input_codes[i]}_{ref_codes[i]}.png').convert('RGB')

for i in range(len(ref_codes)):

    im1 = Image.open(input_path+f'{input_codes[i]}_0.png').convert('RGB')
    im2 = Image.open(ref_path+f'0_{ref_codes[i]}.png').convert('RGB').resize((1024,1024))

    methods = ['jojogan', 'multistyle', 'mtg']
    random.shuffle(methods)

    opa = load_image(methods[0])
    opb = load_image(methods[1])
    opc = load_image(methods[2])

    gt_dict[i] = {'ref_code': ref_codes[i],
                  'input_code': input_codes[i],
                  'A' : methods[0],
                  'B' : methods[1],
                  'C' : methods[2]
                  }

    titles = [
        'Input face',
        'Reference style',
        'Option A',
        'Option B',
        'Option C'
    ]



    def format_axes(fig):
        for i, ax in enumerate(fig.axes):
            ax.tick_params(labelbottom=False, labelleft=False)


    fig = plt.figure(figsize=(6., 4.), constrained_layout=True)

    gs = GridSpec(4, 6, figure=fig)
    ax1 = fig.add_subplot(gs[0:2, :3])
    ax2 = fig.add_subplot(gs[0:2, 3:])
    # identical to ax1 = plt.subplot(gs.new_subplotspec((0, 0), colspan=3))

    axa = fig.add_subplot(gs[2:, :2])
    axb = fig.add_subplot(gs[2:, 2:4])
    axc = fig.add_subplot(gs[2:, 4:])

    for t, ax, im in zip(titles,[ax1, ax2, axa, axb, axc], [im1, im2, opa, opb, opc]):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)
        ax.axis('off')
        ax.set_title(t)

    format_axes(fig)

    #plt.show()
    plt.savefig(save_path + f'image_{i}_input_{input_codes[i]}_ref_{ref_codes[i]}_opa_{gt_dict[i]["A"]}_opb_{gt_dict[i]["B"]}_opc_{gt_dict[i]["C"]}.png', bbox_inches='tight')


json_obj = json.dumps(gt_dict)
with open(save_path + "gt_dict.json",'w')  as f:
    f.write(json_obj)

print("Done!")
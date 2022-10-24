import math
import copy
import torch
import numpy as np
from torch import nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import Train, Model, Utils, Utilities, Augmentations
import signatory

from torch.utils import data
from multiprocessing import cpu_count
from torch.cuda.amp import autocast, GradScaler

from pathlib import Path
from torch.optim import Adam
from torchvision import transforms, utils
from PIL import Image

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

def histogram_plot(real_signature, fake_signature, dim, level, result_folder, grid_number=None, starting_point=0,
                   file_name=None, save_fig=False):
    sig_dim = real_signature.shape[-1]
    if not grid_number:
        grid_number = int(math.sqrt(sig_dim)) + 1
    fig, ax = plt.subplots(grid_number, grid_number)
    fig.set_size_inches(30, 30)
    helper = Utilities.algebra_coordinate_helper(dim, level)
    k = 0
    sns.set()
    for i in range(grid_number):
        for j in range(grid_number):
            k += 1
            if k > sig_dim - 1:
                break
            ax[i, j].hist((np.array(real_signature[:, starting_point + k].cpu()), np.array(fake_signature[:, starting_point + k].cpu()),), bins=20,
                          label=('real', 'fake'))
            ax[i, j].legend()
            ax[i, j].set_title(str('n = {}'.format(starting_point + 8 * i + j)) + ' ' + str(helper[starting_point + k]))
    fig.show()
    if save_fig:
        #         fig.savefig(result_folder + '/histogram_{}.png'.format(starting_point))
        fig.savefig(result_folder + '/' + file_name)
    return


def logsig_histogram_plot(real_logsignature, fake_logsignature, dim, level, result_folder, grid_number=None, starting_point=0,
                          file_name=None, save_fig=False):
    logsig_dim = real_logsignature.shape[-1]
    if not grid_number:
        grid_number = int(math.sqrt(logsig_dim)) + 1
    fig, ax = plt.subplots(grid_number, grid_number)
    fig.set_size_inches(30, 30)

    logsig_length = torch.zeros(level)
    for i in range(1, level + 1):
        logsig_length[i - 1] = signatory.logsignature_channels(dim, i)
    #     helper = Utilities.algebra_coordinate_helper(dim,level)
    k = 0
    sns.set()
    for i in range(grid_number):
        for j in range(grid_number):
            if k > logsig_dim - 1:
                break
            ax[i, j].hist((np.array(real_logsignature[:, starting_point + k].cpu()), np.array(fake_logsignature[:, starting_point + k].cpu()),),
                          bins=20, label=('real', 'fake'))
            ax[i, j].legend()
            ax[i, j].set_title(str('n = {}'.format(starting_point + grid_number * i + j)) + ' degree: ' +
                               str((logsig_length < (starting_point + grid_number * i + j)).count_nonzero() + 1))
            k += 1
    fig.show()
    if save_fig:
        fig.savefig(result_folder + '/' + file_name)
    return


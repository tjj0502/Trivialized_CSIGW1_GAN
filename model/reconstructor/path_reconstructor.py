import torch
import numpy as np
from torch import nn, einsum
from torch.utils import data
from torch.cuda.amp import autocast, GradScaler
from torch import optim
from pathlib import Path
from torch.optim import Adam
from tqdm.auto import tqdm
import signatory
# import Train, Model, Utils, Utilities, Augmentations
import matplotlib.pyplot as plt
import seaborn as sns


class Base_Reconstructor(nn.Module):
    """
    The model aims to learn the inverse mapping from signatures with certain path-augmentation to the original path.
    """

    def __init__(self, conditional_dim, sig_dim, sig_level, path_length, device, augmentation='lead_lag'):
        super(Base_Reconstructor, self).__init__()

        # self.dict = {
        #     'lead_lag': 'LeadLag',
        #     'time': 'AddTime',
        #     'raw': 'Scale'
        # }

        self.device = device
        self.conditional_dim = conditional_dim
        self.sig_dim = sig_dim
        self.sig_level = sig_level
        self.path_length = path_length - 1
        self.augmentation_ = augmentation
        # self.augmentation = Augmentations.parse_augmentations([{'name': self.dict[augmentation]}])
        # self.aug_dim = Augmentations.get_number_of_channels_after_augmentations(self.sig_dim, self.augmentation)
        self.aug_dim = 2 * self.sig_dim  # Lead-Lag augmentation
        self.sig_len = signatory.signature_channels(self.aug_dim, self.level)
        self.device = device
        self.model = nn.Sequential()

    def forward(self, signatures, path_lengths):
        input_ = torch.cat([signatures, path_lengths], 1)
        output = self.model(input_)
        output = torch.reshape(output, (output.shape[0], self.path_length, self.dim))
        # We first the starting point to 0
        output = torch.cat([torch.tensor(torch.zeros(output.shape[0], 1, output.shape[2]).to(output.device)), output], 1)
        return output


class Poisson_Reconstructor(Base_Reconstructor):
    """
    We reconstruct from logsignature to path
    """

    def __init__(self, conditional_dim, sig_dim, sig_level, path_length, device):
        super(Poisson_Reconstructor, self).__init__(conditional_dim, sig_dim, sig_level, path_length, device)

        # Initialize the parameters to be trained
        self.model = nn.Sequential(
            nn.Linear(self.sig_len + self.conditional_dim, self.path_length * self.dim),
            nn.LeakyReLU(),
            nn.Linear(self.path_length * self.dim, self.path_length * self.dim),
            nn.LeakyReLU(),
            nn.Linear(self.path_length * self.dim, self.path_length * self.dim)
            #             nn.Sigmoid()
        )


class Pendigit_Reconstructor(Base_Reconstructor):
    """
    We reconstruct from logsignature to path
    """

    def __init__(self, conditional_dim, sig_dim, sig_level, path_length, device):
        super(Pendigit_Reconstructor, self).__init__(conditional_dim, sig_dim, sig_level, path_length, device)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat))
            # layers.append(nn.Sigmoid())
            #             layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # Initialize the parameters to be trained
        self.model = nn.Sequential(
            *block(self.sig_len + self.conditional_dim, self.path_length * self.dim, normalize=False),
            nn.LeakyReLU(0.3),
            *block(self.path_length * self.dim, self.path_length * self.dim),
            nn.LeakyReLU(0.3),
            *block(self.path_length * self.dim, self.path_length * self.dim),
            nn.Sigmoid()
        )

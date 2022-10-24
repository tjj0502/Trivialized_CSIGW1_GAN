from torch import nn
import torch
import numpy as np
import signatory
# import Utilities, Parametrization


# GAN models
class Base_Logsig_Generator(nn.Module):
    '''
    Signature generator using logsig trivialization, conditional on the path length
    '''
    def __init__(self, input_dim, hidden_dim, conditional_dim, sig_dim, sig_level, device):
        super(Base_Logsig_Generator, self).__init__()
        self.input_dim = input_dim  # Noise dimension
        self.sig_dim = sig_dim  # Time series dimension
        self.sig_level = sig_level  # Signature degree
        self.hidden_dim = hidden_dim  # Hidden dimension
        self.aug_dim = 2 * self.dim  # Lead-lag augmentation
        self.logsig_length = torch.zeros(self.sig_level)
        for i in range(1, self.sig_level + 1):
            self.logsig_length[i - 1] = signatory.logsignature_channels(self.aug_dim, i)
        self.lyndon_matrix = Utilities.lyndon_matrix(self.aug_dim, self.level).to(device)
        #         print(self.dim, self.level, self.lyndon_matrix.shape, self.logsig_length)
        self.lyndon_inverse = torch.tensor(np.linalg.pinv(self.lyndon_matrix.cpu())).to(device)
        self.orthonormal_matrix, _ = torch.linalg.qr(self.lyndon_matrix)
        self.orthonormal_matrix = self.orthonormal_matrix.to(device)
        self.param = Parametrization.differential_exp_class.apply
        self.conditional_dim = conditional_dim  # Dimension of conditional variable
        self.device = device

    def block(self, in_feat, out_feat, normalize = True, activation = 'sigmoid'):
        layers = [nn.Linear(in_feat, out_feat)]
        if normalize:
            layers.append(nn.BatchNorm1d(out_feat))
        if activation == 'relu':
            layers.append(nn.LeakyReLU(0.3))
        elif activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        return layers

    def forward(self, z, lengths):
        raise NotImplementedError

class Poisson_Conditional_Logsig_Generator(Base_Logsig_Generator):
    '''
    Signature generator using logsig trivialization, conditional on the path length
    '''

    def __init__(self, input_dim, hidden_dim, conditional_dim, sig_dim, sig_level, device):
        super(Poisson_Conditional_Logsig_Generator, self).__init__(input_dim, hidden_dim, conditional_dim, sig_dim, sig_level, device)



        # Different models
        self.model = nn.Sequential(
            *self.block(self.input_dim + self.conditional_dim, self.hidden_dim , normalize=False),
            *self.block(self.hidden_dim , self.hidden_dim ),
            *self.block(self.hidden_dim , int(self.logsig_length[-1]))
        )

    def forward(self, z, lengths):
        input_z = torch.cat([z, lengths], dim=-1)
        self.gen_logsig = self.model(input_z)
        return self.param(self.gen_logsig, self.aug_dim, self.level, self.lyndon_matrix, self.orthonormal_matrix, self.lyndon_inverse)


        # Trivialization via autograd
        # tensor_A = torch.matmul(self.lyndon_matrix, self.gen_logsig.t()).t()
        # result = Utilities.tensor_exp(tensor_A, self.aug_dim, self.level, "tensor")
        # return result



class Pendigit_Conditional_Logsig_Generator(Base_Logsig_Generator):
    '''
    Signature generator using logsig trivialization, conditional on the path length
    '''

    def __init__(self, input_dim, hidden_dim, conditional_dim, sig_dim, sig_level, device):
        super(Pendigit_Conditional_Logsig_Generator, self).__init__(input_dim, hidden_dim, conditional_dim, sig_dim, sig_level, device)

        # Different models
        self.model_1 = nn.Sequential(
                *self.block(self.input_dim + self.conditional_dim, self.hidden_dim, normalize=False, activation = 'relu'),
                *self.block(self.hidden_dim, self.hidden_dim, activation = 'relu'),
                *self.block(self.hidden_dim, int(self.logsig_length[-1]))
            )

    def forward(self, z, lengths):
        input_z = torch.cat([z, lengths], dim=-1)
        self.gen_logsig = self.model(input_z)
        return self.param(self.gen_logsig, self.aug_dim, self.level, self.lyndon_matrix, self.orthonormal_matrix, self.lyndon_inverse)


        # Trivialization via autograd
        # tensor_A = torch.matmul(self.lyndon_matrix, self.gen_logsig.t()).t()
        # result = Utilities.tensor_exp(tensor_A, self.aug_dim, self.level, "tensor")
        # return result
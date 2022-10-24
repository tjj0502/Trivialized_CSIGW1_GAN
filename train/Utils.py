# helpers functions
from inspect import isfunction
import torch
import math
from typing import Tuple, Optional
import signatory
import math


class Dataset(torch.utils.data.Dataset):
    """Characterizes a dataset for PyTorch
    We apply the path augmentation to the raw path
    """

    def __init__(self,
                 features,
                 labels,
                 lengths,
                 train=True,
                 device=None):


        """Initialization"""

        self.x = features.clone() # Signatures
        self.y = labels.clone() # Orginal paths, [sample_size, length, channel_dim]
        self.z = lengths.clone() # Conditioning variable
        # paths = torch.load(data_dir + '/filtered_digit_{}.pt'.format(digit))  # [sample_size, length, channel_dim]
        # increment = (paths[:, :, :].roll(-1, 1) - paths[:, :, :])
        # if augmentation == 'lead_lag':
        #     self.x = torch.load(data_dir + '/filtered_digit_{}_'.format(digit) \
        #                         + augmentation + '_signature_{}.pt'.format(level))
        #     # self.y = paths[:,:,:].clone()
        #     self.y = increment.clone()
        # #             self.y = Augmentations.lead_lag_transform(paths[:,:,1:])
        # #             augmentation = 'LeadLag'
        # elif augmentation == 'time':
        #     self.x = torch.load(data_dir + '/' + augmentation + '_aug_signature_{}.pt'.format(level))
        #     self.y = paths
        # else:
        #     raise ("Unknown augmentation.")
        #
        # self.z = torch.load(data_dir + '/filtered_digit_{}_lengths.pt'.format(digit))

        if train:
            self.x = self.x[:int(0.8 * self.x.shape[0])]
            self.y = self.y[:int(0.8 * self.y.shape[0])]
            self.z = self.z[:int(0.8 * self.z.shape[0])]
        else:
            self.x = self.x[int(0.8 * self.x.shape[0]):]
            self.y = self.y[int(0.8 * self.y.shape[0]):]
            self.z = self.z[int(0.8 * self.z.shape[0]):]
        if device:
            self.x = self.x.to(device)
            self.y = self.y.to(device)
            self.z = self.z.to(device)

        self.length = self.x.shape[0]

    def __len__(self):
        """Denotes the total number of samples"""
        return self.length

    def __getitem__(self, index):
        """Generates samples of data"""
        return self.x[index], self.y[index], self.z[index]


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def cycle(dl):
    while True:
        for data in dl:
            yield data


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


# gaussian diffusion trainer class
def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps, lower_bound=0.0001, upper_bound=0.02):
    scale = 1000 / timesteps
    #     scale = 1
    beta_start = scale * lower_bound
    beta_end = scale * upper_bound  # 0.008 for gbm 0.02 for sine
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


# Sig-W1 loss
def lengthwise_sig1metric(x_real, x_fake, sig_dim, sig_level, lengths):
    #     difference = x_real - x_fake
    min_length = int(torch.min(lengths))
    max_length = int(torch.max(lengths))
    loss_mean = 0
    loss_var = 0
    loss_coeff = 0
    for length in range(min_length, max_length):
        #         sig_diff_length_n = difference[(lengths == length).nonzero(as_tuple=False)[:,0]]
        x_real_length_n = x_real[(lengths == length).nonzero(as_tuple=False)[:, 0]]
        x_fake_length_n = x_fake[(lengths == length).nonzero(as_tuple=False)[:, 0]]
        sig_diff_length_n = x_real_length_n - x_fake_length_n

        loss_mean_length_n = compute_expected_signature(sig_diff_length_n, sig_dim, sig_level).pow(2).sum().sqrt()
        loss_var_length_n = rmse(compute_variance_signature(x_real_length_n, sig_dim, sig_level),
                                 compute_variance_signature(x_fake_length_n, sig_dim, sig_level))
        loss_coeff_length_n = rmse(compute_correlation_signature(x_real_length_n),
                                   compute_correlation_signature(x_fake_length_n))

        if not torch.isnan(loss_mean_length_n):
            loss_mean += loss_mean_length_n
        if not torch.isnan(loss_var_length_n):
            loss_var += loss_var_length_n
        if not torch.isnan(loss_coeff_length_n):
            loss_coeff += loss_coeff_length_n

    return loss_mean, loss_var, loss_coeff


# def compute_expected_signature(x_path, depth: int, augmentations: Tuple, normalise: bool = True):
#     x_path_augmented = Augmentation.apply_augmentations(x_path, augmentations)
#     expected_signature = signatory.signature(x_path_augmented, depth=depth).mean(0)
#     dim = x_path_augmented.shape[2]
#     count = 0
#     if normalise:
#         for i in range(depth):
#             expected_signature[count:count + dim**(i+1)] = expected_signature[count:count + dim**(i+1)] * math.factorial(i+1)
#             count = count + dim**(i+1)
#     return expected_signature

def compute_expected_signature(signature, sig_dim, sig_level, normalise: bool = True):
    expected_signature = signature.mean(dim=0)
    count = 1
    if normalise:
        for i in range(sig_level):
            expected_signature[count:count + sig_dim ** (i + 1)] = expected_signature[count:count + sig_dim ** (i + 1)] * math.factorial(i + 1)
            count = count + sig_dim ** (i + 1)
    return expected_signature


def compute_variance_signature(signature, sig_dim, sig_level, normalise: bool = True):
    var_signature = signature.var(dim=0)
    count = 1
    if normalise:
        for i in range(sig_level):
            var_signature[count:count + sig_dim ** (i + 1)] = var_signature[count:count + sig_dim ** (i + 1)] * math.factorial(i + 1)
            count = count + sig_dim ** (i + 1)
    return var_signature


def compute_correlation_signature(real_signature):
    corr_matrix = torch.corrcoef(real_signature[:, 1:].t())
    #     count = 0
    #     if normalise:
    #         for i in range(sig_level):
    #             corr_matrix[count:count + sig_dim**(i+1) , count:count + sig_dim**(i+1)] = corr_matrix[count:count + sig_dim**(i+1), count:count + sig_dim**(i+1)] * math.factorial(i+1)
    #             count = count + sig_dim**(i+1)
    return corr_matrix


def compute_covariance_signature(real_signature):
    cov_matrix = torch.cov(real_signature[:, 1:].t())
    #     count = 0
    #     if normalise:
    #         for i in range(sig_level):
    #             corr_matrix[count:count + sig_dim**(i+1) , count:count + sig_dim**(i+1)] = corr_matrix[count:count + sig_dim**(i+1), count:count + sig_dim**(i+1)] * math.factorial(i+1)
    #             count = count + sig_dim**(i+1)
    return cov_matrix


def rmse(x, y):
    return (x - y).pow(2).sum().sqrt()


def lengthwise_rmse(x, y):
    return (x - y).pow(2).sum(1).sqrt().sum()


def masked_rmse(x, y, mask_rate, device):
    mask = torch.FloatTensor(x.shape[0]).to(device).uniform_() > mask_rate
    mask = mask.int()
    return ((x - y).pow(2) * mask).mean().sqrt()


class SigW1Metric:
    """Including second moment"""

    def __init__(self, dim: int, level: int, real_signature: torch.Tensor, mask_rate: float = 0., normalise: bool = True):
        self.dim = dim
        self.level = level

        self.length = signatory.signature_channels(self.dim, 4) + 1
        #         self.n_lags = x_real.shape[1]
        self.mask_rate = mask_rate

        self.normalise = normalise
        self.expected_signature_mu = compute_expected_signature(real_signature, dim, level, normalise)
        self.var_signature_mu = compute_variance_signature(real_signature, dim, level, normalise)
        self.cov_signature_mu = compute_covariance_signature(real_signature[:, :self.length])  # Only up to level 4

    def __call__(self, fake_signature: torch.Tensor):
        """ Computes the SigW1 metric."""
        device = fake_signature.device
        batch_size = fake_signature.shape[0]
        # expected_signature_nu1 = compute_expected_signature(x_path_nu[:batch_size//2], self.depth, self.augmentations)
        # expected_signature_nu2 = compute_expected_signature(x_path_nu[batch_size//2:], self.depth, self.augmentations)
        # y = self.expected_signature_mu.to(device)
        # loss = (expected_signature_nu1-y)*(expected_signature_nu2-y)
        # loss = loss.sum()
        expected_signature_nu = compute_expected_signature(fake_signature, self.dim, self.level, self.normalise)
        var_signature_nu = compute_variance_signature(fake_signature, self.dim, self.level, self.normalise)
        cov_signature_nu = compute_covariance_signature(fake_signature[:, :self.length])
        loss_mean = rmse(self.expected_signature_mu.to(device), expected_signature_nu)
        loss_var = rmse(self.var_signature_mu.to(device), var_signature_nu)
        loss_cov = rmse(self.cov_signature_mu.to(device), cov_signature_nu)
        # loss = masked_rmse(self.expected_signature_mu.to(
        #    device), expected_signature_nu, self.mask_rate, device)
        return loss_mean, loss_var, loss_cov


class PathwiseSigW1Metric:
    """Including second moment"""

    def __init__(self, dim: int, level: int, real_signature: torch.Tensor, lengths: torch.Tensor, device, mask_rate: float = 0.,
                 normalise: bool = True):

        self.dim = dim
        self.level = level
        self.normalise = normalise
        self.device = device
        self.signature_length = signatory.signature_channels(self.dim, self.level) + 1

        self.min_length = int(torch.min(lengths))
        self.max_length = 11 + 1 # We set the maximum length to 11
        self.expected_mean_mu = torch.zeros([self.max_length - self.min_length, self.signature_length]).to(self.device)
        self.expected_var_mu = torch.zeros([self.max_length - self.min_length, self.signature_length]).to(self.device)
        self.expected_cov_mu = torch.zeros([self.max_length - self.min_length, self.signature_length - 1, self.signature_length - 1]).to(self.device)
        for length in range(self.min_length, self.max_length):
            x_real_length_n = real_signature[(lengths == length).nonzero(as_tuple=False)[:, 0]]
            #             print('n = {} '.format(length), x_real_length_n.shape)
            expected_sig_length_n = compute_expected_signature(x_real_length_n, self.dim, self.level, normalise=self.normalise)
            expected_val_length_n = compute_variance_signature(x_real_length_n, self.dim, self.level, normalise=self.normalise)
            expected_cov_length_n = compute_covariance_signature(x_real_length_n)

            #             print('n = {} '.format(length), torch.isnan(expected_val_length_n).any())

            self.expected_mean_mu[length - self.min_length] = expected_sig_length_n
            self.expected_var_mu[length - self.min_length] = expected_val_length_n

            if not torch.isnan(expected_cov_length_n).any():
                self.expected_cov_mu[length - self.min_length] = expected_cov_length_n

        #         self.n_lags = x_real.shape[1]
        self.mask_rate = mask_rate

    def __call__(self, fake_signature: torch.Tensor, lengths: torch.Tensor):
        """ Computes the SigW1 metric."""
        device = fake_signature.device
        batch_size = fake_signature.shape[0]
        # expected_signature_nu1 = compute_expected_signature(x_path_nu[:batch_size//2], self.depth, self.augmentations)
        # expected_signature_nu2 = compute_expected_signature(x_path_nu[batch_size//2:], self.depth, self.augmentations)
        # y = self.expected_signature_mu.to(device)
        # loss = (expected_signature_nu1-y)*(expected_signature_nu2-y)
        # loss = loss.sum()

        expected_mean_nu = torch.zeros([self.max_length - self.min_length, self.signature_length]).to(device)
        expected_var_nu = torch.zeros([self.max_length - self.min_length, self.signature_length]).to(device)
        expected_cov_nu = torch.zeros([self.max_length - self.min_length, self.signature_length - 1, self.signature_length - 1]).to(device)
        for length in range(self.min_length, self.max_length):
            x_fake_length_n = fake_signature[(lengths == length).nonzero(as_tuple=False)[:, 0]]

            expected_sig_length_n = compute_expected_signature(x_fake_length_n, self.dim, self.level, normalise=self.normalise)
            expected_val_length_n = compute_variance_signature(x_fake_length_n, self.dim, self.level, normalise=self.normalise)
            expected_cov_length_n = compute_covariance_signature(x_fake_length_n)

            expected_mean_nu[length - self.min_length] = expected_sig_length_n
            expected_var_nu[length - self.min_length] = expected_val_length_n
            #             print('n = {} '.format(length), torch.isnan(expected_corr_length_n).any())
            expected_cov_nu[length - self.min_length] = expected_cov_length_n

        loss_mean = lengthwise_rmse(self.expected_mean_mu, expected_mean_nu)
        loss_var = lengthwise_rmse(self.expected_var_mu, expected_var_nu)
        loss_cov = lengthwise_rmse(self.expected_cov_mu, expected_cov_nu)
        # loss = masked_rmse(self.expected_signature_mu.to(
        #    device), expected_signature_nu, self.mask_rate, device)
        return loss_mean, loss_var, loss_cov


class Pendigit_PathwiseSigW1Metric:
    """Including second moment"""

    def __init__(self, dim: int, level: int, real_signature: torch.Tensor, lengths: torch.Tensor, device, mask_rate: float = 0.,
                 normalise: bool = True):

        self.dim = dim
        self.level = level
        self.normalise = normalise
        self.device = device
        self.signature_length = signatory.signature_channels(self.dim, self.level) + 1

        self.number_of_lengths = 5  # Inly choose the first 5 lengths
        _, indices = torch.sort(torch.bincount(lengths.squeeze()), descending=True)
        self.expected_mean_mu = torch.zeros([self.number_of_lengths, self.signature_length]).to(self.device)
        self.expected_var_mu = torch.zeros([self.number_of_lengths, self.signature_length]).to(self.device)
        self.expected_cov_mu = torch.zeros([self.number_of_lengths, self.signature_length - 1, self.signature_length - 1]).to(self.device)
        for idx, length in enumerate(indices[:self.number_of_lengths]):
            x_real_length_n = real_signature[(lengths == length).nonzero(as_tuple=False)[:, 0]]
            #             print('n = {} '.format(length), x_real_length_n.shape)
            expected_sig_length_n = compute_expected_signature(x_real_length_n, self.dim, self.level, normalise=self.normalise)
            expected_val_length_n = compute_variance_signature(x_real_length_n, self.dim, self.level, normalise=self.normalise)
            expected_cov_length_n = compute_covariance_signature(x_real_length_n)

            #             print('n = {} '.format(length), torch.isnan(expected_val_length_n).any())

            self.expected_mean_mu[idx] = expected_sig_length_n
            self.expected_var_mu[idx] = expected_val_length_n

            if not torch.isnan(expected_cov_length_n).any():
                self.expected_cov_mu[idx] = expected_cov_length_n

        #         self.n_lags = x_real.shape[1]
        self.mask_rate = mask_rate

    def __call__(self, fake_signature: torch.Tensor, lengths: torch.Tensor):
        """ Computes the SigW1 metric."""
        device = fake_signature.device
        batch_size = fake_signature.shape[0]
        # expected_signature_nu1 = compute_expected_signature(x_path_nu[:batch_size//2], self.depth, self.augmentations)
        # expected_signature_nu2 = compute_expected_signature(x_path_nu[batch_size//2:], self.depth, self.augmentations)
        # y = self.expected_signature_mu.to(device)
        # loss = (expected_signature_nu1-y)*(expected_signature_nu2-y)
        # loss = loss.sum()
        _, indices = torch.sort(torch.bincount(lengths.squeeze()), descending=True)
        expected_mean_nu = torch.zeros([self.number_of_lengths, self.signature_length]).to(device)
        expected_var_nu = torch.zeros([self.number_of_lengths, self.signature_length]).to(device)
        expected_cov_nu = torch.zeros([self.number_of_lengths, self.signature_length - 1, self.signature_length - 1]).to(device)
        for idx, length in enumerate(indices[:self.number_of_lengths]):
            x_fake_length_n = fake_signature[(lengths == length).nonzero(as_tuple=False)[:, 0]]

            expected_sig_length_n = compute_expected_signature(x_fake_length_n, self.dim, self.level, normalise=self.normalise)
            expected_val_length_n = compute_variance_signature(x_fake_length_n, self.dim, self.level, normalise=self.normalise)
            expected_cov_length_n = compute_covariance_signature(x_fake_length_n)

            expected_mean_nu[idx] = expected_sig_length_n
            expected_var_nu[idx] = expected_val_length_n
            #             print('n = {} '.format(length), torch.isnan(expected_corr_length_n).any())
            expected_cov_nu[idx] = expected_cov_length_n

        loss_mean = lengthwise_rmse(self.expected_mean_mu, expected_mean_nu)
        loss_var = lengthwise_rmse(self.expected_var_mu, expected_var_nu)
        loss_cov = lengthwise_rmse(self.expected_cov_mu, expected_cov_nu)
        # loss = masked_rmse(self.expected_signature_mu.to(
        #    device), expected_signature_nu, self.mask_rate, device)
        return loss_mean, loss_var, loss_cov

    # class LogSigW1Metric:
#     """Including second moment"""
#     def __init__(self, dim: int, level: int, real_logsignature: torch.Tensor, mask_rate: float = 0., normalise: bool = True):

#         self.dim = dim
#         self.level = level

# #         self.length = signatory.signature_channels(self.dim, self.level) + 1

#         self.logsig_length = torch.zeros(level)
#         for i in range(1, level+1):
#             self.logsig_length[i-1] = signatory.logsignature_channels(self.dim, i)
# #         self.n_lags = x_real.shape[1]
#         self.mask_rate = mask_rate

#         self.normalise = normalise
#         self.expected_signature_mu = compute_expected_logsignature(real_signature, dim, level, normalise)
#         self.var_signature_mu = compute_variance_logsignature(real_signature, dim, level, normalise)
#         self.corr_signature_mu = compute_correlation_logsignature(real_signature[:,:self.length]) # Only up to level 4

#     def __call__(self, fake_signature: torch.Tensor):
#         """ Computes the SigW1 metric."""
#         device = fake_signature.device
#         batch_size = fake_signature.shape[0]
#         # expected_signature_nu1 = compute_expected_signature(x_path_nu[:batch_size//2], self.depth, self.augmentations)
#         # expected_signature_nu2 = compute_expected_signature(x_path_nu[batch_size//2:], self.depth, self.augmentations)
#         # y = self.expected_signature_mu.to(device)
#         # loss = (expected_signature_nu1-y)*(expected_signature_nu2-y)
#         # loss = loss.sum()
#         expected_signature_nu = compute_expected_signature(fake_signature, self.dim, self.level, self.normalise)
#         var_signature_nu = compute_variance_signature(fake_signature, self.dim, self.level, self.normalise)
#         corr_signature_mu = compute_correlation_signature(fake_signature[:,:self.length])
#         loss_mean = rmse(self.expected_signature_mu.to(device), expected_signature_nu)
#         loss_var = rmse(self.var_signature_mu.to(device), var_signature_nu)
#         loss_corr = rmse(self.corr_signature_mu.to(device), corr_signature_mu)
#         # loss = masked_rmse(self.expected_signature_mu.to(
#         #    device), expected_signature_nu, self.mask_rate, device)
#         return loss_mean, loss_var, loss_corr


def masking(paths, lengths):
    """
    Given the path as full length, only extract the first m entries
    """
    for index, length in enumerate(lengths):
        paths[index ,1:length,:] = 1
    return paths


def param_norm(model):
    param_norm_loss = torch.zeros(1).to(model.device)
    for param in model.parameters():
        param_norm_loss += torch.norm(param, p=1)
    return param_norm_loss
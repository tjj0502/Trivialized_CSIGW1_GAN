import torch
from torch.utils import data
from tqdm.auto import tqdm
from torch.cuda.amp import autocast
import copy
from base import Generator_trainer, Reconstructor_trainer
import Utils


class Pendigit_generator_trainer(Generator_trainer):
    def __init__(
            self,
            generator,
            data_path,
            digit,
            device,
            train_batch_size=64,
            train_lr=1e-3,
            train_lr_gamma=0.95,
            train_lr_step_size=1280,
            train_num_steps=10000,
            amp=False,
            save_model=True,
            save_every=1000,
            loss_track_every=100,
            results_folder='./results',
            var_coeff=1.,
            cov_coeff=1.
    ):
        super(Pendigit_generator_trainer).__init__(generator,
                                         device,
                                         train_batch_size,
                                         train_lr,
                                         train_lr_gamma,
                                         train_lr_step_size,
                                         train_num_steps,
                                         amp,
                                         save_model,
                                         save_every,
                                         loss_track_every,
                                         results_folder,
                                         var_coeff,
                                         cov_coeff)

        self.dim = generator.sig_dim
        self.level = generator.sig_level
        self.aug_dim = generator.aug_dim
        self.input_dim = generator.input_dim
        self.logsig_length = generator.logsig_length
        self.digit = digit
        self.ds = torch.load(data_path + '/filtered_digit_{}_coordinatewise_normalized_lead_lag_signature_{}.pt' \
                             .format(self.digit, self.level)).to(device)
        self.path_lengths = torch.load(data_path + '/filtered_digit_{}_lengths.pt'.format(self.digit)).to(device)
        self.sig_w1_loss = Utils.SigW1Metric(self.aug_dim, self.level, self.ds, normalise=True)


    def train_step(self):
        noise = torch.randn([self.ds.shape[0], self.input_dim]).to(self.device)
        x_fake = self.model(noise, self.path_lengths)
        loss_mean, loss_var, loss_cov = self.sig_w1_loss(x_fake)
        return loss_mean, loss_var, loss_cov





class Pendigit_reconstructor_trainer(Reconstructor_trainer):
    def __init__(
            self,
            reconstructor,
            data_path,
            digit,
            device,
            train_batch_size=64,
            train_lr=1e-3,
            train_lr_gamma=0.95,
            train_lr_step_size=1280,
            train_num_steps=10000,
            amp=False,
            save_model=True,
            save_every=1000,
            loss_track_every=100,
            results_folder='./results',
            L1_coeff=0.001
    ):
        super(Pendigit_reconstructor_trainer).__init__(reconstructor,
                                                       device,
                                                       train_batch_size,
                                                       train_lr,
                                                       train_lr_gamma,
                                                       train_lr_step_size,
                                                       train_num_steps,
                                                       amp,
                                                       save_model,
                                                       save_every,
                                                       loss_track_every,
                                                       results_folder,
                                                       L1_coeff)

        self.dim = self.model.sig_dim
        self.level = self.model.sig_level
        self.aug_dim = self.model.path_length
        self.path_length = self.model.path_length
        self.logsig_length = self.model.logsig_length
        self.data_path = data_path
        self.digit = digit

        features = torch.load(self.data_path + '/filtered_digit_{}_lead_lag_signature_{}.pt'.format(self.digit, self.level))
        paths = torch.load(self.data_path + '/filtered_digit_{}.pt'.format(digit))  # [sample_size, length, channel_dim]
        lengths = torch.load(self.data_path + '/filtered_digit_{}_lengths.pt'.format(digit))
        # We calculate the increment between points, this will be used in the computation of loss function
        increments = (paths[:, :, :].roll(-1, 1) - paths[:, :, :])
        labels = increments

        # Convert to Dataset format:
        training_set = Utils.Dataset(features,
                                     labels,
                                     lengths,
                                     train=True,
                                     device=device
                                     )

        test_set = Utils.Dataset(features,
                                 labels,
                                 lengths,
                                 train=False,
                                 device=device
                                 )

        self.training_dataloader = Utils.cycle(data.DataLoader(training_set, batch_size=train_batch_size, shuffle=True, drop_last=True))
        self.test_dataloader = Utils.cycle(data.DataLoader(test_set, batch_size=train_batch_size, shuffle=True, drop_last=True))

    def train_step(self):
        signature, increment, path_length = next(self.training_dataloader)
        pred_path = self.model(signature[:, 1:], path_length)
        # Apply a masking to it
        mask = Utils.masking(torch.zeros(pred_path.shape), path_length).to(self.device)
        pred_path = mask * pred_path
        # pred_path = Augmentations.lead_lag_transform(pred_path)
        # path_loss = torch.norm(path - pred_path)
        pred_increment = (pred_path[:, :, :].roll(-1, 1) - pred_path[:, :, :])
        param_norm_loss = Utils.param_norm(self.model)
        path_loss = torch.norm(increment - pred_increment, dim=[1, 2]).mean()
        return path_loss, param_norm_loss

    def eval_step(self):
        signature, increment, path_length = next(self.test_dataloader)
        pred_path = self.model(signature[:, 1:], path_length)
        # Apply a masking to it
        mask = Utils.masking(torch.zeros(pred_path.shape), path_length).to(self.device)
        pred_path = mask * pred_path
        # pred_path = Augmentations.lead_lag_transform(pred_path)
        # path_loss = torch.norm(path - pred_path)
        pred_increment = (pred_path[:, :, :].roll(-1, 1) - pred_path[:, :, :])
        param_norm_loss = Utils.param_norm(self.model)
        path_loss = torch.norm(increment - pred_increment, dim=[1, 2]).mean()
        return path_loss, param_norm_loss



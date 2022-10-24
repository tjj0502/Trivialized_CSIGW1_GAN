import torch
from torch.utils import data
from tqdm.auto import tqdm
from torch.cuda.amp import autocast, GradScaler
from torch import optim, nn
from pathlib import Path
from torch.optim import Adam

import copy
# import Utilities, Parametrization, Dataset, Utils
import time


class Base_trainer():
    """
    This is the base trainer for both the generator and reconstructor
    """

    def __init__(
            self,
            model,
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
            results_folder='./results'
    ):
        super().__init__()

        #         self.width = diffusion_model.width
        self.device = device
        self.model = model.to(device)
        #         self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)

        #         self.step_start_ema = step_start_ema
        self.save_every = save_every
        self.save_model = save_model

        self.batch_size = train_batch_size

        self.train_num_steps = train_num_steps

        self.opt = Adam(self.model.parameters(), lr=train_lr)
        self.train_lr_gamma = train_lr_gamma
        self.train_lr_step_size = train_lr_step_size
        self.scheduler = optim.lr_scheduler.StepLR(optimizer=self.opt, gamma=self.train_lr_gamma, step_size=self.train_lr_step_size)

        self.step = 0

        self.amp = amp
        self.scaler = GradScaler(enabled=amp)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)

        # Track the loss at some time points for analysis
        self.loss_track_every = loss_track_every
        self.loss_tracker = torch.zeros(self.train_num_steps // self.loss_track_every, 4)

        # Track the best model
        self.best_step = 0
        self.best_loss = None
        self.best_loss_model = self.model.state_dict()

    def save(self, milestone=1, model_type='generator'):
        data = {
            'step': self.best_step,
            'model': self.best_loss_model,
            #             'ema': self.ema.state_dict(),
            'scaler': self.scaler.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'loss': self.loss_tracker,
            'train_lr_gamma': self.train_lr_gamma,
            'train_lr_step_size': self.train_lr_step_size
        }
        torch.save(data, str(self.results_folder / f'{model_type}-model-{milestone}.pt'))

    def load(self, milestone=1, model_type='generator'):
        data = torch.load(str(self.results_folder / f'{model_type}-model-{milestone}.pt'))

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        #         self.ema.load_state_dict(data['ema'])
        self.scaler.load_state_dict(data['scaler'])
        self.scheduler.load_state_dict(data['scheduler'])
        self.loss_tracker = data['loss']
        self.train_lr_gamma = data['train_lr_gamma']
        self.train_lr_step_size = data['train_lr_step_size']

    def evaluate(self, conditional_variable):
        """
        In evaluation, we first generate fake signatures give the conditionwe compute all test metrics
        """
        self.load()
        self.model.eval()
        noise = torch.randn(conditional_variable.shape[0], self.input_dim).to(self.device)
        with torch.no_grad():
            normalized_sig_fake = self.model(noise, conditional_variable).detach().cpu()

    def train_step(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError


class Generator_trainer(Base_trainer):
    """
    We specify the training procedure here
    """

    def __init__(
            self,
            generator,
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
        super(Generator_trainer).__init__(generator,
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
                                          results_folder)

        # Penalty terms
        self.cov_coeff = cov_coeff
        self.var_coeff = var_coeff

    def train(self):
        with tqdm(initial=self.step, total=self.train_num_steps) as pbar:

            while self.step < self.train_num_steps:

                self.model.train()

                with autocast(enabled=self.amp):

                    loss_mean, loss_var, loss_cov = self.train_step()
                    self.scaler.scale(loss_mean + self.var_coeff * loss_var + self.cov_coeff * loss_cov).backward()

                if self.best_loss is None or (loss_mean) < self.best_loss:
                    self.best_loss_model = copy.deepcopy(self.model.state_dict())
                    self.best_loss = loss_mean.clone()
                    self.best_step = self.step
                    print('updated: ', self.step, self.best_loss)

                pbar.set_description(f'loss: {loss_mean.item():.4f}, {loss_var.item():.4f}, {loss_cov.item():.4f}')

                self.scaler.step(self.opt)
                self.scheduler.step()
                self.scaler.update()
                self.opt.zero_grad()

                if self.save_model and self.step != 0 and self.step % self.save_every == 0:
                    with torch.no_grad():
                        milestone = self.step // self.save_every
                    self.save(milestone)

                if self.step != 0 and self.step % self.loss_track_every == 0:
                    with torch.no_grad():
                        milestone_ = self.step // self.loss_track_every
                        self.loss_tracker[milestone_ - 1, 0] = self.best_loss
                        self.loss_tracker[milestone_ - 1, 1] = loss_mean
                        self.loss_tracker[milestone_ - 1, 2] = loss_var
                        self.loss_tracker[milestone_ - 1, 3] = loss_cov
                self.step += 1
                torch.cuda.empty_cache()
                pbar.update(1)

        self.save(1)
        print('training complete')


class Reconstructor_trainer(Base_trainer):
    """
    We specify the training procedure here
    """

    def __init__(
            self,
            reconstructor,
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
            L1_coeff = 0.001
    ):
        super(Reconstructor_trainer).__init__(reconstructor,
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
                                              results_folder)

        self.L1_coeff = L1_coeff

    def eval_step(self):
        raise NotImplementedError

    def train(self):
        with tqdm(initial=self.step, total=self.train_num_steps) as pbar:

            while self.step < self.train_num_steps:

                self.model.train()

                with autocast(enabled=self.amp):

                    path_loss, param_norm_loss = self.train_step()
                    # signature, increment, path_length = next(self.training_dataloader)
                    # pred_path = self.model(signature[:, 1:], path_length)
                    # #                     print(path.device, pred_path.device)
                    #
                    # # Apply a masking to it
                    # mask = masking(torch.zeros(pred_path.shape), path_length).to(self.device)
                    # pred_path = mask * pred_path
                    # # pred_path = Augmentations.lead_lag_transform(pred_path)
                    # # path_loss = torch.norm(path - pred_path)
                    # pred_increment = (pred_path[:, :, :].roll(-1, 1) - pred_path[:, :, :])
                    #
                    # param_norm_loss = param_norm(self.model)
                    # path_loss = torch.norm(increment - pred_increment, dim=[1, 2]).mean()
                    # current_loss = path_loss

                    if self.best_loss is None or path_loss < self.best_loss:
                        # Save the best performing model
                        print('updated: {}'.format(self.step))
                        self.best_loss_model = self.model.state_dict()
                        self.best_loss = path_loss.clone()

                    self.scaler.scale(path_loss + self.L1_coeff * param_norm_loss).backward()

                pbar.set_description(f'path_loss: {path_loss.item():.4f}, param_norm_loss: {param_norm_loss.item():.4f}, ')
                self.scaler.step(self.opt)
                self.scheduler.step()
                self.scaler.update()
                self.opt.zero_grad()

                if self.save_model and self.step != 0 and self.step % self.save_every == 0:
                    #                     self.ema.ema_model.eval()
                    with torch.no_grad():
                        milestone = self.step // self.save_every
                    self.save(milestone, 'reconstructor')

                if self.step != 0 and self.step % self.loss_track_every == 0:
                    self.model.eval()
                    with torch.no_grad():
                        test_loss, _ = self.eval_step()
                        milestone_ = self.step // self.loss_track_every
                        # test_signature, test_increment, test_path_length = next(self.test_dataloader)
                        # pred_path = self.model(test_signature[:, 1:], test_path_length)
                        # mask = masking(torch.zeros(pred_path.shape), test_path_length).to(self.device)
                        # pred_path = mask * pred_path
                        # # pred_path = Augmentations.lead_lag_transform(pred_path)
                        # pred_increment = (pred_path[:, :, :].roll(-1, 1) - pred_path[:, :, :])
                        # test_loss = torch.norm(test_increment - pred_increment)
                        # pred_sig = signatory.signature(Augmentations.lead_lag_transform(pred_path), self.level)
                        # test_sig_loss = torch.norm(test_signature[:,1:] - pred_sig)
                        #                         loss = self.model(self.loss_tracker_data)
                        self.loss_tracker[milestone_ - 1, 0] = path_loss
                        self.loss_tracker[milestone_ - 1, 1] = self.best_loss
                        self.loss_tracker[milestone_ - 1, 2] = param_norm_loss
                        #                         self.loss_tracker[milestone_ - 1, 1] = sig_loss
                        self.loss_tracker[milestone_ - 1, 3] = test_loss
                #                         self.loss_tracker[milestone_ - 1, 3] = test_sig_loss
                self.step += 1
                torch.cuda.empty_cache()
                pbar.update(1)

        self.save(1, 'reconstructor')
        print('training complete')

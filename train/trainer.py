from pendigit import Pendigit_generator_trainer, Pendigit_reconstructor_trainer
from poisson import Poisson_generator_trainer
from model.generator.logsig_generator import Poisson_Conditional_Logsig_Generator, Pendigit_Conditional_Logsig_Generator
from model.reconstructor.path_reconstructor import Poisson_Reconstructor, Pendigit_Reconstructor
import torch

GENERATORS = {'poisson': Poisson_Conditional_Logsig_Generator,
              'pendigit': Pendigit_Conditional_Logsig_Generator
              }

RECONSTRUCTOR = {'poisson': Poisson_Reconstructor,
                 'pendigit': Pendigit_Reconstructor
                 }


def get_generator_trainer(config):
    model_name = config.dataset

    generator = GENERATORS[config.generator](input_dim=config.G_input_dim,
                                             hidden_dim=config.G_hidden_dim,
                                             conditional_dim=config.conditional_dim,
                                             sig_dim=config.sig_dim,
                                             sig_level=config.sig_level,
                                             device=config.device)

    trainer = {
        "poisson": Poisson_generator_trainer(generator=generator,
                                             data_path=config.data_path,
                                             device=config.device,
                                             train_batch_size=config.G_train_batch_size,
                                             train_lr=config.G_train_lr,
                                             train_lr_gamma=config.G_train_lr_gamma,
                                             train_lr_step_size=config.G_train_lr_step_size,
                                             train_num_steps=config.G_train_num_steps,
                                             save_model=config.G_save_model,
                                             save_every=config.G_save_every,
                                             loss_track_every=config.G_loss_track_every,
                                             results_folder=config.G_results_folder,
                                             var_coeff=config.G_var_coeff,
                                             cov_coeff=config.G_corr_coeff),
        "pendigit": Pendigit_generator_trainer(generator=generator,
                                               data_path=config.data_path,
                                               digit=config.digit,
                                               device=config.device,
                                               train_batch_size=config.G_train_batch_size,
                                               train_lr=config.G_train_lr,
                                               train_lr_gamma=config.G_train_lr_gamma,
                                               train_lr_step_size=config.G_train_lr_step_size,
                                               train_num_steps=config.G_train_num_steps,
                                               save_model=config.G_save_model,
                                               save_every=config.G_save_and_every,
                                               loss_track_every=config.G_loss_track_every,
                                               results_folder=config.G_results_folder,
                                               var_coeff=config.G_var_coeff,
                                               cov_coeff=config.G_corr_coeff)
    }[model_name]

    return trainer


def get_reconstructor_trainer(config):

    model_name = config.dataset
    if model_name == 'poisson':
        lengths = torch.load(config.data_path + '/random_walk_lengths.pt')
    elif model_name == 'pendigit':
        lengths = torch.load(config.data_path + '/filtered_digit_{}_lengths.pt'.format(config.digit))
    else:
        raise ValueError

    max_length = int(torch.max(lengths))
    config.update({"R_path_length": max_length}, allow_val_change=True)
    reconstructor = RECONSTRUCTOR[config.reconstructor](conditional_dim=config.conditional_dim,
                                                        sig_dim=config.sig_dim,
                                                        sig_level=config.sig_level,
                                                        path_length=config.R_path_length,
                                                        device=config.device)

    trainer = {
        "poisson": Poisson_reconstructor_trainer(generator=reconstructor,
                                                 data_path=config.data_path,
                                                 device=config.device,
                                                 train_batch_size=config.G_train_batch_size,
                                                 train_lr=config.G_train_lr,
                                                 train_lr_gamma=config.G_train_lr_gamma,
                                                 train_lr_step_size=config.G_train_lr_step_size,
                                                 train_num_steps=config.G_train_num_steps,
                                                 save_model=config.G_save_model,
                                                 save_every=config.G_save_every,
                                                 loss_track_every=config.G_loss_track_every,
                                                 results_folder=config.G_results_folder,
                                                 var_coeff=config.G_var_coeff,
                                                 cov_coeff=config.G_corr_coeff),
        "pendigit": Pendigit_reconstructor_trainer(reconstructor=reconstructor,
                                                   data_path=config.data_path,
                                                   digit=config.digit,
                                                   device=config.device,
                                                   train_batch_size=config.R_train_batch_size,
                                                   train_lr=config.R_train_lr,
                                                   train_lr_gamma=config.R_train_lr_gamma,
                                                   train_lr_step_size=config.R_train_lr_step_size,
                                                   train_num_steps=config.R_train_num_steps,
                                                   save_model=config.R_save_model,
                                                   save_every=config.R_save_and_every,
                                                   loss_track_every=config.R_loss_track_every,
                                                   results_folder=config.R_results_folder,
                                                   L1_coeff=config.R_L1_coeff)
    }[model_name]

    return trainer

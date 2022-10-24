import copy
import wandb
import yaml
import ml_collections
import os
from os import path as pt
import torch
from torch import nn
from train.trainer import get_generator_trainer



def train_generator(config):
    trainer = get_generator_trainer(config)
    trainer.train()

def train_reconstructor(config):
    trainer = get_generator_trainer(config)
    trainer.train()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    config_dir = 'configs/' + 'train_pendigit.yaml'
    with open(config_dir) as file:
        config = ml_collections.ConfigDict(yaml.safe_load(file))
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    print(os.environ["CUDA_VISIBLE_DEVICES"])
    if (config.device ==
            "cuda" and torch.cuda.is_available()):
        config.update({"device": "cuda:0"}, allow_val_change=True)
    else:
        config.update({"device": "cpu"}, allow_val_change=True)

    # Set seed
    torch.manual_seed(config.seed)

    train_generator(config)



# See PyCharm help at https://www.jetbrains.com/help/pycharm/

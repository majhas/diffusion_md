import torch.nn as nn

from pathlib import Path

from .config_dict import ConfigDict


config = ConfigDict(**{})


##############################################################################################################


#                                               Data Config


##############################################################################################################

data_config = ConfigDict(**{})
data_config.data_dir = Path('data/ala2/')
data_config.traj_filepath = data_config.data_dir.joinpath('ala2_cg_data.npz')
data_config.pdb_filepath = data_config.data_dir.joinpath('ala2_cg.pdb')

data_config.batch_size = 256

config.data_config = data_config
##############################################################################################################


#                                               Model Config


##############################################################################################################

model_config = ConfigDict(**{})
model_config.in_node_nf = 0
model_config.n_dim = 3
model_config.timesteps = 1000
model_config.noise_schedule = 'polynomial3'

# # Encoder layers for the input node features
# model_config.n_encoder_layers = 1
# model_config.encoder_mlp_size = 2

# # Processor layers for the graph network to process
# model_config.n_processor_layers = 2
# model_config.processor_mlp_size = 2

config.model_config = model_config


config.dynamics_config = ConfigDict(**{})

##############################################################################################################


#                                               Trainer Config


##############################################################################################################

# Main Trainer parameters
trainer_config = {
                'n_steps': int(200_000),
                'checkpoint_path': 'checkpoints/',
                'model_filename': 'model.pt',
                'ema_decay': 0.995,
                'early_stopping_patience': 20
            }

config.trainer_config = ConfigDict(**trainer_config)

# Optimizer hyperparameters for Adam 
optimizer_config = {
                'lr': 1e-4,
                'weight_decay': 1e-12,
                'betas': (0.9, 0.99)
            }

config.optimizer_config = ConfigDict(**optimizer_config)

# Learning rate scheduler for OneCycleLR with cosine annealling schedule
lr_scheduler_config = {
                'max_lr': config.optimizer_config.lr,
                'total_steps': config.trainer_config.n_steps,
                'pct_start': 0.2
                }

config.scheduler_config = ConfigDict(**lr_scheduler_config)


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
model_config.n_nodes = 5
model_config.in_node_nf = 64
model_config.in_edge_nf = 1
model_config.hidden_nf = 96
model_config.n_processor_layers = 2
model_config.rel_pos_emb=True
model_config.norm_edges=True
model_config.with_feedforwards=True
model_config.gated_residual=True

diffusion_config = ConfigDict(**{})

diffusion_config.n_dim = 3
diffusion_config.timesteps = 1000
diffusion_config.noise_schedule = 'cosine'

config.model_config = model_config
config.diffusion_config = diffusion_config

config.lr = 1.e-4
config.weight_decay = 1e-12
config.n_steps = 100_000
config.batch_size = 512
config.device = 'cuda'
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


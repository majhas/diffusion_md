import argparse
import importlib

import torch
import numpy as np

from copy import copy
from tqdm import trange
from torch.utils.data import DataLoader

from src.data_modules import read_trajectories, TrajectoryLoader
from src.models import DDPM, GTDynamics, apply_random_rotation


def parse_args():
    parser = argparse.ArgumentParser('Train diffusion model on MD simulation')
    parser.add_argument('--config', help='filepath to config file')

    return parser.parse_args()



def train(
        ddpm, 
        dataloader, 
        n_steps,
        lr=1e-4, 
        weight_decay=0.,
        betas=(0.9, 0.99),
        device='cuda'
    ):

    ddpm = ddpm.float()
    ddpm.to(device)
    optimizer = torch.optim.Adam(
                            ddpm.parameters(),
                            lr=lr, 
                            weight_decay=weight_decay,
                            betas=betas
                        )
    
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=n_steps, max_lr=lr, pct_start=0.2)

    dataloader_iterator = iter(dataloader)
    pbar = trange(n_steps)

    
    for step in pbar:

        pos_batch = next(dataloader_iterator, None)
        if pos_batch is None:
            dataloader_iterator = iter(dataloader_iterator)
            pos_batch = next(dataloader_iterator)

        x = torch.broadcast_to(torch.arange(pos_batch.size(1)).view(1, -1, 1), size=(pos_batch.size(0), pos_batch.size(1), 1))
        

        pos_batch = apply_random_rotation(pos_batch)
        x = x.to(device)
        pos_batch = pos_batch.to(device)

        loss = ddpm.get_loss(x=x, pos=pos_batch)
        loss.backward()

        optimizer.step()
        lr_scheduler.step()

        pbar.set_description(f'train_loss: {loss:6f}')




def main(config):
    
    trajectories = read_trajectories(
                            traj_filepath=config.data_config.traj_filepath, 
                            pdb_filepath=config.data_config.pdb_filepath)


    # For development purposes, manually split
    # TODO cross-validation or read train/test splits from files
    # 1_000_000 trajectories comprised of 4 runs of 250k steps

    train_trajectories = trajectories[:750_000]
    val_trajectories = trajectories[750_000:]

    train_loader = TrajectoryLoader(dataset=train_trajectories)
    train_loader = DataLoader(train_loader, batch_size=config.batch_size, shuffle=True)

    dynamics = GTDynamics(**vars(config.model_config))

    ddpm = DDPM(
                dynamics=dynamics,
                **vars(config.diffusion_config)
            )

    train(
        ddpm=ddpm, 
        dataloader=train_loader, 
        n_steps=config.n_steps,
        lr=config.lr, 
        weight_decay=config.get('weight_decay', 0.),
        betas=config.get('betas', (0.9, 0.99)),
        device=config.device
    )

if __name__ == '__main__':
    args = parse_args()
    config = copy(importlib.import_module(args.config).config)

    main(config)

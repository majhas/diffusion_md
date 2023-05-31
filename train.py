import argparse
import importlib

import numpy as np

from copy import copy

from src.data_modules import read_trajectories, SimulationLoader

from torch.utils.data import DataLoader

def parse_args():
    parser = argparse.ArgumentParser('Train diffusion model on MD simulation')
    parser.add_argument('--config', help='filepath to config file')

    return parser.parse_args()


def main(config):
    

    trajectories = read_trajectories(
                            traj_filepath=config.data_config.traj_filepath, 
                            pdb_filepath=config.data_config.pdb_filepath)


    trajectories = np.asarray(trajectories)

    loader = SimulationLoader(dataset=trajectories)
    dataloader = DataLoader(loader, batch_size=512, shuffle=True)

    dataloader_iterator = iter(dataloader)
    print(next(dataloader_iterator).size())

if __name__ == '__main__':
    args = parse_args()
    config = copy(importlib.import_module(args.config).config)

    main(config)

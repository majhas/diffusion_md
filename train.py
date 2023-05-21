import argparse
import importlib

from copy import copy

def parse_args():
    parser = argparse.ArgumentParser('Train diffusion model on MD simulation')
    parser.add_argument('--config', help='filepath to config file')

    return parser.parse_args()


def main(config):
    

    print(config)



if __name__ == '__main__':
    args = parse_args()
    config = copy(importlib.import_module(args.config).config)

    main(config)

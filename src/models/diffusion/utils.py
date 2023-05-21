# code from: https://github.com/ehoogeboom/e3_diffusion_for_molecules/blob/main/equivariant_diffusion/utils.py

import torch


def remove_mean(x):
    mean = torch.mean(x, dim=1, keepdim=True)
    return x - mean



def remove_mean_with_mask(x, node_mask):

    N = node_mask.sum(dim=1, keepdim=True)
    mean = torch.sum(x, dim=1, keepdim=True) / N

    return x - mean*node_mask



def sample_center_gravity_zero_gaussian_with_mask(size, node_mask, device='cpu'):
    x = sample_gaussian(size, device=device)

    x_masked = x*node_mask
    return remove_mean_with_mask(x_masked, node_mask)



def sample_gaussian_with_mask(size, node_mask, device='cpu'):
    x = sample_gaussian(size, device=device)

    x_masked = x * node_mask
    return remove_mean_with_mask(x_masked, node_mask)



def sample_gaussian(size, device='cpu'):
    return torch.randn(size, device=device)



def sample_center_gravity_zero_gaussian(size, device='cpu'):

    x = sample_gaussian(size, device=device)

    return remove_mean(x)



# code from: https://github.com/ehoogeboom/e3_diffusion_for_molecules/blob/main/equivariant_diffusion/utils.py

import math
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


def apply_random_rotation(x):

    alpha_degree = torch.randint(low=0, high=360, size=(x.size(0), 1, 1))
    beta_degree = torch.randint(low=0, high=360, size=(x.size(0), 1, 1))
    gamma_degree = torch.randint(low=0, high=360, size=(x.size(0), 1, 1))
    
    alpha = alpha_degree * math.pi / 180
    beta = beta_degree * math.pi / 180
    gamma = gamma_degree * math.pi / 180

    cos_alpha = torch.cos(alpha)
    sin_alpha = torch.sin(alpha)

    cos_beta = torch.cos(beta)
    sin_beta = torch.sin(beta)

    cos_gamma = torch.cos(gamma)
    sin_gamma = torch.sin(gamma)

    zeros = torch.zeros(cos_alpha.shape)
    ones = torch.ones(cos_alpha.shape)

    # print(cos_gamma.shape)
    # print(torch.concat([cos_gamma, -sin_gamma, zeros], dim=-1).shape)
    yaw = torch.concat([
                        torch.concat([cos_gamma, -sin_gamma, zeros], dim=2),
                        torch.concat([sin_gamma, cos_gamma, zeros], dim=2),
                        torch.concat([zeros, zeros, ones], dim=2)
                    ],
                    dim=1
                ).to(x.device)

    pitch = torch.concat([
                        torch.concat([cos_beta, zeros, sin_beta], dim=2),
                        torch.concat([zeros, ones, zeros], dim=2),
                        torch.concat([-sin_beta, zeros, cos_beta], dim=2)
                    ],
                    dim=1
                ).to(x.device)

    
    roll = torch.concat([
                        torch.concat([ones, zeros, zeros], dim=2),
                        torch.concat([zeros, cos_alpha, -sin_alpha], dim=2),
                        torch.concat([zeros, sin_alpha, cos_alpha], dim=2)
                    ],
                    dim=1
                ).to(x.device)

    R = torch.bmm(yaw, pitch)
    R = torch.bmm(R, roll)

    return torch.bmm(x, R)


if __name__ == '__main__':

    x= torch.randn((128, 5, 3))
    rot_x = apply_random_rotation(x)
    print(rot_x.shape)
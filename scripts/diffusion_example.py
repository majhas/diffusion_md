import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from typing import Tuple
from tqdm import trange
from functools import partial

def clip_noise_schedule(alphas, clip_value=0.001):
    alphas = torch.concat([torch.ones(1), alphas], axis=0)
    alphas_step = (alphas[1:]/alphas[:-1])
    alphas_step = torch.clip(alphas_step, min=clip_value, max=1.)

    alphas = torch.cumprod(alphas_step, axis=0)
    return alphas

def polynomial_beta_schedule(timesteps, s=1e-4, power=3):

    steps = timesteps + 1
    x = torch.linspace(0, steps, steps)
    alphas = (1 - torch.pow(x /steps, power))**2
    alphas = clip_noise_schedule(alphas, clip_value=0.001)
    precision = 1 - 2*s
    alphas = precision*alphas + s
    return alphas

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

    alphas_cumprod = clip_noise_schedule(alphas_cumprod)

    return alphas_cumprod


def inflate_batch_array(array, target):
    target_shape = (array.size(0),) + (1,)*(len(target.size()) - 1)
    return array.view(target_shape)
    
def get_sigma(gamma):
    return torch.sqrt(torch.sigmoid(gamma))

def get_alpha(gamma):
    return torch.sqrt(torch.sigmoid(-gamma))

def SNR(gamma):
    return torch.exp(-gamma)

def sample_gaussian(n_samples: int):
    
    return torch.randn((n_samples, 1))


def log_px_given_z0(model, x, gamma):
    t_zeros = torch.zeros(x.size(0), 1).long().to(x.device)
    gamma_0 = inflate_batch_array(gamma[t_zeros], x)
    alpha_0 = get_alpha(gamma_0)
    sigma_0 = get_sigma(gamma_0)

    # Sample z_0 given x, h for timestep t, from q(z_t | x, h)
    eps_0 = sample_gaussian(n_samples=x.size(0)).to(x.device)

    z_0 = alpha_0 * x + sigma_0 * eps_0

    out = model(z_0, t_zeros)
    return (0.5*(eps_0 - out)**2).mean(0)



def get_loss(model, x, gamma, total_timesteps):

    # sample timesteps for each input
    t = torch.randint(1, total_timesteps+1, size=(x.size(0), 1), device=x.device).float()
    s = t - 1

    t_is_zero = (t == 0).float()

    # compute gamma_s and gamma_t 
    gamma_t = gamma[t.long()]
    gamma_s = gamma[s.long()]
    
    # normalize timesteps
    t /= total_timesteps
    s /= total_timesteps


    alpha_t = get_alpha(gamma_t)
    sigma_t = get_sigma(gamma_t)

    alpha_s = get_alpha(gamma_s)
    sigma_s = get_sigma(gamma_s)
    
    # sample the noise
    eps = sample_gaussian(n_samples=x.size(0)).to(x.device)
                          
    z_t = alpha_t*x + sigma_t*eps
    out = model(x=z_t, t=t)

    loss = (0.5*(eps-out)**2).mean(dim=0)
    loss_at_zero = log_px_given_z0(model=model, x=x, gamma=gamma)

    return loss + loss_at_zero

def get_terms_from_noise_schedule(alphas):
    return torch.sqrt(alphas), torch.sqrt(1-alphas)


def train(model, dataloader, noise_schedule, timesteps=1000, lr=1e-4, n_steps=int(10_000), device='cuda'):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    alphas = noise_schedule(timesteps=1000)

    sigmas = 1 - alphas
    log_alphas = torch.log(alphas)
    log_sigmas = torch.log(sigmas)

    log_alphas_to_sigmas = (log_alphas - log_sigmas).float()
    gamma = - log_alphas_to_sigmas    

    gamma = gamma.to(device)

    pbar = trange(n_steps)
    dataloader_iterator = iter(dataloader)
    for i in pbar:

        batch = next(dataloader_iterator, None)
        if batch is None:
            dataloader_iterator = iter(dataloader)
            batch = next(dataloader_iterator)

        optimizer.zero_grad()

        batch = batch.to(device)
        loss = get_loss(model, batch, gamma=gamma, total_timesteps=1000)
        loss.backward()
        optimizer.step()

        if i % 500 == 0:
            pbar.set_description(f'Loss: {loss.item()}')


    torch.save(obj={'state_dict': model.state_dict()}, f='checkpoints/model.pt')



class MLP(torch.nn.Module):
    def __init__(self, nf):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(nn.Linear(2, nf), nn.Linear(nf, nf), nn.Linear(nf, 1))

    def forward(self, x, t):
        return self.layers(torch.concat([x, t], dim=1))



def main():
    distribution1 = torch.randn((50000,1)) - 1
    distribution2 = torch.randn((50000,1)) + 1

    x = torch.concat([distribution1, distribution2])

    train_index = torch.randint(high=100000, size=(90000,))
    test_index = torch.tensor([i for i in range(x.size(0)) if i not in train_index], dtype=torch.long)

    train_dataset = x[train_index]
    test_dataset = x[test_index]

    train_dataloader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=512, shuffle=False)
    noise_schedule = partial(polynomial_beta_schedule, power=3)

    model = MLP(nf=32)
    train(model=model, noise_schedule=noise_schedule, dataloader=train_dataloader, lr=1.e-4, n_steps=500000)

    
if __name__ == '__main__':
    main()
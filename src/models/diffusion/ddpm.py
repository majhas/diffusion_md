# code from https://github.com/ehoogeboom/e3_diffusion_for_molecules/blob/main/equivariant_diffusion/en_diffusion.py

import math

import torch
import torch.nn as nn

from typing import Tuple

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


class DDPM(nn.Module):
    def __init__(
            self,
            dynamics, 
            n_dim: int =3,
            timesteps: int =1000, 
            noise_schedule: str ='cosine',
            noise_precision: float =1e-4,
            loss_type = 'mse'
        ):
        super(DDPM, self).__init__()
        self.loss_type = loss_type
        self.dynamics = dynamics

        self.n_dim = n_dim               # dimension size for node positions

        self.T = timesteps

        if self.loss_type == 'mse':
            self.loss_f = torch.nn.functional.mse_loss
        else:
            NotImplemented

        if noise_schedule == 'learned':
            NotImplemented
        elif noise_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.nn.functional.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        one_minus_alphas_cumprod = 1. - alphas_cumprod

        self.register_buffer('alphas', alphas)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', sqrt_alphas_cumprod)
        self.register_buffer('one_minus_alphas_cumprod', one_minus_alphas_cumprod)

        sqrt_alphas_cumprod_recip = 1. / sqrt_alphas_cumprod
        reverse_eps_coef = betas * (torch.sqrt(1. - alphas_cumprod)) / (1. - alphas_cumprod)
        sqrt_variance = torch.sqrt(betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod))

        self.register_buffer('sqrt_alphas_cumprod_recip', sqrt_alphas_cumprod_recip)
        self.register_buffer('reverse_eps_coef', reverse_eps_coef)
        self.register_buffer('sqrt_variance', sqrt_variance)




    def forward(self, x, pos, t):
        return self.dynamics(x=x, pos=pos, t=t)

    
    def forward_process(self, x, t):
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].unsqueeze(-1)
        one_minus_alphas_cumprod_t = self.one_minus_alphas_cumprod[t].unsqueeze(-1)

        eps = torch.randn(x.shape).to(x.device)

        z_t = sqrt_alphas_cumprod_t*x + one_minus_alphas_cumprod_t*eps
        return z_t, eps


    def reverse_process(self, z_t, timesteps=None):

        if timesteps is None:
            timesteps = self.T

        reversed_timesteps = torch.tensor(reversed(list(range(timesteps+1))), dtype=torch.long)
        for t in reversed_timesteps:

            sqrt_alphas_cumprod_recip_t = self.sqrt_alphas_cumprod_recip[t]         
            reverse_eps_coef_t = self.reverse_eps_coef[t]     
            sqrt_variance_t = self.sqrt_variance.gather[t]            
            noise = torch.randn(1)

            z_t = (sqrt_alphas_cumprod_recip_t)*(z_t - reverse_eps_coef_t*eps) + sqrt_variance_t*noise

        return z_t    



    def get_loss(self, x, pos):

        t = torch.randint(0, self.T+1, size=(len(pos), 1), device=pos.device) 

        z_t, eps = self.forward_process(pos, t)

        predicted_eps = self.forward(x=x, pos=z_t, t=t/self.T)

        return self.loss_f(predicted_eps, eps)


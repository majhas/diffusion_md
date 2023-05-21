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


class PredefinedNoiseSchedule(nn.Module):
    def __init__(self, noise_schedule, timesteps, noise_precision=1e-4):

        if noise_schedule == 'cosine':
            alphas = cosine_beta_schedule(timesteps=timesteps, s=noise_precision)
        elif 'polynomial' in noise_schedule:
            power = noise_schedule.split('polynomial')[-1]
            power = float(power)
            alphas = polynomial_beta_schedule(timesteps=timesteps, power=power, s=noise_precision)

        sigmas = 1 - alphas
        log_alphas = torch.log(alphas)
        log_sigmas = torch.log(sigmas)

        log_alphas_to_sigmas = log_alphas - log_sigmas
        
        self.register_buffer('gamma', -log_alphas_to_sigmas.float())

    def forward(self, t):
        t = torch.round(t*self.timesteps).long()
        return self.gamma[t]


class En_Diffusion(nn.Module):
    def __init__(
            self,
            dynamics, 
            in_node_nf: int, 
            n_dim: int =3,
            timesteps: int =1000, 
            noise_schedule: str ='cosine',
            noise_precision: float =1e-4, 
            norm_values: Tuple[float, float, float] =(1., 1., 1.),
            norm_biases: Tuple[float, float, float] =(None, 0., 0.)
        ):
    
        self.loss_type = loss_type

        if noise_schedule == 'learned':
            NotImplemented
        else:
            self.gamma = PredefinedNoiseSchedule(noise_schedule, timesteps=timesteps, precision=noise_precision)

        self.dynamics = dynamics

        self.in_node_nf = in_node_nf     # dimension size for node input features
        self.n_dim = n_dim               # dimension size for node positions

        self.T = timesteps
        self.norm_value = norm_values
        self.norm_biases = norm_biases

        self.register_buffer('buffer', torch.zeros(1))


    def phi(self, x, t, node_mask, edge_mask, context):
        return self.dynamics(t, x, node_mask,edge_mask, context)

    def inflate_batch_array(self, array, target):
        target_shape = (array.size(0),) + (1,)*(len(target.size()) - 1)
        return array.view(target_shape)
        
    def get_sigma(self, gamma):
        return torch.sqrt(torch.sigmoid(gamma))

    def get_alpha(self, gamma):
        return torch.sqrt(torch.sigmoid(-gamma))

    def SNR(self, gamma):
        return torch.exp(-gamma)

    def sample_combined_position_feature_noise(self, n_samples: int, n_nodes: int, node_mask: torch.Tensor):
        
        z_x = utils.sample_center_gravity_zero_gaussian_with_mask(
                                                            size=(n_samples, n_nodes, self.n_dim),
                                                            node_mask=node_mask,
                                                            device=node_mask.device
                                                        )

        z_h = utils.sample_gaussian_with_mask(
                                       size=(n_samples, n_nodes, self.in_node_nf),
                                       node_mask=node_mask,
                                       device=node_mask.device
                                    )

        return torch.cat([z_x, z_h], dim=-1)


    def get_error(self, predicted_eps, eps):

        # the output dimension is of size node_num_features + x_size. Need to factor this in to the L2 Loss.
        # Result is node_num_features + x_size + num_nodes
        denom = self.in_node_nf + self.n_dim + predicted_eps.size(1)

        # L2 loss
        squared_diff = (eps - predicted_eps)**2
        sum_of_squares = squared_diff.view(predicted_eps.size(0), -1).sum(dim=-1)
        error = sum_of_squares / denom

        return error

    def log_pxh_given_z0_without_constants(self, x, z_t, gamma_0, eps, net_out, node_mask, epsilon=1e-10):
        
        # Current z_t, net_out, eps only contain information relevant to positions - no other features

        # Compute sigma_0 and rescale to the integer scale of the data.
        sigma_0 = self.inflate_batch_array(self.get_sigma(gamma_0), z_t)
        sigma_0_cat = sigma_0 * self.norm_values[1]
        sigma_0_int = sigma_0 * self.norm_values[2]

        # Computes the error for the distribution N(x | 1 / alpha_0 z_0 + sigma_0/alpha_0 eps_0, sigma_0 / alpha_0),
        # the weighting in the epsilon parametrization is exactly '1'.
        log_p_x_given_z_without_constants = -0.5 * self.compute_error(net_out, eps)


        # Select the log_prob of the current category usign the onehot
        # representation.
        log_ph_cat = sum_except_batch(log_probabilities * onehot * node_mask)

        # Combine categorical and integer log-probabilities.
        log_p_h_given_z = log_ph_integer + log_ph_cat

        # Combine log probabilities for x and h.
        log_p_xh_given_z = log_p_x_given_z_without_constants + log_p_h_given_z

        return log_p_xh_given_z


        
    def get_loss(self, x, node_mask, edge_mask):

        # sample timesteps for each input
        t = torch.randint(1, self.T+1, size=(x.size(0), 1), device=x.device).float()
        s = t - 1

        t_is_zero = (t == 0).float()

        # normalize timesteps
        t /= self.T
        s /= self.T

        # compute gamma_t 

        gamma_t = self.inflate_batch_array(self.gamma(t), x)

        alpha_t = self.get_alpha(gamma_t)
        sigma_t = self.get_sigma(gamma_t)


        # sample the noise
        eps = self.sample_combined_position_feature_noise(
                                                n_samples=x.size(0), 
                                                n_nodes=x.size(1), 
                                                node_mask=node_mask
                                            )

        z_t = alpha_t*x + sigma_t*eps

        out = self.phi(z_t, t, node_mask, edge_mask)

        error = self.get_error(out, eps)

        # found that weight of one performs better than weighting by
        # signal-to-noise ratio. 
        SNR_weight = torch.ones_like(error)

        # loss term for sampled timesteps greater than zero
        loss_t_greater_zero = (0.5 * SNR_weight * error).mean()

        # Get loss for t = 0
        t_zero = torch.zeros_like(size=t)
        gamma_0 = self.inflate_batch_array(self.gamma(t_zero), x)
        alpha_0 = self.get_alpha(gamma_0)
        sigma_0 = self.get_sigma(gamma_0)

        eps_0 = self.sample_combined_position_feature_noise(
                                                    n_samples=x.size(0), 
                                                    n_nodes=x.size(1), 
                                                    node_mask=node_mask
                                                )
        z_0 = alpha_0*x + sigma_0*eps_0

        out_t_zero = self.phi(z_0, t_zero, node_mask, edge_mask)
        loss_at_zero = 0.5 * SNR_weight* self.compute_error(predicted_eps=out_t_zero, eps=eps_0)

        return loss_t_greater_zero + loss_at_zero

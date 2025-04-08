import torch
import torch.nn as nn
from typing import Optional


class DDPMScheduler(nn.Module):
    def __init__(self,
                 total_timesteps: int = 1000, # diffusion training timesteps
                 beta_1: float = 1e-4,
                 beta_T: float = 0.02,
                 var_type: str ="lower_bound", # lower_bound or upper bound
    ) -> None:
        super().__init__()
        self.total_timesteps = total_timesteps

        assert var_type in ["lower_bound", "upper_bound"], f"var_type {var_type} not supported"

        # Timestep starts at 0 and is revesed for sampling. [total_timesteps, ]
        # Note timesteps are T-1, T-2, ..., 0, which differs from the original paper
        self.timesteps = torch.arange(self.total_timesteps - 1, -1, -1, dtype=torch.long)

        betas = torch.linspace(beta_1, beta_T, total_timesteps) # [total_timesteps,]

        alphas = 1 - betas

        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # add betas, alphas, alphas_cumprod to the persistent buffers of the module
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

        if var_type == "lower_bound":
            # lower bound variance
            # for t > 0 sigmas is valid. If t == 0, sigmas has no meaning
            alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]], dim=0)
            sigmas = ((1 - alphas_cumprod_prev) / ( 1- alphas_cumprod) * betas) ** 0.5
        else:
            # upper bound variance
            sigmas = torch.sqrt(betas)

        self.register_buffer("sigmas", sigmas)

    def denoise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        '''
        One step denoising given noise: x_t -> x_{t-1}
        :param x_t, noisy image, [batch_size, channels, height, width]
        :param t, the timestep: [batch_size,]
        :param noise, the noise to be removed: [batch_size, channels, height, width]
        :note the timestep starts at 0 and is reversed for sampling
        :timesteps = [total_timesteps - 1, total_timesteps - 2, ..., 0] differing from the original
        paper: https://arxiv.org/pdf/2006.11239
        '''
        assert torch.all(t >= 0) and torch.all(t < self.total_timesteps), f"timestep {t} out of range [0, {self.total_timesteps})"
        B = x_t.shape[0]

        x_t_1 =  (x_t - (1 - self.alphas[t].view(B, 1, 1, 1)) / \
                (torch.sqrt(1 - self.alphas_cumprod[t].view(B, 1, 1, 1))) * noise) / \
                (1 / torch.sqrt(self.alphas[t]).view(B, 1, 1, 1))

        z = torch.randn_like(x_t)
        mask = (t > 0).float().view(B, 1, 1, 1)
        x_t_1 += mask * self.sigmas[t].view(B, 1, 1, 1) * z
        return x_t_1

    def add_noise(self, x_0: torch.Tensor, t: torch.Tensor, epsilon: Optional[torch.Tensor] = None) -> torch.Tensor:
        '''
        A forward noising step: x_0 -> x_t
        :param x_0, the original image: [batch_size, channels, height, width]
        :param t, the timestep: [batch_size,]
        :param epsilon, the noise to be added: [batch_size, channels, height, width] Optionally
        '''
        assert torch.all(t >=0) and torch.all(t < self.total_timesteps), f"timestep {t} out of range [0, {self.total_timesteps})"
        assert t.dtype == torch.long, f"timestep {t} should be of type long"
        B= x_0.shape[0]

        if epsilon is None:
            epsilon = torch.randn_like(x_0)

        x_t = torch.sqrt(self.alphas_cumprod[t]).view(B, 1, 1, 1) * x_0 + \
                torch.sqrt(1 - self.alphas_cumprod[t]).view(B, 1, 1, 1) * epsilon

        return x_t

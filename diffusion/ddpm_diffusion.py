import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from models.unet import UNet
from diffusion.ddpm_scheduler import DDPMScheduler



class DDPM(nn.Module):
    def __init__(self, denoise_net: UNet, var_scheduler: DDPMScheduler) -> None:
        super().__init__()
        self.denoise_net = denoise_net
        self.var_scheduler = var_scheduler

    def compute_loss(self, x0: torch.tensor):
        '''
        :param  x0 the orignal clean image (batch_size, num_channels, height, width) [B, 3, H, W]
        :return: loss
        :note Different from the DDPM paper, our timesteps are [T-1, T-2, ..., 0]
        '''
        B, C, _, _ = x0.shape
        assert C == 3, f"Only support RGB images, but got {C} channels"

        # 1. sample a batch of random timestep t from [0, T-1]
        t = torch.randint(0, self.var_scheduler.total_timesteps, (B,), device=x0.device).long()

        # 2. sample a batch of random noise
        epsilon = torch.randn_like(x0)

        # 3. compute the noisy image x_t
        x_t = self.var_scheduler.add_noise(x0, t, epsilon)

        # 4. compute the predicted noise
        epsilon_theta = self.denoise_net(x_t, t)

        # 5. compute the loss
        # L = ||epsilon - epsilon_theta||^2
        loss = F.mse_loss(epsilon_theta, epsilon)

        return loss

    @property
    def device(self):
        ''' Access method like a property '''
        return next(self.denoise_net.parameters()).device

    @property
    def img_resolution(self):
        ''' Assume image is square '''
        return self.denoise_net.img_res

    @torch.no_grad()
    def sample(self, batch_size: int):
        '''
        :param batch_size: the number of images to sample
        :return: a batch of images (batch_size, 3, height, width) [B, 3, H, W]
        '''
        noisy_images = []
        # 1. sample a batch of random noise
        x_t = torch.randn((batch_size, 3, self.img_resolution, self.img_resolution), device=self.device)
        noisy_images.append(x_t.clone()) # tensor is mutable and pass by reference, so we need to clone it

        # 2. loop over the timesteps in reverse order
        for t in self.var_scheduler.timesteps:
            t = t.unsqueeze(0).expand(batch_size).to(self.device) # [batch_size, ]

            # 2.1 compute the predicted noise
            epsilon_theta = self.denoise_net(x_t, t)

            # 2.2 compute the previous image x_{t-1}
            x_t = self.var_scheduler.denoise(x_t, t, epsilon_theta)
            noisy_images.append(x_t.clone())


        return x_t, noisy_images

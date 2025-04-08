import unittest
import torch
import random

from diffusion.ddpm_scheduler import DDPMScheduler


class TestDDPMScheduler(unittest.TestCase):
    def setUp(self):
        self.total_timesteps = 1000
        self.scheduler = DDPMScheduler(total_timesteps=self.total_timesteps, beta_1=1e-4,
                                       beta_T=0.02, var_type="lower_bound")

    def test_initialization(self):
        self.assertEqual(self.scheduler.total_timesteps, 1000)
        self.assertEqual(self.scheduler.betas.shape[0], 1000)
        self.assertEqual(self.scheduler.alphas.shape[0], 1000)
        self.assertEqual(self.scheduler.alphas_cumprod.shape[0], 1000)
        self.assertEqual(self.scheduler.sigmas.shape[0], 1000)

    def test_denoise(self):
        x_t = torch.randn(2, 3, 64, 64)
        # t = random.randint(0, self.total_timesteps - 1) # [0, self.total_timesteps - 1]
        t = torch.randint(0, self.total_timesteps, (2,)) # [2,]
        noise = torch.randn(2, 3, 64, 64)
        x_t_1 = self.scheduler.denoise(x_t, t, noise)
        self.assertEqual(x_t_1.shape, x_t.shape)

    def test_add_noise(self):
        x_0 = torch.randn(2, 3, 64, 64)
        t = torch.randint(0, self.total_timesteps, (2,)) # [2,]
        noise = torch.randn(2, 3, 64, 64)
        x_t = self.scheduler.add_noise(x_0, t, noise)
        self.assertEqual(x_t.shape, x_0.shape)

if __name__ == "__main__":
    unittest.main()

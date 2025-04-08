import unittest
import torch
from models.unet import UNet
from diffusion.ddpm_scheduler import DDPMScheduler
from diffusion.ddpm_diffusion import DDPM


class TestDDPMRealComponents(unittest.TestCase):
    def setUp(self):
        # Define UNet and DDPMScheduler with minimal settings
        self.img_res = 64
        self.total_timesteps = 10

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.unet = UNet(img_res=self.img_res)

        self.scheduler = DDPMScheduler(total_timesteps=self.total_timesteps)

        self.ddpm = DDPM(self.unet, self.scheduler)
        self.ddpm.to(self.device)

    def test_device_property(self):
        self.assertEqual(self.ddpm.device.type, 'cuda')

    def test_img_resolution_property(self):
        self.assertEqual(self.ddpm.img_resolution, self.img_res)

    def test_compute_loss_runs(self):
        B, C, H, W = 2, 3, self.img_res, self.img_res
        x0 = torch.randn(B, C, H, W, device=self.device)
        loss = self.ddpm.compute_loss(x0)
        self.assertIsInstance(loss.item(), float)

    def test_sample_output_shape(self):
        B = 2
        x_t, noisy_images = self.ddpm.sample(B)
        self.assertEqual(x_t.shape, (B, 3, self.img_res, self.img_res))
        self.assertEqual(len(noisy_images), self.total_timesteps + 1)


if __name__ == "__main__":
    unittest.main()

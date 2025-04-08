import unittest
import torch
from models.unet import UNet  # adjust the path if needed


class TestUNet(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.img_res = 64
        self.model = UNet(img_res=self.img_res)
        self.model.eval()  # disable dropout etc.

        self.x = torch.randn(self.batch_size, 3, self.img_res, self.img_res)

        self.timesteps = torch.randint(low=0, high=1000, size=(1,)).expand(self.batch_size)

    def test_forward_shape(self):
        with torch.no_grad():
            out = self.model(self.x, self.timesteps)
            self.assertEqual(out.shape, self.x.shape)

    def test_output_stability(self):
        torch.manual_seed(42)
        x = torch.ones_like(self.x)
        timesteps = torch.zeros_like(self.timesteps)

        with torch.no_grad():
            out1 = self.model(x, timesteps)
            out2 = self.model(x, timesteps)

        # Outputs should be identical (since model is in eval mode and deterministic)
        self.assertTrue(torch.allclose(out1, out2, atol=1e-5))

    def test_gradient_flow(self):
        x = self.x.clone().requires_grad_()
        timesteps = self.timesteps

        out = self.model(x, timesteps)
        loss = out.mean()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)
        self.assertFalse(torch.isnan(x.grad).any(), "Gradient has NaNs")


    def test_model_has_parameters(self):
        self.assertTrue(any(p.requires_grad for p in self.model.parameters()))

if __name__ == '__main__':
    unittest.main()

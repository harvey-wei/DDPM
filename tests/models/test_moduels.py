import unittest
from pathlib import Path
import sys
import os
import torch


from models.modules import (
        Downsample,
        Upsample,
        SelfAttention,
        ResBlock,
        TimeEmbedding
        )


# Unit test class for Downsample
class TestDownsample(unittest.TestCase):
    def setUp(self):
        self.downsample = Downsample(4)

    def test_init(self):
        self.assertEqual(self.downsample.conv.in_channels, 4)
        self.assertEqual(self.downsample.conv.out_channels, 4)

    def test_forward(self):
        x = torch.randn(1, 4, 16, 16)
        output = self.downsample(x)
        self.assertEqual(output.shape, (1, 4, 8, 8))

    def test_forward_values(self):
        with torch.no_grad():
            # All ones input
            x = torch.ones(1, 4, 16, 16)

            # Set weights to 1 and bias to 0 for deterministic output
            self.downsample.conv.weight.fill_(1.0)
            self.downsample.conv.bias.zero_()

            output = self.downsample(x)

            # Compute expected value: sum over 3x3 kernel × 4 channels = 9 * 4 = 36
            expected_value = 36.0
            expected = torch.full_like(output[1:-1, 1:-1], expected_value)

            self.assertTrue(torch.allclose(output[1:-1, 1:-1], expected, atol=1e-4))


class TestUpsample(unittest.TestCase):
    def setUp(self):
        self.upsample = Upsample(8)

    def test_init(self):
        self.assertEqual(self.upsample.conv.in_channels, 8)
        self.assertEqual(self.upsample.conv.out_channels, 8)

    def test_forward(self):
        x = torch.randn(1, 8, 8, 8)
        output = self.upsample(x)
        self.assertEqual(output.shape, (1, 8, 16, 16))


class TestSelfAttention(unittest.TestCase):
    def setUp(self):
        self.self_attention = SelfAttention(64)

    def test_init(self):
        self.assertEqual(self.self_attention.in_ch, 64)

    def test_forward(self):
        # Note num_channels must be divisble by num_groups in GroupNorm
        x = torch.randn(1, 64, 16, 16)
        output = self.self_attention(x)
        self.assertEqual(output.shape, (1, 64, 16, 16))


class TestResBlock(unittest.TestCase):
    def setUp(self):
        self.res_block = ResBlock(32, 64, 16, 0.4, True)

    def test_init(self):
        self.assertEqual(self.res_block.in_ch, 32)
        self.assertEqual(self.res_block.out_ch, 64)

    def test_forward(self):
        x = torch.randn(8, 32, 64, 64)
        time_embed = torch.randn(8, 16)
        output = self.res_block(x, time_embed)
        self.assertEqual(output.shape, (8, 64, 64, 64))


class TestTimeEmbedding(unittest.TestCase):
    def setUp(self):
        self.time_embedding = TimeEmbedding(8, 16)

    def test_init(self):
        self.assertEqual(self.time_embedding.time_embed_hidden_dim, 8)
        self.assertEqual(self.time_embedding.time_embed_input_dim, 16)

    def test_forward(self):
        x = torch.randn(12,)
        output = self.time_embedding(x)
        self.assertEqual(output.shape, (12, 8))


if __name__ == '__main__':
    unittest.main()

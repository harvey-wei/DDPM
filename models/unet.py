import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import Downsample, Upsample, TimeEmbedding, SelfAttention, ResBlock


'''
Input (3x64x64)
      │
      ▼
Conv2D (3 → 128) ──────┐
      │                │
      ▼                │
+-------------------+  │
| DownBlock Stage 0 |  │
| ResBlock x4       |  │
+-------------------+  │
      │                │
      ▼                │
Downsample (128 → 256) │
      │                │
+-------------------+  │
| DownBlock Stage 1 |  │
| ResBlock x4       |  │
+-------------------+  │
      │                │
      ▼                │
Downsample (256 → 256) │
      │                │
+-------------------+  │
| DownBlock Stage 2 |  │
| ResBlock x4       |  │
+-------------------+  │
      │                │
      ▼                │
Downsample (256 → 256) │
      │                │
+-------------------+  │
| DownBlock Stage 3 |  │
| ResBlock x4       |  │
+-------------------+  │
      │                │
      ▼                │
+----------------------------+
|     Middle Block           |
|  ResBlock (attn=True)      |
|  ResBlock (attn=False)     |
+----------------------------+
      │
      ▼
+-------------------+
| UpBlock Stage 3   | <─┐ skip from DownBlock 3
| ResBlock x5       |   │
+-------------------+   │
      │                 │
Upsample (256 → 256)     │
      │                 │
+-------------------+   │
| UpBlock Stage 2   | <─┘
| ResBlock x5       | <─┐ skip from DownBlock 2
+-------------------+   │
      │                 │
Upsample (256 → 256)     │
      │                 │
+-------------------+   │
| UpBlock Stage 1   | <─┘ skip from DownBlock 1
| ResBlock x5       |
+-------------------+
      │
Upsample (256 → 128)
      │
+-------------------+
| UpBlock Stage 0   | <── skip from DownBlock 0
| ResBlock x5       |
+-------------------+
      │
      ▼
GroupNorm → Swish → Conv2D (128 → 3)
      │
      ▼
   Output (3x64x64)
'''

class UNet(nn.Module):
    def __init__(self,
                 img_res=64,
                 num_base_ch=128,
                 ch_multiplier=[1, 2, 2, 2], # channel_multipler for each stage
                 attention_stages=[1], # stage index for self attention layer
                 num_resblock_per_stage=4,
                 dropout=0.1,
                 T=1000, # Total number of sampling steps in DDPM
    ) -> None:
        super().__init__()

        # We assume image is of spatial size(img_res, img_res)
        self.img_res = img_res
        self.T = T
        self.timestep_embed_dim = 4 * num_base_ch
        self.timestep_embed_layer = TimeEmbedding(self.timestep_embed_dim)


        # Extend the image channel from 3 to num_base_ch while maintaiing spatial dim.
        self.in_layer = nn.Conv2d(3, num_base_ch, kernel_size=3, stride=1, padding=1)

        # Shrink the channel number from num_base_ch to 3
        # self.out_layer = nn.Sequential(
        #         nn.GroupNorm(32, )
        #         )

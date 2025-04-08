import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules import Downsample, Upsample, TimeEmbedding, SelfAttention, ResBlock


'''
For downsampling block, the downsample follows each block except the last block.
For upsampling block, the upsampling follows each block except the first block for spatial dim
compatibility

For each stage, the channel changes occurrs at the first block.

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
Upsample (256 → 256)
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
GroupNorm → SiLU → Conv2D (128 → 3)
      │
      ▼
   Output (3x64x64)
'''

class UNet(nn.Module):
    def __init__(self,
                 img_res=64,
                 num_base_ch=128,
                 stage_out_ch_mult=[1, 2, 2, 2], # out_channel_multipler for each stage
                 attention_stages=[1], # stage index for self attention layer
                 num_resblock_per_stage=4,
                 dropout=0.1,
    ) -> None:
        super().__init__()

        # We assume image is of spatial size(img_res, img_res)
        self.img_res = img_res
        self.timestep_embed_hidden_dim = 4 * num_base_ch
        self.timestep_embed_layer = TimeEmbedding(self.timestep_embed_hidden_dim)


        # Extend the image channel from 3 to num_base_ch while maintaiing spatial dim.
        self.in_layer = nn.Conv2d(3, num_base_ch, kernel_size=3, stride=1, padding=1)

        self.down_stages = nn.ModuleList()
        self.middle_stage = nn.ModuleList()
        self.up_stages = nn.ModuleList()

        # Tracks the in_ch and out_ch of moving blocks
        in_ch = num_base_ch
        out_ch = None
        down_block_out_chs = [num_base_ch]

        for stage_idx, ch_multi in enumerate(stage_out_ch_mult):
            out_ch = num_base_ch * ch_multi
            for _ in range(num_resblock_per_stage):
                self.down_stages.append(ResBlock(in_ch,
                                                 out_ch,
                                                 self.timestep_embed_hidden_dim,
                                                 dropout=dropout,
                                                 has_attention=(stage_idx in attention_stages)))
                in_ch = out_ch
                down_block_out_chs.append(out_ch)

            if stage_idx != len(stage_out_ch_mult) - 1:
                self.down_stages.append(Downsample(in_ch))
                down_block_out_chs.append(in_ch)

        self.middle_stage.append(ResBlock(in_ch, in_ch, self.timestep_embed_hidden_dim,
                                          dropout=dropout, has_attention=True))
        self.middle_stage.append(ResBlock(in_ch, in_ch, self.timestep_embed_hidden_dim,
                                          dropout=dropout, has_attention=False))

        for stage_idx, ch_multi in enumerate(reversed(stage_out_ch_mult)):
            out_ch = num_base_ch * ch_multi

            #  One more blocks to consider downsample block
            for _ in range(num_resblock_per_stage + 1):
                self.up_stages.append(ResBlock(
                    in_ch + down_block_out_chs.pop(),
                    out_ch,
                    self.timestep_embed_hidden_dim,
                    dropout=dropout,
                    has_attention=(stage_idx in attention_stages)
                ))

                in_ch = out_ch

            if  stage_idx != len(stage_out_ch_mult) - 1:
                self.up_stages.append(Upsample(in_ch))

        assert len(down_block_out_chs) == 0

        # Both UpStages and DownStages do the same number of downsampling & upsampling
        # The spatial dim is recovered.
        # Shrink the channel number from num_base_ch to 3
        self.out_layer = nn.Sequential(
                nn.GroupNorm(32, in_ch),
                nn.SiLU(),
                nn.Conv2d(in_ch, 3, kernel_size=3, stride=1, padding=1)
                )

        self.initialize()

    def initialize(self):
        nn.init.xavier_uniform_(self.in_layer.weight)
        nn.init.zeros_(self.in_layer.bias)

        nn.init.xavier_uniform_(self.out_layer[-1].weight, gain=1e-5)
        nn.init.zeros_(self.out_layer[-1].bias)

    def forward(self, x: torch.tensor, timestep: torch.tensor):
        '''
        :param x: (batch_size, 3, img_res, img_res), [B, 3, H, W]
        :param timestep: (batch_size,), [B,]
        :return : (batch_size, 3, img_res, img_res), [B, 3, H, W]
        :notes: denoise an batch of noisier images to a less noisy image
        '''
        assert x.dim() == 4, f"Input x should be of shape [B, 3, H, W], but got {x.shape}"

        B, _, _, _ = x.shape

        assert timestep.shape == (B,), f"timestep shape {timestep.shape} should be of shape [B,]"

        time_embed = self.timestep_embed_layer(timestep) # [B, 4*base_ch]

        x = self.in_layer(x) # [B, num_base_ch, H, W]

        down_stages_outs = [x]

        for block in self.down_stages:
            if isinstance(block, ResBlock):
                x = block(x, time_embed)
            else:
                # Downsample
                x = block(x)

            down_stages_outs.append(x)

        # for down_out in down_stages_outs:
        #     print(f'down out shape {down_out.shape}')

        for block in self.middle_stage:
            x = block(x, time_embed)

        for block in self.up_stages:
            if isinstance(block, ResBlock):
                x = block(torch.cat((x , down_stages_outs.pop()), dim=1), time_embed)
            else:
                # Upsample Block
                x = block(x)

        x = self.out_layer(x) # [B, 3, H, W]

        return x

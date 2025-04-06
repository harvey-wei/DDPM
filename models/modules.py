import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Downsample(nn.Module):
    '''
    Downsample module for UNet that reduces the spatial dimensions of the input tensor by a factor of 2.
    This is done by applying a 2D convolution with a kernel size of 3 and a stride of 2, padding of
    1.
    Ref: https://arxiv.org/abs/1603.07285
    '''
    def __init__(self, in_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=2, padding=1)
        self.initialize()

    def initialize(self) -> None:
        # Initialize the weights of the convolutional layer using xavier uniform distribution
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4, "Input tensor must be 4D (batch_size, channels, height, width)"
        assert x.shape[1] == self.conv.in_channels, f"""Input tensor must have
                {self.conv.in_channels} channels, but got {x.shape[1]} channels"""

        # Apply the convolutional layer and return the output
        x = self.conv(x)

        return x

class Upsample(nn.Module):
    '''
    Upsample module for UNet that scale up the spatial dimensions of the input tensor by a factor of
    2. This is done by nearest neighbor interpolation followed by a 2D convolution with a kernel
    size of 3 and a stride of 1, padding of 1.
    '''
    def __init__(self, in_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1)

        self.initialize()

    def initialize(self) -> None:
        # Initialize the weights of the convolutional layer using xavier uniform distribution
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.tensor) -> torch.Tensor:
        assert x.dim() == 4, "Input tensor must be 4D (batch_size, channels, height, width)"
        assert x.shape[1] == self.conv.in_channels, f"""Input tensor must have {self.conv.in_channels}
                channels, but got {x.shape[1]} channels"""

        # Apply nearest neighbor interpolation to scale up the spatial dimensions
        x = F.interpolate(x, scale_factor=2, mode='nearest')

        # Apply the convolutional layer and return the output
        x = self.conv(x)
        return x


class SelfAttention(nn.Module):
    '''
    Perform Group Normalization on the input tensor. The number of groups is set to 32 by default.
    This attention preserves the number of channels and spatial dimensions of the input tensor.
    '''
    def __init__(self, in_ch: int):
        super().__init__()
        self.in_ch = in_ch

        self.group_norm = nn.GroupNorm(32, in_ch)

        # nn.Conv2d is faster than nn.Linear for 2D data
        # The learnable parameters are dependent on the number of channels instead of the number of
        # spatial dimensions.
        self.q_project = nn.Conv2d(in_ch, in_ch, kernel_size=1)
        self.k_project = nn.Conv2d(in_ch, in_ch, kernel_size=1)
        self.v_project = nn.Conv2d(in_ch, in_ch, kernel_size=1)
        self.out_project = nn.Conv2d(in_ch, in_ch, kernel_size=1)

        self.inintialize()

    def inintialize(self) -> None:
        for module in [self.q_project, self.k_project, self.v_project, self.out_project]:
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

        # Use small gain for the output projection to avoid exploding gradients
        nn.init.xavier_uniform_(self.out_project.weight, gain=1e-5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        :param x: Input tensor of shape (batch_size, channels, height, width), [B, C, H, W] for short.
        :return: Output tensor of the same shape as input tensor.
        signal flow: x -> GroupNorm -> SelfAttention -> linear -> ReLU -> + x (Residual connection)
        '''
        assert x.dim() == 4, "Input tensor must be 4D (batch_size, channels, height, width)"

        B, C, H, W = x.shape

        assert C == self.in_ch, f"Input tensor must have {self.in_ch} channels, but got {C} channels"

        # Apply Group Normalization
        h = self.group_norm(x)

        q = self.q_project(h) # [B, C, H, W]
        k = self.k_project(h) # [B, C, H, W]
        v = self.v_project(h) # [B, C, H, W]

        q = q.reshape(B, C, -1) # [B, C, H*W]
        v = v.reshape(B, C, -1) # [B, C, H*W]
        k = k.reshape(B, C, -1) # [B, C, H*W]

        # att_scores[b, i, j] = sum_{c} q[b, c, i] * k[b, c, j]
        att_scores = torch.einsum('bci, bcj -> bij', q, k) # [B, H*W, H*W]
        att_scores = F.softmax(att_scores * (C) ** (-0.5), dim=-1) # [B, H*W, H*W]

        # out[b, c, i] = sum_{j} att_scores[b, i, j] * v[b, c, j]
        out = torch.einsum('bij, bcj->bci', att_scores, v) # [B, C, H*W]
        out = out.reshape(B, C, H, W) # [B, C, H, W]

        # Recall that the self.out_project is scaled down by a gain of 1e-5
        # This can make the return value very close to input, i.e. leading SelfAttention to be
        # identity function, which make training more stable, especially for large models.
        out = self.out_project(out) # [B, C, H, W]

        return out + x

class TimeEmbedding(nn.Module):
    def __init__(self,
                 time_embed_hidden_dim: int,
                 time_embed_input_dim: int=256) -> None:
        super().__init__()
        self.time_embed_hidden_dim = time_embed_hidden_dim
        self.time_embed_input_dim = time_embed_input_dim

        self.mlp = nn.Sequential(
            nn.Linear(time_embed_input_dim, time_embed_hidden_dim),
            nn.SiLU(),
            nn.Linear(time_embed_hidden_dim, time_embed_hidden_dim)
        )

    def timestep_embedding(self,
                           timestep: torch.tensor,
                           time_embed_input_dim:int,
                           max_period: int = 10000):
        '''
        :param timestep: Tensor of shape (batch_size,)
        :param time_embed_dim: Dimension of the time embedding
        :param max_period: Maximum period for the sine and cosine functions
        :return: Tensor of shape (batch_size, time_embed_dim)

        In the paper "Attention is All You Need", the authors use a sinusoidal embedding for the
        position. The emdedding is a function of both position and feature dimension.
        both even(2i, i = 0, 1, 3) and odd(2i + 1, i = 0, 1, 2) dimensions share the same frequency.
        freq(2i or 2i + 1) = 1 / (max_period) ^ (2i / C)
        To avoid numerical issues, we use a = exp(ln(a)) to compute the frequency.
        freq = exp((-2i) * ln(max_period) / C)

        To vectrize the computation, we use the following formula:
        freq = torch.exp(torch.arange(0, C, 2 dtype=torch.float32) * -(math.log(max_period) /
                                                                         half_dim)) #[C/2,]
        pos = torch.arange(0, N, dtype=torch.float32) #[N,]

        freq = freq[None,:] #[1, C/2]
        pos = pos[:, None] #[N, 1]

        # pos_enc is of shape [B, N, C]
        pos_enc = torch.zeros((N, C), dtype=torch.float32) #[N, C]
        pos_enc[:, 0::2] = torch.sin(pos * freq) #[N, C/2]
        pos_enc[:, 1::2] = torch.cos(pos * freq) #[N, C/2]

        pos_enc = pos_enc[None, :, :] #[1, N, C]
        pos_enc = pos_enc.expand(B, -1, -1) #[B, N, C]

        In diffusion model, we use the similar  idea to encode the time step.
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        feq = exp([0, 1, 2, ... , half_dim] * ln(max_period) / C) # [C/2,]

        input timestep is of shape [B] simply one time step for each batch.
        we multiply the timestep with the frequency, take sin and cos, concatenate the result

        temp = timestep[:, None] * freq[None, :] # [B, C/2]
        '''
        assert timestep.dim() == 1, "Input tensor must be 1D (batch_size,)"

        half_dim = time_embed_input_dim // 2

        freq = torch.exp(torch.arange(0, half_dim, dtype=torch.float32) * -(math.log(max_period) /
                                                                            half_dim))
        # Ensure freq is on the same device as timestep
        freq = freq.to(timestep.device) # [half_dim,]

        # Always ensure both tensors are on the same device before performing operations.
        temp = timestep[:, None] * freq[None, :] # [B, half_dim]

        time_embed = torch.cat((torch.cos(temp), torch.sin(temp) ), dim=-1) # [B, 2 * half_dim]

        if time_embed_input_dim % 2 == 1:
            # If time_embed_dim is odd, we add a zero to the end of the tensor
            time_embed = torch.cat((time_embed, torch.zeros(time_embed.shape[0], 1,
                                                            device=time_embed.device,
                                                            dtype=time_embed.dtype)),
                                   dim=-1)


        return time_embed

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        '''
        :param timestep: Tensor of shape (batch_size, 1)
        :return: Tensor of shape (batch_size, time_embed_dim)
        signal flow: timestep -> timestep_embedding -> time_embed -> output
        '''
        if timestep.dim() == 0:
            # If timestep is a torch scalar, we need to unsqueeze it to make it 1D
            timestep = timestep.unsqueeze(0) # [1,] batch_size = 1

        assert timestep.dim() == 1, "Input tensor must be 1D (batch_size,)"

        time_embed = self.mlp(self.timestep_embedding(timestep, self.time_embed_input_dim)) # [B, time_embed_dim]

        return time_embed


class ResBlock(nn.Module):
    '''
    The ResBlock takes as input both feature map and timestep embedding, and applies pre-activation
    Residual Blocks followed by a attention layer optionally.
    Note, it does not change the spatial dimensions of the input tensor.
    It use pre-activation residual block:
        GroupNorm -> SiLU -> (Dropout) -> Conv2d
    '''
    def __init__(self, in_ch, out_ch, time_embed_dim, dropout, has_attention =False ) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.block1 = nn.Sequential(
                nn.GroupNorm(32, in_ch),
                nn.SiLU(),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
                )

        # Add one 1x1 conv to ensure time embedding and feature map have the same number of channels
        self.time_embed_proj = nn.Sequential(
                nn.SiLU(),
                nn.Conv2d(time_embed_dim, out_ch, kernel_size=1, stride=1, padding=0),
                )

        self.block2 = nn.Sequential(
                nn.GroupNorm(32, out_ch),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
                )

        # Shortcut connection
        if out_ch != in_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()

        # Add attention layer
        if has_attention:
            self.attention = SelfAttention(out_ch)
        else:
            self.attention = nn.Identity()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

        # Scale down the last conv layer to avoid exploding gradients in residual connection
        nn.init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)

    def forward(self, x: torch.Tensor, time_embed: torch.Tensor) -> torch.Tensor:
        '''
        :param x: Input tensor of shape (B, C, H, W)
        :param time_embed: Tensor of shape (B, time_embed_dim)
        signal flow: x -> block 1-> + time_embedding -> block2 + shortcut(x) -> attention -> output
        The ops of timestep to embedding is done by a higher level module.
        '''
        assert x.dim() == 4, "Input tensor must be 4D (batch_size, channels, height, width)"
        assert time_embed.dim() == 2, "Input tensor must be 2D (batch_size, time_embed_dim)"

        assert x.shape[1] == self.in_ch, f"Input tensor must havv {self.in_ch} channels, but got {x.shape[1]} channels"

        h = self.block1(x) # [B, out_ch, H, W]

        h += self.time_embed_proj(time_embed[:, :, None, None]) # [B, out_ch, H, W]

        h = self.block2(h) # [B, out_ch, H, W]

        h += self.shortcut(x) # [B, out_ch, H, W]

        h = self.attention(h) # [B, out_ch, H, W]

        return h

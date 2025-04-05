import torch
from PIL import Image
import numpy as np


def tensor_to_PIL_img(x: torch.tensor):
    '''
    :param x, the image tensor of shape [N, C, H, W] or [C, H, W], value ranges from -1 to 1
    :return List of PIL Images HxWxC with value [0, 1]
    '''
    is_one_img = False

    if 3 == x.dim():
        x = x[None, : , :, :]
        is_one_img = True

    # Rescale pixel value from [-1, 1] to [0, 1]
    x = ((x + 1.0) * 0.5).clamp(0, 1).detach().cpu().permute(0, 2, 3, 1).numpy() # [N, H, W, C]

    # Convert [0, 1] to 0-255(uint8)
    imgs = (x * 255).astype(np.uint8)

    # Convet img numpy to PIL images
    imgs  = [Image.fromarray(img) for img in imgs]

    return imgs[0] if is_one_img else imgs


if __name__ == '__main__':
    # Test tensor_to_PIL_img
    img = torch.rand(3, 256, 256) * 2 - 1
    PIL_img = tensor_to_PIL_img(img)

    print("Type:", type(PIL_img))
    print("Size (W x H):", PIL_img.size)   # (width, height)
    print("Mode:", PIL_img.mode)           # RGB, L, etc.


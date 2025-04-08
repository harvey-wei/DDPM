# DDPM for Image Generation

This repository contains a PyTorch implementation of Denoising Diffusion Probabilistic Models (DDPM)
for image generation.

The code is based on the paper [Denoising Diffusion Probabilistic
Models](https://arxiv.org/abs/2006.11239) by Ho et al.

The Denoising Network adopts the architectural design spirit seminal UNet for meidical image segmentation
but leverage preactivatioin ResBlocks and Self-Attention Layers for better performance.

## Training
The training script is located in `train.py`. You can run the training script with the following command:

```bash
python train.py
```

## Sampling new images
The sampling script is located in `sample.py`. You can run the sampling script with the following command:

```bash
python sample.py \
  --checkpoint_path path_to_ckpt_pth \
  --num_samples 2 \
  --save_dir generated_images
```

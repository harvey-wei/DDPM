import argparse
import os
from pathlib import Path

import torch
from models.unet import UNet
from diffusion.ddpm_diffusion import DDPM
from diffusion.ddpm_scheduler import DDPMScheduler
from utils.tensor_to_PIL_img import tensor_to_PIL_img

def main():
    parser = argparse.ArgumentParser(description="Sample images from a trained DDPM model.")

    parser.add_argument("--checkpoint_path", type=str, required=True,
        help="Path to the checkpoint file.")
    parser.add_argument("--num_samples", type=int, default=4,
        help="Number of images to generate.")
    parser.add_argument("--save_dir", type=str, default="samples",
        help="Directory to save sampled images.")
    parser.add_argument("--gpu", type=int, default=0,
        help="GPU index to use (default: 0)")

    args = parser.parse_args()

    # Set device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint_path, map_location=device)

    # Extract saved hyperparameters
    beta_1 = checkpoint["beta_1"]
    beta_T = checkpoint["beta_T"]
    T = checkpoint["num_diffusion_timesteps"]
    var_type = checkpoint["var_type"]
    img_size = checkpoint["img_size"]

    # Rebuild model and scheduler
    unet = UNet(img_res=img_size)
    scheduler = DDPMScheduler(
        total_timesteps=T,
        beta_1=beta_1,
        beta_T=beta_T,
        var_type=var_type,
    )
    ddpm = DDPM(unet, scheduler).to(device)
    ddpm.load_state_dict(checkpoint["model_state_dict"])
    ddpm.eval()

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Generate samples
    print(f"Sampling {args.num_samples} image(s)...")
    with torch.no_grad():
        samples, _ = ddpm.sample(batch_size=args.num_samples)

    imgs = tensor_to_PIL_img(samples)

    for i, img in enumerate(imgs):
        img.save(Path(args.save_dir) / f"sample_{i}.png")

    print(f"Saved {args.num_samples} samples to {args.save_dir}/")

if __name__ == "__main__":
    main()

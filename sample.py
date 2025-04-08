import argparse
from pathlib import Path

import torch
from utils import tensor_to_PIL_img
from diffusion.ddpm_diffusion import DDPM
from diffusion.ddpm_scheduler import DDPMScheduler


def main():
    parser = argparse.ArgumentParser(description="Sample from a rained diffusion model.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for sampling.")
    parser.add_argument(
        "--num_samples", type=int, default=1, help="Number of samples to generate."
    )
    parser.add_argument("--model_path", type=str, help="Path to the trained model checkpoint.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory to save the generated samples.",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained model
    model = DDPM(None, None)
    model.load_state_dict(torch.load(args.model_path, map_location=device))


if __name__ == '__main__':
    main()

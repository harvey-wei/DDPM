import argparse
import matplotlib.pyplot as plt
import os

import torch
from torchvision.transforms.functional import to_pil_image

from data.dataset import AFHQDatasetHelper
from utils.get_infinite_generator import get_infinite_generator
from utils.tensor_to_PIL_img import tensor_to_PIL_img
from models.unet import UNet
from diffusion.ddpm_diffusion import DDPM
from diffusion.ddpm_scheduler import DDPMScheduler

from tqdm import tqdm

def get_now_str():
    from datetime import datetime
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")

def main():
    parser = argparse.ArgumentParser(description="Train a DDPM model.")

    parser.add_argument("--resume_checkpoint", type=str, default=None,
        help="Path to the checkpoint file to resume training.",
    )

    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_num_img_per_class", type=int, default=10000)
    parser.add_argument("--num_model_training_steps", type=int, default=100)
    parser.add_argument("--num_diffusion_timesteps", type=int, default=1000)
    parser.add_argument("--warm_steps", type=float, default=200)
    parser.add_argument("--beta_1", type=float, default=1e-4)
    parser.add_argument("--beta_T", type=float, default=0.02)
    parser.add_argument("--log_every", type=int, default=10)
    # parser.add_argument("--sample_every", type=int, default=1000)
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--var_type", type=str, default="lower_bound",)

    args = parser.parse_args()

    # Parse the arguments
    batch_size = args.batch_size
    img_size = args.img_size
    max_num_img_per_class = args.max_num_img_per_class
    num_model_training_steps = args.num_model_training_steps
    num_diffusion_timesteps = args.num_diffusion_timesteps
    warm_steps = args.warm_steps
    beta_1 = args.beta_1
    beta_T = args.beta_T
    log_every = args.log_every
    # sample_every = args.sample_every
    checkpoint_dir = args.ckpt_dir + "/" + get_now_str()
    num_workers = args.num_workers

    # Create the checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    #  Set device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # 1. Load dataset and create data loader
    dataset_helper = AFHQDatasetHelper(
        root_dir="datasets",
        img_resolution=img_size,
        batch_size=batch_size,
        max_samples_per_class=max_num_img_per_class,
        num_workers=num_workers,
    )

    train_dataloader = dataset_helper.get_train_loader()
    train_dataloader_gen = get_infinite_generator(train_dataloader)

    # 2. DDPM model, unet and scheduler
    unet = UNet(img_res=img_size)
    scheduler = DDPMScheduler(total_timesteps=num_diffusion_timesteps,
                              beta_1=beta_1,
                              beta_T=beta_T,
                              var_type=args.var_type,
    )

    ddpm = DDPM(unet, scheduler)
    ddpm.to(device)

    # 3. Optimizer and learning rate scheduler
    optimizer = torch.optim.Adam(
        ddpm.parameters(),
        lr=args.lr,
    )

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min((step + 1) / warm_steps, 1),
    )

    init_train_step = 0
    losses = []

    # 4. Load checkpoint if provided
    if args.resume_checkpoint is not None:
        print(f"Loading checkpoint from {args.resume_checkpoint}")
        checkpoint = torch.load(args.resume_checkpoint, map_location=device)
        ddpm.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        init_train_step = checkpoint["stop_step"]
        losses = checkpoint["losses"]
        print(f"Loaded checkpoint from {args.resume_checkpoint}")

    # 5. Training loop
    with tqdm(initial=init_train_step, total=num_model_training_steps, desc="Training") as pbar:
        for step in range(init_train_step, num_model_training_steps):
            if step > init_train_step and step % log_every == 0:
                ddpm.eval()
                plt.plot(losses)
                plt.title("loss curve")
                plt.xlabel("step")
                plt.ylabel("loss")
                plt.savefig(f"{checkpoint_dir}/loss_curve.png")

                # Save a sample image
                with torch.no_grad():
                    sample = ddpm.sample(batch_size=1)
                    sample_pil = tensor_to_PIL_img(sample[0])
                    sample_pil[0].save(f"{checkpoint_dir}/sample_{step}.png")

                # Save the model checkpoint
                torch.save({
                    "model_state_dict": ddpm.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                    "stop_step": step,
                    "losses": losses,
                    "beta_1": beta_1,
                    "beta_T": beta_T,
                    "num_diffusion_timesteps": num_diffusion_timesteps,
                    "var_type": args.var_type,
                    "img_size": img_size,
                }, f"{checkpoint_dir}/checkpoint_{step}.pth")

                ddpm.train()

            # Get the next batch of images
            real_images, _ = next(train_dataloader_gen)
            real_images = real_images.to(device)
            print(f"real_images shape: {real_images.shape}")

            # Perform a training step
            loss = ddpm.compute_loss(real_images)
            losses.append(loss.item())

            # Update the optimizer and learning rate scheduler
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            pbar.set_description(f"Training step {step}/{num_model_training_steps}, Loss: {loss.item():.4f}")
            pbar.update(1)

if __name__ == "__main__":
    main()

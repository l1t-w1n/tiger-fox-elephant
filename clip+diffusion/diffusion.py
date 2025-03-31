"""https://images.cv/dataset/cat-image-classification-dataset 
    https://arxiv.org/pdf/2006.11239
    
    Denoising Diffusion Probabilistic Models
    Jonathan Ho, Ajay Jain, Pieter Abbeel
"""

import torch
import diffusers
from diffusers import DDPMPipeline
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from diffusers import UNet2DModel, DDPMScheduler
from accelerate import Accelerator
from tqdm import tqdm
import numpy as np
from pathlib import Path
from PIL import Image
import sys
import os

project_root = Path.cwd()
sys.path.append(str(project_root))

class Config:
    DATA_DIR = project_root / "data/diffusion/cat_and_dog_face"
    WEIGHTS_DIR = project_root / "clip+diffusion/weights/diffusion"
    LOG_DIR = project_root / "clip+diffusion/logs/diffusion"
    output_dir = project_root / "clip+diffusion/diffusion_samples"
    
    Path(WEIGHTS_DIR).mkdir(parents=True, exist_ok=True)
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    image_size = 128 
    train_batch_size = 64
    eval_batch_size = 16 
    num_epochs = 450
    gradient_accumulation_steps = 1
    learning_rate = 1e-5
    lr_warmup_steps = 500
    save_image_epochs = 5
    save_model_epochs = 5
    mixed_precision = "fp16"
    scheduler_timesteps = 1000
    
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    

class foxDataset(Dataset):
    def __init__(self, root_dir):
        self.image_paths = []
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        root_path = Path(root_dir)
        
        # Case-insensitive extension check with rglob
        self.image_paths = [
            p for p in root_path.rglob("*")
            if p.suffix.lower() in valid_extensions and p.is_file()
        ]
        self.image_paths = sorted(self.image_paths)
        
        self.transform = transforms.Compose([
            transforms.Resize(Config.image_size),
            transforms.CenterCrop(Config.image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.02),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            return self.transform(img)

def create_unet():
    return UNet2DModel(
        sample_size=Config.image_size,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 512, 512, 1024),
        down_block_types=(
            "DownBlock2D", 
            "AttnDownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
        ),
        up_block_types=(
            "AttnUpBlock2D",
            "UpBlock2D",
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
        ),
    )

def evaluate(config, epoch, pipeline):
    images = pipeline(
        batch_size=config.eval_batch_size,
    ).images

    image_grid = diffusers.utils.make_image_grid(images, rows=4, cols=4)
    test_dir = os.path.join(config.output_dir)
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=config.LOG_DIR
    )
    
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="diffusion",
            config={"num_epochs": config.num_epochs,
                    "learning_rate": config.learning_rate,
                    "scheduler_timesteps": config.scheduler_timesteps,
                    "image_size": config.image_size,
                    "train_batch_size": config.train_batch_size,
                    "mixed_precision": config.mixed_precision,
                    "gradient_accumulation_steps": config.gradient_accumulation_steps,
                    "lr_warmup_steps": config.lr_warmup_steps,
                    "seed": config.seed
                    })

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0
    for epoch in range(300, config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch
            noise = torch.randn(clean_images.shape, device=clean_images.device)
            bs = clean_images.shape[0]

            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device,
                dtype=torch.int64
            )

            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"epoch": epoch, "loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        if accelerator.is_main_process:
            pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(config, epoch, pipeline)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                pipeline.save_pretrained(config.WEIGHTS_DIR)

def main(load_checkpoint = False):
    config = Config()
    dataset = foxDataset(config.DATA_DIR)
    train_dataloader = DataLoader(
        dataset, 
        batch_size=config.train_batch_size, 
        shuffle=True, 
        num_workers=16,
        pin_memory=True
    )
    
    model = create_unet()
    with SummaryWriter(config.LOG_DIR) as writer:
        writer.add_scalar("model parameters", count_parameters(model))
    
    if load_checkpoint:
        pipeline = DDPMPipeline.from_pretrained(config.WEIGHTS_DIR)
        model = pipeline.unet
        print(f"Loaded model from {config.WEIGHTS_DIR}")
    
    noise_scheduler = DDPMScheduler(num_train_timesteps=config.scheduler_timesteps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    # Correct training steps calculation
    num_training_steps = (len(train_dataloader) * config.num_epochs) // config.gradient_accumulation_steps
    lr_scheduler = diffusers.optimization.get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=num_training_steps,
    )
    
    train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)
    
def accelerated_inference(config, num_iters=5):
    # Initialize accelerator
    accelerator = Accelerator(mixed_precision=config.mixed_precision)
    
    # Load model and scheduler
    pipeline = DDPMPipeline.from_pretrained(config.WEIGHTS_DIR)
    print("Loaded model")
    model = pipeline.unet
    noise_scheduler = pipeline.scheduler
    
    # Prepare model with Accelerator
    model = accelerator.prepare(model)
    
    # Generate samples
    for iter in range(num_iters):
        if accelerator.is_main_process:
            print(f"Generation iteration {iter+1}/{num_iters}")
            
        # Sample noise on correct device
        noise = torch.randn(
            (config.eval_batch_size, 3, config.image_size, config.image_size),
            device=accelerator.device
        )
        
        # Denoising loop with progress bar
        timesteps = noise_scheduler.timesteps
        with tqdm(total=len(timesteps), disable=not accelerator.is_local_main_process) as pbar:
            for t in timesteps:
                # Model prediction (conditioned on timestep)
                with torch.inference_mode():
                    noise_pred = model(noise, t).sample

                # Update sample with scheduler
                noise = noise_scheduler.step(
                    noise_pred, t, noise
                ).prev_sample
                pbar.update(1)
        
        # Gather all images across processes
        noise = accelerator.gather(noise)
        
        # Save images only from main process
        if accelerator.is_main_process:
            # Convert to PIL images
            images = (noise / 2 + 0.5).clamp(0, 1)
            images = images.detach().cpu().permute(0, 2, 3, 1).numpy()
            images = (images * 255).round().astype("uint8")
            pil_images = [Image.fromarray(img) for img in images]
            
            # Create and save grid
            image_grid = diffusers.utils.make_image_grid(pil_images, rows=4, cols=4)
            image_grid.save(f"{config.output_dir}/inference_{iter}.png")
    
if __name__ == "__main__":
    main(load_checkpoint = True)
    config = Config()
    accelerated_inference(config, num_iters=5)

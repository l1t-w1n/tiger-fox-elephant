"""
diffusion_train.py - Optimized for cloud GPU training
"""
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from diffusers import UNet2DModel, DDIMScheduler, DDPMScheduler
import matplotlib
from tqdm import tqdm
matplotlib.use('Agg')  # Headless mode
import matplotlib.pyplot as plt
import logging
import numpy as np
from pathlib import Path
from PIL import Image
import argparse
import sys
from pathlib import Path
project_root = Path.cwd()
sys.path.append(str(project_root))

# ================= CONFIGURATION =================
class Config:
    # Cloud-optimized paths (assumes script runs in repo root)
    """DATA_DIR = Path("data")  # Matches unzipped dataset location
    WEIGHTS_DIR = Path("weights/diffusion")
    HIST_DIR = Path("histories/diffusion")
    LOG_DIR = Path("logs")"""
    
    DATA_DIR = project_root / "data/diffusion/fox"
    WEIGHTS_DIR = project_root / "weights/diffusion"
    HIST_DIR = project_root / "histories/diffusion"
    LOG_DIR = project_root / "logs"
    SAMPLE_DIR = project_root / "diffusion_samples"
    
    # Training parameters (adjust via command line)
    IMAGE_SIZE = 64
    BATCH_SIZE = 42
    NUM_EPOCHS = 30
    LR = 1e-4
    NUM_TIMESTEPS = 1000
    SAMPLING_STEPS = 200
    PLOT_EVERY = 2  # Set to NUM_EPOCHS+1 to disable plotting
    SEED = 42

    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize directories and seed
Path(Config.WEIGHTS_DIR).mkdir(parents=True, exist_ok=True)
Path(Config.HIST_DIR).mkdir(parents=True, exist_ok=True)
Path(Config.LOG_DIR).mkdir(parents=True, exist_ok=True)
torch.manual_seed(Config.SEED)
np.random.seed(Config.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(Config.SEED)

# ================= LOGGING =================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler(Config.LOG_DIR / "diffusion.v2.2_training.log", mode="a"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# ================= DATASET =================
class DiffusionDataset(Dataset):
    def __init__(self, root_dir):
        self.image_paths = list(Path(root_dir).glob("*.*"))
        self.transform = transforms.Compose([
            transforms.Resize(Config.IMAGE_SIZE),
            transforms.CenterCrop(Config.IMAGE_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
            transforms.ToTensor(),          
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img), 0

# ================= MODEL & UTILITIES =================
def create_unet():
    return UNet2DModel(
        sample_size=Config.IMAGE_SIZE,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 256, 512, 512),
        dropout=0.1,
        down_block_types=(
            "DownBlock2D", 
            "AttnDownBlock2D", 
            "AttnDownBlock2D", 
            "AttnDownBlock2D"
        ),
        up_block_types=(
            "AttnUpBlock2D", 
            "AttnUpBlock2D",
            "AttnUpBlock2D", 
            "UpBlock2D"
        ),
        norm_num_groups=8,
        attention_head_dim=16
    ).to(Config.device)

def save_checkpoint(model, loss):
    """Save model weights and update training history"""
    weight_path = Config.WEIGHTS_DIR / f"diffusion_v2.2.pth"
    torch.save(model.state_dict(), weight_path)
    
    hist_path = Config.HIST_DIR / "diffusion_v2.2.npz"
    history = np.load(hist_path) if hist_path.exists() else {"losses": []}
    updated_losses = np.append(history.get("losses", []), loss)
    np.savez(hist_path, losses=updated_losses)
    logger.info(f"Checkpoint saved: {weight_path}")

# ================= TRAINING LOOP =================
def train(model, dataloader, scheduler, resume_epoch=0):
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR)
    scaler = torch.amp.GradScaler("cuda")

    for epoch in range(resume_epoch, Config.NUM_EPOCHS):
        epoch_loss = 0.0
        model.train()
        
        batch_bar = tqdm(dataloader, 
                         desc=f"Epoch {epoch+1}/{Config.NUM_EPOCHS} [Train]",
                         bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
        
        for batch_idx, batch in enumerate(batch_bar):
            images = batch[0].to(Config.device)
            noise = torch.randn_like(images)
            timesteps = torch.randint(
                0, scheduler.config.num_train_timesteps,
                (images.shape[0],),
                device=Config.device
            )
            
            with torch.amp.autocast("cuda"):
                noisy_images = scheduler.add_noise(images, noise, timesteps)
                pred_noise = model(noisy_images, timesteps).sample
                
                # Calculate MSE loss
                loss = F.mse_loss(pred_noise, noise)
                
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)          
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            batch_bar.set_postfix({
                'Loss': f"{loss.item():.5f}",
                'LR': f"{optimizer.param_groups[0]['lr']:.3e}"
            })

        avg_loss = epoch_loss / len(dataloader)
        logger.info(f"Epoch [{epoch+1}/{Config.NUM_EPOCHS}] Loss: {avg_loss:.4f}")
        save_checkpoint(model, avg_loss)
        
        if (epoch+1) % Config.PLOT_EVERY == 0:
            generate_samples(model, scheduler, epoch+1)
            
def generate_samples(model, scheduler, epoch, num_images=4):
    """Save samples to file instead of plotting"""
    if epoch == 0:  # Save initial noise
        plt.figure(figsize=(2, 2))
        plt.imshow(torch.randn(3, Config.IMAGE_SIZE, Config.IMAGE_SIZE).permute(1,2,0).cpu())
        plt.savefig(Config.SAMPLE_DIR / "initial_noise.png")
        plt.close()
        
    model.eval()
    samples = torch.randn((num_images, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE))
    samples = samples.to(Config.device)
    
    scheduler.set_timesteps(Config.SAMPLING_STEPS)
    with torch.inference_mode():
        for t in scheduler.timesteps:
            residual = model(samples, t).sample
            samples = scheduler.step(residual, t, samples).prev_sample
    
    # Save sample grid
    fig, axs = plt.subplots(1, num_images, figsize=(15, 3))
    for i, ax in enumerate(axs):
        ax.imshow((samples[i].cpu().permute(1,2,0) * 0.5 + 0.5).clip(0,1))
        ax.axis('off')
    #plt.savefig(Config.WEIGHTS_DIR / f"samples_epoch_{epoch}.png")
    plt.savefig(Config.SAMPLE_DIR / f"samples_epoch_{epoch}.png")
    plt.close()
    
def count_parameters(model):
    """Utility function to count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ================= MAIN EXECUTION =================
def main(args):
    logger.info("Initializing training...")
    save_checkpoint = None
    #save_checkpoint = Config.WEIGHTS_DIR / "diffusion_v2.2.pth"
    # Initialize components
    dataset = DiffusionDataset(Config.DATA_DIR)
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=20)
    model = create_unet()
    scheduler = DDPMScheduler(
        num_train_timesteps=Config.NUM_TIMESTEPS,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon"
    )
    logger.info(f"Model: {model}")
    logger.info(f"Model parameters: {count_parameters(model)}")
    # Load checkpoint if specified
    if args.checkpoint or save_checkpoint:
        model.load_state_dict(torch.load(args.checkpoint))
        logger.info(f"Resuming from checkpoint: {args.checkpoint}")

    # Start training
    train(model, dataloader, scheduler, resume_epoch=args.resume_epoch)
    logger.info("Training completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train diffusion model')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint file')
    parser.add_argument('--resume-epoch', type=int, default=0, help='Epoch to resume from')
    args = parser.parse_args()
    main(args)
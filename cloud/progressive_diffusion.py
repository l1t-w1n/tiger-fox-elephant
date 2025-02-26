import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from diffusers import UNet2DModel, DDPMScheduler
import matplotlib.pyplot as plt
import logging
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# ================= CONFIG =================
class Config:
    # Path configurations
    DATA_DIR = Path("data/diffusion/fox")
    WEIGHTS_DIR = Path("weights/diffusion")
    HIST_DIR = Path("histories/diffusion")
    LOG_DIR = Path("logs")
    SAMPLE_DIR = Path("diffusion_samples/diffusion_samples.v2.4")
    
    # Progressive training parameters
    PROGRESSIVE_STAGES = [
        {'size': 64, 'epochs': 50, 'lr': 3e-4, 'batch_size': 330},
        {'size': 128, 'epochs': 50, 'lr': 1e-4, 'batch_size': 110},
        {'size': 224, 'epochs': 100, 'lr': 3e-5, 'batch_size': 30}
    ]
    
    # Model parameters
    NUM_TIMESTEPS = 1000
    SAMPLING_STEPS = 250
    PLOT_EVERY = 2
    SEED = 42
    GRAD_CLIP = 0.5
    device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize directories
for p in [Config.WEIGHTS_DIR, Config.HIST_DIR, Config.LOG_DIR, Config.SAMPLE_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# ================= MODEL =================
def create_unet(input_size):
    return UNet2DModel(
        sample_size=input_size,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 256, 512, 1024),
        down_block_types=(
            "DownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
        ),
        up_block_types=(
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
        ),
        attention_head_dim=8,
        norm_num_groups=32
    ).to(Config.device)

# ================= DATASET =================
class ProgressiveDataset(Dataset):
    def __init__(self, root_dir):
        self.image_paths = list(Path(root_dir).glob("*.*"))
        self.current_size = Config.PROGRESSIVE_STAGES[0]['size']
        
    def update_transform(self, new_size):
        self.current_size = new_size
        self.transform = transforms.Compose([
            transforms.Resize(new_size),
            transforms.RandomResizedCrop(new_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
        
    def __len__(self): 
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img), 0

# ================= TRAINING UTILITIES =================
def save_checkpoint(model, loss):
    weight_path = Config.WEIGHTS_DIR / f"diffusion_v2.4.pth"
    torch.save(model.state_dict(), weight_path)
    
    hist_path = Config.HIST_DIR / "diffusion_v2.4.npz"
    history = np.load(hist_path)["losses"].tolist() if hist_path.exists() else []
    history.append(loss)
    np.savez(hist_path, losses=history)
    logging.info(f"Saved checkpoint: {weight_path}")

def generate_samples(model, scheduler, stage_size, epoch):
    model.eval()
    samples = torch.randn((4, 3, stage_size, stage_size), device=Config.device)
    
    scheduler.set_timesteps(Config.SAMPLING_STEPS)
    with torch.inference_mode():
        for t in scheduler.timesteps:
            residual = model(samples, t).sample
            samples = scheduler.step(residual, t, samples).prev_sample
    
    fig, axs = plt.subplots(1, 4, figsize=(15, 3))
    for i, ax in enumerate(axs):
        ax.imshow((samples[i].cpu().permute(1,2,0) * 0.5 + 0.5).clip(0,1))
        ax.axis('off')
    plt.savefig(Config.SAMPLE_DIR / f"stage_{stage_size}_epoch_{epoch}.png")
    plt.close()

# ================= PROGRESSIVE TRAINING =================
def train_progressive():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler(Config.LOG_DIR / "diffusion_v2.4.log", mode="a"),
            logging.StreamHandler()
        ]
    )
    
    # Seed everything
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Config.SEED)
        
    # Create one dataset instance
    dataset = ProgressiveDataset(Config.DATA_DIR)
    
    # Create one UNet at the largest resolution 
    # (or pick the average/largest resolution you want):
    model = create_unet(input_size=224)
    
    # Single optimizer for the entire training
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    
    # (You may later adjust learning rate stage by stage if you wish.)
    
    # Single DDPM Scheduler
    scheduler = DDPMScheduler(
        num_train_timesteps=Config.NUM_TIMESTEPS,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon"
    )

    scaler = torch.amp.GradScaler(enabled=(Config.device=="cuda"))

    for stage_idx, stage_cfg in enumerate(Config.PROGRESSIVE_STAGES):
        current_size = stage_cfg['size']
        logging.info(f"\n=== Starting Stage {stage_idx+1} ({current_size}px) ===")

        # Update dataset transforms to produce images at `current_size`
        dataset.update_transform(current_size)
        
        # Create a DataLoader at this stage's batch_size
        dataloader = DataLoader(
            dataset, 
            batch_size=stage_cfg['batch_size'],
            shuffle=True, 
            num_workers=16
        )
        
        # Optionally set a stage‐specific learning rate
        for g in optimizer.param_groups:
            g['lr'] = stage_cfg['lr']

        # Train for the specified number of epochs at this size
        for epoch in range(1, stage_cfg['epochs']+1):
            model.train()
            epoch_loss = 0.0
            
            for images, _ in tqdm(dataloader, desc=f"Stage {stage_idx+1} Epoch {epoch}"):
                images = images.to(Config.device)
                noise = torch.randn_like(images)
                timesteps = torch.randint(
                    0, 
                    Config.NUM_TIMESTEPS, 
                    (images.size(0),), 
                    device=Config.device
                )
                
                with torch.amp.autocast(enabled=(Config.device=="cuda")):
                    noisy = scheduler.add_noise(images, noise, timesteps)
                    pred = model(noisy, timesteps).sample
                    loss = F.mse_loss(pred, noise)

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
                
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            logging.info(f"Stage {stage_idx+1} | Epoch {epoch}/{stage_cfg['epochs']} | Loss: {avg_loss:.4f}")
            
            # Save checkpoint or generate samples if you want
            save_checkpoint(model, avg_loss)
            if epoch % Config.PLOT_EVERY == 0:
                generate_samples(model, scheduler, current_size, epoch)

    logging.info("Progressive training completed!")


if __name__ == "__main__":
    train_progressive()
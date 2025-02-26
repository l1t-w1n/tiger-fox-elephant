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
        {'size': 64, 'epochs': 50, 'lr': 3e-4, 'batch_size': 64},
        {'size': 128, 'epochs': 50, 'lr': 1e-4, 'batch_size': 32},
        {'size': 224, 'epochs': 100, 'lr': 3e-5, 'batch_size': 16}
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

def transfer_weights(src_model, dest_model, new_size):
    """Transfer weights with spatial interpolation and layer matching"""
    try:
        # Get scale factor based on input sizes
        scale_factor = new_size / src_model.sample_size
        
        # Transfer initial convolution with interpolation
        with torch.no_grad():
            dest_model.conv_in.weight.data = F.interpolate(
                src_model.conv_in.weight.data,
                scale_factor=scale_factor,
                mode='bicubic',
                align_corners=False
            )
            dest_model.conv_in.bias.data.copy_(src_model.conv_in.bias.data)

        # Transfer compatible middle blocks
        dest_model.mid_block.load_state_dict(src_model.mid_block.state_dict())

        # Transfer down/up blocks with matching depth
        for src_block, dest_block in zip(src_model.down_blocks, dest_model.down_blocks):
            if isinstance(src_block, type(dest_block)):
                dest_block.load_state_dict(src_block.state_dict())

        for src_block, dest_block in zip(src_model.up_blocks, dest_model.up_blocks):
            if isinstance(src_block, type(dest_block)):
                dest_block.load_state_dict(src_block.state_dict())

        # Initialize new attention layers
        for name, param in dest_model.named_parameters():
            if 'attn' in name and param not in src_model.state_dict():
                if 'weight' in name:
                    torch.nn.init.normal_(param, mean=0.0, std=0.02)
                elif 'bias' in name:
                    torch.nn.init.constant_(param, 0.0)

    except Exception as e:
        logging.error(f"Weight transfer failed: {str(e)}")
        raise
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
        
    dataset = ProgressiveDataset(Config.DATA_DIR)
    previous_model = None
    
    for stage_idx, stage in enumerate(Config.PROGRESSIVE_STAGES):
        current_size = stage['size']
        dataset.update_transform(current_size)
        
        # Create model with weight transfer
        model = create_unet(current_size, previous_model)
        previous_model = model  # Store for next stage
        
        dataloader = DataLoader(
            dataset, 
            batch_size=stage['batch_size'],
            shuffle=True, 
            num_workers=16
        )
        
        optimizer = torch.optim.AdamW([
            {'params': [p for n,p in model.named_parameters() if 'attn' not in n], 'lr': stage['lr']},
            {'params': [p for n,p in model.named_parameters() if 'attn' in n], 'lr': stage['lr']*2}
        ])
        
        scheduler = DDPMScheduler(
            num_train_timesteps=Config.NUM_TIMESTEPS,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="epsilon"
        )
        
        scaler = torch.amp.GradScaler("cuda")
        logging.info(f"\n=== Starting Stage {stage_idx+1} ({current_size}px) ===")
        
        for epoch in range(1, stage['epochs']+1):
            model.train()
            epoch_loss = 0.0
            
            for batch in tqdm(dataloader, desc=f"Stage {stage_idx+1} Epoch {epoch}"):
                images = batch[0].to(Config.device)
                noise = torch.randn_like(images)
                timesteps = torch.randint(0, Config.NUM_TIMESTEPS, (images.size(0),), device=Config.device)
                
                with torch.amp.autocast("cuda"):
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
            logging.info(f"Stage {stage_idx+1} Epoch {epoch}/{stage['epochs']} Loss: {avg_loss:.4f}")
            
            save_checkpoint(model, avg_loss)
            if epoch % Config.PLOT_EVERY == 0:
                generate_samples(model, scheduler, current_size, epoch)

    logging.info("Progressive training completed!")

if __name__ == "__main__":
    train_progressive()
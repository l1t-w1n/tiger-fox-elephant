import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import cv2
from tqdm import tqdm
import sys
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import datetime

project_root = Path.cwd()
sys.path.append(str(project_root))

from config import Config
from dataset import SRDataset
from loss import Loss
from model import FECAN 

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = config.device
        
        # Initialize model
        self.model = FECAN(upscale_factor=config.scale_factor).to(self.device)
        
        # Loss and optimizer
        self.criterion = Loss(l1_weight=config.l1_weight, freq_weight=config.freq_weight)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr, betas=config.betas)
        self.scheduler = CosineAnnealingLR(
            self.optimizer, 
            T_max=config.max_iter, 
            eta_min=config.min_lr
        )
        self.scaler = torch.amp.GradScaler(self.device)
        
        # Datasets and loaders
        self.train_dataset = SRDataset(config.train_hr_path, scale=config.scale_factor, train=True)
        self.val_dataset = SRDataset(config.val_hr_path, scale=config.scale_factor, train=False)
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=1,  # Full image validation
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True
        )
        
        # TensorBoard
        log_dir = config.log_dir / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.writer = SummaryWriter(log_dir=log_dir)
        
        # Training state
        self.current_iter = 0
        self.best_psnr = 0.0

    def _log_images(self, lr, sr, hr, tag="train"):
        """Log image examples to TensorBoard"""
        # Convert tensors to numpy arrays
        lr_img = lr[0].cpu().detach().numpy().transpose(1, 2, 0)
        sr_img = sr[0].cpu().detach().numpy().transpose(1, 2, 0)
        hr_img = hr[0].cpu().detach().numpy().transpose(1, 2, 0)
        
        # Denormalize if needed (assuming input is [0,1])
        self.writer.add_images(f"{tag}/LR", [lr_img], self.current_iter)
        self.writer.add_images(f"{tag}/SR", [sr_img], self.current_iter)
        self.writer.add_images(f"{tag}/HR", [hr_img], self.current_iter)

    def _calculate_psnr_ssim(self, sr, hr):
        """Calculate PSNR and SSIM on Y channel"""
        sr_y = self._rgb_to_y(sr)
        hr_y = self._rgb_to_y(hr)
        
        psnr = peak_signal_noise_ratio(hr_y, sr_y, data_range=1.0)
        ssim = structural_similarity(hr_y, sr_y, data_range=1.0)
        return psnr, ssim

    def _rgb_to_y(self, img):
        """Convert RGB tensor to Y channel (numpy)"""
        img_np = img.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        ycbcr = cv2.cvtColor(img_np, cv2.COLOR_RGB2YCrCb)
        return ycbcr[:, :, 0].clip(0, 1)

    def _validate(self, pbar=None):
        """Run validation with optional progress bar"""
        self.model.eval()
        total_psnr = 0.0
        total_ssim = 0.0
        
        with torch.inference_mode():
            for idx, (lr, hr) in enumerate(self.val_loader):
                lr = lr.to(self.device)
                hr = hr.to(self.device)
                
                with torch.amp.autocast("cuda"):
                    sr = self.model(lr)
                
                # Calculate metrics
                psnr, ssim = self._calculate_psnr_ssim(sr, hr)
                total_psnr += psnr
                total_ssim += ssim
                
                # Update validation progress bar if provided
                if pbar:
                    pbar.update(1)
                    pbar.set_postfix({
                        'current_psnr': f"{psnr:.2f}",
                        'current_ssim': f"{ssim:.4f}"
                    })

        avg_psnr = total_psnr / len(self.val_loader)
        avg_ssim = total_ssim / len(self.val_loader)
        
        # Log validation metrics
        self.writer.add_scalar("Validation/PSNR", avg_psnr, self.current_iter)
        self.writer.add_scalar("Validation/SSIM", avg_ssim, self.current_iter)
        
        # Log images
        self._log_images(lr, sr, hr, tag="val")
        
        # Save best model
        if avg_psnr > self.best_psnr:
            self.best_psnr = avg_psnr
            self._save_checkpoint(best=True)
        
        self.model.train()
        return avg_psnr, avg_ssim

    def _save_checkpoint(self, best=False):
        """Save model checkpoint"""
        state = {
            "iter": self.current_iter,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "best_psnr": self.best_psnr,
        }
        
        filename = f"checkpoint_{self.current_iter:07d}.pth"
        if best:
            filename = "best.pth"
            
        save_path = project_root / "FECAN" / "checkpoints"
        torch.save(state, save_path)
        print(f"Saved checkpoint to {save_path}")

    def train(self):
        self.model.train()
        
        # Create main epoch progress bar
        epoch_pbar = tqdm(
            range(self.config.num_epochs),
            desc="[Training Progress]",
            unit="epoch",
            dynamic_ncols=True,
            postfix={
                'loss': 'N/A', 
                'lr': 'N/A', 
                'psnr': 'N/A', 
                'ssim': 'N/A',
                'epoch': '0/{}'.format(self.config.num_epochs)
            }
        )

        try:
            for epoch in epoch_pbar:
                # Update epoch counter in progress bar
                epoch_pbar.postfix['epoch'] = f'{epoch+1}/{self.config.num_epochs}'
                
                # Create batch progress bar
                batch_pbar = tqdm(
                    self.train_loader,
                    desc=f"Epoch {epoch+1}/{self.config.num_epochs}",
                    unit="batch",
                    leave=False,
                    dynamic_ncols=True,
                    postfix={'batch_loss': 'N/A'}
                )

                epoch_loss = 0.0
                processed_batches = 0

                for lr, hr in batch_pbar:
                    # Training step
                    lr = lr.to(self.device)
                    hr = hr.to(self.device)

                    # Forward pass with mixed precision
                    with torch.amp.autocast("cuda"):
                        sr = self.model(lr)
                        loss = self.criterion(sr, hr)

                    # Backward pass
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()

                    # Update metrics
                    batch_loss = loss.item()
                    epoch_loss += batch_loss
                    processed_batches += 1

                    # Update batch progress
                    batch_pbar.set_postfix({
                        'batch_loss': f"{batch_loss:.4f}",
                        'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
                    })

                    # Log training metrics
                    if processed_batches % self.config.log_interval == 0:
                        self.writer.add_scalar(
                            "Loss/train_batch", 
                            batch_loss, 
                            epoch * len(self.train_loader) + processed_batches
                        )

                batch_pbar.close()

                # Calculate epoch metrics
                avg_epoch_loss = epoch_loss / len(self.train_loader)
                self.writer.add_scalar("Loss/train_epoch", avg_epoch_loss, epoch+1)

                # Run validation
                val_pbar = tqdm(
                    self.val_loader,
                    desc=f"Validating Epoch {epoch+1}",
                    leave=False,
                    unit="img",
                    dynamic_ncols=True
                )
                avg_psnr, avg_ssim = self._validate(val_pbar)
                val_pbar.close()

                # Update epoch progress bar
                epoch_pbar.set_postfix({
                    'loss': f"{avg_epoch_loss:.4f}",
                    'lr': f"{self.scheduler.get_last_lr()[0]:.2e}",
                    'psnr': f"{avg_psnr:.2f}",
                    'ssim': f"{avg_ssim:.4f}"
                })

                # Save checkpoint
                if (epoch + 1) % self.config.save_interval == 0 or (epoch + 1) == self.config.num_epochs:
                    self._save_checkpoint()

                # Early stopping check
                if self.current_iter >= self.config.max_iter:
                    break

        except KeyboardInterrupt:
            tqdm.write("\nTraining interrupted! Saving current state...")
        finally:
            epoch_pbar.close()
            self.writer.close()
            
if __name__ == "__main__":
    # Create checkpoints directory
    (project_root / "FECAN" / "checkpoints").mkdir(exist_ok=True)
    
    # Initialize and run training
    config = Config()
    trainer = Trainer(config)
    trainer.train()
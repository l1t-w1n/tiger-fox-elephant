import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
import cv2
from tqdm import tqdm
import numpy as np
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
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=config.lr,
            total_steps=config.max_iter,
            pct_start=0.05,
            anneal_strategy='cos',
            cycle_momentum=False,
            div_factor=config.div_factor,
            final_div_factor=config.final_div_factor
        )
        self.scaler = torch.amp.GradScaler(device=self.device)
        
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
        self.current_iter = 1
        self.best_psnr = 0.0

    def _log_images(self, lr, sr, hr, tag="train"):
        """
        Log image examples to TensorBoard.
        Assumes `lr[0]`, `sr[0]`, and `hr[0]` each have shape (C,H,W) in [0,1].
        """
        # Convert tensors to NumPy arrays and move channels last => (H,W,C)
        lr_img = lr[0].cpu().detach().numpy().transpose(1, 2, 0)
        sr_img = sr[0].cpu().detach().numpy().transpose(1, 2, 0)
        hr_img = hr[0].cpu().detach().numpy().transpose(1, 2, 0)

        # Now transpose back to (C,H,W)
        lr_img = lr_img.transpose(2, 0, 1)  # (3,H,W)
        sr_img = sr_img.transpose(2, 0, 1)  # (3,H,W)
        hr_img = hr_img.transpose(2, 0, 1)  # (3,H,W)

        # Expand dimensions so we have (N,C,H,W) with N=1
        lr_img = np.expand_dims(lr_img, axis=0)
        sr_img = np.expand_dims(sr_img, axis=0)
        hr_img = np.expand_dims(hr_img, axis=0)

        # Finally, log them with dataformats="NCHW" so TB knows how to interpret it
        self.writer.add_images(f"{tag}/LR", lr_img, self.current_iter, dataformats='NCHW')
        self.writer.add_images(f"{tag}/SR", sr_img, self.current_iter, dataformats='NCHW')
        self.writer.add_images(f"{tag}/HR", hr_img, self.current_iter, dataformats='NCHW')


    def _calculate_psnr_ssim(self, sr, hr):
        """Calculate PSNR and SSIM on Y channel"""
        sr_y = self._rgb_to_y(sr)
        hr_y = self._rgb_to_y(hr)
        
        psnr = peak_signal_noise_ratio(hr_y, sr_y, data_range=255)
        ssim = structural_similarity(hr_y, sr_y, data_range=255)
        return psnr, ssim

    def _rgb_to_y(self, img):
        """Convert RGB tensor to Y channel (0-255 range)"""
        img_np = img.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        img_np = (img_np * 255).clip(0, 255).astype(np.uint8)  # Denormalize
        ycbcr = cv2.cvtColor(img_np, cv2.COLOR_RGB2YCrCb)
        return ycbcr[:, :, 0].astype(np.float32)  # Keep as float32 for calculations

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

    def _save_checkpoint(self, best=False, final=False):
        """Save model checkpoint"""
        state = {
            "iter": self.current_iter,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "best_psnr": self.best_psnr,
        }
        
        filename = f"checkpoint.pth"
        if best:
            filename = "best.pth"
        elif final:
            filename = f"final.pth"
            
        save_path = project_root / "FECAN" / "checkpoints"
        (save_path).mkdir(exist_ok=True)
        
        torch.save(state, save_path / filename)
        print(f"Saved checkpoint to {save_path / filename}")
    
    def _count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def train(self):
        self.model.train()
        print(f"The model has {self._count_parameters()} trainable parameters")
        
        # Create main progress bar
        main_pbar = tqdm(
            total=self.config.max_iter,
            desc="[Training]",
            unit="iter",
            dynamic_ncols=True,
            postfix={
                'loss': 'N/A', 
                'lr': 'N/A', 
                'psnr': 'N/A', 
                'ssim': 'N/A'
            }
        )

        # Create data loader iterator with restart capability
        data_iter = iter(self.train_loader)
        
        # Gradient accumulation
        grad_accum_steps = self.config.grad_accumulation
        accum_loss = 0.0
        
        try:
            while self.current_iter < self.config.max_iter + 1:
                try:
                    lr, hr = next(data_iter)
                except StopIteration:
                    # Restart the iterator when exhausted
                    data_iter = iter(self.train_loader)
                    lr, hr = next(data_iter)

                # Training step
                lr = lr.to(self.device, non_blocking=True)
                hr = hr.to(self.device, non_blocking=True)

                # Forward pass with mixed precision
                with torch.amp.autocast(device_type=config.device, dtype=torch.float16):
                    sr = self.model(lr)
                    loss = self.criterion(sr, hr) / grad_accum_steps

                # Backward pass with gradient accumulation
                self.scaler.scale(loss).backward()
                accum_loss += loss.item()

                # Update weights if accumulation steps completed
                if (self.current_iter + 1) % grad_accum_steps == 0:
                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    
                    # Optimizer step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    
                    # Learning rate scheduling
                    self.scheduler.step()
                    
                    # Update main progress bar
                    main_pbar.set_postfix({
                        'loss': f"{accum_loss:.4f}",
                        'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
                    })

                    accum_loss = 0.0

                    # Validation and logging
                    self.writer.add_scalar("Loss/train", loss.item() * grad_accum_steps, self.current_iter)
                    self.writer.add_scalar("LR", self.scheduler.get_last_lr()[0], self.current_iter)
                    
                if self.current_iter % self.config.save_interval == 0:
                    # Create validation progress bar
                    val_pbar = tqdm(
                        total=len(self.val_loader),
                        desc=f"[Validation @ iter {self.current_iter}]",
                        unit="img",
                        position=1,
                        leave=False,
                        dynamic_ncols=True
                    )
                    
                    avg_psnr, avg_ssim = self._validate(val_pbar)
                    val_pbar.close()
                    
                    # Update main progress bar
                    main_pbar.set_postfix({
                        'loss': main_pbar.postfix['loss'],
                        'lr': f"{self.scheduler.get_last_lr()[0]:.2e}",
                        'psnr': f"{avg_psnr:.2f}",
                        'ssim': f"{avg_ssim:.4f}"
                    })

                # Save checkpoint
                if self.current_iter % self.config.save_interval == 0:
                    self._save_checkpoint()

                # Update iteration counter
                self.current_iter += 1
                main_pbar.update(1)

                # Early stopping
                if self.current_iter >= self.config.max_iter:
                    break

        except KeyboardInterrupt:
            tqdm.write("\nTraining interrupted! Saving final state...")
        finally:
            main_pbar.close()
            self.writer.close()
            self._save_checkpoint(final=True)
            tqdm.write(f"Training completed. Best PSNR: {self.best_psnr:.2f} dB")
            
if __name__ == "__main__":   
    # Initialize and run training
    config = Config()
    trainer = Trainer(config)
    trainer.train()
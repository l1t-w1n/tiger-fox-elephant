import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
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
        self.num_val_images = config.num_val_images
        # Initialize model
        self.model = FECAN(upscale_factor=config.scale_factor).to(self.device)
        
        # Loss and optimizer
        self.criterion = Loss(l1_weight=config.l1_weight, freq_weight=config.freq_weight)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.lr,
            betas=config.betas
        )
        
        self.scaler = torch.amp.GradScaler(device=self.device)
        
        # Datasets and loaders
        self.train_dataset = SRDataset(config.train_hr_path_div2k, scale=config.scale_factor, train=True)
        self.val_dataset = SRDataset(config.val_hr_path_div2k, scale=config.scale_factor, train=False)
        
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
        self.current_epoch = 0
        self.global_step = 0  # Counts total iterations across epochs
        self.best_psnr = 0.0
        total_steps = config.num_epochs * len(self.train_dataset) // config.batch_size
        
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=config.min_lr
        )


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
        self.writer.add_images(f"{tag}/LR", lr_img, self.global_step, dataformats='NCHW')
        self.writer.add_images(f"{tag}/SR", sr_img, self.global_step, dataformats='NCHW')
        self.writer.add_images(f"{tag}/HR", hr_img, self.global_step, dataformats='NCHW')

    def _calculate_psnr_ssim(self, sr, hr):
        """Calculate PSNR and SSIM on Y channel"""
        sr_y = self._rgb_to_y(sr)
        hr_y = self._rgb_to_y(hr)
        
        psnr = peak_signal_noise_ratio(hr_y, sr_y, data_range=255)
        ssim = structural_similarity(hr_y, sr_y, data_range=255)
        return psnr, ssim

    def _rgb_to_y(self, img):
        """Convert RGB tensor to Y channel (0-255 range)"""
        img = (img * 0.5) + 0.5  # Denormalize to [0,1]
        img_np = img.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        img_np = (img_np * 255).clip(0, 255).astype(np.uint8)  # Denormalize
        ycbcr = cv2.cvtColor(img_np, cv2.COLOR_RGB2YCrCb)
        return ycbcr[:, :, 0].astype(np.float32)  # Keep as float32 for calculations

    def _validate(self):
        """Run validation on a subset of validation images."""
        self.model.eval()
        total_psnr = 0.0
        total_ssim = 0.0
        num_val_images = self.config.num_val_images
        
        total_val_batches = min(num_val_images, len(self.val_loader))
        if total_val_batches == 0:
            return 0.0, 0.0

        val_pbar = tqdm(total=total_val_batches, desc=f"[Validation]", leave=False)
        
        # Use first batch for logging example images (optional)
        example_logged = False

        with torch.inference_mode():
            for i, (lr, hr) in enumerate(self.val_loader):
                if i >= num_val_images:
                    break
                    
                lr = lr.to(self.device)
                hr = hr.to(self.device)
                
                with torch.amp.autocast("cuda"):
                    sr = self.model(lr)
                
                psnr, ssim = self._calculate_psnr_ssim(sr, hr)
                total_psnr += psnr
                total_ssim += ssim
                
                # Log example images from the first batch only (optional)
                if not example_logged:
                    self._log_images(lr, sr, hr, tag="val")
                    example_logged = True
                
                val_pbar.update(1)
                val_pbar.set_postfix({'psnr': f"{psnr:.2f}", 'ssim': f"{ssim:.4f}"})

        val_pbar.close()

        avg_psnr = float(total_psnr / total_val_batches)
        avg_ssim = float(total_ssim / total_val_batches)

        # Update best metric with native float
        if avg_psnr > self.best_psnr:
            self.best_psnr = avg_psnr

        # Log validation metrics to TensorBoard
        self.writer.add_scalar("PSNR/val", avg_psnr, self.global_step)
        self.writer.add_scalar("SSIM/val", avg_ssim, self.global_step)

        return avg_psnr, avg_ssim

    
    def _save_checkpoint(self, best=False, final=False):
        """Save model checkpoint."""
        state = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "best_psnr": self.best_psnr,
        }
        
        filename = "checkpoint.pth"
        if best:
            filename = "best.pth"
        elif final:
            filename = "final.pth"
            
        save_path = project_root / "FECAN" / "checkpoints"
        save_path.mkdir(exist_ok=True)
        
        torch.save(state, save_path / filename)
        print(f"Saved checkpoint to {save_path / filename}")
    
    def load_checkpoint(self, checkpoint_path):
        """
        Load a checkpoint to resume training.
        
        Args:
            checkpoint_path (str or Path): Path to the checkpoint file.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        # Load with weights_only=True and allow necessary globals
        with torch.serialization.safe_globals([np._core.multiarray.scalar]):
            checkpoint = torch.load(
                checkpoint_path,
                map_location=self.device,
                weights_only=True
            )

        # Restore training state
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.best_psnr = float(checkpoint["best_psnr"])  # Ensure float conversion
        
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]

        print(
            f"Checkpoint loaded from '{checkpoint_path}': "
            f"epoch={self.current_epoch}, best_psnr={self.best_psnr:.2f}"
        )

    def _count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def train(self):
        print(f"The model has {self._count_parameters()} trainable parameters")       

        # Loop over epochs
        for epoch in range(self.current_epoch + 1, self.config.num_epochs + 1):
            self.current_epoch = epoch

            # Create an epoch-level progress bar
            epoch_pbar = tqdm(
                enumerate(self.train_loader, start=1),
                total=len(self.train_loader),
                desc=f"Epoch [{epoch}/{self.config.num_epochs}]",
                dynamic_ncols=True
            )
            
            # Training phase
            self.model.train()
            for batch_idx, (lr, hr) in epoch_pbar:
                lr = lr.to(self.device, non_blocking=True)
                hr = hr.to(self.device, non_blocking=True)

                with torch.amp.autocast(device_type=self.config.device, dtype=torch.float16):
                    sr = self.model(lr)
                    loss = self.criterion(sr, hr)

                # Backward
                self.scaler.scale(loss).backward()

                # Gradient clipping
                #self.scaler.unscale_(self.optimizer)
                #torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # Step optimizer
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                
                # LR scheduler step (if stepping each iteration)
                self.scheduler.step()

                # Update progress bar
                epoch_pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"
                })

                # TensorBoard logging
                self.global_step += 1
                self.writer.add_scalar("Loss/train", loss.item(), self.global_step)
                self.writer.add_scalar("LR", self.scheduler.get_last_lr()[0], self.global_step)
            
            # --- End of one epoch ---
            epoch_pbar.close()
            
            # Validate after each epoch
            avg_psnr, avg_ssim = self._validate()
            
            # Log or print epoch-end info
            tqdm.write(
                f"Epoch {epoch}/{self.config.num_epochs} - "
                f"PSNR: {avg_psnr:.2f} dB, SSIM: {avg_ssim:.4f}, Best PSNR: {self.best_psnr:.2f} dB"
            )

            # (Optional) save checkpoint each epoch (or every N epochs)
            if epoch % self.config.save_interval == 0:
                self._save_checkpoint()

        # End of all epochs
        self._save_checkpoint(final=True)
        self.writer.close()
        print(f"Training completed. Best PSNR: {self.best_psnr:.2f} dB")


if __name__ == "__main__":   
    config = Config()
    trainer = Trainer(config)

    trainer.load_checkpoint(project_root / "FECAN/checkpoints/checkpoint.pth")

    trainer.train()

"""
Classical FECAN (4×) training – exact protocol of
Huang et al., Sci Rep 15 (2025).

• 1.6 M iterations, cosine LR 5e-4 → 1e-7
• Adam(β1 0.9, β2 0.99), mixed precision
• HR patch 192×192 → LR 48×48, batch 32
• FFT-enhanced L1 loss (weights 1.0 + 0.05)
• checkpoint every 5 k + when PSNR improves
"""

from pathlib import Path
import datetime
import cv2
import numpy as np
import random
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm

from config import Config
from dataset import SRDataset
from loss import Loss
from model import FECAN


# ─────────────────── utility ─────────────────── #

def rgb_to_y(img_tensor):
    """RGB (-1..1) → Y channel 0-255 float32"""
    img = (img_tensor * 0.5) + 0.5           # → 0..1
    img = (img.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255
           ).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)[:, :, 0].astype(np.float32)


# ─────────────────── Trainer ─────────────────── #

class Trainer:
    def __init__(self, cfg: Config):
        self.cfg       = cfg
        self.step      = 0
        self.best_psnr = 0.0
        self.device    = torch.device(cfg.device)

        # ─ model ─
        self.model = FECAN(upscale_factor=cfg.scale_factor).to(self.device)

        # ─ loss ─
        self.crit = Loss(cfg.l1_weight, cfg.freq_weight)

        # ─ optim/sched ─
        self.opt    = optim.Adam(self.model.parameters(), lr=cfg.lr, betas=cfg.betas)
        self.sched  = CosineAnnealingLR(self.opt,
                                        T_max=cfg.total_iters,
                                        eta_min=cfg.min_lr)
        self.scaler = torch.amp.GradScaler("cuda")

        # ───── dataset split (single folder) ─────
        full_ds = SRDataset(cfg.hr_path, scale=cfg.scale_factor, train=True)
        all_idx = list(range(len(full_ds)))
        random.shuffle(all_idx)

        val_idx   = all_idx[:cfg.val_subset]    # fixed 100-image subset
        train_idx = all_idx[cfg.val_subset:]

        self.train_loader = DataLoader(
            Subset(full_ds, train_idx),
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
            drop_last=True
        )

        # validation dataset uses *full-image* transform
        val_ds_full = SRDataset(cfg.hr_path, scale=cfg.scale_factor, train=False)
        self.val_loader = DataLoader(
            Subset(val_ds_full, val_idx),
            batch_size=1,
            shuffle=False,
            num_workers=2
        )

        # ─ logging ─
        self.tb = SummaryWriter(cfg.log_dir / datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

        self.ckpt_dir = Path("FECAN/checkpoints")
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        print(f"trainable params: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

    # ──────────────── helper: TensorBoard images ──────────────── #
    def _log_images(self, lr, sr, hr):
        lr = (lr.clamp(-1, 1) + 1) / 2
        sr = (sr.clamp(-1, 1) + 1) / 2
        hr = (hr.clamp(-1, 1) + 1) / 2
        self.tb.add_images("LR", lr, self.step)
        self.tb.add_images("SR", sr, self.step)
        self.tb.add_images("HR", hr, self.step)

    # ──────────────── validation ──────────────── #
    @torch.inference_mode()
    def _validate(self):
        self.model.eval()
        psnr_tot = ssim_tot = 0.0
        for lr, hr in self.val_loader:
            lr, hr = lr.to(self.device), hr.to(self.device)
            with torch.amp.autocast("cuda"):
                sr = self.model(lr)
            psnr, ssim = self._metrics(sr, hr)
            psnr_tot += psnr
            ssim_tot += ssim
        n = len(self.val_loader)
        return float(psnr_tot / n), float(ssim_tot / n)

    @staticmethod
    def _metrics(sr, hr):
        sr_y = rgb_to_y(sr)
        hr_y = rgb_to_y(hr)
        psnr = peak_signal_noise_ratio(hr_y, sr_y, data_range=255)
        ssim = structural_similarity(hr_y, sr_y, data_range=255)
        return psnr, ssim

    # ─────────────── training loop ─────────────── #
    def train(self):
        cfg    = self.cfg
        loader = iter(self.train_loader)

        pbar = tqdm(
            total=cfg.total_iters,
            initial=self.step,              # resume progress bar
            dynamic_ncols=True,
            colour="cyan"
        )
        while self.step < cfg.total_iters:
            try:
                lr, hr = next(loader)
            except StopIteration:
                loader = iter(self.train_loader)
                lr, hr = next(loader)

            lr, hr = lr.to(self.device, non_blocking=True), hr.to(self.device, non_blocking=True)

            self.model.train()
            with torch.amp.autocast("cuda"):
                sr   = self.model(lr)
                loss = self.crit(sr, hr)

            # AMP + correct update order
            self.scaler.scale(loss).backward()
            self.scaler.step(self.opt)                     # optimizer step
            self.opt.zero_grad(set_to_none=True)           # clear grads
            self.scaler.update()                           # scaler book-keeping
            self.sched.step()                              # LR scheduler

            # ─ log scalars ─
            self.step += 1
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{self.sched.get_last_lr()[0]:.2e}")

            self.tb.add_scalar("Loss/train", loss.item(), self.step)
            self.tb.add_scalar("LR", self.sched.get_last_lr()[0], self.step)

            # ─ log images & hists ─
            if self.step % cfg.tb_img_every == 0:
                self._log_images(lr, sr, hr)

            if (self.step % cfg.tb_hist_every == 0) and loss.requires_grad:
                for name, p in self.model.named_parameters():
                    if p.grad is not None:
                        self.tb.add_histogram(f"grads/{name}", p.grad, self.step)
                        self.tb.add_histogram(f"weights/{name}", p.data, self.step)

            # ─ checkpoint / val ─
            if self.step % cfg.ckpt_every == 0 or self.step == cfg.total_iters:
                psnr, ssim = self._validate()
                self.tb.add_scalar("PSNR/val", psnr, self.step)
                self.tb.add_scalar("SSIM/val", ssim, self.step)

                is_best = False
                if psnr > self.best_psnr:
                    self.best_psnr = psnr
                    is_best = True

                self._save_ckpt(best=is_best)
                tqdm.write(f"[{self.step:>7}]  PSNR {psnr:.2f}  SSIM {ssim:.4f}  "
                           f"{'[BEST]' if is_best else ''}")

        pbar.close()
        self.tb.close()
        print(f"Training complete – best PSNR: {self.best_psnr:.2f} dB")

    # ─────────────── checkpoint ─────────────── #
    def _save_ckpt(self, *, best=False):
        state = {
            "step":      self.step,
            "best_psnr": self.best_psnr,
            "model":     self.model.state_dict(),
            "opt":       self.opt.state_dict(),
            "sched":     self.sched.state_dict(),
            "scaler":    self.scaler.state_dict(),
        }
        torch.save(state, self.ckpt_dir / f"iter_{self.step:07}.pth")
        if best:
            torch.save(state, self.ckpt_dir / "best.pth")

    # ─────────────── resume from checkpoint ─────────────── #
    def load_checkpoint(self, ckpt_path: str):
        """
        Load training state from a checkpoint and resume.
        """
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.step      = ckpt['step']
        self.best_psnr = ckpt['best_psnr']
        self.model.load_state_dict( ckpt['model'] )
        self.opt.load_state_dict(   ckpt['opt']   )
        self.sched.load_state_dict( ckpt['sched'] )
        self.scaler.load_state_dict(ckpt['scaler'])
        print(f"Loaded checkpoint '{ckpt_path}' at step {self.step} (best PSNR={self.best_psnr:.2f})")


# ──────────────────── entry ──────────────────── #

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    trainer = Trainer(Config())

    # ← path to your last checkpoint (update as needed)
    ckpt_path = "FECAN/checkpoints/iter_0010000.pth"
    if Path(ckpt_path).exists():
        trainer.load_checkpoint(ckpt_path)

    trainer.train()

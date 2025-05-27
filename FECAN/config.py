import torch
from pathlib import Path
import sys
import os
import random
project_root = Path.cwd()
sys.path.append(str(project_root))

class Config:
    # Dataset 
    train_cat = project_root / "data/cat_face_512/train"
    val_cat = project_root / "data/cat_face_512/val"
    
    data = project_root / "data/flickr+div_hr"
    Path(data).mkdir(parents=True, exist_ok=True) 
    
    val_subset = 100          # how many images to hold out for PSNR/SSIM
    random.seed(42)           # reproducible split
    
    batch_size = 32
    num_workers = os.cpu_count() 
    
    # Training
    scale_factor = 4
    lr = 5e-4
    min_lr = 1e-7
    betas = (0.9, 0.99)
    total_iters = 1_600_000
    ckpt_every = 2_000
          
    tb_img_every   = 100   # log LR/SR/HR image triplet every N steps
    tb_hist_every  = 5_000   # log weight / grad histograms every N steps
    
    # Loss weights
    l1_weight = 1.0
    freq_weight = 0.05
    
    # Augmentation
    patch_size = 192
    rot_angles = [0, 90, 180, 270]
    hflip_prob = 0.5
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    log_dir = project_root / "FECAN/logs"
    Path(log_dir).mkdir(parents=True, exist_ok=True)

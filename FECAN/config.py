import torch
from pathlib import Path
import sys
project_root = Path.cwd()
sys.path.append(str(project_root))

class Config:
    # Dataset 
    train_hr_path = project_root / "data/cat_face_512/train"
    val_hr_path = project_root / "data/cat_face_512/val"
    Path(train_hr_path).mkdir(parents=True, exist_ok=True)
    Path(val_hr_path).mkdir(parents=True, exist_ok=True)
    
    batch_size = 20
    grad_accumulation = 1
    num_workers = 16
    
    # Training
    scale_factor = 4
    lr = 5e-4
    div_factor = 10
    final_div_factor = 50
    betas = (0.9, 0.99)
    max_iter = 2000
    save_interval = 100
    
    # Loss weights
    l1_weight = 1.0
    freq_weight = 0.05
    
    # Augmentation
    rot_angles = [0, 90, 180, 270]
    hflip_prob = 0.5
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    log_dir = project_root / "FECAN/logs"
    Path(log_dir).mkdir(parents=True, exist_ok=True)

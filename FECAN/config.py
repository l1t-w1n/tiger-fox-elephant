import torch
from pathlib import Path
import sys
project_root = Path.cwd()
sys.path.append(str(project_root))

class Config:
    # Dataset 
    train_hr_path_cat = project_root / "data/cat_face_512/train"
    val_hr_path_cat = project_root / "data/cat_face_512/val"
    train_hr_path_div2k = project_root / "data/div2k/DIV2K_train_HR/DIV2K_train_HR"
    val_hr_path_div2k = project_root / "data/div2k/DIV2K_valid_HR/DIV2K_valid_HR"
    train_hr_path_flickr2k = project_root / "data/flickr2k/train"
    val_hr_path_flickr2k = project_root / "data/flickr2k/val"
    
    
    Path(train_hr_path_cat).mkdir(parents=True, exist_ok=True)
    Path(val_hr_path_cat).mkdir(parents=True, exist_ok=True)
    Path(train_hr_path_div2k).mkdir(parents=True, exist_ok=True)
    Path(val_hr_path_div2k).mkdir(parents=True, exist_ok=True)
    Path(train_hr_path_flickr2k).mkdir(parents=True, exist_ok=True)
    Path(val_hr_path_flickr2k).mkdir(parents=True, exist_ok=True)
    
    batch_size = 32
    num_workers = 20
    num_val_images = 2
    
    # Training
    scale_factor = 4
    lr = 1e-4
    min_lr = 1e-7
    betas = (0.9, 0.99)
    num_epochs = 100
    save_interval = 5
    
    # Loss weights
    l1_weight = 1.0
    freq_weight = 0.1
    
    # Augmentation
    patch_size = 128
    rot_angles = [0, 90, 180, 270]
    hflip_prob = 0.5
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    log_dir = project_root / "FECAN/logs"
    Path(log_dir).mkdir(parents=True, exist_ok=True)

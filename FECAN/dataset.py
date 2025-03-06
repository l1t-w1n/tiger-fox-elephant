import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image
from pathlib import Path
import sys
project_root = Path.cwd()
sys.path.append(str(project_root))
from config import Config

class SRDataset(Dataset):
    def __init__(self, hr_path, scale=4, train=True):
        self.hr_images = self.load_images(hr_path)
        self.scale = scale
        self.train = train
        
        # Transformations
        self.train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=Config.hflip_prob),
            transforms.RandomChoice([transforms.RandomRotation((angle, angle)) for angle in Config.rot_angles]),
            transforms.ToTensor()
        ])
        self.val_transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def load_images(self, path):
        return [os.path.join(path, f) for f in os.listdir(path) 
               if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    def __len__(self):
        return len(self.hr_images)
    
    def __getitem__(self, idx):
        # Load full HR image
        hr_img = Image.open(self.hr_images[idx]).convert('RGB')
        
        # Apply augmentations
        if self.train:
            hr_img = self.train_transform(hr_img)
        else:
            hr_img = self.val_transform(hr_img)
        
        # Generate LR from full HR image
        lr = F.interpolate(
            hr_img.unsqueeze(0),
            scale_factor=1/self.scale,
            mode='bicubic',
            align_corners=False
        ).squeeze(0)
        
        return lr, hr_img

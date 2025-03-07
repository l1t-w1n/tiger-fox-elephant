import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as T
import os
from PIL import Image
from pathlib import Path
import sys
project_root = Path.cwd()
sys.path.append(str(project_root))
from config import Config


class SRDataset(Dataset):
    """
    Patch-based dataset for Super-Resolution.

    - During training, we:
      (1) Randomly crop a patch_size×patch_size region from the HR image,
      (2) Randomly rotate the patch by [0°, 90°, 180°, or 270°],
      (3) Randomly flip it horizontally (p=0.5),
      (4) Convert to tensor & normalize to [-1,1],
      (5) Downsample that patch by 'scale' to get LR.

    - During validation, we keep the entire image (no crop/rot/flip),
      just convert to tensor & normalize, then downsample.
    """

    def __init__(self, hr_path, scale=4, patch_size=Config.patch_size, train=True, hflip_prob=0.5):
        super().__init__()
        
        self.hr_folder   = hr_path
        self.scale       = scale
        self.patch_size  = patch_size
        self.train       = train
        self.hflip_prob  = hflip_prob

        self.hr_images = sorted([
            f for f in os.listdir(hr_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        
        # Discrete rotation among [0,90,180,270]:
        rotations = []
        for angle in Config.rot_angles:
            rotations.append(T.RandomRotation([angle, angle]))
            
        # Define the augmentation transforms
        if self.train:
            
                
            # For training: random crop, random rotation among {0,90,180,270}, random flip
            self.transform = T.Compose([
                T.RandomCrop((patch_size, patch_size)),
                T.RandomChoice(rotations),
                T.RandomHorizontalFlip(p=self.hflip_prob),                
                T.ToTensor(),
                T.Normalize(mean=(0.5, 0.5, 0.5),
                            std=(0.5, 0.5, 0.5))
            ])
        else:
            # For validation: no crop, no random rotation/flip
            self.transform = T.Compose([
                T.RandomCrop((patch_size, patch_size)),
                T.ToTensor(),
                T.Normalize(mean=(0.5, 0.5, 0.5),
                            std=(0.5, 0.5, 0.5))
            ])

    def __len__(self):
        return len(self.hr_images)

    def __getitem__(self, idx):
        # Load an HR image
        hr_img_path = os.path.join(self.hr_folder, self.hr_images[idx])
        hr_img = Image.open(hr_img_path).convert("RGB")

        # Apply transforms (random crop/rot/flip if train)
        # -> becomes a Tensor in [-1,1] of shape [C,H,W].
        hr_tensor = self.transform(hr_img)

        # If we are doing *validation* and want the entire image,
        # be mindful that T.RandomCrop doesn't exist in the val transform.
        # So hr_tensor might be the whole image. That's fine.

        # Downsample HR patch/image to LR using bicubic
        # Suppose the hr_tensor shape is [3, H, W].
        lr_tensor = F.interpolate(
            hr_tensor.unsqueeze(0),
            scale_factor=1.0 / self.scale,
            mode='bicubic',
            align_corners=False
        ).squeeze(0)

        return lr_tensor, hr_tensor


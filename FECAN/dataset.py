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
    Random-patch dataset for 4× super-resolution.
    """

    def __init__(self, hr_path, *, scale=4, patch_size=Config.patch_size,
                 train=True, hflip_prob=Config.hflip_prob):
        super().__init__()

        # defensive – make sure patch size fits scale
        assert patch_size % scale == 0, "patch_size must be divisible by scale_factor"

        self.hr_folder  = hr_path
        self.scale      = scale
        self.patch_size = patch_size
        self.train      = train

        # gather image list
        self.hr_images = sorted(
            f for f in os.listdir(hr_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        )

        # ───── transforms ───── #
        rotations = [T.RandomRotation([a, a]) for a in Config.rot_angles]

        if self.train:
            self.transform = T.Compose([
                T.RandomCrop((patch_size, patch_size)),
                T.RandomChoice(rotations),
                T.RandomHorizontalFlip(p=hflip_prob),
                T.ToTensor(),
                T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ])
        else:
            self.transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ])

    def __len__(self):
        return len(self.hr_images)

    def __getitem__(self, idx):
        hr_path = os.path.join(self.hr_folder, self.hr_images[idx])
        hr_img  = Image.open(hr_path).convert("RGB")

        hr = self.transform(hr_img)

        lr = F.interpolate(hr.unsqueeze(0),
                           scale_factor=1 / self.scale,
                           mode='bicubic',
                           align_corners=False).squeeze(0)
        return lr, hr

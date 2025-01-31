import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
from config.config import Config

class BinaryImageDataset(Dataset):
    """
    Custom Dataset for binary classification.
    Expects a directory structure:
        data_dir/
          positive/ -> images with label 1
          negative/ -> images with label 0
    """
    def __init__(self, data_dir, transform=None, logger=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.logger = logger or logging.getLogger(__name__)
        self.samples = []  # List of tuples: (image_path, label)
        
        # Load positive samples
        pos_dir = self.data_dir / "positive"
        if pos_dir.exists():
            for fname in os.listdir(pos_dir):
                if self._is_image_file(fname):
                    self.samples.append((pos_dir / fname, 1))
        else:
            logger.warning(f"Positive directory {pos_dir} does not exist.")
        
        # Load negative samples
        neg_dir = self.data_dir / "negative"
        if neg_dir.exists():
            for fname in os.listdir(neg_dir):
                if self._is_image_file(fname):
                    self.samples.append((neg_dir / fname, 0))
        else:
            logger.warning(f"Negative directory {neg_dir} does not exist.")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Image {img_path} could not be read.")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            self.logger.error(f"Error reading image {img_path}: {e}")
            img = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
        
        if self.transform:
            img = self.transform(img)
        
        return img, torch.tensor(label, dtype=torch.float32)
    
    def _is_image_file(self, filename):
        return filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))


class ImprovedBinaryCNN(nn.Module):
    def __init__(self, num_channels=3):
        super(ImprovedBinaryCNN, self).__init__()
        # Convolutional Block 1
        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.act1  = nn.LeakyReLU(0.1)
        
        # Convolutional Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.act2  = nn.LeakyReLU(0.1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 224 -> 112
        
        # Convolutional Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        self.act3  = nn.LeakyReLU(0.1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 112 -> 56
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # 56x56 -> 1x1
        
        # Fully Connected Layer with Dropout
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128, 1)  # 128 channels -> 1 logit

    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.act3(self.bn3(self.conv3(x)))
        x = self.pool2(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x  # raw logits

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return running_loss / total, correct / total
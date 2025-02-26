import os
import torch 
import requests
import logging
import pandas as pd
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import matplotlib.pyplot as plt
from datasets import load_dataset, load_from_disk
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms 

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ButterflyDataset(Dataset):
    def __init__(self, 
                 data_dir: str = '/home/l1t-w1n/tiger-fox-elephant/ddpm/src/data/butterflies_dataset', 
                 transform: transforms.Compose = None):
        
        self.data_dir = data_dir
        self.transform = transform
        
        if not os.path.exists(self.data_dir):
            self._load_dataset()
            
        csv_path = os.path.join(os.path.dirname(self.data_dir), 'butterflies.csv')
        if os.path.exists(csv_path):
            self.df = pd.read_csv(csv_path)
        else:
            self.df = self._clean_dataset()
            
        self.img_dir = os.path.join(self.data_dir, 'images')
        if not os.path.exists(self.img_dir) or len(os.listdir(self.img_dir)) == 0:
            self._download_images()

    def __len__(self):
        return len(self.df)
    
    # def __getitem__(self, idx):
    #     file_id = self.df.iloc[idx]['id'].split('/')[-1]
    #     id, name = self.df.iloc[idx]['id'], self.df.iloc[idx]['name']
    #     image_path = os.path.join(self.img_dir, f"{file_id}.jpg")
        
    #     image = Image.open(image_path)
    #     if self.transform:
    #         image = self.transform(image)
    #     return (image, name)
    
    def __getitem__(self, idx):
        file_id = self.df.iloc[idx]['id'].split('/')[-1]
        id = self.df.iloc[idx]['id']
        name = self.df.iloc[idx]['name']
        image_path = os.path.join(self.img_dir, f"{file_id}.jpg")
        
        # Загрузка изображения
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        return {
            "image": image,  # Tensor [C, H, W]
            "name": name,     # Текст (название бабочки)
            "id": id          # Идентификатор
        }

    def _load_dataset(self):
        """ Load dataset from Hugging Face """
        logger.info("Downloading dataset from Hugging Face...")
        dataset = load_dataset("huggan/smithsonian_butterflies_subset")
        os.makedirs(self.data_dir, exist_ok=True)
        dataset.save_to_disk(self.data_dir)
        logger.info(f"Dataset saved to {self.data_dir}")

    def _clean_dataset(self) -> pd.DataFrame:
        """ Data cleaning """
        logger.info("Cleaning metadata...")
        dataset = load_from_disk(self.data_dir)
        df = pd.DataFrame(dataset['train'])
        df = df[['image_url', 'id', 'name']]
        csv_path = os.path.join(os.path.dirname(self.data_dir), 'butterflies.csv')
        df.to_csv(csv_path, index=False)
        return df

    def _download_images(self):
        """ Download images """
        logger.info("Downloading images...")
        os.makedirs(self.img_dir, exist_ok=True)
        
        for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Downloading"):
            try:
                response = requests.get(row['image_url'], timeout=15)
                response.raise_for_status()
                
                image = Image.open(BytesIO(response.content))
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                file_id = row['id'].split('/')[-1]
                image.save(os.path.join(self.img_dir, f"{file_id}.jpg"))
                
            except Exception as e:
                logger.error(f"Error downloading {row['image_url']}: {str(e)}")

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((config.LATENT_HEIGHT, config.LATENT_WIDTH)),
        transforms.ToTensor()
    ])
    
    dataset = ButterflyDataset(transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Recieve 1 batch of data
    images, ids, names = next(iter(dataloader))
    
    # # Visualize first 4 samples
    # fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    # for i, ax in enumerate(axes.flat):
    #     ax.imshow(images[i].permute(1, 2, 0))
    #     ax.set_title(f"{names[i]}\n{ids[i]}")
    #     ax.axis('off')
    # plt.tight_layout()
    # plt.show()
    
    for batch_idx, batch in enumerate(dataloader):
        images = batch['image']
        ids = batch['id']
        names = batch['name']

        if batch_idx == 0:
            print(f"Image shape: {images.shape}")
            break

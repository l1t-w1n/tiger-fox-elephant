import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os
from datetime import datetime

import torch.nn.functional as F
import torchvision.transforms as transforms

project_root = Path.cwd()
sys.path.append(str(project_root))

from config.config import Config
from utils.helper_functions import create_X_y, save_model

class FoxDataset(Dataset):
    def __init__(self, X, transform=None):
        # Normalisation des données en les divisant par 255
        self.X = torch.FloatTensor(X).permute(0, 3, 1, 2) / 255.0
        self.transform = transform
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        image = self.X[idx]
        if self.transform:
            image = self.transform(image)
        return image, 0  # Label non utilisé pour le GAN

class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.init_size = Config.IMG_SIZE // 4
        self.init_channels = 256
        
        self.l1 = nn.Sequential(
            nn.Linear(latent_dim, self.init_channels * self.init_size * self.init_size)
        )
        
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(self.init_channels),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.init_channels, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 3, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], self.init_channels, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [
                nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25)
            ]
            if bn:
                block.append(nn.BatchNorm2d(out_filters))
            return block

        self.model = nn.Sequential(
            *discriminator_block(3, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        ds_size = Config.IMG_SIZE // 2**4
        self.adv_layer = nn.Sequential(
            nn.Linear(128 * ds_size * ds_size, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        features = self.model(img)
        features = features.view(features.shape[0], -1)
        validity = self.adv_layer(features)
        return validity

class FoxGAN:
    def __init__(self, results_dir, latent_dim=100):
        self.device = torch.device(Config.device)
        self.latent_dim = latent_dim
        self.results_dir = results_dir
        
        # Création des dossiers pour les résultats
        self.images_dir = results_dir / 'generated_images'
        self.images_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialisation des listes pour stocker les pertes
        self.d_losses = []
        self.g_losses = []
        
        # Initialisation des modèles
        self.generator = Generator(latent_dim).to(self.device)
        self.discriminator = Discriminator().to(self.device)
        
        self.adversarial_loss = nn.BCELoss()
        
        self.optimizer_G = torch.optim.Adam(
            self.generator.parameters(), 
            lr=0.0002, 
            betas=(0.5, 0.999)
        )
        self.optimizer_D = torch.optim.Adam(
            self.discriminator.parameters(), 
            lr=0.0002, 
            betas=(0.5, 0.999)
        )

    def train(self, dataloader, n_epochs):
        for epoch in range(n_epochs):
            epoch_d_losses = []
            epoch_g_losses = []
            
            for i, (real_imgs, _) in enumerate(dataloader):
                real_imgs = real_imgs.to(self.device)
                batch_size = real_imgs.shape[0]
                
                valid = torch.ones((batch_size, 1), requires_grad=False).to(self.device)
                fake = torch.zeros((batch_size, 1), requires_grad=False).to(self.device)

                # Train Generator
                self.optimizer_G.zero_grad()
                z = torch.randn(batch_size, self.latent_dim).to(self.device)
                gen_imgs = self.generator(z)
                g_loss = self.adversarial_loss(self.discriminator(gen_imgs), valid)
                g_loss.backward()
                self.optimizer_G.step()

                # Train Discriminator
                self.optimizer_D.zero_grad()
                real_loss = self.adversarial_loss(self.discriminator(real_imgs), valid)
                fake_loss = self.adversarial_loss(
                    self.discriminator(gen_imgs.detach()), fake
                )
                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                self.optimizer_D.step()

                epoch_d_losses.append(d_loss.item())
                epoch_g_losses.append(g_loss.item())

                if i % 100 == 0:
                    print(
                        f"[Epoch {epoch}/{n_epochs}] "
                        f"[Batch {i}/{len(dataloader)}] "
                        f"[D loss: {d_loss.item():.4f}] "
                        f"[G loss: {g_loss.item():.4f}]"
                    )

            # Enregistrement des pertes moyennes de l'époque
            self.d_losses.append(np.mean(epoch_d_losses))
            self.g_losses.append(np.mean(epoch_g_losses))

            if epoch % 10 == 0:
                self.save_sample_images(epoch)
                self.plot_losses()

    def generate_images(self, num_images):
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(num_images, self.latent_dim).to(self.device)
            generated = self.generator(z)
            generated = (generated * 0.5 + 0.5).cpu().numpy()
            generated = np.transpose(generated, (0, 2, 3, 1))
            generated = np.clip(generated, 0, 1)
        return generated

    def save_sample_images(self, epoch, n_row=5, n_col=5):
        self.generator.eval()
        with torch.no_grad():
            n_samples = n_row * n_col
            z = torch.randn(n_samples, self.latent_dim).to(self.device)
            generated = self.generator(z)
            generated = (generated * 0.5 + 0.5).cpu()
            
            fig, axs = plt.subplots(n_row, n_col, figsize=(10, 10))
            for idx in range(n_samples):
                i, j = divmod(idx, n_col)
                img = generated[idx].numpy().transpose(1, 2, 0)
                axs[i, j].imshow(img)
                axs[i, j].axis('off')
            
            plt.savefig(self.images_dir / f'samples_epoch_{epoch}.png')
            plt.close()

    def plot_losses(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.d_losses, label='Discriminator Loss')
        plt.plot(self.g_losses, label='Generator Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.results_dir / 'loss_plot.png')
        plt.close()

def main():
    # Création du dossier de résultats avec date et heure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("./models/gan/gan_results") / f"results_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Chargement et prétraitement des données Fox
    fox_class = ['fox', 'Fox_negative_class']
    X, _ = create_X_y(fox_class)
    
    # Création du dataset et du dataloader
    transform = transforms.Compose([
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    dataset = FoxDataset(X, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=2
    )
    
    # Initialisation et entraînement du GAN
    gan = FoxGAN(results_dir, latent_dim=100)
    gan.train(dataloader, n_epochs=500)
    
    # Génération et sauvegarde des images finales
    generated_images = gan.generate_images(10)
    
    # Sauvegarde des images finales
    fig, axs = plt.subplots(2, 5, figsize=(15, 6))
    for i, img in enumerate(generated_images):
        row, col = divmod(i, 5)
        axs[row, col].imshow(img)
        axs[row, col].axis('off')
    plt.savefig(results_dir / 'final_generated_foxes.png')
    plt.close()
    
    # Sauvegarde du modèle
    torch.save({
        'generator_state_dict': gan.generator.state_dict(),
        'discriminator_state_dict': gan.discriminator.state_dict(),
    }, results_dir / 'fox_gan.pth')

if __name__ == '__main__':
    main()
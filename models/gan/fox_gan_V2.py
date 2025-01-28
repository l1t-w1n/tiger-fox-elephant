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
project_root = Path.cwd()
sys.path.append(str(project_root))

from config.config import Config
from utils.helper_functions import create_X_y, save_model
from models.gan.fox_gan import FoxGAN, FoxDataset
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, 3, padding=1),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, in_features, 3, padding=1),
            nn.InstanceNorm2d(in_features)
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, latent_dim, num_residual_blocks=6):
        super(Generator, self).__init__()
        
        # Initial projection et reshape
        self.initial = nn.Sequential(
            nn.Linear(latent_dim, 16 * 16 * 256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Couches de convolution principales
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Blocs résiduels
        resblocks = []
        for _ in range(num_residual_blocks):
            resblocks.append(ResidualBlock(256))
        self.resblocks = nn.Sequential(*resblocks)
        
        # Upsampling blocks
        self.upsampling = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 3, 7, 1, 3),
            nn.Tanh()
        )
        
    def forward(self, z):
        out = self.initial(z)
        out = out.view(out.shape[0], 256, 16, 16)
        out = self.conv_blocks(out)
        out = self.resblocks(out)
        img = self.upsampling(out)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(3, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
        )
        
        # PatchGAN - classifier
        self.patch = nn.Conv2d(512, 1, 4, padding=1)
        
    def forward(self, img):
        features = self.model(img)
        validity = self.patch(features)
        return validity

class ImprovedFoxGAN:
    def __init__(self, results_dir, latent_dim=100):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.latent_dim = latent_dim
        self.results_dir = results_dir
        
        # Création des dossiers pour les résultats
        self.images_dir = results_dir / 'generated_images'
        self.images_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialisation des modèles
        self.generator = Generator(latent_dim).to(self.device)
        self.discriminator = Discriminator().to(self.device)
        
        # Optimiseurs avec taux d'apprentissage adaptatifs
        self.optimizer_G = optim.Adam(
            self.generator.parameters(),
            lr=0.0002,
            betas=(0.5, 0.999)
        )
        self.optimizer_D = optim.Adam(
            self.discriminator.parameters(),
            lr=0.0002,
            betas=(0.5, 0.999)
        )
        
        # Learning rate schedulers
        self.scheduler_G = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_G,
            T_max=200,
            eta_min=1e-5
        )
        self.scheduler_D = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_D,
            T_max=200,
            eta_min=1e-5
        )
        
        # Loss tracking
        self.g_losses = []
        self.d_losses = []

    def compute_gradient_penalty(self, real_samples, fake_samples):
        """Calcul de la pénalité de gradient pour WGAN-GP"""
        alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(self.device)
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = self.discriminator(interpolates)
        
        fake = torch.ones(d_interpolates.size()).to(self.device)
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def train(self, dataloader, n_epochs, n_critic=5):
        # Historique pour le suivi
        history = {
            'D_losses': [],
            'G_losses': [],
            'D_real': [],
            'D_fake': []
        }
        
        batches_done = 0
        
        for epoch in range(n_epochs):
            for i, (real_imgs, _) in enumerate(dataloader):
                batch_size = real_imgs.size(0)
                real_imgs = real_imgs.to(self.device)
                
                # ---------------------
                #  Train Discriminator
                # ---------------------
                
                self.optimizer_D.zero_grad()
                
                # Générer un batch d'images
                z = torch.randn(batch_size, self.latent_dim).to(self.device)
                fake_imgs = self.generator(z)
                
                # Sorties du discriminateur
                real_validity = self.discriminator(real_imgs)
                fake_validity = self.discriminator(fake_imgs.detach())
                
                # Gradient penalty
                gradient_penalty = self.compute_gradient_penalty(real_imgs, fake_imgs.detach())
                
                # Loss du discriminateur
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + \
                        10 * gradient_penalty
                
                d_loss.backward()
                self.optimizer_D.step()
                
                # Train Generator every n_critic iterations
                if i % n_critic == 0:
                    # -----------------
                    #  Train Generator
                    # -----------------
                    
                    self.optimizer_G.zero_grad()
                    
                    # Generate images
                    fake_imgs = self.generator(z)
                    fake_validity = self.discriminator(fake_imgs)
                    
                    # Generator loss
                    g_loss = -torch.mean(fake_validity)
                    
                    g_loss.backward()
                    self.optimizer_G.step()
                    
                    # Log Progress
                    if i % 50 == 0:
                        print(
                            f"[Epoch {epoch}/{n_epochs}] "
                            f"[Batch {i}/{len(dataloader)}] "
                            f"[D loss: {d_loss.item():.4f}] "
                            f"[G loss: {g_loss.item():.4f}]"
                        )
                        
                        history['D_losses'].append(d_loss.item())
                        history['G_losses'].append(g_loss.item())
                        history['D_real'].append(real_validity.mean().item())
                        history['D_fake'].append(fake_validity.mean().item())
            
            # Update learning rates
            self.scheduler_G.step()
            self.scheduler_D.step()
            
            # Save images périodiquement
            if epoch % 10 == 0:
                self.save_sample_images(epoch)
                self.plot_losses(history)
        
        return history

    def generate_images(self, num_images):
        """Génère un nombre spécifique d'images"""
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(num_images, self.latent_dim).to(self.device)
            gen_imgs = self.generator(z)
            gen_imgs = (gen_imgs + 1) / 2  # Dénormalisation
            return gen_imgs.cpu().numpy()

    def save_sample_images(self, epoch, n_row=3, n_col=3):
        """Sauvegarde une grille d'images générées"""
        gen_imgs = self.generate_images(n_row * n_col)
        
        fig, axs = plt.subplots(n_row, n_col, figsize=(12, 12))
        for idx, img in enumerate(gen_imgs):
            i = idx // n_col
            j = idx % n_col
            axs[i, j].imshow(np.transpose(img, (1, 2, 0)))
            axs[i, j].axis('off')
        
        plt.savefig(self.images_dir / f'epoch_{epoch}.png')
        plt.close()

    def plot_losses(self, history):
        """Plot et sauvegarde les courbes de loss"""
        plt.figure(figsize=(10, 5))
        plt.plot(history['D_losses'], label='D loss')
        plt.plot(history['G_losses'], label='G loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(self.results_dir / 'loss_plot.png')
        plt.close()

def main():
    # Configuration
    LATENT_DIM = 100
    N_EPOCHS = 500
    BATCH_SIZE = 32
    
    # Création du répertoire de résultats
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("./results") / f"improved_fox_gan_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Chargement des données
    transform = transforms.Compose([
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    dataset = FoxDataset(transform=transform)  # Définir votre dataset
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )
    
    # Initialisation et entraînement du GAN
    gan = ImprovedFoxGAN(results_dir, LATENT_DIM)
    history = gan.train(dataloader, N_EPOCHS)
    
    # Sauvegarde du modèle final
    torch.save({
        'generator_state_dict': gan.generator.state_dict(),
        'discriminator_state_dict': gan.discriminator.state_dict(),
        'gen_optimizer_state_dict': gan.optimizer_G.state_dict(),
        'disc_optimizer_state_dict': gan.optimizer_D.state_dict(),
        'history': history
    }, results_dir / 'final_model.pth')

if __name__ == "__main__":
    main()
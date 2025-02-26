import os
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from transformers import CLIPTokenizer
from accelerate import Accelerator
import matplotlib.pyplot as plt

from ddpm import DDPMSampler
import config
import model_loader
from dataset import ButterflyDataset

class TrainPipeline:
    def __init__(
        self, 
        tokenizer_merges: str = "path/to/merges.txt",
        tokenizer_vocab: str = "path/to/vocab.json",
        dataset: ButterflyDataset = None,
        pretrained_weights: bool = True
    ):
        """
        Args:
            tokenizer_merges: Path to merges.txt for CLIPTokenizer
            tokenizer_vocab:  Path to vocab.json for CLIPTokenizer
            dataset:          Your custom dataset providing (image, text) pairs
            pretrained_weights: Whether to load pretrained weights or not
        """
        
        # Set up Tokenizer
        self.tokenizer = CLIPTokenizer(
            merges_file=tokenizer_merges,
            vocab_file=tokenizer_vocab
        )
        
        # Set up Dataset
        self.dataset = dataset
        
        # Accelerator for multi-GPU / mixed precision 
        self.accelerator = Accelerator()
        
        # Training Configurations
        self.device = config.DEVICE
        self.pretrained_weights = pretrained_weights
        self.batch_size = config.BATCH_SIZE
        self.num_epochs = config.NUM_EPOCHS
        self.learning_rate = config.LEARNING_RATE
        self.num_time_steps = config.NUM_TIME_STEPS
        self.seed = config.SEED
        self.num_workers = config.NUM_WORKERS
        self.checkpoint_path = config.checkpoint_path
        self.latent_height = config.LATENT_HEIGHT
        self.latent_width = config.LATENT_WIDTH
        
        self.save_dir = "/home/l1t-w1n/tiger-fox-elephant/ddpm/samples"
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Generator for random noise
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(self.seed)
        
        # DDPMSampler for training steps
        self.ddpm_sampler = DDPMSampler(
            generator=self.generator,
            num_training_steps=self.num_time_steps
        )
        
        # Load pretrained models (Diffusion, CLIP, Encoder, Decoder, etc.)
        if self.pretrained_weights:
            self.models = model_loader.preload_models_from_standard_weights(self.checkpoint_path, self.device)
        
        # Diffusion Model
        self.diffusion_model = self.models['diffusion'].to(self.device)
        
        # Optimizer
        self.optimizer = AdamW(
            self.diffusion_model.parameters(),
            lr=self.learning_rate
        )
        
        # Learning Rate Scheduler
        # (You can change T_max to self.num_epochs if that suits your training schedule)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=self.optimizer,
            T_max=100
        )
        
        # Loss Function
        self.loss_function = nn.MSELoss()
        
        # DataLoader
        self.dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        # Grad Scaler for Mixed Precision
        self.scaler = GradScaler()
        
        # Prepare for distributed / mixed-precision
        (
            self.diffusion_model,
            self.optimizer,
            self.dataloader
        ) = self.accelerator.prepare(
            self.diffusion_model,
            self.optimizer,
            self.dataloader
        )
        
    def _get_time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """
        Создает синусоидальные временные эмбеддинги для каждого шага в батче `t`.
        """
        half_dim = 160  # Размерность эмбеддинга: 2 * half_dim = 320
        freqs = torch.pow(
            10000, 
            -torch.arange(0, half_dim, dtype=torch.float32, device=t.device) / half_dim
        )
        # Изменяем форму тензоров для broadcasting
        x = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # [B, 1] * [1, half_dim] → [B, half_dim]
        return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)  # [B, 320]
        
    def train(self):
        self.diffusion_model.train()
        
        # If you have CLIP / Encoder, set them up
        clip = self.models.get('clip', None)
        if clip:
            clip = clip.to(self.device)
            clip.eval()  # Often we don't train CLIP
        
        encoder = self.models.get('encoder', None)
        if encoder:
            encoder = encoder.to(self.device)
            encoder.eval()  # If you're not training it, set to eval

        for epoch in range(self.num_epochs):
            total_loss = 0.0 
            
            for step, batch in enumerate(tqdm(self.dataloader, desc=f'Epoch {epoch+1}/{self.num_epochs}')):
                # Retrieve images and text from your dataset
                # images = batch['image'].to(self.device)  # shape: (B, C, H, W)
                # prompts = batch['text']                  # list of strings, can't do .to(device)
                
                images = batch["image"].to(self.device)  # Tensor [B, C, H, W]
                prompts = batch["name"]
    
                # If your pipeline uses an encoder (VAE, etc.), create latents
                if encoder is not None:
                    # Example shape of latents (B, 4, latent_height, latent_width)
                    latent_shape = (images.size(0), 4, self.latent_height, self.latent_width)
                    noise_for_encoder = torch.randn(latent_shape, generator=self.generator, device=self.device)
                    # Encode images into latents
                    latents = encoder(images, noise_for_encoder)
                else:
                    # Otherwise, treat images as latents directly (e.g. training in pixel space)
                    latents = images

                # Tokenize text
                prompt_tokens = self.tokenizer.batch_encode_plus(
                    prompts,
                    padding='max_length',
                    max_length=77
                ).input_ids  # shape: (B, 77)
                
                # Convert to tensor
                prompt_token_tensors = torch.tensor(prompt_tokens, dtype=torch.long, device=self.device)

                # If we have a CLIP model, get text embeddings
                if clip:
                    context = clip(prompt_token_tensors)
                else:
                    context = None

                # Sample random timesteps: range [0 .. num_time_steps)
                t = torch.randint(
                    low=0,
                    high=self.ddpm_sampler.num_train_timesteps,
                    size=(latents.size(0),),
                    device=self.device
                )

                # Add noise at step t
                # Adjust if your DDPMSampler function name / signature is different
                noisy_latents, noise = self.ddpm_sampler.add_noise(latents, t)
                
                # Create time embeddings
                time_embedding = self._get_time_embedding(t)

                with autocast():
                    # Forward pass: predict the noise that was added
                    predicted_noise = self.diffusion_model(
                        latent=noisy_latents,
                        context=context,
                        time=time_embedding
                    )
                    
                    # Compute MSE Loss between predicted noise and true noise
                    loss = self.loss_function(predicted_noise, noise)
                
                # Backprop with GradScaler
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                
                if step % 100 == 0:
                    self._save_sample(epoch, step, noise)
                
                total_loss += loss.item()

                # Step the scheduler (can do it per epoch instead if you prefer)
                self.scheduler.step()

            # Compute average loss for the epoch
            avg_epoch_loss = total_loss / len(self.dataloader)
            print(f"Epoch [{epoch+1}/{self.num_epochs}] - Loss: {avg_epoch_loss:.4f}")

            ckpt_path = None  

            if (epoch + 1) % 5 == 0:
                checkpoint_dir = os.path.dirname(self.checkpoint_path)
                os.makedirs(checkpoint_dir, exist_ok=True)
                
                ckpt_path = self.checkpoint_path.replace('.ckpt', f"_epoch_{epoch+1}.pth")
                torch.save(self.diffusion_model.state_dict(), ckpt_path)
                print(f"Checkpoint saved at {ckpt_path}")

            if ckpt_path:
                print(f"Checkpoint saved at {ckpt_path}")        
        print("Training completed.")
        
    def _save_sample(self, epoch, step, latent):
        with torch.no_grad():
            decoder = self.models['decoder']
            decoder = decoder.to(self.device)

            os.makedirs(self.save_dir, exist_ok=True)

            image = decoder(latent)

            image = (image + 1) / 2
            image = image.clamp(0, 1)

            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.imshow(image[0].permute(1, 2, 0).cpu().numpy())
            ax.axis('off')
            
            filename = f"epoch_{epoch}_step_{step}.png"
            save_path = os.path.join(self.save_dir, filename)
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            
            print(f"[INFO] Сэмпл сохранён: {save_path}")

        
if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((config.HEIGHT, config.WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Optional normalization
    ])
    dataset = ButterflyDataset(transform=transform)
    
    trainer = TrainPipeline(
        tokenizer_merges="/home/l1t-w1n/tiger-fox-elephant/ddpm/src/data/tokenizer_merges.txt",
        tokenizer_vocab="/home/l1t-w1n/tiger-fox-elephant/ddpm/src/data/tokenizer_vocab.json",
        dataset=dataset,
        pretrained_weights=True
    )
    
    trainer.train()

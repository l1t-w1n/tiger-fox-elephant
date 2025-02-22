import torch 
import torch.nn as nn 
import torch.nn.functional as F

from decoder import VAE_AttentionBlock, VAE_ResidualBlock

class VAE_Encoder(nn.Sequential):
    """
        VAE Encoder part - exists for transfer simple image representation into latent space, which will be less sparse and will have less dimensinality of the feature space. 
    """
    def __inti__(self):
        super(VAE_Encoder, self).__init__(
            # (batch_size, 3, height, width) -> (batch_size, 128, height, width)
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=1, padding=1),
            
            # (batch_size, 128, height, width) -> (batch_size, 128, height, width)
            VAE_ResidualBlock(128, 128),
            
            # (batch_size, 128, height, width) -> (batch_size, 128, height, width)
            VAE_ResidualBlock(128, 128),
            
            # (batch_size, 128, height, width) -> (batch_size, 128, height/2, width/2)
            nn.Conv2d(in_Channels=128, out_channels=128, kernel_size=3, stride=2, padding=0),
            
            # (batch_size, 128, height/2, width/2) -> (batch_size, 256, height/2, width/2)
            VAE_ResidualBlock(128, 256),
            
            # (batch_size, 256, height/2, width/2) -> (batch_size, 256, height/2, width/2)
            VAE_ResidualBlock(256, 256),
            
            # (batch_size, 256, height/2, width/2) -> (batch_size, 256, height/4, width/4)
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=0),
            
            # (batch_size, 256, height/4, width/4) -> (batch_size, 512, height/4, width/4)
            VAE_ResidualBlock(256, 512),
            
            # (batch_size, 512, height/4, width/4) -> (batch_size, 512, height/4, width/4)
            VAE_ResidualBlock(512, 512),
            
            # (batch_size, 512, height/4, width/4) -> (batch_size, 512, height/8, width/8)
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=0),
            
            # (batch_size, 512, height/8, width/8) -> (batch_size, 512, height/8, width/8)
            VAE_ResidualBlock(512, 512),
            
            # (batch_size, 512, height/8, width/8) -> (batch_size, 512, height/8, width/8)
            VAE_ResidualBlock(512, 512),
            
            # (batch_size, 512, height/8, width/8) -> (batch_size, 512, height/8, width/8)
            VAE_ResidualBlock(512, 512),
            
            VAE_AttentionBlock(512),
            
            # (batch_size, 512, height/8, width/8) -> (batch_size, 512, height/8, width/8)
            VAE_ResidualBlock(512, 512),
            
            nn.GroupNorm(num_groups=32, num_channels=512),
            
            nn.SiLU(),
            
            # (batch_size, 512, height/8, width/8) -> (batch_size, 8, height/8, width/8)
            nn.Conv2d(in_channels=512, out_channels=8, kernel_size=3, stride=1, padding=1),
            
            # (batch_size, 8, height/8, width/8) -> (batch_size, 8, height/8, width/8)
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=0)
        )
        
    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
            x - input image tensor
            noise N(0, I) - noise tensor, which will be added to the input retrieved (mean and std) from input image tensor, to correctly synthesize Gaussiane distribution for VAE. 
        """
        
        for layer in self:
             # Padding at downsampling should be asymmetri, only for right and bottom edges
            if getattr(layer, 'stride', None) == (2, 2):
                x = F.pad(x, (0, 1, 0, 1))
            x = layer(x)
        
        # x = (batch_size, 8, height/8, width/8) -> 2 * (batch_size, 4, height/8, width/8) 
        # devide output tensor to 2 parts - mean and variance 
        mean, log_variance = torch.chunk(x, 2, dim=1)
        
        # clamp variance into limits [-30, 20]
        log_variance = torch.clamp(log_variance, min=-30, max=20)
        
        variance = torch.exp(log_variance)
        
        standard_deviation = torch.sqrt(variance)
    
        # Transform N(0, 1) -> N(mean, stdev)  (Batch_Size, 4, Height / 8, Width / 8)
        # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
        
        x = mean + standard_deviation * noise
        
        # Normalization constant
        x *= 0.18215
        
        return x 
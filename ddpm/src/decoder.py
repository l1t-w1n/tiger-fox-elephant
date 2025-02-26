import torch 
import torch.nn as nn
import torch.nn.functional as F

from attention import SelfAttention

class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(VAE_ResidualBlock, self).__init__()
        
        self.groupnorm_1 = nn.GroupNorm(num_groups=32, num_channels=in_channels)
        self.conv_1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels, 
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.groupnorm_2 = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        self.conv_2 = nn.Conv2d(
            in_channels=out_channels, 
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            stride=1
        )
        
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0
            )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, in_channels, height, width)
        
        residual  = x 
        x = self.groupnorm_1(x)
        x = F.selu(x)
        x = self.conv_1(x)
        
        x = self.groupnorm_2(x)
        x = F.selu(x)
        x = self.conv_2(x)
        
        return x + self.residual_layer(residual)        
    
class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super(VAE_AttentionBlock, self).__init__()
        
        self.groupnormalization = nn.GroupNorm(num_groups=32, num_channels=channels)
        self.attention = SelfAttention(1, channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        residual = x 
        
        # (batch_size, channels, height, width)
        initial_shape = x.shape
        
        b, c, h, w = x.shape
        
        # (batch_size, channels, height, width) -> (batch_size, channels, height * width)
        x = x.view(b, c, h * w)
        
        # (batch_size, channels, height * width) -> (batch_size, height * width, channels)
        x = x.transpose(-1, -2)
        
        x = self.attention(x)
        
        # (batch_size, height * width, channels) -> (batch_size, channels, height * width)
        x = x.transopse(-1, -2)
    
        # (batch_size, channels, height * width) -> (batch_size, channels, height, width)
        x = x.view(initial_shape)
        
        x += residual
        return x 
    

class VAE_Decoder(nn.Sequential):
    def __init__(self):
        super(VAE_Decoder, self).__init__(
            # (batch_size, 4, height/8, width/8) -> (batch_size, 4, height/8, width/8)
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=1, stride=1, padding=0),
            
            # (batch_size, 4, height/8, width/8) -> (batch_size, 512, height/8, width/8)
            nn.Conv2d(in_channels=4, out_channels=512, kernel_size=3, stride=1, padding=1),
            
            # (batch_size, 512, height/8, width/8) -> (batch_size, 512, height/8, width/8)
            VAE_ResidualBlock(512, 512),
            
            # (batch_size, 512, height/8, width/8) -> (batch_size, 512, height/8, width/8)
            VAE_AttentionBlock(512),
            
            # (batch_size, 512, height/8, width/8) -> (batch_size, 512, height/8, width/8)
            VAE_ResidualBlock(512, 512),
            
            # (batch_size, 512, height/8, width/8) -> (batch_size, 512, height/8, width/8)
            VAE_ResidualBlock(512, 512),
            
            # (batch_size, 512, height/8, width/8) -> (batch_size, 512, height/8, width/8)
            VAE_ResidualBlock(512, 512),
            
            # (batch_size, 512, height/8, width/8) -> (batch_size, 512, height/8, width/8)
            VAE_ResidualBlock(512, 512),
            
            # (batch_size, 512, height/8, width/8) -> (batch_size, 512, height/4, width/4)
            nn.Upsample(scale_factor=2, mode='nearest'),
            
            # (batch_size, 512, height/4, width/4) -> (batch_size, 512, height/4, width/4)
            VAE_ResidualBlock(512, 512),
            
            # (batch_size, 512, height/4, width/4) -> (batch_size, 512, height/4, width/4)
            VAE_ResidualBlock(512, 512),
            
            # (batch_size, 512, height/4, width/4) -> (batch_size, 512, height/4, width/4)
            VAE_ResidualBlock(512, 512),
            
            # (batch_size, 512, height/4, width/4) -> (batch_size, 256, height/2, width/2)
            nn.Upsample(scale_factor=2, mode='nearest'),
            
            # (batch_size, 256, height/2, width/2) -> (batch_size, 256, height/2, width/2)
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            
            # (Batch_Size, 512, Height / 2, Width / 2) -> (Batch_Size, 256, Height / 2, Width / 2)
            VAE_ResidualBlock(512, 256), 
            
            # (Batch_Size, 256, Height / 2, Width / 2) -> (Batch_Size, 256, Height / 2, Width / 2)
            VAE_ResidualBlock(256, 256), 
            
            # (Batch_Size, 256, Height / 2, Width / 2) -> (Batch_Size, 256, Height / 2, Width / 2)
            VAE_ResidualBlock(256, 256), 
            
            # (Batch_size, 256, height/2, width/2) -> (batch_size, 128, height, width)
            nn.Upsample(scale_factor=2, mode='nearest'),
            
            # (batch_size, 128, height, width) -> (batch_size, 128, height, width)
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            
            # (Batch_Size, 256, Height, Width) -> (Batch_Size, 128, Height, Width)
            VAE_ResidualBlock(256, 128), 
            
            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height, Width)
            VAE_ResidualBlock(128, 128), 
            
            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height, Width)
            VAE_ResidualBlock(128, 128), 
            
            nn.GroupNorm(num_groups=32, num_channels=128),
            
            nn.SiLU(),
            
            nn.Conv2d(in_channels=128, out_channels=3, kernel_size=3, stride=1, padding=1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, 8, height/8, wiodth/8)
        
        x /= 0.18215
        
        for layer in self:
            x = layer(x)
        
        # (Batch_size, 3, height, width)
        return x

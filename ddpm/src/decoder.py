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
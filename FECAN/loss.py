"""Loss is taken from:
    ShuffleMixer: An Efficient ConvNet for Image Super-Resolution
    Long Sun, Jinshan Pan, Jinhui Tang
    https://arxiv.org/abs/2205.15175
"""
import torch
import torch.nn as nn

class Loss(nn.Module):
    def __init__(self, l1_weight=1.0, freq_weight=0.05):
        super().__init__()
        self.l1_weight = l1_weight
        self.freq_weight = freq_weight
        self.l1_loss = nn.L1Loss()
        
    def forward(self, sr, hr):
        # Pixel-wise L1 loss
        l1_loss = self.l1_loss(sr, hr)
        
        # Frequency domain L1 loss
        fft_sr = torch.fft.fft2(sr, dim=(-2, -1))
        fft_hr = torch.fft.fft2(hr, dim=(-2, -1))
        
        # Convert complex tensors to real-valued tensors with separate channels
        fft_sr_real = torch.view_as_real(fft_sr)
        fft_hr_real = torch.view_as_real(fft_hr)
        
        l_f = self.l1_loss(fft_sr_real, fft_hr_real)
        
        # Combined loss
        total_loss = self.l1_weight * l1_loss + self.freq_weight * l_f
        return total_loss
""" Huang, F., Liu, H., Chen, L. et al. 
    Feature enhanced cascading attention network for lightweight image super-resolution. 
    Sci Rep 15, 2051 (2025). 
    https://doi.org/10.1038/s41598-025-85548-4
    
    SA-Net: Shuffle Attention for Deep Convolutional Neural Networks
    Qing-Long Zhang Yu-Bin Yang
    https://arxiv.org/abs/2102.00240
    https://github.com/wofmanaf/SA-Net
    
    Visual Attention Network
    Meng-Hao Guo, Cheng-Ze Lu, Zheng-Ning Liu, Ming-Ming Cheng, Shi-Min Hu
    https://arxiv.org/abs/2202.09741
    
    """
    
import torch
import torch.nn as nn
import torch.nn.functional as F

class SA(nn.Module):
    """Shuffle Attention"""
    def __init__(self, channels, sa_groups):
        super().__init__()
        self.sa_groups = sa_groups
        self.split_channels = channels // (2 * sa_groups)  # Split each group into two parts

        self.cweight = nn.Parameter(torch.zeros(1, self.split_channels, 1, 1))
        self.cbias = nn.Parameter(torch.ones(1, self.split_channels, 1, 1))
        self.sweight = nn.Parameter(torch.zeros(1, self.split_channels, 1, 1))
        self.sbias = nn.Parameter(torch.ones(1, self.split_channels, 1, 1))

        self.sigmoid = nn.Sigmoid()
        self.group_norm = nn.GroupNorm(self.split_channels, self.split_channels)
    
    def channel_shuffle(self, x, sa_groups):
        batch, channels, height, width = x.size()
        x = x.view(batch, sa_groups, channels // sa_groups, height, width)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        return x.view(batch, channels, height, width)

    def forward(self, x):
        batch, c, h, w = x.size()
        
        #group into subfeatures
        x = x.view(batch*self.sa_groups, -1, h, w)
        
        #channel split
        x_0, x_1 = torch.chunk(x, 2, dim=1)
        
        #channel attention
        xn = F.adaptive_avg_pool2d(x_0, 1)
        xn = self.cweight * xn + self.cbias
        xn = x_0 * self.sigmoid(xn)
        
        #spatial attention
        xs = self.group_norm(x_1)
        xs = self.sweight * xs + self.sbias
        xs = x_1 * self.sigmoid(xs)
        
        #concatenate
        out = torch.cat([xn, xs], dim=1)
        out = out.view(batch, -1, h, w)
        
        return self.channel_shuffle(out, 2)

class ESA(nn.Module):
    """Enhanced Shuffle Attention"""
    def __init__(self, channels, sa_groups):
        super().__init__()
        # 1x1 convolutions from paper
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.sa = SA(channels, sa_groups=sa_groups)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=1)
        self.gelu = nn.GELU()

    def forward(self, x):
        identity = x  # Preserve original input
        
        # First convolution
        x = self.conv1(x)
        
        # Shuffle attention with enhanced residual
        sa_out = self.sa(x)
        x = sa_out * identity + identity  # Element-wise multiply then add
        
        # Second convolution
        x = self.conv2(x)
        
        return self.gelu(x)
    

class DepthWiseConv2d(nn.Module):
    """Configurable depth-wise convolution block
    Args:
        in_channels: Number of input channels
        kernel_size: Convolution kernel size (int or tuple)
        stride: Convolution stride
        padding: Explicit padding value
        dilation: Dilation factor
        auto_padding: Calculate padding automatically
    """
    def __init__(self, in_channels, kernel_size, stride=1,
                 padding=None, dilation=1, auto_padding=True):
        super().__init__()
        
        # Handle kernel size (supports asymmetric kernels)
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
            
        # Automatic padding calculation (maintains spatial dimensions)
        if auto_padding and padding is None:
            self.padding = (
                (self.kernel_size[0] - 1) // 2 * dilation,
                (self.kernel_size[1] - 1) // 2 * dilation
            )
        else:
            self.padding = padding or 0

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=self.kernel_size,
            stride=stride,
            padding=self.padding,
            dilation=dilation,
            groups=in_channels,  # Depth-wise separation
            bias=False
        )

    def forward(self, x):
        return self.conv(x)

class GLSKA(nn.Module):
    """Gated Large Separable Kernel Attention"""
    def __init__(self, channels, hdw1_kernel, hdw2_kernel,
                 hdwd1_kernel, hdwd2_kernel, hdw_conv_kernel):
        super().__init__()
        # Calculate dilation factor from HDWD1 kernel following K = 2d-1 for a K×K convolution
        self.d = (hdwd1_kernel[1] + 1) // 2

        # First non-dilated components
        self.hdw1 = DepthWiseConv2d(
            channels, kernel_size=hdw1_kernel,
            auto_padding=True
        )
        self.hdw2 = DepthWiseConv2d(
            channels, kernel_size=hdw2_kernel,
            auto_padding=True
        )

        # Dilated components
        self.hdwd1 = DepthWiseConv2d(
            channels, kernel_size=hdwd1_kernel,
            dilation=self.d, auto_padding=True
        )
        self.hdwd2 = DepthWiseConv2d(
            channels, kernel_size=hdwd2_kernel,
            dilation=self.d, auto_padding=True
        )

        # Gating components
        self.conv1x1 = nn.Conv2d(channels, channels, 1)
        self.gate_conv = DepthWiseConv2d(
            channels, kernel_size=hdw_conv_kernel,
            auto_padding=True
        )

    def forward(self, x):
        identity = x
        z = self.hdw1(x)
        z = self.hdw2(z)
        z = self.hdwd1(z)
        z = self.hdwd2(z)
        z = self.conv1x1(z)
        
        # Gating mechanism
        gate = self.gate_conv(identity)
        return z * gate

class MLSKA(nn.Module):
    """Multi-scale Large Separable Kernel Attention"""
    def __init__(self, channels):
        super().__init__()
        assert channels % 2 == 0, "MLSKA requires even input channels"
        
        self.channels = channels
        self.split = channels // 2  # Split input into two equal parts
    
        # Calculate group channels with remainder distribution
        group_base, remainder = divmod(self.split, 3)
        self.group_channels = [
            group_base + (1 if i < remainder else 0)
            for i in range(3)
        ]
        
        # Three GLSKA modules with different kernel patterns
        self.scales = nn.ModuleList([
            GLSKA(
                self.group_channels[0],
                hdw1_kernel=(1, 3),
                hdw2_kernel=(3, 1),
                hdwd1_kernel=(1, 5),
                hdwd2_kernel=(5, 1),
                hdw_conv_kernel=3
            ),
            GLSKA(
                self.group_channels[1],
                hdw1_kernel=(1, 5),
                hdw2_kernel=(5, 1),
                hdwd1_kernel=(1, 7),
                hdwd2_kernel=(7, 1),
                hdw_conv_kernel=5
            ),
            GLSKA(
                self.group_channels[2],
                hdw1_kernel=(1, 7),
                hdw2_kernel=(7, 1),
                hdwd1_kernel=(1, 9),
                hdwd2_kernel=(9, 1),
                hdw_conv_kernel=7
            )
        ])

        # Feature aggregation components
        self.aggregate = nn.Sequential(
            nn.Conv2d(self.split, self.split, kernel_size=1),
            nn.GELU()
        )
        self.final_conv = nn.Conv2d(self.split, self.split, kernel_size=1)

    def forward(self, xin):
        # Split input into X and Y
        x, y = torch.chunk(xin, 2, dim=1)
        
        # Split X into three groups
        x_parts = torch.split(x, self.group_channels, dim=1)
        
        # Process each group through corresponding GLSKA
        z = []
        for part, scale in zip(x_parts, self.scales):
            z.append(scale(part))
        
        # Concatenate and aggregate features
        z = torch.cat(z, dim=1)
        z = self.aggregate(z)
        
        # Combine with Y branch
        y = self.final_conv(F.gelu(z) * y)
        
        # Maintain original channel count
        return torch.cat([x, y], dim=1) 

class FECA(nn.Module):
    """Feature Enhanced Cascading Attention"""
    def __init__(self, channels, sa_groups):
        super().__init__()
        self.esa = ESA(channels, sa_groups)
        self.mlska = MLSKA(channels)
        self.dw_conv = DepthWiseConv2d(channels, kernel_size=3, auto_padding=True)

    def forward(self, x):
        identity = self.dw_conv(x)
        x = self.esa(x)
        x = self.mlska(x)
        return x + identity
    
class SGFF(nn.Module):
    """Spatial Gated Feed-Forward Module"""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        # Expansion convolution
        self.conv_expand = nn.Conv2d(channels, 2*channels, kernel_size=1)
        self.gelu = nn.GELU()
        
        # Spatial gate components
        self.dw_conv = DepthWiseConv2d(
            channels, 
            kernel_size=3,  # 3x3 depth-wise convolution
            auto_padding=True
        )
        
        # Final projection
        self.conv_reduce = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        # Expand channels
        x = self.conv_expand(x)
        x = self.gelu(x)
        
        # Split into two branches
        x1, x2 = torch.chunk(x, 2, dim=1)
        
        # Spatial gating mechanism
        # Element-wise multiplication
        x = x1 * self.dw_conv(x2)
        
        # Final projection
        return self.conv_reduce(x)

class FCB(nn.Module):
    """Feature Connect Block"""
    def __init__(self, channels):
        super().__init__()
        self.dw_conv = DepthWiseConv2d(
            channels,
            kernel_size=3,
            auto_padding=True 
        )

    def forward(self, X, X0):
        """Inputs:
        X: Current feature map [batch, channels, H, W]
        X0: Original input feature [batch, channels, H, W]
        """
        return self.dw_conv(X) + X0

class HFEM(nn.Module):
    """High-Frequency Enhancement Module"""
    def __init__(self, channels, sa_groups):
        super().__init__()
        self.l_norm1 = nn.LayerNorm([channels])
        self.feca = FECA(channels, sa_groups)
        self.l_norm2 = nn.LayerNorm([channels])
        self.sgff = SGFF(channels)
        self.fcb = FCB(channels)

    def forward(self, x):
        identity1 = x
        
        # First normalization + FECA
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.l_norm1(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.feca(x)
        x = x + identity1
        
        # Second normalization + SGFF
        identity2 = x
        x = x.permute(0, 2, 3, 1).contiguous() 
        x = self.l_norm2(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.sgff(x)
        x = x + identity2
        
        # Feature Connect Block
        return self.fcb(x, identity1)

class LSKAT(nn.Module):
    """Large Separable Kernel Attention Tail"""
    def __init__(self, channels):
        super().__init__()
        
        # Residual path (gate)
        self.gate_path = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.GELU()
        )
        
        # Main processing path
        self.dw1 = DepthWiseConv2d(channels, (1, 7), auto_padding=True)
        self.dw2 = DepthWiseConv2d(channels, (7, 1), auto_padding=True)
        self.dwd1 = DepthWiseConv2d(channels, (1, 9), dilation=5, auto_padding=True)
        self.dwd2 = DepthWiseConv2d(channels, (9, 1), dilation=5, auto_padding=True)
        
        # Final projection
        self.conv1x1 = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        gate = self.gate_path(x)
        
        # Main processing
        x = self.dw1(x)
        x = self.dw2(x)
        x = self.dwd1(x)
        x = self.dwd2(x)
        
        # Combine paths
        x = x * gate  # Element-wise multiplication
        return self.conv1x1(x)

class FECAN(nn.Module):
    """Feature Enhanced Cascading Attention Network (Classical Version)"""
    def __init__(self, in_channels=3, num_hfem=36, channels=96, sa_groups=6, upscale_factor=4):
        super().__init__()
        self.upscale_factor = upscale_factor

        # Shallow Feature Extraction Module (SFEM)
        self.sfem = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)

        # Deep Feature Extraction Module (DFEM)
        self.dfem = nn.Sequential(
            *[HFEM(channels, sa_groups) for _ in range(num_hfem)],
            LSKAT(channels)
        )

        # Image Reconstruction Module (IRM)
        self.upsample = nn.Sequential(
            nn.Conv2d(channels, 3 * (upscale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor)
        )
        
        # Bicubic initialization
        self.bicubic = nn.Upsample(scale_factor=upscale_factor, mode='bicubic', align_corners=False)

    def forward(self, x):
        # Shallow features
        m0 = self.sfem(x)
        
        # Deep features
        md = self.dfem(m0)
        
        # Reconstruction
        hr = self.upsample(md + m0)  # Feature fusion
        lr_bicubic = self.bicubic(x)  # Bicubic upsampling
        
        return hr + lr_bicubic  # Final output
    
if __name__ == "__main__":
    # Test EnhancedShuffleAttention
    esa = ESA(96, 6)
    x = torch.randn(2, 96, 32, 32)
    try:
        assert esa(x).shape == x.shape
        print("ESA test passed ✅")
    except Exception as e:
        print(f"ESA test failed ❌: {e}")

    # Test SGFF
    sgff = SGFF(96)
    x = torch.randn(2, 96, 32, 32)
    try:
        assert sgff(x).shape == x.shape
        print("SGFF test passed ✅")
    except Exception as e:
        print(f"SGFF test failed ❌: {e}")

    # Test MLSKA
    mlska = MLSKA(64)  # Even channels
    x = torch.randn(2, 64, 32, 32)
    try:
        assert mlska(x).shape == x.shape
        print("MLSKA test passed ✅")
    except Exception as e:
        print(f"MLSKA test failed ❌: {e}")

    # Test invalid MLSKA
    try:
        MLSKA(63)  # Should throw error
        print("MLSKA invalid test failed ❌")
    except AssertionError:
        print("MLSKA invalid test passed ✅")
    
    # Test FCB
    fcb = FCB(64)
    x = torch.randn(2, 64, 32, 32)
    x0 = torch.randn_like(x)  # Original input feature
    try:
        output = fcb(x, x0)
        assert output.shape == x.shape
        print("FCB test passed ✅")
    except Exception as e:
        print(f"FCB test failed ❌: {e}")

    # Test FCB dimension mismatch
    try:
        bad_x = torch.randn(2, 64, 16, 16)  # Wrong spatial size
        fcb(bad_x, x0)
        print("FCB dimension test failed ❌")
    except RuntimeError:
        print("FCB dimension test passed ✅")
    
    # Test FECA
    feca = FECA(96, 6)
    x = torch.randn(2, 96, 32, 32)
    try:
        output = feca(x)
        assert output.shape == x.shape
        print("FECA test passed ✅")
    except Exception as e:
        print(f"FECA test failed ❌: {e}")

    # Test HFEM
    hfem = HFEM(96, 6)
    x = torch.randn(2, 96, 32, 32)
    try:
        output = hfem(x)
        assert output.shape == x.shape
        print("HFEM test passed ✅")
    except Exception as e:
        print(f"HFEM test failed ❌: {e}")
        
     # Test LSKAT
    lskat = LSKAT(64)
    x = torch.randn(2, 64, 32, 32)
    try:
        output = lskat(x)
        assert output.shape == x.shape
        print("LSKAT test passed ✅")
    except Exception as e:
        print(f"LSKAT test failed ❌: {e}")

    # Test FECAN
    model = FECAN(num_hfem=36, channels=96, sa_groups=6)
    
    x = torch.randn(4, 3, 32, 32)  #32x32 → 128x128
    try:
        out = model(x)
        assert out.shape == (4, 3, 128, 128), (
            f"Unexpected output shape {out.shape}, should be [4, 3, 128, 128]."
        )
        print("FECAN test passed ✅")
    except Exception as e:
        print(f"FECAN test failed ❌: {e}")
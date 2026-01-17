"""
CBAM (Convolutional Block Attention Module) Implementation for YOLOv11m
Based on: Woo et al. "CBAM: Convolutional Block Attention Module" ECCV 2018

Usage:
    from models.cbam import CBAM
    attention = CBAM(in_channels=512, reduction=16)
    output = attention(input_tensor)
"""

import torch
import torch.nn as nn


class ChannelAttentionModule(nn.Module):
    """
    Channel Attention Module
    Focuses on *what* is meaningful given an input image
    Uses both average and max pooling to capture richer channel statistics
    """
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttentionModule, self).__init__()
        
        # Shared MLP layers
        hidden_channels = in_channels // reduction_ratio
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, in_channels, 1, bias=False)
        )
        
        # Pooling operations
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Activation
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Average pooling branch
        avg_out = self.mlp(self.avg_pool(x))
        
        # Max pooling branch
        max_out = self.mlp(self.max_pool(x))
        
        # Combine both branches
        channel_attention = self.sigmoid(avg_out + max_out)
        
        # Apply attention
        return x * channel_attention


class SpatialAttentionModule(nn.Module):
    """
    Spatial Attention Module
    Focuses on *where* is meaningful given an input feature map
    Uses both average and max pooling across channels
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttentionModule, self).__init__()
        
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = (kernel_size - 1) // 2
        
        self.conv = nn.Conv2d(
            2,  # Input: concatenated avg and max pooling (2 channels)
            1,  # Output: single attention map
            kernel_size,
            padding=padding,
            bias=False
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Average pooling across channel dimension
        avg_out = torch.mean(x, dim=1, keepdim=True)
        
        # Max pooling across channel dimension
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        
        # Generate spatial attention map
        spatial_attention = self.sigmoid(self.conv(spatial_input))
        
        # Apply attention
        return x * spatial_attention


class CBAM(nn.Module):
    """
    Complete CBAM Module (Channel + Spatial Attention)
    
    Args:
        in_channels (int): Number of input channels
        reduction_ratio (int): Reduction ratio for channel attention. Default: 16
        kernel_size (int): Kernel size for spatial attention. Default: 7
        
    Example:
        >>> cbam = CBAM(in_channels=512, reduction_ratio=16)
        >>> x = torch.randn(1, 512, 20, 20)
        >>> out = cbam(x)  # Shape: (1, 512, 20, 20)
    """
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        
        self.channel_attention = ChannelAttentionModule(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttentionModule(kernel_size)
    
    def forward(self, x):
        # Apply channel attention first
        x = self.channel_attention(x)
        
        # Then apply spatial attention
        x = self.spatial_attention(x)
        
        return x


# Test code
if __name__ == "__main__":
    print("Testing CBAM Module...")
    
    # Test configuration
    batch_size = 2
    channels = 512
    height, width = 20, 20
    
    # Create sample input
    x = torch.randn(batch_size, channels, height, width)
    print(f"Input shape: {x.shape}")
    
    # Test CBAM
    cbam = CBAM(in_channels=channels, reduction_ratio=16, kernel_size=7)
    output = cbam(x)
    print(f"Output shape: {output.shape}")
    
    # Verify shape is unchanged
    assert output.shape == x.shape, "Output shape mismatch!"
    
    print("✅ CBAM test passed!")
    
    # Test parameter count
    total_params = sum(p.numel() for p in cbam.parameters())
    print(f"Total CBAM parameters: {total_params:,}")

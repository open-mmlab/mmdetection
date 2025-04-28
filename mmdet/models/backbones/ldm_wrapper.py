# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from ldm.modules.diffusionmodules.model import Encoder as LDMEncoder

class LDMEncoderWrapper(nn.Module):
    """A wrapper for the LDM Encoder to ensure compatibility with MMDet.
    
    This wrapper handles the return values and features from LDM Encoder
    and makes it compatible with MMDetection's expected format.
    """
    
    def __init__(self, in_channels=3, **kwargs):
        super().__init__()
        self.encoder = LDMEncoder(in_channels=in_channels, **kwargs)
        
        # Get the output dimension of the encoder's final block
        # Adjust based on actual LDM encoder architecture
        self.out_channels = [32]  # Example, adjust based on actual architecture
        
    def forward(self, x):
        """Forward function.
        
        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).
            
        Returns:
            tuple[torch.Tensor]: Multi-level feature maps.
        """
        # Forward pass through LDM encoder
        features = self.encoder(x)
        
        # Format the output to match MMDet's expected format
        # This may need adjustment based on actual LDM encoder output
        if not isinstance(features, (list, tuple)):
            features = [features]
            
        return tuple(features)
# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from torch import Tensor

from mmdet.registry import MODELS
from ldm.modules.diffusionmodules.model import Encoder as LDMEncoder

@MODELS.register_module()
class LDMEncoderBackbone(BaseModule):
    """LDM Encoder as backbone for DETR.
    
    Args:
        in_channels (int): Number of input channels. Default: 3.
        out_indices (Sequence[int]): Output from which stages.
            Default: (0,).
        pretrained (str): Path to pre-trained weights. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
        freeze (bool): Whether to freeze the backbone. Default: True.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_indices: Tuple[int] = (0,),
        pretrained: Optional[str] = None,
        init_cfg: Optional[Union[dict, List[dict]]] = None,
        freeze: bool = True,
        **kwargs
    ):
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.out_indices = out_indices
        self.freeze = freeze
        
        # Initialize LDM Encoder
        self.encoder = LDMEncoder(
            in_channels=in_channels, 
            double_z=True,
            z_channels=16,
            resolution=256,
            out_ch=3,
            ch=128,
            ch_mult=[1, 1, 2, 2, 4],  # num_down = len(ch_mult)-1
            num_res_blocks=2,
            attn_resolutions=[16],
            dropout=0.0,
            **kwargs
        )
        
        # Get the output dimension of the encoder
        # Assuming it returns a feature with 512 channels at the final level
        # This will need to be adjusted based on the actual LDM encoder architecture
        self.out_channels = [32]  # Adjust based on actual LDM encoder
        
        # Load pretrained weights
        if pretrained is not None:
            self._load_pretrained(pretrained)
            
        # Freeze the encoder if specified


        if self.freeze:
            self._freeze_encoder()
            
    def _load_pretrained(self, pretrained: str) -> None:
        """Load pretrained weights for the encoder."""
        checkpoint = torch.load(pretrained, map_location='cpu')
        
        # If the checkpoint contains a state_dict, use that directly
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        # Filter only the encoder part from the state_dict
        encoder_dict = {}
        for k, v in state_dict.items():
            if k.startswith('first_stage_model.encoder') or k.startswith('encoder'):
                # Remove the prefix to match our encoder structure
                key = k.replace('first_stage_model.encoder.', '').replace('encoder.', '')
                encoder_dict[key] = v
                
        # Load the weights to the encoder
        # Using strict=False to ignore missing or unexpected keys
        self.encoder.load_state_dict(encoder_dict, strict=False)
        print(f"Loaded pretrained LDM encoder weights from {pretrained}")
        
    def _freeze_encoder(self) -> None:
        """Freeze all parameters in the encoder."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()
        print("LDM encoder has been frozen.")
            
    def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        """Forward function."""
        # Use the LDM encoder to get features
        with torch.no_grad() if self.freeze else torch.enable_grad():
            features = self.encoder(x)
        
        # If the encoder returns multiple features, select the ones at out_indices
        if isinstance(features, (list, tuple)):
            outs = [features[i] for i in self.out_indices]
        else:
            # If it returns a single feature map, wrap it in a list
            outs = [features]
            
        return tuple(outs)
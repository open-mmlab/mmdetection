# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os

import torch
from mmengine.config import Config
from mmengine.registry import MODELS
from mmengine.runner import Runner

def parse_args():
    parser = argparse.ArgumentParser(description='Check LDM Encoder Backbone')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--checkpoint', help='path to pretrained LDM checkpoint',
                        default='pretrained_checkpoints/kl-f16/model.ckpt')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    # Load config
    cfg = Config.fromfile(args.config)
    
    # Build the model
    model = MODELS.build(cfg.model)
    
    # Check if the backbone is initialized and frozen
    backbone = model.backbone
    
    # Check parameters
    total_params = sum(p.numel() for p in backbone.parameters())
    trainable_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    
    print(f"LDM Encoder Backbone Summary:")
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    print(f"Frozen: {trainable_params == 0}")
    
    # Perform a simple forward pass to verify functionality
    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        outputs = backbone(dummy_input)
    
    print("\nForward Pass Output Shapes:")
    for i, output in enumerate(outputs):
        print(f"Output {i}: {output.shape}")
    
    print("\nBackbone Configuration:")
    print(f"Input channels: {backbone.in_channels}")
    print(f"Output channels: {backbone.out_channels}")
    print(f"Out indices: {backbone.out_indices}")

if __name__ == '__main__':
    main()
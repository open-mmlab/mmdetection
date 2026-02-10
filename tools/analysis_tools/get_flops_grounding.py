# Copyright (c) OpenMMLab. All rights reserved.
"""Get FLOPs for GroundingDINO models.

This script is specifically designed for models that require text inputs,
such as GroundingDINO and other vision-language models.

Gradient Checkpointing (with_cp) Compatibility:
    This script uses fvcore's FlopCountAnalysis for accurate backbone
    FLOPs, which internally uses PyTorch JIT tracing. However, gradient
    checkpointing (with_cp=True) is INCOMPATIBLE with JIT tracing.
    This script automatically disables gradient checkpointing when
    building the model for analysis. This does NOT affect model accuracy
    or your original config file.

Usage:
    python tools/analysis_tools/get_flops_grounding.py <config_file>

Example:
    python tools/analysis_tools/get_flops_grounding.py \\
        configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_cat.py
"""
import argparse
import tempfile
from pathlib import Path

import torch
from mmengine.config import Config, DictAction
from mmengine.logging import MMLogger
from mmengine.model import revert_sync_batchnorm
from mmengine.registry import init_default_scope

from mmdet.registry import MODELS

try:
    from fvcore.nn import FlopCountAnalysis
except ImportError:
    FlopCountAnalysis = None


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Get FLOPs for grounding detection models')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[800, 1333],
        help='input image size (height width)')
    parser.add_argument(
        '--text',
        type=str,
        default='car',
        help='text prompt for computing text encoder FLOPs')
    parser.add_argument(
        '--num-classes',
        type=int,
        default=None,
        help='number of classes (for estimating text encoder FLOPs)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config')
    args = parser.parse_args()
    return args


def format_flops(flops):
    """Format FLOPs to human readable string.

    Args:
        flops (int or float): Number of FLOPs.

    Returns:
        str: Human readable FLOPs string.
    """
    if flops >= 1e12:
        return f'{flops / 1e12:.2f} T'
    elif flops >= 1e9:
        return f'{flops / 1e9:.2f} G'
    elif flops >= 1e6:
        return f'{flops / 1e6:.2f} M'
    else:
        return f'{flops:.2f}'


def format_params(params):
    """Format parameters to human readable string.

    Args:
        params (int or float): Number of parameters.

    Returns:
        str: Human readable parameter count string.
    """
    if params >= 1e9:
        return f'{params / 1e9:.2f} B'
    elif params >= 1e6:
        return f'{params / 1e6:.2f} M'
    elif params >= 1e3:
        return f'{params / 1e3:.2f} K'
    else:
        return f'{params:.0f}'


def _get_backbone_out_channels(cfg):
    """Extract backbone output channels from config.

    Tries to read from the neck's in_channels, then falls back to
    common defaults based on backbone type.

    Args:
        cfg (Config): Model config.

    Returns:
        list[int]: Output channels per feature level.
    """
    # Try reading from neck config (most reliable)
    if hasattr(cfg.model, 'neck'):
        neck_cfg = cfg.model.neck
        if hasattr(neck_cfg, 'in_channels'):
            return list(neck_cfg.in_channels)

    # Fallback: infer from backbone type
    backbone_type = cfg.model.backbone.get('type', '')
    if 'Swin' in backbone_type:
        embed_dims = cfg.model.backbone.get('embed_dims', 96)
        depths = cfg.model.backbone.get('depths', [2, 2, 6, 2])
        num_levels = len(depths)
        return [embed_dims * (2**i) for i in range(num_levels)]

    # Generic default
    return [256, 512, 1024, 2048]


def _get_feature_strides(cfg):
    """Get feature map strides from config.

    Args:
        cfg (Config): Model config.

    Returns:
        list[int]: Strides for each feature level.
    """
    if hasattr(cfg.model, 'neck'):
        neck_cfg = cfg.model.neck
        if hasattr(neck_cfg, 'in_channels'):
            num_levels = len(neck_cfg.in_channels)
            # Common strides: start at 8, double each level
            return [8 * (2**i) for i in range(num_levels)]
    return [8, 16, 32]


def _get_neck_out_channels(cfg):
    """Get neck output channels from config.

    Args:
        cfg (Config): Model config.

    Returns:
        int: Output channels of the neck.
    """
    if hasattr(cfg.model, 'neck'):
        neck_cfg = cfg.model.neck
        if hasattr(neck_cfg, 'out_channels'):
            return neck_cfg.out_channels
    return 256


def _get_transformer_config(cfg):
    """Extract transformer encoder/decoder config.

    Reads layer counts, embed_dim, ffn_dim, and num_queries from the
    model config. Falls back to GroundingDINO defaults.

    Args:
        cfg (Config): Model config.

    Returns:
        dict: Transformer configuration with keys: embed_dim,
            num_encoder_layers, num_decoder_layers, ffn_dim,
            num_queries.
    """
    result = dict(
        embed_dim=256,
        num_encoder_layers=6,
        num_decoder_layers=6,
        ffn_dim=2048,
        num_queries=900)

    # Try to read from encoder config
    if hasattr(cfg.model, 'encoder'):
        enc = cfg.model.encoder
        if hasattr(enc, 'num_layers'):
            result['num_encoder_layers'] = enc.num_layers
        if hasattr(enc, 'layer_cfg'):
            layer = enc.layer_cfg
            if hasattr(layer, 'self_attn_cfg'):
                attn = layer.self_attn_cfg
                if hasattr(attn, 'embed_dims'):
                    result['embed_dim'] = attn.embed_dims
            if hasattr(layer, 'ffn_cfg'):
                ffn = layer.ffn_cfg
                if hasattr(ffn, 'feedforward_channels'):
                    result['ffn_dim'] = ffn.feedforward_channels

    # Try to read from decoder config
    if hasattr(cfg.model, 'decoder'):
        dec = cfg.model.decoder
        if hasattr(dec, 'num_layers'):
            result['num_decoder_layers'] = dec.num_layers

    # Try to read num_queries
    if hasattr(cfg.model, 'num_queries'):
        result['num_queries'] = cfg.model.num_queries
    elif hasattr(cfg.model, 'bbox_head'):
        head = cfg.model.bbox_head
        if hasattr(head, 'num_queries'):
            result['num_queries'] = head.num_queries

    return result


def count_backbone_flops(model, input_shape, device, logger):
    """Count FLOPs for the vision backbone using fvcore.

    Args:
        model (nn.Module): The detection model.
        input_shape (tuple[int]): Input image (H, W).
        device (torch.device): Device to run on.
        logger (MMLogger): Logger instance.

    Returns:
        tuple: (flops, params) or (None, params) if fvcore
            is not available.
    """
    if not hasattr(model, 'backbone'):
        return None, None

    backbone = model.backbone
    backbone.eval()

    h, w = input_shape
    x = torch.randn(1, 3, h, w).to(device)

    if FlopCountAnalysis is not None:
        flops_analyzer = FlopCountAnalysis(backbone, x)
        flops_analyzer.unsupported_ops_warnings(False)
        flops_analyzer.uncalled_modules_warnings(False)
        flops = flops_analyzer.total()
        params = sum(p.numel() for p in backbone.parameters())
        return flops, params
    else:
        logger.warning('fvcore is not installed, backbone FLOPs cannot be '
                       'computed accurately. Install with: pip install fvcore')
        params = sum(p.numel() for p in backbone.parameters())
        return None, params


def count_text_encoder_flops(model, num_classes):
    """Estimate FLOPs for the text encoder.

    Uses rough per-forward-pass estimates based on model type since
    text encoders often have complex control flow that prevents
    accurate tracing.

    Args:
        model (nn.Module): The detection model.
        num_classes (int): Number of classes.

    Returns:
        tuple: (flops, params) or (None, None) if no language
            model is found.
    """
    if not hasattr(model, 'language_model'):
        return None, None

    lang_model = model.language_model
    params = sum(p.numel() for p in lang_model.parameters())

    # Estimate FLOPs based on model type
    model_name = getattr(lang_model, 'name', '')
    if 'clip' in model_name.lower():
        if 'large' in model_name.lower():
            flops_per_forward = 10e9
        elif 'base' in model_name.lower():
            flops_per_forward = 4e9
        else:
            flops_per_forward = 4e9
    elif 'bert' in model_name.lower():
        flops_per_forward = 11e9
    else:
        flops_per_forward = 5e9

    if num_classes:
        flops = flops_per_forward * num_classes
    else:
        flops = flops_per_forward

    return flops, params


def count_neck_flops(model, input_shape, cfg):
    """Estimate FLOPs for the neck (e.g. ChannelMapper).

    Reads in_channels and out_channels from the config to avoid
    hardcoded assumptions.

    Args:
        model (nn.Module): The detection model.
        input_shape (tuple[int]): Input image (H, W).
        cfg (Config): Model config.

    Returns:
        tuple: (flops, params).
    """
    if not hasattr(model, 'neck') or model.neck is None:
        return 0, 0

    neck = model.neck
    params = sum(p.numel() for p in neck.parameters())

    h, w = input_shape
    in_channels = _get_backbone_out_channels(cfg)
    strides = _get_feature_strides(cfg)
    out_channels = _get_neck_out_channels(cfg)

    flops = 0
    for in_c, stride in zip(in_channels, strides):
        fh, fw = h // stride, w // stride
        # 1x1 conv: in_c * out_c * H * W * 2 (multiply-add)
        flops += in_c * out_channels * fh * fw * 2

    return flops, params


def count_transformer_flops(model, input_shape, cfg):
    """Estimate FLOPs for the transformer encoder and decoder.

    Reads architecture parameters from the config dynamically
    instead of hardcoding values.

    Args:
        model (nn.Module): The detection model.
        input_shape (tuple[int]): Input image (H, W).
        cfg (Config): Model config.

    Returns:
        tuple: (encoder_flops, decoder_flops,
            encoder_params, decoder_params).
    """
    h, w = input_shape
    trans_cfg = _get_transformer_config(cfg)
    embed_dim = trans_cfg['embed_dim']
    num_enc = trans_cfg['num_encoder_layers']
    num_dec = trans_cfg['num_decoder_layers']
    ffn_dim = trans_cfg['ffn_dim']
    num_queries = trans_cfg['num_queries']

    strides = _get_feature_strides(cfg)
    feat_sizes = [(h // s, w // s) for s in strides]
    total_tokens = sum(fh * fw for fh, fw in feat_sizes)

    # Encoder self-attention: 4 * n^2 * d (Q,K,V proj + output)
    enc_attn = (4 * total_tokens * total_tokens * embed_dim * num_enc)
    # Encoder FFN: 2 * n * d * ffn_dim
    enc_ffn = 2 * total_tokens * embed_dim * ffn_dim * num_enc
    encoder_flops = enc_attn + enc_ffn

    # Decoder self-attention on queries
    dec_self = (4 * num_queries * num_queries * embed_dim * num_dec)
    # Decoder cross-attention to image features
    dec_cross = (4 * num_queries * total_tokens * embed_dim * num_dec)
    # Decoder FFN
    dec_ffn = 2 * num_queries * embed_dim * ffn_dim * num_dec
    decoder_flops = dec_self + dec_cross + dec_ffn

    encoder_params = 0
    if hasattr(model, 'encoder'):
        encoder_params = sum(p.numel() for p in model.encoder.parameters())

    decoder_params = 0
    if hasattr(model, 'decoder'):
        decoder_params = sum(p.numel() for p in model.decoder.parameters())

    return encoder_flops, decoder_flops, encoder_params, decoder_params


def main():
    args = parse_args()
    logger = MMLogger.get_instance(name='MMLogger')

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f'{config_path} not found.')
        return

    cfg = Config.fromfile(args.config)
    cfg.work_dir = tempfile.TemporaryDirectory().name

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # Disable gradient checkpointing for FLOPs analysis.
    # with_cp=True is incompatible with JIT tracing used by fvcore.
    if hasattr(cfg.model, 'backbone'):
        if cfg.model.backbone.get('with_cp', False):
            cfg.model.backbone.with_cp = False
            logger.warning('Auto-disabled gradient checkpointing '
                           '(with_cp=False) for FLOPs analysis. '
                           'This does NOT affect your config file.')

    init_default_scope(cfg.get('default_scope', 'mmdet'))

    # Build model
    model = MODELS.build(cfg.model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model = revert_sync_batchnorm(model)
    model.eval()

    # Get input shape
    if len(args.shape) == 1:
        h = w = args.shape[0]
    else:
        h, w = args.shape[:2]

    # Get number of classes
    num_classes = args.num_classes
    if num_classes is None:
        if (hasattr(cfg.model, 'bbox_head')
                and hasattr(cfg.model.bbox_head, 'num_classes')):
            num_classes = cfg.model.bbox_head.num_classes
        else:
            num_classes = 80  # Default COCO

    split_line = '=' * 60
    logger.info(split_line)
    logger.info('GroundingDINO FLOPs Analysis')
    logger.info(split_line)
    logger.info(f'Config: {args.config}')
    logger.info(f'Input shape: ({h}, {w})')
    logger.info(f'Number of classes: {num_classes}')
    logger.info(split_line)

    total_flops = 0
    total_params = 0

    # 1. Vision Backbone
    backbone_flops, backbone_params = count_backbone_flops(
        model, (h, w), device, logger)
    if backbone_flops is not None:
        logger.info('\n[Vision Backbone]')
        logger.info(f'  FLOPs:  {format_flops(backbone_flops)}')
        logger.info(f'  Params: {format_params(backbone_params)}')
        total_flops += backbone_flops
        total_params += backbone_params
    else:
        logger.info('\n[Vision Backbone]')
        logger.info('  FLOPs:  (fvcore not installed, cannot compute)')
        if backbone_params:
            logger.info(f'  Params: {format_params(backbone_params)}')
            total_params += backbone_params

    # 2. Text Encoder
    text_flops, text_params = count_text_encoder_flops(model, num_classes)
    if text_flops is not None:
        logger.info('\n[Text Encoder]')
        logger.info(f'  FLOPs:  {format_flops(text_flops)} '
                    f'(estimated, {num_classes} classes)')
        logger.info(f'  Params: {format_params(text_params)}')
        total_flops += text_flops
        total_params += text_params

    # 3. Neck
    neck_flops, neck_params = count_neck_flops(model, (h, w), cfg)
    if neck_params > 0:
        logger.info('\n[Neck (ChannelMapper)]')
        logger.info(f'  FLOPs:  {format_flops(neck_flops)} (estimated)')
        logger.info(f'  Params: {format_params(neck_params)}')
        total_flops += neck_flops
        total_params += neck_params

    # 4. Transformer Encoder/Decoder
    enc_flops, dec_flops, enc_params, dec_params = \
        count_transformer_flops(model, (h, w), cfg)
    logger.info('\n[Transformer Encoder]')
    logger.info(f'  FLOPs:  {format_flops(enc_flops)} (estimated)')
    logger.info(f'  Params: {format_params(enc_params)}')
    total_flops += enc_flops
    total_params += enc_params

    logger.info('\n[Transformer Decoder]')
    logger.info(f'  FLOPs:  {format_flops(dec_flops)} (estimated)')
    logger.info(f'  Params: {format_params(dec_params)}')
    total_flops += dec_flops
    total_params += dec_params

    # 5. Bbox Head
    if hasattr(model, 'bbox_head'):
        head_params = sum(p.numel() for p in model.bbox_head.parameters())
        logger.info('\n[Detection Head]')
        logger.info(f'  Params: {format_params(head_params)}')
        total_params += head_params

    # Total
    logger.info('\n' + split_line)
    logger.info(f'TOTAL FLOPs:      {format_flops(total_flops)}')
    logger.info(f'TOTAL Parameters: {format_params(total_params)}')
    logger.info(split_line)

    logger.warning('Note: Some FLOPs are estimated based on model '
                   'architecture. Backbone FLOPs are accurate if fvcore is '
                   'installed. Text encoder and transformer FLOPs are '
                   'theoretical estimates.')

    if FlopCountAnalysis is None:
        logger.info('Tip: Install fvcore for accurate backbone FLOPs: '
                    'pip install fvcore')


if __name__ == '__main__':
    main()

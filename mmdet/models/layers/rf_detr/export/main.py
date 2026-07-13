# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied and modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------
"""CLI orchestrator for ONNX and TensorRT model export."""

from __future__ import annotations

import argparse
import os
import random
import warnings
from typing import cast

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch import Tensor
from torchvision.transforms.v2 import Compose, Resize, ToDtype, ToImage

from mmdet.models.layers.rf_detr.datasets.transforms import Normalize
from mmdet.models.layers.rf_detr.export._onnx.exporter import export_onnx
from mmdet.models.layers.rf_detr.export._tensorrt import trtexec
from mmdet.models.layers.rf_detr.models import BuilderArgs, build_model
from mmdet.models.layers.rf_detr.models.backbone.backbone import Backbone
from mmdet.models.layers.rf_detr.models.lwdetr import LWDETR
from mmdet.models.layers.rf_detr.utilities.distributed import get_rank
from mmdet.models.layers.rf_detr.utilities.logger import get_logger
from mmdet.models.layers.rf_detr.utilities.package import get_sha, get_version

logger = get_logger()


def _num_parameters(module: nn.Module) -> int:
    """Count trainable and non-trainable parameters in a module."""
    return sum(int(p.numel()) for p in module.parameters())


def make_infer_image(
    infer_dir: str | None,
    shape: tuple[int, int],
    batch_size: int,
    device: str | torch.device = "cuda",
    num_channels: int = 3,
) -> Tensor:
    if infer_dir is None:
        if num_channels == 3:
            dummy = np.random.randint(0, 256, (shape[0], shape[1], 3), dtype=np.uint8)
            image = Image.fromarray(dummy, mode="RGB")
        else:
            # Non-RGB: build a random float tensor directly, bypassing PIL.
            # Normalization is intentionally skipped here — export tracing only
            # requires tensors of the correct shape and dtype (float32), not the
            # correct distribution.  Real inference normalizes via predict() before
            # the tensor reaches the exported model, so the ONNX/TensorRT graph
            # never sees raw [0, 1] inputs in production.
            inps = torch.rand(batch_size, num_channels, shape[0], shape[1], device=device)
            return inps
    else:
        if num_channels != 3:
            raise ValueError(
                "Providing `infer_dir` is only supported for RGB models (num_channels=3). "
                "For non-RGB models, omit `infer_dir` to use a synthetic dummy input."
            )
        image = Image.open(infer_dir).convert("RGB")

    transforms = Compose(
        [
            Resize((shape[0], shape[1])),
            ToImage(),
            ToDtype(torch.float32, scale=True),
            Normalize(),
        ]
    )

    inps, _ = transforms(image, None)
    inps = inps.to(device)
    # inps = utils.nested_tensor_from_tensor_list([inps for _ in range(args.batch_size)])
    inps = torch.stack([inps for _ in range(batch_size)])
    return inps


def no_batch_norm(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            raise ValueError("BatchNorm2d found in the model. Please remove it.")


def main(args: argparse.Namespace) -> None:
    git_info = get_sha()
    if git_info != "unknown":
        logger.info(f"Running from git repository: {git_info}")
    else:
        version = get_version()
        logger.info(f"Running RF-DETR version: {version or 'unknown'}")
    logger.info(f"Export config: {vars(args)}")
    # convert device to device_id
    if args.device == "cuda":
        device_id = "0"
    elif args.device == "cpu":
        device_id = ""
    else:
        device_id = str(int(args.device))
        args.device = f"cuda:{device_id}"

    # device for export onnx
    # TODO: export onnx with cuda failed with onnx error
    device = torch.device("cpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = device_id

    # fix the seed for reproducibility
    seed: int = args.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    result = build_model(cast(BuilderArgs, args))
    model = cast(LWDETR, result[0] if isinstance(result, tuple) else result)
    backbone = model.backbone
    backbone_model = cast(Backbone, backbone[0])
    n_parameters = _num_parameters(model)
    logger.info(f"number of parameters: {n_parameters}")
    n_backbone_parameters = _num_parameters(backbone)
    logger.info(f"number of backbone parameters: {n_backbone_parameters}")
    n_projector_parameters = _num_parameters(backbone_model.projector)
    logger.info(f"number of projector parameters: {n_projector_parameters}")
    n_backbone_encoder_parameters = _num_parameters(backbone_model.encoder)
    logger.info(f"number of backbone encoder parameters: {n_backbone_encoder_parameters}")
    n_transformer_parameters = _num_parameters(cast(nn.Module, model.transformer))
    logger.info(f"number of transformer parameters: {n_transformer_parameters}")
    if args.resume:
        # --resume points at RF-DETR-produced checkpoints that may embed
        # argparse.Namespace objects requiring full-pickle deserialization.
        # Mirror RFDETR.from_checkpoint(trust_checkpoint=...): honour an optional
        # `trust_checkpoint` attribute on args (default True for backward
        # compatibility, since --resume is a developer/ops flag).  Setting
        # args.trust_checkpoint=False enforces safe-tensors-only loading and
        # raises if the checkpoint needs pickle.
        from mmdet.models.layers.rf_detr.util.io import _safe_torch_load

        trust_checkpoint = getattr(args, "trust_checkpoint", True)
        if trust_checkpoint:
            # Explicit pre-load warning: --resume has no CLI opt-in gate, so make
            # the pickle-deserialization risk (CWE-502) visible before loading.
            warnings.warn(
                f"Loading --resume checkpoint {args.resume!r} with full pickle "
                "deserialization (weights_only=False fallback). This can execute "
                "arbitrary code if the checkpoint comes from an untrusted source. "
                "Set args.trust_checkpoint=False to require safe-tensors-only loading.",
                UserWarning,
                stacklevel=2,
            )
        checkpoint = _safe_torch_load(args.resume, trust=trust_checkpoint)
        result = model.load_state_dict(checkpoint["model"], strict=False)
        if result.missing_keys or result.unexpected_keys:
            logger.warning(
                "load_state_dict strict=False: missing=%s unexpected=%s",
                result.missing_keys[:5],
                result.unexpected_keys[:5],
            )
        logger.info(f"load checkpoints {args.resume}")

    if args.layer_norm:
        no_batch_norm(model)

    model.to(device)

    input_tensors = make_infer_image(args.infer_dir, args.shape, args.batch_size, device)
    input_names = ["input"]
    if args.backbone_only:
        output_names = ["features"]
    elif args.segmentation_head:
        output_names = ["dets", "labels", "masks"]
    else:
        output_names = ["dets", "labels"]
    if getattr(args, "dynamic_batch", False):
        dynamic_axes = {name: {0: "batch"} for name in input_names + output_names}
    else:
        dynamic_axes = None
    # Run model inference in pytorch mode.
    # Use the export device (args.device) — not a hard-coded "cuda" — so that
    # CPU-only export paths work correctly.  Fall back to CPU when CUDA was
    # requested but is not available (e.g. in a CPU-only CI environment).
    run_device = torch.device(args.device)
    if run_device.type == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available; falling back to CPU for sanity forward pass.")
        run_device = torch.device("cpu")
    model.eval().to(run_device)
    input_tensors = input_tensors.to(run_device)
    with torch.no_grad():
        if args.backbone_only:
            features = model(input_tensors)
            logger.debug(f"PyTorch inference output shape: {features.shape}")
        elif args.segmentation_head:
            outputs = model(input_tensors)
            dets = outputs["pred_boxes"]
            labels = outputs["pred_logits"]
            masks = outputs["pred_masks"]
            if isinstance(masks, torch.Tensor):
                logger.debug(
                    f"PyTorch inference output shapes - Boxes: {dets.shape}, Labels: {labels.shape}, "
                    f"Masks: {masks.shape}"
                )
            else:
                # masks is a dict with spatial_features, query_features, bias
                logger.debug(f"PyTorch inference output shapes - Boxes: {dets.shape}, Labels: {labels.shape}")
                logger.debug(
                    "Mask spatial_features: "
                    f"{masks['spatial_features'].shape}, "
                    f"query_features: {masks['query_features'].shape}, "
                    f"bias: {masks['bias'].shape}"
                )
        else:
            outputs = model(input_tensors)
            dets = outputs["pred_boxes"]
            labels = outputs["pred_logits"]
            logger.debug(f"PyTorch inference output shapes - Boxes: {dets.shape}, Labels: {labels.shape}")
    model.cpu()
    input_tensors = input_tensors.cpu()

    output_file = export_onnx(
        args.output_dir,
        model,
        input_names,
        input_tensors,
        output_names,
        dynamic_axes,
        backbone_only=args.backbone_only,
        verbose=args.verbose,
        opset_version=args.opset_version,
        variant_name=getattr(args, "variant_name", None),
    )

    onnx_path = output_file  # preserve ONNX path before any post-processing step overwrites it

    if args.tensorrt:
        output_file = trtexec(onnx_path, args)

    # TODO: register --tflite, --quantization, --calibration-data, --max-images in the
    # argparser to enable TFLite export via CLI.  Until then, use RFDETR.export(format="tflite").
    _ = onnx_path  # referenced above; suppress unused-variable warning until CLI is wired up

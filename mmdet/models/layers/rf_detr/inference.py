# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""ModelContext and model-context builder for RF-DETR inference."""

from __future__ import annotations

__all__ = ["ModelContext"]

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import torch

from mmdet.models.layers.rf_detr.config import TrainConfig
from mmdet.models.layers.rf_detr.models import PostProcess, build_model
from mmdet.models.layers.rf_detr.models.backbone.backbone import Backbone
from mmdet.models.layers.rf_detr.models.lwdetr import LWDETR
from mmdet.models.layers.rf_detr.models.weights import apply_lora, load_pretrain_weights

if TYPE_CHECKING:
    from mmdet.models.layers.rf_detr.config import ModelConfig


class ModelContext:
    """Lightweight model wrapper returned by RFDETR.get_model().

    Provides the same attribute interface as the legacy ``main.py:Model`` but without importing or depending on
    ``populate_args()`` or the legacy stack.

    Args:
        model: The underlying ``nn.Module`` (LWDETR instance).
        postprocess: PostProcess instance for converting raw outputs to boxes.
        device: Device the model lives on.
        resolution: Input resolution (square side length in pixels).
        args: Namespace of resolved training/model configuration.
        class_names: Optional list of class name strings loaded from checkpoint.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        postprocess: PostProcess,
        device: torch.device,
        resolution: int,
        args: Any,
        class_names: list[str] | None = None,
    ) -> None:
        self.model = model
        self.postprocess = postprocess
        self.device = device
        self.resolution = resolution
        self.args = args
        self.class_names = class_names
        self.inference_model = None

    def reinitialize_detection_head(self, num_classes: int) -> None:
        """Reinitialize the detection head for a different number of classes.

        Args:
            num_classes: New number of output classes (including background).
        """
        reinitialize_head = cast("Callable[[int], None]", self.model.reinitialize_detection_head)
        reinitialize_head(num_classes)
        self.args.num_classes = num_classes


_ModelContext = ModelContext  # backward-compat alias


def _adapt_input_conv(num_channels: int, conv_weight: torch.Tensor) -> torch.Tensor:
    """Adapt a 3-channel pretrained conv weight tensor to *num_channels* input channels.

    When ``num_channels == 3``: returns the weight unchanged. When ``num_channels == 1``: averages weights across the
    original 3 channels.
    Otherwise (``num_channels != 1`` and ``num_channels != 3``): tiles the 3-channel
    pattern and scales by ``3 / num_channels`` to preserve activation magnitude.

    Args:
        num_channels: Target number of input channels.
        conv_weight: Original weight tensor of shape ``[out_ch, 3, H, W]``.

    Returns:
        Adapted weight tensor of shape ``[out_ch, num_channels, H, W]``.
    """
    if num_channels == 3:
        return conv_weight
    if num_channels == 1:
        return conv_weight.mean(dim=1, keepdim=True)
    # General case: tile and scale
    repeats = (num_channels + 2) // 3
    weight_out = torch.cat([conv_weight] * repeats, dim=1)[:, :num_channels]
    weight_out = weight_out * (3.0 / num_channels)
    return weight_out


def _build_model_context(model_config: ModelConfig) -> ModelContext:
    """Build a ModelContext from ModelConfig without using legacy main.py:Model.

    Replicates ``Model.__init__`` logic: builds the nn.Module, optionally loads pretrain weights and applies LoRA.  The
    model is intentionally kept on CPU; :func:`_ensure_model_on_device` in ``detr.py`` performs the deferred
    ``.to(device)`` on the first ``predict()`` / ``export()`` / ``optimize_for_inference()`` call.  Keeping construction
    CPU-only prevents CUDA initialisation during ``__init__``, which would block DDP strategies (``ddp_notebook``,
    ``ddp_spawn``) from spawning child processes in notebook environments.

    Args:
        model_config: Architecture configuration.

    Returns:
        ModelContext with the model on CPU, ready for lazy device placement.
    """
    from mmdet.models.layers.rf_detr._namespace import _namespace_from_configs

    # A dummy TrainConfig is needed only for _namespace_from_configs' required fields;
    # dataset_dir/output_dir are unused during model construction.
    dummy_train_config = TrainConfig(dataset_dir=None, output_dir="output")
    args = _namespace_from_configs(model_config, dummy_train_config)
    # ``TrainConfig.expand_paths`` realpaths these to the caller's absolute CWD, which would be embedded into
    # ``args`` and serialized into exported ``weights.pt`` (see ``RFDETR.export_for_roboflow``). Reset them on the
    # namespace to placeholders so inference-built checkpoints never leak the caller's filesystem layout.
    args.dataset_dir = None
    args.output_dir = "output"
    nn_model = build_model(args)
    assert isinstance(nn_model, LWDETR), (
        "build_model() returned a non-LWDETR result even though encoder_only/backbone_only were not set."
    )

    class_names: list[str] = []
    if model_config.pretrain_weights is not None:
        class_names = load_pretrain_weights(nn_model, model_config)
        # ``load_pretrain_weights`` can mutate ``model_config.num_classes`` and
        # ``model_config.num_keypoints_per_class`` when aligning to checkpoint schema.
        # Keep the derived namespace in sync so postprocess and predict() use correct values.
        if hasattr(args, "num_classes") and args.num_classes != model_config.num_classes:
            args.num_classes = model_config.num_classes
        _mc_kp = list(getattr(model_config, "num_keypoints_per_class", []) or [])
        if (
            hasattr(args, "num_keypoints_per_class")
            and list(getattr(args, "num_keypoints_per_class", []) or []) != _mc_kp
        ):
            args.num_keypoints_per_class = _mc_kp

    if model_config.backbone_lora:
        apply_lora(nn_model)

    # Adapt patch-embedding projection for non-RGB channel counts
    if model_config.num_channels != 3:
        import copy

        backbone = cast(Backbone, nn_model.backbone[0])
        proj = backbone.encoder.encoder.embeddings.patch_embeddings.projection
        new_proj = copy.deepcopy(proj)
        new_proj.in_channels = model_config.num_channels
        new_weight = _adapt_input_conv(model_config.num_channels, proj.weight)
        new_proj.weight = torch.nn.Parameter(new_weight)
        new_proj.weight.requires_grad = proj.weight.requires_grad
        backbone.encoder.encoder.embeddings.patch_embeddings.projection = new_proj
        backbone.encoder.encoder.embeddings.patch_embeddings.num_channels = model_config.num_channels

    device = torch.device(args.device)
    # Keep the model on CPU here; predict() / export() / optimize_for_inference()
    # will lazily move it to the target device on first use.  Eagerly calling
    # .to("cuda") would initialise the CUDA runtime during __init__(), which
    # prevents DDP strategies (ddp_notebook, ddp_spawn) from forking/spawning
    # child processes in notebook environments.
    postprocess = PostProcess(
        num_select=args.num_select,
        num_keypoints_per_class=getattr(args, "num_keypoints_per_class", []),
        # Older detection-only namespaces may omit keypoint postprocess knobs; keep the ModelConfig default.
        trace_alpha=getattr(args, "postprocess_trace_alpha", 0.2),
    )

    return ModelContext(
        model=nn_model,
        postprocess=postprocess,
        device=device,
        resolution=model_config.resolution,
        args=args,
        class_names=class_names or None,
    )

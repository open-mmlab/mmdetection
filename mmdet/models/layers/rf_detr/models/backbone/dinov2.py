# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import json
import math
import os
import types
from typing import Any, cast

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from transformers import AutoBackbone

from mmdet.models.layers.rf_detr.models.backbone.dinov2_with_windowed_attn import (
    WindowedDinov2WithRegistersBackbone,
    WindowedDinov2WithRegistersConfig,
)
from mmdet.models.layers.rf_detr.utilities.logger import get_logger

logger = get_logger()

size_to_width = {
    "tiny": 192,
    "small": 384,
    "base": 768,
    "large": 1024,
}

size_to_config = {
    "small": "dinov2_small.json",
    "base": "dinov2_base.json",
    "large": "dinov2_large.json",
}

size_to_config_with_registers = {
    "small": "dinov2_with_registers_small.json",
    "base": "dinov2_with_registers_base.json",
    "large": "dinov2_with_registers_large.json",
}


def get_config(size: str, use_registers: bool) -> dict[str, Any]:
    config_dict = size_to_config_with_registers if use_registers else size_to_config
    current_dir = os.path.dirname(os.path.abspath(__file__))
    configs_dir = os.path.join(current_dir, "dinov2_configs")
    config_path = os.path.join(configs_dir, config_dict[size])
    with open(config_path) as f:
        dino_config: dict[str, Any] = json.load(f)
    return dino_config


class DinoV2(nn.Module):
    def __init__(
        self,
        shape: tuple[int, int] = (640, 640),
        out_feature_indexes: list[int] | None = None,
        size: str = "base",
        use_registers: bool = True,
        use_windowed_attn: bool = True,
        gradient_checkpointing: bool = False,
        load_dinov2_weights: bool = True,
        patch_size: int = 14,
        num_windows: int = 4,
        positional_encoding_size: int = 37,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()

        if out_feature_indexes is None:
            out_feature_indexes = [2, 4, 5, 9]

        name = f"facebook/dinov2-with-registers-{size}" if use_registers else f"facebook/dinov2-{size}"

        self.shape = shape
        self.patch_size = patch_size
        self.num_windows = num_windows

        # Create the encoder

        if not use_windowed_attn:
            assert not gradient_checkpointing, "Gradient checkpointing is not supported for non-windowed attention"
            assert load_dinov2_weights, "Using non-windowed attention requires loading dinov2 weights from hub"
            if drop_path_rate > 0.0:
                logger.warning(
                    "drop_path_rate > 0.0 is not supported for non-windowed DinoV2 backbones."
                    " drop_path will be ignored."
                )
            self.encoder = AutoBackbone.from_pretrained(  # type: ignore[no-untyped-call]
                name,
                out_features=[f"stage{i}" for i in out_feature_indexes],
                return_dict=False,
            )
        else:
            window_block_indexes_set = set(range(out_feature_indexes[-1] + 1))
            window_block_indexes_set.difference_update(out_feature_indexes)
            window_block_indexes = list(window_block_indexes_set)

            dino_config = get_config(size, use_registers)

            dino_config["return_dict"] = False
            dino_config["out_features"] = [f"stage{i}" for i in out_feature_indexes]
            dino_config["drop_path_rate"] = drop_path_rate

            implied_resolution = positional_encoding_size * patch_size

            if implied_resolution != dino_config["image_size"]:
                logger.warning(
                    "Using a different number of positional encodings than DINOv2, which means"
                    " we're not loading DINOv2 backbone weights. This is not a problem if"
                    " finetuning a pretrained RF-DETR model."
                )
                dino_config["image_size"] = implied_resolution
                load_dinov2_weights = False

            if patch_size != 14:
                logger.warning(
                    f"Using patch size {patch_size} instead of 14, which means we're not loading"
                    " DINOv2 backbone weights. This is not a problem if finetuning a pretrained"
                    " RF-DETR model."
                )
                dino_config["patch_size"] = patch_size
                load_dinov2_weights = False

            if use_registers:
                windowed_dino_config = WindowedDinov2WithRegistersConfig(
                    **dino_config,
                    num_windows=num_windows,
                    window_block_indexes=window_block_indexes,
                    gradient_checkpointing=gradient_checkpointing,
                )
            else:
                windowed_dino_config = WindowedDinov2WithRegistersConfig(
                    **dino_config,
                    num_windows=num_windows,
                    window_block_indexes=window_block_indexes,
                    num_register_tokens=0,
                    gradient_checkpointing=gradient_checkpointing,
                )
            self.encoder = (
                WindowedDinov2WithRegistersBackbone.from_pretrained(
                    name,
                    config=windowed_dino_config,
                )
                if load_dinov2_weights
                else WindowedDinov2WithRegistersBackbone(windowed_dino_config)
            )

        self._out_feature_channels = [size_to_width[size]] * len(out_feature_indexes)
        self._export = False

    def export(self) -> None:
        if self._export:
            return
        self._export = True
        shape = self.shape

        def make_new_interpolated_pos_encoding(
            position_embeddings: Tensor, patch_size: int, height: int, width: int
        ) -> Tensor:

            num_positions = position_embeddings.shape[1] - 1
            dim = position_embeddings.shape[-1]
            height = height // patch_size
            width = width // patch_size

            class_pos_embed = position_embeddings[:, 0]
            patch_pos_embed = position_embeddings[:, 1:]

            # Reshape and permute
            patch_pos_embed = patch_pos_embed.reshape(
                1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim
            )
            patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)

            # Use bicubic interpolation, disabling antialias only on MPS devices
            patch_pos_embed = F.interpolate(
                patch_pos_embed,
                size=(height, width),
                mode="bicubic",
                align_corners=False,
                antialias=patch_pos_embed.device.type != "mps",
            )

            # Reshape back
            patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, dim)
            return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

        # If the shape of self.encoder.embeddings.position_embeddings
        # matches the shape of your new tensor, use copy_:
        with torch.no_grad():
            new_positions = make_new_interpolated_pos_encoding(
                self.encoder.embeddings.position_embeddings,
                self.encoder.config.patch_size,
                shape[0],
                shape[1],
            )
        # Create a new Parameter with the new size
        old_interpolate_pos_encoding = self.encoder.embeddings.interpolate_pos_encoding

        def new_interpolate_pos_encoding(self_mod: Any, embeddings: Tensor, height: int, width: int) -> Tensor:
            num_patches = embeddings.shape[1] - 1
            num_positions = self_mod.position_embeddings.shape[1] - 1
            if num_patches == num_positions and height == width:
                return cast(Tensor, self_mod.position_embeddings)
            return cast(Tensor, old_interpolate_pos_encoding(embeddings, height, width))

        self.encoder.embeddings.position_embeddings = nn.Parameter(new_positions)
        self.encoder.embeddings.interpolate_pos_encoding = types.MethodType(
            new_interpolate_pos_encoding, self.encoder.embeddings
        )

    def forward(self, x: Tensor) -> list[Tensor]:
        block_size = self.patch_size * self.num_windows
        assert x.shape[2] % block_size == 0 and x.shape[3] % block_size == 0, (
            f"Backbone requires input shape to be divisible by {block_size}, but got {x.shape}"
        )
        x = self.encoder(x)
        return list(x[0])


if __name__ == "__main__":
    model = DinoV2()
    model.export()
    x = torch.randn(1, 3, 640, 640)
    logger.info(model(x))
    for j in model(x):
        logger.info(j.shape)

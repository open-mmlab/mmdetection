# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from HuggingFace Transformers (https://github.com/huggingface/transformers)
# Copyright 2022 The HuggingFace Team. All rights reserved.        (pytorch_utils.py)
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.   (backbone_utils.py)
# Copyright 2024 Meta Inc. and the HuggingFace Inc. team. All rights reserved. (DINOv2)
# Licensed under the Apache License, Version 2.0
# ------------------------------------------------------------------------
"""DINOv2-with-Registers backbone with windowed self-attention.

This module is a local copy of the HuggingFace Transformers DINOv2-with-Registers implementation, extended with windowed
attention support for RF-DETR.  It targets the transformers v5 API (``transformers>=5.0.0``).

Transformers v5 API changes vs v4
----------------------------------
``head_mask`` removed:
    The ``head_mask`` parameter that appeared on every ``forward()`` in v4 has been
    dropped in v5.  It defaulted to ``None`` throughout the call chain and callers universally passed ``None``, so
    removing it produces **identical numerics**. Permanent head pruning is still available via ``model._prune_heads()``.

``BackboneMixin._init_transformers_backbone`` signature:
    In v4 this method accepted ``(self, config)``.  In v5 it accepts only ``(self)``;
    the config is accessed via ``self.config`` internally.

Helper functions copied locally:
    ``get_aligned_output_features_output_indices`` and ``find_pruneable_heads_and_indices`` were removed
    from the transformers v5 public API.  Private copies (``_get_aligned_output_features_output_indices``
    and ``_find_pruneable_heads_and_indices``) are kept in this module.
"""

from __future__ import annotations

import collections.abc
import math
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import (  # type: ignore[attr-defined] # public API; stable across all transformers v5.x
    BackboneConfigMixin,
    BackboneMixin,
)
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import (
    BackboneOutput,
    BaseModelOutput,
    BaseModelOutputWithPooling,
    ImageClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import prune_linear_layer
from transformers.utils import (  # type: ignore[attr-defined]
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
    torch_int,
)

logger = logging.get_logger(__name__)


def _find_pruneable_heads_and_indices(
    heads: set[int], n_heads: int, head_size: int, already_pruned_heads: set[int]
) -> tuple[set[int], Tensor]:
    """Return the set of pruneable heads and their index mask for weight pruning.

    Copied from transformers.pytorch_utils.find_pruneable_heads_and_indices (removed from public API in transformers
    v5.0).
    Source: https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/pytorch_utils.py#L127
    MAINTENANCE: if this function is moved to another module or deleted, update the
    "Copyright 2022 The HuggingFace Team" line in the file header accordingly.

    Args:
        heads: Indices of heads to prune.
        n_heads: Total number of heads in the layer.
        head_size: Size of each attention head.
        already_pruned_heads: Heads that have already been pruned.

    Returns:
        A tuple of (heads, index) where heads is the adjusted set of head indices and index is a LongTensor boolean mask
        selecting the remaining weights.
    """
    mask = torch.ones(n_heads, head_size)
    heads = set(heads) - already_pruned_heads
    for head in heads:
        head -= sum(1 if h < head else 0 for h in already_pruned_heads)
        mask[head] = 0
    mask = mask.view(-1).contiguous().eq(1)
    index = torch.arange(len(mask))[mask].long()
    return heads, index


def _align_output_features_output_indices(
    out_features: list[str] | None,
    out_indices: list[int] | tuple[int, ...] | None,
    stage_names: list[str],
) -> tuple[list[str] | None, list[int] | tuple[int, ...] | None]:
    if out_indices is None and out_features is None:
        out_indices = [len(stage_names) - 1]
        out_features = [stage_names[-1]]
    elif out_indices is None and out_features is not None:
        out_indices = [stage_names.index(layer) for layer in out_features]
    elif out_features is None and out_indices is not None:
        out_features = [stage_names[idx] for idx in out_indices]
    return out_features, out_indices


def _get_aligned_output_features_output_indices(
    out_features: list[str] | None,
    out_indices: list[int] | tuple[int, ...] | None,
    stage_names: list[str],
) -> tuple[list[str], list[int]]:
    """Align out_features and out_indices against stage_names, filling in defaults when either is None.

    Copied from transformers.utils.backbone_utils.get_aligned_output_features_output_indices (removed from public API in
    transformers v5.0).
    Source: https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/utils/backbone_utils.py#L30
    MAINTENANCE: if this function is moved to another module or deleted, update the
    "Copyright 2023 The HuggingFace Inc. team" line in the file header accordingly.

    Args:
        out_features: Names of the backbone stages to return features from, or None to derive from out_indices.
        out_indices: Integer indices of the stages to return features from, or None to derive from out_features.
        stage_names: Ordered list of all stage names defined by the backbone config.

    Returns:
        A tuple of (out_features, out_indices) with both fields populated consistently.
    """
    out_indices = list(out_indices) if out_indices is not None else None
    out_features, out_indices = _align_output_features_output_indices(
        out_features=out_features, out_indices=out_indices, stage_names=stage_names
    )
    # By construction, at least one of out_features/out_indices was provided (or both
    # default to the last stage above), so both are always fully resolved here.
    assert out_features is not None
    assert out_indices is not None
    return out_features, cast(list[int], out_indices)


# General docstring
_CONFIG_FOR_DOC = "WindowedDinov2WithRegistersConfig"


class WindowedDinov2WithRegistersConfig(BackboneConfigMixin, PretrainedConfig):  # type: ignore[no-untyped-call,unused-ignore]
    r"""
    This is the configuration class to store the configuration of a [`Dinov2WithRegistersModel`].
    It is used to instantiate a Dinov2WithRegisters model according to the specified arguments,
    defining the model architecture. Instantiating a configuration with the defaults will yield a
    similar configuration to that of the DINOv2 with Registers
    [facebook/dinov2-with-registers-base](https://huggingface.co/facebook/dinov2-with-registers-base) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        mlp_ratio (`int`, *optional*, defaults to 4):
            Ratio of the hidden size of the MLPs relative to the `hidden_size`.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the layer normalization layers.
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the queries, keys and values.
        layerscale_value (`float`, *optional*, defaults to 1.0):
           Initial value to use for layer scale.
        drop_path_rate (`float`, *optional*, defaults to 0.0):
            Stochastic depth rate per sample (when applied in the main path of residual layers).
        use_swiglu_ffn (`bool`, *optional*, defaults to `False`):
            Whether to use the SwiGLU feedforward neural network.
        num_register_tokens (`int`, *optional*, defaults to 4):
            Number of register tokens to use.
        out_features (`List[str]`, *optional*):
            If used as backbone, list of features to output. Can be any of `"stem"`, `"stage1"`, `"stage2"`, etc.
            (depending on how many stages the model has). If unset and `out_indices` is set, will default to the
            corresponding stages. If unset and `out_indices` is unset, will default to the last stage. Must be in the
            same order as defined in the `stage_names` attribute.
        out_indices (`List[int]`, *optional*):
            If used as backbone, list of indices of features to output. Can be any of 0, 1, 2, etc. (depending on how
            many stages the model has). If unset and `out_features` is set, will default to the corresponding stages.
            If unset and `out_features` is unset, will default to the last stage. Must be in the
            same order as defined in the `stage_names` attribute.
        apply_layernorm (`bool`, *optional*, defaults to `True`):
            Whether to apply layer normalization to the feature maps in case the model is used as backbone.
        reshape_hidden_states (`bool`, *optional*, defaults to `True`):
            Whether to reshape the feature maps to 4D tensors of shape `(batch_size, hidden_size, height, width)` in
            case the model is used as backbone. If `False`, the feature maps will be 3D tensors of shape `(batch_size,
            seq_len, hidden_size)`.

    Example:

    >>> from mmdet.models.layers.rf_detr.models.backbone.dinov2_with_windowed_attn import WindowedDinov2WithRegistersConfig

    >>> # Initializing a tiny configuration suitable for doctests
    >>> configuration = WindowedDinov2WithRegistersConfig(
    ...     image_size=32,
    ...     patch_size=16,
    ...     hidden_size=32,
    ...     num_hidden_layers=2,
    ...     num_attention_heads=4,
    ...     num_register_tokens=2,
    ... )

    >>> configuration.hidden_size
    32

    """

    model_type = "dinov2_with_registers"

    def __init__(
        self,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        mlp_ratio: int = 4,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.0,
        attention_probs_dropout_prob: float = 0.0,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-6,
        image_size: int = 224,
        patch_size: int = 16,
        num_channels: int = 3,
        qkv_bias: bool = True,
        layerscale_value: float = 1.0,
        drop_path_rate: float = 0.0,
        use_swiglu_ffn: bool = False,
        num_register_tokens: int = 4,
        out_features: list[str] | None = None,
        out_indices: list[int] | tuple[int, ...] | None = None,
        apply_layernorm: bool = True,
        reshape_hidden_states: bool = True,
        num_windows: int = 1,
        window_block_indexes: list[int] | None = None,
        gradient_checkpointing: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.mlp_ratio = mlp_ratio
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.qkv_bias = qkv_bias
        self.layerscale_value = layerscale_value
        self.drop_path_rate = drop_path_rate
        self.use_swiglu_ffn = use_swiglu_ffn
        self.num_register_tokens = num_register_tokens
        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, num_hidden_layers + 1)]
        self._out_features, self._out_indices = _get_aligned_output_features_output_indices(
            out_features=out_features, out_indices=out_indices, stage_names=self.stage_names
        )
        self.apply_layernorm = apply_layernorm
        self.reshape_hidden_states = reshape_hidden_states
        self.num_windows = num_windows
        self.window_block_indexes = (
            list(range(num_hidden_layers)) if window_block_indexes is None else window_block_indexes
        )
        self.gradient_checkpointing = gradient_checkpointing


class Dinov2WithRegistersPatchEmbeddings(nn.Module):
    """This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer."""

    def __init__(self, config: "WindowedDinov2WithRegistersConfig") -> None:
        super().__init__()
        image_size_cfg, patch_size_cfg = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size

        image_size = cast(
            "tuple[int, int]",
            image_size_cfg
            if isinstance(image_size_cfg, collections.abc.Iterable)
            else (image_size_cfg, image_size_cfg),
        )
        patch_size = cast(
            "tuple[int, int]",
            patch_size_cfg
            if isinstance(patch_size_cfg, collections.abc.Iterable)
            else (patch_size_cfg, patch_size_cfg),
        )
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values: Tensor) -> Tensor:
        num_channels = pixel_values.shape[1]
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
                f" Expected {self.num_channels} but got {num_channels}."
            )
        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return cast(Tensor, embeddings)


class WindowedDinov2WithRegistersEmbeddings(nn.Module):
    """Construct the CLS token, mask token, register tokens, position and patch embeddings."""

    def __init__(self, config: WindowedDinov2WithRegistersConfig) -> None:
        super().__init__()

        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.mask_token = nn.Parameter(torch.zeros(1, config.hidden_size))
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, config.num_register_tokens, config.hidden_size))
            if config.num_register_tokens > 0
            else None
        )
        self.patch_embeddings = Dinov2WithRegistersPatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = nn.Parameter(torch.randn(1, num_patches + 1, config.hidden_size))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.patch_size = config.patch_size
        self.config = config

    def interpolate_pos_encoding(self, embeddings: Tensor, height: int, width: int) -> Tensor:
        """This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
        resolution images. This implementation supports torch.jit tracing while maintaining backwards compatibility with
        the original implementation.

        Adapted from:
        - https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
        - https://github.com/facebookresearch/dinov2/blob/main/dinov2/models/vision_transformer.py
        """
        num_patches = embeddings.shape[1] - 1
        num_positions = self.position_embeddings.shape[1] - 1

        # Skip interpolation for matching dimensions (unless tracing)
        if (
            not torch.jit.is_tracing()  # type: ignore[attr-defined,no-untyped-call]
            and num_patches == num_positions
            and height == width
        ):
            return cast(Tensor, self.position_embeddings)

        # Handle class token and patch embeddings separately
        class_pos_embed = self.position_embeddings[:, 0]
        patch_pos_embed = self.position_embeddings[:, 1:]
        dim = embeddings.shape[-1]

        # Calculate new dimensions
        height = height // self.config.patch_size
        width = width // self.config.patch_size

        # Reshape for interpolation
        sqrt_num_positions = torch_int(num_positions**0.5)  # type: ignore[no-untyped-call]
        patch_pos_embed = patch_pos_embed.reshape(1, sqrt_num_positions, sqrt_num_positions, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)

        # Store original dtype for restoration after interpolation
        target_dtype = patch_pos_embed.dtype

        # Interpolate at float32 precision
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.to(dtype=torch.float32),
            size=(
                torch_int(height),  # type: ignore[no-untyped-call]
                torch_int(width),  # type: ignore[no-untyped-call]
            ),  # Explicit size instead of scale_factor
            mode="bicubic",
            align_corners=False,
            antialias=patch_pos_embed.device.type != "mps",
        ).to(dtype=target_dtype)

        # Validate output dimensions if not tracing
        if not torch.jit.is_tracing():  # type: ignore[attr-defined,no-untyped-call]
            if int(height) != patch_pos_embed.shape[-2] or int(width) != patch_pos_embed.shape[-1]:
                raise ValueError("Width or height does not match with the interpolated position embeddings")

        # Reshape back to original format
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

        # Combine class and patch embeddings
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def forward(self, pixel_values: Tensor, bool_masked_pos: Tensor | None = None) -> Tensor:
        """Compute windowed patch embeddings for the given pixel values.

        Args:
            pixel_values: Image tensor of shape ``(B, C, H, W)``. Both ``H`` and
                ``W`` must be divisible by ``patch_size * num_windows``.
            bool_masked_pos: Optional boolean mask of shape ``(B, num_patches)``.
                Masked positions are replaced with the learnable ``mask_token``.

        Returns:
            Patch embedding tensor. When ``num_windows > 1`` the batch dimension is expanded to ``B * num_windows ** 2``
            and the sequence length corresponds to patches within a single window (plus CLS token and any register
            tokens).

        Raises:
            ValueError: If ``H`` or ``W`` is not divisible by
                ``patch_size * num_windows``.
        """
        batch_size, _, height, width = pixel_values.shape
        divisor = self.patch_size * self.config.num_windows
        if height % divisor != 0 or width % divisor != 0:
            raise ValueError(
                f"Input spatial dimensions must be divisible by patch_size * num_windows "
                f"({self.patch_size} * {self.config.num_windows} = {divisor}), "
                f"but got height={height}, width={width}."
            )
        target_dtype = self.patch_embeddings.projection.weight.dtype
        embeddings = self.patch_embeddings(pixel_values.to(dtype=target_dtype))

        if bool_masked_pos is not None:
            embeddings = torch.where(
                bool_masked_pos.unsqueeze(-1), self.mask_token.to(embeddings.dtype).unsqueeze(0), embeddings
            )

        # add the [CLS] token to the embedded patch tokens
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # add positional encoding to each token
        embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)

        if self.config.num_windows > 1:
            # reshape for windows
            num_h_patches = height // self.config.patch_size
            num_w_patches = width // self.config.patch_size
            cls_token_with_pos_embed = embeddings[:, :1]
            pixel_tokens_with_pos_embed = embeddings[:, 1:]
            pixel_tokens_with_pos_embed = pixel_tokens_with_pos_embed.view(batch_size, num_h_patches, num_w_patches, -1)
            num_w_patches_per_window = num_w_patches // self.config.num_windows
            num_h_patches_per_window = num_h_patches // self.config.num_windows
            num_windows = self.config.num_windows
            windowed_pixel_tokens = pixel_tokens_with_pos_embed.reshape(
                batch_size * num_windows, num_h_patches_per_window, num_windows, num_w_patches_per_window, -1
            )
            windowed_pixel_tokens = windowed_pixel_tokens.permute(0, 2, 1, 3, 4)
            windowed_pixel_tokens = windowed_pixel_tokens.reshape(
                batch_size * num_windows**2, num_h_patches_per_window * num_w_patches_per_window, -1
            )
            windowed_cls_token_with_pos_embed = cls_token_with_pos_embed.repeat(num_windows**2, 1, 1)
            embeddings = torch.cat((windowed_cls_token_with_pos_embed, windowed_pixel_tokens), dim=1)

        # add register tokens
        if self.config.num_register_tokens > 0:
            assert self.register_tokens is not None
            embeddings = torch.cat(
                (embeddings[:, :1], self.register_tokens.expand(embeddings.shape[0], -1, -1), embeddings[:, 1:]), dim=1
            )

        embeddings = self.dropout(embeddings)

        return cast(Tensor, embeddings)


class Dinov2WithRegistersSelfAttention(nn.Module):
    def __init__(self, config: WindowedDinov2WithRegistersConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {(config.hidden_size,)} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: Tensor) -> Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self, hidden_states: Tensor, output_attentions: bool = False
    ) -> tuple[Tensor, Tensor | None] | tuple[Tensor]:
        # Note: head_mask was removed in the transformers v5 migration.
        # In v4 the parameter defaulted to None and callers universally passed None,
        # so dropping it produces identical numerics.  Permanent head pruning is still
        # available via model._prune_heads().
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


class Dinov2WithRegistersSdpaSelfAttention(Dinov2WithRegistersSelfAttention):
    def __init__(self, config: WindowedDinov2WithRegistersConfig) -> None:
        super().__init__(config)
        self.attention_probs_dropout_prob = config.attention_probs_dropout_prob

    def forward(
        self, hidden_states: Tensor, output_attentions: bool = False
    ) -> tuple[Tensor, Tensor | None] | tuple[Tensor]:
        if output_attentions:
            logger.warning_once(
                "Dinov2WithRegistersModel is using Dinov2WithRegistersSdpaSelfAttention, "
                "but `torch.nn.functional.scaled_dot_product_attention` does not support "
                "`output_attentions=True`. Falling back to the manual attention implementation. "
                "To avoid this fallback, call `model.set_attn_implementation('eager')` "
                'or pass `attn_implementation="eager"` when instantiating the model.'
            )
            return super().forward(hidden_states=hidden_states, output_attentions=output_attentions)

        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        context_layer = torch.nn.functional.scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            None,
            self.attention_probs_dropout_prob if self.training else 0.0,
            is_causal=False,
            scale=None,
        )

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        return context_layer, None


class Dinov2WithRegistersSelfOutput(nn.Module):
    """The residual connection is defined in Dinov2WithRegistersLayer instead of here (as is the case with other
    models), due to the layernorm applied before each block."""

    def __init__(self, config: WindowedDinov2WithRegistersConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: Tensor, input_tensor: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class Dinov2WithRegistersAttention(nn.Module):
    def __init__(self, config: WindowedDinov2WithRegistersConfig) -> None:
        super().__init__()
        self.attention = Dinov2WithRegistersSelfAttention(config)
        self.output = Dinov2WithRegistersSelfOutput(config)
        self.pruned_heads: set[int] = set()

    def prune_heads(self, heads: set[int]) -> None:
        if len(heads) == 0:
            return
        heads, index_tensor = _find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )
        index = cast(torch.LongTensor, index_tensor)

        # Prune linear layers
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: Tensor,
        output_attentions: bool = False,
    ) -> tuple[Tensor, Tensor | None] | tuple[Tensor]:
        self_outputs = self.attention(hidden_states, output_attentions)

        attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return cast("tuple[Tensor, Tensor | None] | tuple[Tensor]", outputs)


class Dinov2WithRegistersSdpaAttention(Dinov2WithRegistersAttention):
    def __init__(self, config: WindowedDinov2WithRegistersConfig) -> None:
        super().__init__(config)
        self.attention = Dinov2WithRegistersSdpaSelfAttention(config)


class Dinov2WithRegistersLayerScale(nn.Module):
    def __init__(self, config: "WindowedDinov2WithRegistersConfig") -> None:
        super().__init__()
        self.lambda1 = nn.Parameter(config.layerscale_value * torch.ones(config.hidden_size))

    def forward(self, hidden_state: Tensor) -> Tensor:
        return hidden_state * self.lambda1


def drop_path(input: Tensor, drop_prob: float = 0.0, training: bool = False) -> Tensor:
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Comment by Ross Wightman: This is the same as the DropConnect impl I created for EfficientNet, etc networks,
    however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper... See
    discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the
    layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the
    argument.
    """
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()  # binarize
    output = input.div(keep_prob) * random_tensor
    return output


class Dinov2WithRegistersDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float | None = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob if drop_prob is not None else 0.0

    def forward(self, hidden_states: Tensor) -> Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return f"p={self.drop_prob}"


class Dinov2WithRegistersMLP(nn.Module):
    def __init__(self, config: "WindowedDinov2WithRegistersConfig") -> None:
        super().__init__()
        in_features = out_features = config.hidden_size
        hidden_features = int(config.hidden_size * config.mlp_ratio)
        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        if isinstance(config.hidden_act, str):
            self.activation = ACT2FN[config.hidden_act]
        else:
            self.activation = config.hidden_act
        self.fc2 = nn.Linear(hidden_features, out_features, bias=True)

    def forward(self, hidden_state: Tensor) -> Tensor:
        hidden_state = self.fc1(hidden_state)
        hidden_state = self.activation(hidden_state)
        hidden_state = self.fc2(hidden_state)
        return hidden_state


class Dinov2WithRegistersSwiGLUFFN(nn.Module):
    def __init__(self, config: "WindowedDinov2WithRegistersConfig") -> None:
        super().__init__()
        in_features = out_features = config.hidden_size
        hidden_features = int(config.hidden_size * config.mlp_ratio)
        hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8

        self.weights_in = nn.Linear(in_features, 2 * hidden_features, bias=True)
        self.weights_out = nn.Linear(hidden_features, out_features, bias=True)

    def forward(self, hidden_state: Tensor) -> Tensor:
        hidden_state = self.weights_in(hidden_state)
        x1, x2 = hidden_state.chunk(2, dim=-1)
        hidden = nn.functional.silu(x1) * x2
        return cast(Tensor, self.weights_out(hidden))


DINOV2_WITH_REGISTERS_ATTENTION_CLASSES = {
    "eager": Dinov2WithRegistersAttention,
    "sdpa": Dinov2WithRegistersSdpaAttention,
}


class WindowedDinov2WithRegistersLayer(nn.Module):
    """This corresponds to the Block class in the original implementation."""

    def __init__(self, config: WindowedDinov2WithRegistersConfig) -> None:
        super().__init__()

        self.num_windows = config.num_windows

        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = DINOV2_WITH_REGISTERS_ATTENTION_CLASSES[config._attn_implementation](config)
        self.layer_scale1 = Dinov2WithRegistersLayerScale(config)
        self.drop_path = (
            Dinov2WithRegistersDropPath(config.drop_path_rate) if config.drop_path_rate > 0.0 else nn.Identity()
        )

        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.mlp: nn.Module
        if config.use_swiglu_ffn:
            self.mlp = Dinov2WithRegistersSwiGLUFFN(config)
        else:
            self.mlp = Dinov2WithRegistersMLP(config)
        self.layer_scale2 = Dinov2WithRegistersLayerScale(config)

    def forward(
        self,
        hidden_states: Tensor,
        output_attentions: bool = False,
        run_full_attention: bool = False,
    ) -> tuple[Tensor, Tensor | None] | tuple[Tensor]:
        assert not output_attentions, "output_attentions is not supported for windowed attention"
        shortcut = hidden_states
        if run_full_attention:
            # reshape x to remove windows
            batch_windows, tokens_per_window, channels = hidden_states.shape
            num_windows_squared = self.num_windows**2
            hidden_states = hidden_states.view(
                batch_windows // num_windows_squared, num_windows_squared * tokens_per_window, channels
            )

        self_attention_outputs = self.attention(
            self.norm1(hidden_states),  # in Dinov2WithRegisters, layernorm is applied before self-attention
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]

        if run_full_attention:
            # reshape x to add windows back
            batch_windows, tokens_per_window, channels = hidden_states.shape
            num_windows_squared = self.num_windows**2
            # hidden_states = hidden_states.view(B * num_windows_squared, HW // num_windows_squared, C)
            attention_output = attention_output.view(
                batch_windows * num_windows_squared, tokens_per_window // num_windows_squared, channels
            )

        attention_output = self.layer_scale1(attention_output)
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # first residual connection
        hidden_states = self.drop_path(attention_output) + shortcut

        # in Dinov2WithRegisters, layernorm is also applied after self-attention
        layer_output = self.norm2(hidden_states)
        layer_output = self.mlp(layer_output)
        layer_output = self.layer_scale2(layer_output)

        # second residual connection
        layer_output = self.drop_path(layer_output) + hidden_states

        outputs = (layer_output,) + outputs

        return cast("tuple[Tensor, Tensor | None] | tuple[Tensor]", outputs)


class WindowedDinov2WithRegistersEncoder(nn.Module):
    def __init__(self, config: WindowedDinov2WithRegistersConfig) -> None:
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([WindowedDinov2WithRegistersLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = config.gradient_checkpointing

    def forward(
        self,
        hidden_states: Tensor,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> tuple[Any, ...] | BaseModelOutput:
        all_hidden_states: tuple[Tensor, ...] | None = () if output_hidden_states else None
        all_self_attentions: tuple[Tensor, ...] | None = () if output_attentions else None

        last_feature = self.config.out_features[-1]
        # Feature names are "stage<N>"; only those carry a layer index to early-stop on.
        early_stop_layer = int(last_feature[5:]) if last_feature.startswith("stage") else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                assert all_hidden_states is not None
                all_hidden_states = all_hidden_states + (hidden_states,)

            if early_stop_layer is not None and i > early_stop_layer:
                # early stop if we have reached the last output feature
                break

            run_full_attention = i not in self.config.window_block_indexes

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(  # type: ignore[operator]
                    layer_module.__call__,
                    hidden_states,
                    output_attentions,
                    run_full_attention,
                )
            else:
                layer_outputs = layer_module(hidden_states, output_attentions, run_full_attention)

            hidden_states = layer_outputs[0]

            if output_attentions:
                assert all_self_attentions is not None
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            assert all_hidden_states is not None
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=cast("torch.FloatTensor", hidden_states),
            hidden_states=cast("tuple[torch.FloatTensor, ...] | None", all_hidden_states),
            attentions=cast("tuple[torch.FloatTensor, ...] | None", all_self_attentions),
        )


class WindowedDinov2WithRegistersPreTrainedModel(PreTrainedModel):  # type: ignore[no-untyped-call]
    """An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models."""

    config_class = WindowedDinov2WithRegistersConfig
    base_model_prefix = "dinov2_with_registers"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Dinov2WithRegistersSwiGLUFFN"]
    _supports_sdpa = True

    def _init_weights(self, module: nn.Linear | nn.Conv2d | nn.LayerNorm) -> None:
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Upcast the input in `fp32` and cast it back to desired `dtype` to avoid
            # `trunc_normal_cpu` not implemented in `half` issues
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data.to(torch.float32), mean=0.0, std=self.config.initializer_range
            ).to(module.weight.dtype)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, WindowedDinov2WithRegistersEmbeddings):
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.position_embeddings.dtype)

            module.cls_token.data = nn.init.trunc_normal_(
                module.cls_token.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.cls_token.dtype)


DINOV2_WITH_REGISTERS_START_DOCSTRING = """
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`Dinov2WithRegistersConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

DINOV2_WITH_REGISTERS_BASE_INPUTS_DOCSTRING = """
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`BitImageProcessor.preprocess`] for details.

        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, sequence_length)`):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0). Only relevant for
            pre-training.

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(  # type: ignore[no-untyped-call]
    "The bare Dinov2WithRegisters Model transformer outputting raw hidden-states without any specific head on top.",
    DINOV2_WITH_REGISTERS_START_DOCSTRING,
)
class WindowedDinov2WithRegistersModel(WindowedDinov2WithRegistersPreTrainedModel):  # type: ignore[no-untyped-call]
    def __init__(self, config: WindowedDinov2WithRegistersConfig) -> None:
        super().__init__(config)
        self.config = config

        self.embeddings = WindowedDinov2WithRegistersEmbeddings(config)
        self.encoder = WindowedDinov2WithRegistersEncoder(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Initialize weights and apply final processing
        self.post_init()  # type: ignore[no-untyped-call]

    def get_input_embeddings(self) -> Dinov2WithRegistersPatchEmbeddings:
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune: dict[int, list[int]]) -> None:
        """Prunes heads of the model.

        heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            cast(WindowedDinov2WithRegistersLayer, self.encoder.layer[layer]).attention.prune_heads(set(heads))

    def set_attn_implementation(self, attn_implementation: str) -> None:  # type: ignore[override]
        """Switch the attention implementation without reloading the model.

        This is useful when you want to change the attention implementation after the model has been instantiated — for
        example, to use ``"eager"`` (manual) attention when inspecting attention weights, without having to reconstruct
        the entire model from scratch.

        Args:
            attn_implementation: One of ``"eager"`` (manual attention) or ``"sdpa"``
                (:func:`torch.nn.functional.scaled_dot_product_attention`).

        Raises:
            ValueError: If *attn_implementation* is not a supported key.

        Example::

            >>> from mmdet.models.layers.rf_detr.models.backbone.dinov2_with_windowed_attn import (
            ...     WindowedDinov2WithRegistersConfig,
            ...     WindowedDinov2WithRegistersModel,
            ... )
            >>> config = WindowedDinov2WithRegistersConfig(
            ...     image_size=32,
            ...     patch_size=16,
            ...     hidden_size=32,
            ...     num_hidden_layers=1,
            ...     num_attention_heads=4,
            ...     num_register_tokens=2,
            ... )
            >>> model = WindowedDinov2WithRegistersModel(config)
            >>> model.set_attn_implementation("eager")
            >>> model.config._attn_implementation
            'eager'
        """
        if attn_implementation not in DINOV2_WITH_REGISTERS_ATTENTION_CLASSES:
            raise ValueError(
                f"Unknown attn_implementation {attn_implementation!r}. "
                f"Choose from {sorted(DINOV2_WITH_REGISTERS_ATTENTION_CLASSES)}."
            )
        self.config._attn_implementation = attn_implementation
        for layer_module in self.encoder.layer:
            layer = cast(WindowedDinov2WithRegistersLayer, layer_module)
            new_attn = DINOV2_WITH_REGISTERS_ATTENTION_CLASSES[attn_implementation](
                cast(WindowedDinov2WithRegistersConfig, self.config)
            )
            # Transfer trained weights: eager and sdpa variants share identical parameter keys
            # (both contain attention.{query,key,value,dropout} and output.{dense,dropout}).
            # strict=True is intentional — a key mismatch would indicate a structural divergence
            # that should be caught and fixed rather than silently skipped.
            new_attn.load_state_dict(layer.attention.state_dict())
            layer.attention = new_attn

    @add_start_docstrings_to_model_forward(  # type: ignore[untyped-decorator,no-untyped-call]
        DINOV2_WITH_REGISTERS_BASE_INPUTS_DOCSTRING
    )
    @replace_return_docstrings(  # type: ignore[untyped-decorator,no-untyped-call]
        output_type=BaseModelOutputWithPooling, config_class=_CONFIG_FOR_DOC
    )
    def forward(
        self,
        pixel_values: Tensor | None = None,
        bool_masked_pos: Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple[Any, ...] | BaseModelOutputWithPooling:
        """
        Returns:

        Examples:

        >>> import torch
        >>> from mmdet.models.layers.rf_detr.models.backbone.dinov2_with_windowed_attn import (
        ...     WindowedDinov2WithRegistersConfig,
        ...     WindowedDinov2WithRegistersModel,
        ... )
        >>> config = WindowedDinov2WithRegistersConfig(
        ...     image_size=32,
        ...     patch_size=16,
        ...     hidden_size=32,
        ...     num_hidden_layers=2,
        ...     num_attention_heads=4,
        ...     num_register_tokens=2,
        ... )
        >>> model = WindowedDinov2WithRegistersModel(config)
        >>> pixel_values = torch.randn(1, 3, 32, 32)
        >>> outputs = model(pixel_values)
        >>> list(outputs.last_hidden_state.shape)
        [1, 7, 32]
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        embedding_output = self.embeddings(pixel_values, bool_masked_pos=bool_masked_pos)

        encoder_outputs = self.encoder(
            embedding_output,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = sequence_output[:, 0, :]

        if not return_dict:
            head_outputs = (sequence_output, pooled_output)
            return cast("tuple[Any, ...]", head_outputs + encoder_outputs[1:])

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


DINOV2_WITH_REGISTERS_INPUTS_DOCSTRING = """
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`BitImageProcessor.preprocess`] for details.

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(  # type: ignore[no-untyped-call]
    (
        "Dinov2WithRegisters Model transformer with an image classification head on top "
        "(a linear layer on top of the final hidden state of the [CLS] token) e.g. for ImageNet."
    ),
    DINOV2_WITH_REGISTERS_START_DOCSTRING,
)
class WindowedDinov2WithRegistersForImageClassification(
    WindowedDinov2WithRegistersPreTrainedModel  # type: ignore[no-untyped-call]
):
    def __init__(self, config: WindowedDinov2WithRegistersConfig) -> None:
        super().__init__(config)

        self.num_labels = config.num_labels
        self.dinov2_with_registers = WindowedDinov2WithRegistersModel(config)

        # Classifier head
        self.classifier = (
            nn.Linear(config.hidden_size * 2, config.num_labels) if config.num_labels > 0 else nn.Identity()
        )

        # Initialize weights and apply final processing
        self.post_init()  # type: ignore[no-untyped-call]

    @add_start_docstrings_to_model_forward(  # type: ignore[untyped-decorator,no-untyped-call]
        DINOV2_WITH_REGISTERS_INPUTS_DOCSTRING
    )
    @replace_return_docstrings(  # type: ignore[untyped-decorator,no-untyped-call]
        output_type=ImageClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        pixel_values: Tensor | None = None,
        labels: Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple[Any, ...] | ImageClassifierOutput:
        r"""Labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):

            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Returns:

        Example:

        >>> import torch
        >>> from mmdet.models.layers.rf_detr.models.backbone.dinov2_with_windowed_attn import (
        ...     WindowedDinov2WithRegistersConfig,
        ...     WindowedDinov2WithRegistersForImageClassification,
        ... )
        >>> config = WindowedDinov2WithRegistersConfig(
        ...     image_size=32,
        ...     patch_size=16,
        ...     hidden_size=32,
        ...     num_hidden_layers=2,
        ...     num_attention_heads=4,
        ...     num_register_tokens=2,
        ...     num_labels=3,
        ... )
        >>> model = WindowedDinov2WithRegistersForImageClassification(config)
        >>> pixel_values = torch.randn(1, 3, 32, 32)
        >>> outputs = model(pixel_values)
        >>> list(outputs.logits.shape)
        [1, 3]
        """
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        outputs = self.dinov2_with_registers(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]  # batch_size, sequence_length, hidden_size

        cls_token = sequence_output[:, 0]
        patch_tokens = sequence_output[:, 1:]

        linear_input = torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=1)

        logits = self.classifier(linear_input)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            loss_fct: nn.Module
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return cast("tuple[Any, ...]", (loss, *output) if loss is not None else output)

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(  # type: ignore[no-untyped-call]
    "Dinov2WithRegisters backbone, to be used with frameworks like DETR and MaskFormer.",
    DINOV2_WITH_REGISTERS_START_DOCSTRING,
)
class WindowedDinov2WithRegistersBackbone(
    WindowedDinov2WithRegistersPreTrainedModel,  # type: ignore[no-untyped-call]
    BackboneMixin,
):
    def __init__(self, config: WindowedDinov2WithRegistersConfig) -> None:
        super().__init__(config)
        self._init_transformers_backbone()
        self.num_features = [config.hidden_size for _ in range(config.num_hidden_layers + 1)]
        self.embeddings = WindowedDinov2WithRegistersEmbeddings(config)
        self.encoder = WindowedDinov2WithRegistersEncoder(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.num_register_tokens = config.num_register_tokens

        # Initialize weights and apply final processing
        self.post_init()  # type: ignore[no-untyped-call]

    def get_input_embeddings(self) -> Dinov2WithRegistersPatchEmbeddings:
        return self.embeddings.patch_embeddings

    @add_start_docstrings_to_model_forward(  # type: ignore[untyped-decorator,no-untyped-call]
        DINOV2_WITH_REGISTERS_INPUTS_DOCSTRING
    )
    @replace_return_docstrings(  # type: ignore[untyped-decorator,no-untyped-call]
        output_type=BackboneOutput, config_class=_CONFIG_FOR_DOC
    )
    def forward(
        self,
        pixel_values: Tensor,
        output_hidden_states: bool | None = None,
        output_attentions: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple[Any, ...] | BackboneOutput:
        """
        Returns:

        Examples:

        >>> import torch
        >>> from mmdet.models.layers.rf_detr.models.backbone.dinov2_with_windowed_attn import (
        ...     WindowedDinov2WithRegistersBackbone,
        ...     WindowedDinov2WithRegistersConfig,
        ... )
        >>> config = WindowedDinov2WithRegistersConfig(
        ...     image_size=32,
        ...     patch_size=16,
        ...     hidden_size=32,
        ...     num_hidden_layers=2,
        ...     num_attention_heads=4,
        ...     num_register_tokens=2,
        ...     out_indices=[2],
        ... )
        >>> model = WindowedDinov2WithRegistersBackbone(config)
        >>> pixel_values = torch.randn(1, 3, 32, 32)
        >>> outputs = model(pixel_values)
        >>> len(outputs.feature_maps)
        1
        >>> list(outputs.feature_maps[0].shape)
        [1, 32, 2, 2]

        """
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        embedding_output = self.embeddings(pixel_values)

        outputs = self.encoder(
            embedding_output, output_hidden_states=True, output_attentions=output_attentions, return_dict=return_dict
        )

        hidden_states = outputs.hidden_states if return_dict else outputs[1]

        feature_maps: tuple[torch.Tensor, ...] = ()
        for stage, hidden_state in zip(self.stage_names, hidden_states):
            if stage in self.out_features:
                if self.config.apply_layernorm:
                    hidden_state = self.layernorm(hidden_state)
                if self.config.reshape_hidden_states:
                    hidden_state = hidden_state[:, self.num_register_tokens + 1 :]
                    # this was actually a bug in the original implementation that we copied here,
                    # cause normally the order is height, width
                    batch_size, _, height, width = pixel_values.shape
                    patch_size = self.config.patch_size

                    num_h_patches = height // patch_size
                    num_w_patches = width // patch_size

                    if self.config.num_windows > 1:
                        # undo windowing
                        num_windows_squared = self.config.num_windows**2
                        batch_windows, tokens_per_window, channels = hidden_state.shape
                        num_h_patches_per_window = num_h_patches // self.config.num_windows
                        num_w_patches_per_window = num_w_patches // self.config.num_windows
                        hidden_state = hidden_state.reshape(
                            batch_windows // num_windows_squared,
                            num_windows_squared * tokens_per_window,
                            channels,
                        )
                        hidden_state = hidden_state.reshape(
                            (batch_windows // num_windows_squared) * self.config.num_windows,
                            self.config.num_windows,
                            num_h_patches_per_window,
                            num_w_patches_per_window,
                            channels,
                        )
                        hidden_state = hidden_state.permute(0, 2, 1, 3, 4)

                    hidden_state = hidden_state.reshape(batch_size, num_h_patches, num_w_patches, -1)
                    hidden_state = hidden_state.permute(0, 3, 1, 2).contiguous()

                feature_maps += (hidden_state,)

        if not return_dict:
            if output_hidden_states:
                output = (feature_maps,) + outputs[1:]
            else:
                output = (feature_maps,) + outputs[2:]
            return cast("tuple[Any, ...]", output)

        return BackboneOutput(
            # The HF stub types `feature_maps` as a fixed 1-tuple, but a multi-stage
            # backbone genuinely returns a variable-length tuple of feature maps.
            feature_maps=cast(Any, feature_maps),
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions if output_attentions else None,
        )


__all__ = [
    "WindowedDinov2WithRegistersPreTrainedModel",
    "WindowedDinov2WithRegistersModel",
    "WindowedDinov2WithRegistersForImageClassification",
    "WindowedDinov2WithRegistersBackbone",
]

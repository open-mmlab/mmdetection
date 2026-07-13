# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Shared weight-loading and LoRA application utilities.

Provides the canonical implementations of pretrained checkpoint loading and LoRA adapter injection, used by both the L1
inference facade (``rfdetr.detr``) and the L2 LightningModule (``rfdetr.training.module_model``).

The weight-loading logic is taken from ``RFDETRModelModule._load_pretrain_weights`` in ``module_model.py`` (more
complete: Pydantic-aware user-override detection, auto-alignment for fine-tuned checkpoints) and augmented with class-
name extraction from ``detr.py:_load_pretrain_weights_into``.
"""

from __future__ import annotations

import math
import os
from typing import Any, cast

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from mmdet.models.layers.rf_detr.assets.model_weights import download_pretrain_weights, validate_pretrain_weights
from mmdet.models.layers.rf_detr.config import ModelConfig, TrainConfig
from mmdet.models.layers.rf_detr.models.backbone.backbone import Backbone
from mmdet.models.layers.rf_detr.models.lwdetr import LWDETR
from mmdet.models.layers.rf_detr.utilities.decorators import TargetMode, deprecated
from mmdet.models.layers.rf_detr.utilities.logger import get_logger
from mmdet.models.layers.rf_detr.utilities.state_dict import _ckpt_args_get, remap_projector_to_cross_attn, validate_checkpoint_compatibility

logger = get_logger()

__all__ = ["load_pretrain_weights", "apply_lora", "interpolate_position_embeddings"]

_PE_KEY_SUFFIX = "embeddings.position_embeddings"

# Query-related parameters that LWDETR packs as nn.Embedding(num_queries * group_detr, ...).
# Any new query parameter packed the same way must be added here.
_QUERY_PARAM_SUFFIXES: tuple[str, ...] = ("refpoint_embed.weight", "query_feat.weight")


def _slice_query_param_per_group(
    tensor: Tensor,
    ckpt_num_queries: int,
    ckpt_group_detr: int,
    target_num_queries: int,
    target_group_detr: int,
) -> Tensor:
    """Slice a ``refpoint_embed`` / ``query_feat`` weight preserving per-group structure.

    ``LWDETR`` packs query embeddings as ``nn.Embedding(num_queries * group_detr, ...)`` where group ``g`` occupies the
    contiguous slot range ``[g * num_queries, (g + 1) * num_queries)`` (see ``LWDETR.__init__`` and ``LWDETR.forward``
    in ``models/lwdetr.py``).  When ``num_queries`` decreases and ``group_detr > 1``, a flat ``tensor[:
    target_num_queries * target_group_detr]`` slice silently scrambles groups: the tail of group 0 winds up in what
    should be group 1's slots, and so on.  At inference only group 0 is read so the bug is invisible, but for
    training-resume it corrupts groups 1+.

    This helper does the right thing per group:

    * ``num_queries`` decrease (``target_num_queries < ckpt_num_queries``) →
      keep the first ``target_num_queries`` slots of each retained group.
    * ``group_detr`` decrease (``target_group_detr < ckpt_group_detr``) →
      drop tail groups; retained groups stay pretrained.
    * Either dimension expands, or one shrinks while the other expands →
      return whatever per-group sub-tensor can be built (``min(target, ckpt)`` along each axis). The result has fewer
      rows than the model expects, so ``load_state_dict`` will raise a shape mismatch immediately.

    When the tensor's flat length disagrees with ``ckpt_num_queries * ckpt_group_detr`` (corrupt or unexpected
    checkpoint shape), fall back to the legacy flat slice so loading continues with the same behavior the codebase had
    before this fix.

    Args:
        tensor: The checkpoint tensor for ``refpoint_embed.weight`` or
            ``query_feat.weight``.
        ckpt_num_queries: ``num_queries`` recorded in the checkpoint's training args.
        ckpt_group_detr: ``group_detr`` recorded in the checkpoint's training args.
        target_num_queries: ``num_queries`` configured for the model.
        target_group_detr: ``group_detr`` configured for the model.

    Returns:
        A tensor whose layout matches the model's configured packing for the decrease-or-equal cases, or a per-group
        sub-tensor built from ``min(target, ckpt)`` along each axis for the expansion case (which ``load_state_dict``
        will then reject on shape mismatch).

    Raises:
        ValueError: If any of ``ckpt_num_queries``, ``ckpt_group_detr``,
            ``target_num_queries``, or ``target_group_detr`` is ≤ 0.
    """
    if ckpt_num_queries <= 0 or ckpt_group_detr <= 0 or target_num_queries <= 0 or target_group_detr <= 0:
        raise ValueError(
            f"_slice_query_param_per_group: all dimension args must be positive; "
            f"got ckpt_num_queries={ckpt_num_queries}, ckpt_group_detr={ckpt_group_detr}, "
            f"target_num_queries={target_num_queries}, target_group_detr={target_group_detr}."
        )

    expected_total = ckpt_num_queries * ckpt_group_detr
    if tensor.shape[0] != expected_total:
        # Args inconsistent with tensor shape — fall back to legacy flat slice.
        logger.warning(
            "_slice_query_param_per_group: checkpoint args claim %d × %d = %d rows "
            "but tensor has %d rows; falling back to flat slice. Per-group structure "
            "may be scrambled if group_detr > 1.",
            ckpt_num_queries,
            ckpt_group_detr,
            expected_total,
            tensor.shape[0],
        )
        return tensor[: target_num_queries * target_group_detr]

    if target_num_queries == ckpt_num_queries and target_group_detr == ckpt_group_detr:
        return tensor

    keep_groups = min(target_group_detr, ckpt_group_detr)
    keep_per_group = min(target_num_queries, ckpt_num_queries)
    pieces = [tensor[g * ckpt_num_queries : g * ckpt_num_queries + keep_per_group] for g in range(keep_groups)]
    return torch.cat(pieces, dim=0)


def _filter_intentional_keys(keys: list[str]) -> list[str]:
    """Return *keys* with intentional-reinit/trim entries removed.

    Matching is boundary-aware: a pattern matches a key when the pattern appears at the start of the key or immediately
    after a module separator (``.``).  This prevents substring collisions where a pattern like ``"class_embed."`` would
    inadvertently match a key belonging to an unrelated module (e.g. ``"class_embed_projection.weight"`` is safe because
    ``class_embed_projection.`` ≠ ``class_embed.``, but using a plain ``in`` check against longer ambiguous strings is
    fragile by design).
    """
    # Substrings identifying state-dict keys that ``load_pretrain_weights`` is
    # *expected* to have to reconcile (head reinitialisation and per-group query
    # trimming).  Keys matching any of these are filtered from the partial-load
    # warning so it only fires on *unexpected* mismatches that indicate a real
    # config / checkpoint incompatibility.
    intentional_patterns: tuple[str, ...] = (
        "class_embed.",
        "bbox_embed.",
        *_QUERY_PARAM_SUFFIXES,
        "enc_out_class_embed.",
        "enc_out_bbox_embed.",
        # The preview keypoint checkpoint still stores the old standalone
        # MLP projection head, but the current GroupPose inference path no
        # longer consumes it.
        "keypoint_head.keypoint_proj.",
    )

    def _is_intentional(key: str) -> bool:
        return any(key.startswith(pat) or f".{pat}" in key for pat in intentional_patterns)

    return [k for k in keys if not _is_intentional(k)]


def _warn_on_partial_load(incompatible: Any, pretrain_weights_path: str) -> None:
    """Emit a ``logger.warning`` when ``load_state_dict`` left non-trivial gaps.

    ``load_state_dict(strict=False)`` silently ignores keys that the model has but the checkpoint does not
    (``missing_keys``) and keys present in the checkpoint but absent from the model (``unexpected_keys``).  When this
    happens for parameters outside the head / query embeddings — which the loader intentionally reinitialises or trims —
    the corresponding model weights were left at their random initial values and the user is silently getting a much
    weaker model. This helper surfaces that condition with a single, actionable warning. Same-key shape mismatches do
    not reach this function — they raise :class:`RuntimeError` directly from ``load_state_dict`` and are therefore
    impossible to miss.

    Args:
        incompatible: The ``_IncompatibleKeys`` namedtuple returned by
            :meth:`torch.nn.Module.load_state_dict`.
        pretrain_weights_path: Path to the checkpoint that was loaded; included
            in the warning so the user can identify which load partially succeeded.
    """
    missing_keys_raw = getattr(incompatible, "missing_keys", None)
    unexpected_keys_raw = getattr(incompatible, "unexpected_keys", None)
    try:
        missing_keys = [str(k) for k in missing_keys_raw] if missing_keys_raw else []
        unexpected_keys = [str(k) for k in unexpected_keys_raw] if unexpected_keys_raw else []
    except TypeError:
        # Result wasn't iterable (e.g. a MagicMock in unit tests) — quietly skip.
        return
    missing = _filter_intentional_keys(missing_keys)
    unexpected = _filter_intentional_keys(unexpected_keys)
    if not missing and not unexpected:
        return

    parts: list[str] = []
    if missing:
        sample = ", ".join(missing[:5])
        if len(missing) > 5:
            sample += ", ..."
        parts.append(f"{len(missing)} model parameter(s) not in checkpoint (left at random init): [{sample}]")
    if unexpected:
        sample = ", ".join(unexpected[:5])
        if len(unexpected) > 5:
            sample += ", ..."
        parts.append(f"{len(unexpected)} checkpoint key(s) not consumed by model: [{sample}]")

    logger.warning(
        "Pretrained weights at %r loaded only partially — this typically produces "
        "lower accuracy. %s. Check that the model configuration (encoder, hidden_dim, "
        "out_feature_indexes, projector_scale, ...) matches the architecture the "
        "checkpoint was trained with.",
        pretrain_weights_path,
        " ".join(parts),
    )


def interpolate_position_embeddings(
    checkpoint_state: dict[str, Any],
    pe_size: int,
) -> None:
    """Interpolate DINOv2 positional embeddings in *checkpoint_state* to match *pe_size*.

    When the model is configured with a custom ``resolution`` that differs from the checkpoint's training resolution,
    the DINOv2 backbone's ``position_embeddings`` parameter has an incompatible shape.
    ``load_state_dict(strict=False)`` does **not** skip shape mismatches on matching keys — it raises ``RuntimeError``.

    This function bicubic-interpolates every PE tensor in the checkpoint whose shape differs from the target grid,
    modifying *checkpoint_state* in-place before ``load_state_dict`` is called.

    Args:
        checkpoint_state: The ``"model"`` sub-dict from a loaded checkpoint.
        pe_size: Target grid side length in patches (number of patches per spatial
            dimension, assuming a square grid).  Typically ``model_config.positional_encoding_size``.
    """
    n_target = pe_size * pe_size  # target number of patch tokens

    pe_keys = [k for k in checkpoint_state if k.endswith(_PE_KEY_SUFFIX)]
    for key in pe_keys:
        ckpt_pe = checkpoint_state[key]  # [1, N_src+1, dim]
        n_source = ckpt_pe.shape[1] - 1  # exclude class token
        if n_source == n_target:
            continue  # no mismatch — skip

        h_src = int(math.isqrt(n_source))
        h_tgt = int(math.isqrt(n_target))
        if h_src * h_src != n_source or h_tgt * h_tgt != n_target:
            logger.warning(
                f"Skipping PE interpolation for {key}:"
                f" grid size is not a perfect square (source {n_source}, target {n_target}).",
            )
            continue

        dim = ckpt_pe.shape[-1]
        class_token = ckpt_pe[:, :1]  # [1, 1, dim] — keeps the sequence dimension
        patch_pe = ckpt_pe[:, 1:]  # [1, N_src, dim]

        patch_pe = patch_pe.reshape(1, h_src, h_src, dim).permute(0, 3, 1, 2)  # [1, dim, H, W]
        patch_pe = F.interpolate(
            patch_pe.float(),
            size=(h_tgt, h_tgt),
            mode="bicubic",
            align_corners=False,
            antialias=patch_pe.device.type != "mps",
        ).to(ckpt_pe.dtype)
        patch_pe = patch_pe.permute(0, 2, 3, 1).reshape(1, n_target, dim)  # [1, N_tgt, dim]

        checkpoint_state[key] = torch.cat([class_token, patch_pe], dim=1)
        logger.debug(
            "Interpolated positional embeddings %s: %s → %s.",
            key,
            tuple(ckpt_pe.shape),
            tuple(checkpoint_state[key].shape),
        )


@deprecated(  # type: ignore[untyped-decorator,unused-ignore]
    target=TargetMode.ARGS_REMAP,
    args_mapping={"train_config": None},
    deprecated_in="1.7.0",
    remove_in="1.9.0",
    num_warns=-1,
)
def load_pretrain_weights(
    nn_model: LWDETR,
    model_config: ModelConfig,
    train_config: TrainConfig | None = None,
) -> list[str]:
    """Load pretrained checkpoint weights into *nn_model* in-place.

    Canonical implementation shared by the L1 facade (``_build_model_context`` in ``rfdetr.detr``) and the L2
    LightningModule (``RFDETRModelModule.__init__`` in ``rfdetr.training.module_model``).

    Uses the Pydantic-aware logic from ``module_model.py``:

    - When the user did **not** explicitly set ``num_classes`` (it is left unset),
      the checkpoint class count is treated as authoritative and the model head is auto-aligned to it.
    - When the user **did** explicitly set ``num_classes`` (to any value, including the
      class default) larger than the checkpoint provides, the head is temporarily aligned to the checkpoint for
      loading, then expanded back to the configured size.
    - When the checkpoint has more classes than configured (backbone-pretrain
      scenario), both reinitializations are applied: expand to checkpoint size for loading, then trim to configured
      size.

    Class names stored in the checkpoint ``args`` are extracted and returned.

    Args:
        nn_model: The model whose weights will be updated in-place.
        model_config: Pydantic ``ModelConfig`` instance. Must have
            ``pretrain_weights``, ``num_classes``, ``num_queries``, and ``group_detr`` attributes.
        train_config: Deprecated since v1.7.0 — no longer used internally.
            Passing a non-``None`` value emits a ``DeprecationWarning``.
            Omit the argument; it will be removed in v1.9.0.

    Returns:
        List of class name strings from the checkpoint, or an empty list if none are present or if
        ``model_config.pretrain_weights`` is ``None``.

    Raises:
        Exception: If the checkpoint file cannot be loaded even after a re-download.
    """
    mc = model_config
    if mc.pretrain_weights is None:
        return []
    # `expand_path`/`_coerce_resume_path` pydantic validators on ModelConfig already normalize
    # this field to a `str` at runtime; the `str()` here just satisfies the static `PathLikeStr` type.
    pretrain_weights = str(mc.pretrain_weights)
    class_names: list[str] = []

    from mmdet.models.layers.rf_detr.util.io import _safe_torch_load

    # Download first (no-op if already present and hash is valid).
    download_pretrain_weights(pretrain_weights)
    # If the first download attempt didn't produce the file (e.g. stale MD5
    # caused an earlier ValueError that was silently swallowed), retry once.
    # MD5 validation is kept on the retry — if it fails again the error is real.
    if not os.path.isfile(pretrain_weights):
        logger.warning("Pretrain weights not found after initial download; retrying.")
        download_pretrain_weights(pretrain_weights, redownload=True)
    validate_pretrain_weights(pretrain_weights, strict=False)

    try:
        checkpoint = _safe_torch_load(pretrain_weights)
    except Exception:
        logger.info("Failed to load pretrain weights, re-downloading")
        download_pretrain_weights(pretrain_weights, redownload=True)
        checkpoint = _safe_torch_load(pretrain_weights)

    # Normalize PyTorch Lightning native .ckpt format to the expected {"model": {...}}
    # structure.  PTL stores model weights in "state_dict" with keys prefixed by
    # "model." (matching the attribute path inside RFDETRModelModule).  Legacy and
    # BestModelCallback checkpoints already have a top-level "model" key.
    if "model" not in checkpoint and "state_dict" in checkpoint:
        logger.debug("Normalizing PTL .ckpt checkpoint format (state_dict -> model)")
        prefix = "model."
        # When the model was wrapped with torch.compile, PTL stores weights with keys
        # like "model._orig_mod.<param>".  Strip the extra "_orig_mod." segment so the
        # resulting keys match the expected bare parameter names.
        compile_prefix = "_orig_mod."
        model_state = {}
        for k, v in checkpoint["state_dict"].items():
            if k.startswith(prefix):
                stripped = k[len(prefix) :]
                if stripped.startswith(compile_prefix):
                    stripped = stripped[len(compile_prefix) :]
                model_state[stripped] = v
        if not model_state:
            raise ValueError(
                f"The checkpoint at {pretrain_weights!r} appears to be in PyTorch Lightning "
                "format ('state_dict' key present, 'model' key absent), but 'state_dict' "
                "contains no keys with the expected 'model.' prefix. "
                "The checkpoint may be corrupt or in an unsupported format."
            )
        checkpoint["model"] = model_state
        # PTL stores training hyper-parameters under "hyper_parameters".  Map them
        # to the "args" key expected by class-name extraction and compatibility checks
        # (only when "args" is not already present).
        if "args" not in checkpoint and "hyper_parameters" in checkpoint:
            checkpoint["args"] = checkpoint["hyper_parameters"]

    # Extract class_names from the checkpoint if available (ported from detr.py).
    if "args" in checkpoint:
        raw_class_names = _ckpt_args_get(checkpoint["args"], "class_names")
        if raw_class_names:
            # Normalize to a new List[str] to avoid leaking mutable references and
            # to respect the annotated return type.
            if isinstance(raw_class_names, str):
                class_names = [raw_class_names]
            else:
                try:
                    iterator = iter(raw_class_names)
                except TypeError:
                    # Non-iterable, ignore and keep the default empty list.
                    class_names = []
                else:
                    class_names = [name for name in iterator if isinstance(name, str)]

    validate_checkpoint_compatibility(checkpoint, mc)

    # Determine whether the user explicitly set num_classes on the ModelConfig.
    # "num_classes" in model_fields_set is True only when the field was explicitly provided
    # (or assigned afterwards, e.g. by dataset alignment); an explicit value is honored
    # regardless of whether it equals the class default, so an intentional num_classes always
    # wins over the checkpoint's class count.
    # Capture BEFORE any mc.num_classes assignment below — those re-add "num_classes" to
    # model_fields_set, which would falsely appear as user-set to the second guard (~line 524).
    user_overrode = "num_classes" in getattr(mc, "model_fields_set", set())
    num_classes = mc.num_classes

    checkpoint_num_classes = checkpoint["model"]["class_embed.bias"].shape[0]
    configured_num_classes_plus_bg = num_classes + 1
    if checkpoint_num_classes != configured_num_classes_plus_bg:
        # Align model head size before loading checkpoint weights.
        if checkpoint_num_classes < configured_num_classes_plus_bg:
            # Checkpoint has FEWER classes than configured.
            if not user_overrode:
                # Auto-align to the checkpoint when the user did NOT explicitly set
                # num_classes (i.e., left it unset): treat the checkpoint as authoritative.
                num_classes = checkpoint_num_classes - 1
                configured_num_classes_plus_bg = checkpoint_num_classes
                mc.num_classes = num_classes
        # In all mismatch cases we need the head to match the checkpoint's
        # class count so load_state_dict succeeds without size mismatches.
        nn_model.reinitialize_detection_head(checkpoint_num_classes)

    # Reshape query embeddings to the configured query count, preserving per-group
    # structure when the checkpoint records its training-time num_queries / group_detr.
    # See _slice_query_param_per_group for why a flat slice is wrong with group_detr > 1.
    ckpt_args = checkpoint.get("args")
    ckpt_num_queries_raw = _ckpt_args_get(ckpt_args, "num_queries") if ckpt_args is not None else None
    ckpt_group_detr_raw = _ckpt_args_get(ckpt_args, "group_detr") if ckpt_args is not None else None
    try:
        ckpt_num_queries = int(ckpt_num_queries_raw) if ckpt_num_queries_raw is not None else None
        ckpt_group_detr = int(ckpt_group_detr_raw) if ckpt_group_detr_raw is not None else None
    except (TypeError, ValueError):
        logger.warning(
            "load_pretrain_weights: checkpoint args.num_queries / args.group_detr not coercible "
            "to int; falling back to legacy flat slice."
        )
        ckpt_num_queries = None
        ckpt_group_detr = None
    # When exactly one of the pair is present, infer the missing value from the
    # first matching tensor's shape.  This handles PTL checkpoints where
    # BestModelCallback writes TrainConfig.model_dump() into checkpoint["args"]
    # but TrainConfig does not include num_queries (it lives on ModelConfig).
    if (ckpt_num_queries is None) != (ckpt_group_detr is None):
        _first_query_key = next(
            (k for k in checkpoint["model"] if any(k.endswith(s) for s in _QUERY_PARAM_SUFFIXES)),
            None,
        )
        if _first_query_key is not None:
            _n = checkpoint["model"][_first_query_key].shape[0]
            _absent: str | None = None
            if ckpt_num_queries is not None and ckpt_num_queries > 0 and _n % ckpt_num_queries == 0:
                ckpt_group_detr = _n // ckpt_num_queries
                _absent, _inferred, _known, _known_val = "group_detr", ckpt_group_detr, "num_queries", ckpt_num_queries
            elif ckpt_group_detr is not None and ckpt_group_detr > 0 and _n % ckpt_group_detr == 0:
                ckpt_num_queries = _n // ckpt_group_detr
                _absent, _inferred, _known, _known_val = "num_queries", ckpt_num_queries, "group_detr", ckpt_group_detr
            if _absent is not None:
                logger.warning(
                    "load_pretrain_weights: args.%s absent; inferred ckpt_%s=%d from tensor rows %d ÷ ckpt_%s=%d.",
                    _absent,
                    _absent,
                    _inferred,
                    _n,
                    _known,
                    _known_val,
                )
    # Warn once (not once per suffix key) when falling back to the legacy flat slice.
    if mc.group_detr > 1 and (ckpt_num_queries is None or ckpt_group_detr is None):
        logger.warning(
            "load_pretrain_weights: checkpoint lacks args.num_queries / "
            "args.group_detr; falling back to flat slice. With "
            "group_detr=%d this may scramble per-group query structure if "
            "the checkpoint was trained with group_detr > 1.",
            mc.group_detr,
        )
    for name in list(checkpoint["model"].keys()):
        if any(name.endswith(x) for x in _QUERY_PARAM_SUFFIXES):
            tensor = checkpoint["model"][name]
            if ckpt_num_queries is not None and ckpt_group_detr is not None:
                checkpoint["model"][name] = _slice_query_param_per_group(
                    tensor,
                    ckpt_num_queries=ckpt_num_queries,
                    ckpt_group_detr=ckpt_group_detr,
                    target_num_queries=mc.num_queries,
                    target_group_detr=mc.group_detr,
                )
            else:
                # Legacy checkpoint with no num_queries/group_detr in args:
                # preserve the original flat slice for backward compatibility.
                # NOTE: the flat slice is incorrect for group_detr > 1 — it scrambles
                # groups 1+ when num_queries decreases. Legacy checkpoints predate
                # multi-group training, so in practice they are all group_detr == 1.
                checkpoint["model"][name] = tensor[: mc.num_queries * mc.group_detr]

    checkpoint["model"] = remap_projector_to_cross_attn(checkpoint["model"], nn_model)

    # Auto-align num_keypoints_per_class from checkpoint when the user did not explicitly set it.
    # Without this, loading a bg-first checkpoint ([0, 17]) into a model configured with the
    # active-first default ([17]) causes a semantic mismatch: the detection head is trimmed to 2
    # rows but keeps background weights at row 0 — the slot active-first inference expects to be
    # person — producing AP ≈ 0.0 on pretrained keypoint models.
    # Mirrors the existing num_classes auto-align pattern (lines ~385-392).
    _user_overrode_kp_schema = "num_keypoints_per_class" in getattr(mc, "model_fields_set", set())
    if (
        not _user_overrode_kp_schema
        and getattr(mc, "use_grouppose_keypoints", False)
        and hasattr(mc, "num_keypoints_per_class")
    ):
        _early_kp_mask = checkpoint["model"].get("_kp_active_mask")
        if isinstance(_early_kp_mask, Tensor) and _early_kp_mask.ndim == 2:
            _ckpt_kp_schema = [int(n) for n in _early_kp_mask.sum(dim=1).tolist()]
            _cfg_kp_schema = list(getattr(mc, "num_keypoints_per_class", []) or [])
            if not any(n > 0 for n in _ckpt_kp_schema):
                logger.warning(
                    "load_pretrain_weights: _kp_active_mask in checkpoint has no active slots "
                    "(schema=%s) — skipping auto-align to avoid overwriting config with empty schema.",
                    _ckpt_kp_schema,
                )
            elif _ckpt_kp_schema != _cfg_kp_schema:
                logger.debug(
                    "load_pretrain_weights: auto-aligning num_keypoints_per_class %s → %s "
                    "(inferred from checkpoint _kp_active_mask; user did not set explicitly).",
                    _cfg_kp_schema,
                    _ckpt_kp_schema,
                )
                mc.num_keypoints_per_class = _ckpt_kp_schema
        elif isinstance(_early_kp_mask, Tensor):
            logger.warning(
                "load_pretrain_weights: _kp_active_mask has unexpected shape %s (expected 2-D) "
                "— skipping auto-align; schema mismatch may cause AP≈0 on keypoint models.",
                tuple(_early_kp_mask.shape),
            )

    # Detection checkpoints/configs may omit keypoint schema fields; absence means no keypoint reconciliation.
    configured_keypoint_schema = list(getattr(mc, "num_keypoints_per_class", []) or [])
    checkpoint_keypoint_schema = None
    should_restore_config_keypoint_schema = False
    if getattr(mc, "use_grouppose_keypoints", False) and hasattr(
        nn_model, "get_num_keypoints_per_class_from_checkpoint"
    ):
        checkpoint_keypoint_schema = nn_model.get_num_keypoints_per_class_from_checkpoint(checkpoint["model"])
        if checkpoint_keypoint_schema:
            get_model_schema = getattr(nn_model, "get_num_keypoints_per_class", None)
            model_keypoint_schema = get_model_schema() if callable(get_model_schema) else configured_keypoint_schema
            if checkpoint_keypoint_schema != model_keypoint_schema and hasattr(nn_model, "reinitialize_keypoint_head"):
                logger.warning(
                    "load_pretrain_weights: temporarily resizing keypoint schema from %s to checkpoint schema %s.",
                    model_keypoint_schema,
                    checkpoint_keypoint_schema,
                )
                nn_model.reinitialize_keypoint_head(checkpoint_keypoint_schema)
                should_restore_config_keypoint_schema = checkpoint_keypoint_schema != configured_keypoint_schema

    # `_kp_active_mask` is a deterministic buffer derived from
    # `num_keypoints_per_class` in the current config. Some checkpoints store a
    # shape that reflects an earlier schema (for example [2, 17] vs [1, 17]).
    # Dropping only this key avoids hard load failures while preserving all
    # learned weights.
    ckpt_kp_active_mask = checkpoint["model"].get("_kp_active_mask")
    model_state_dict = nn_model.state_dict() if hasattr(nn_model, "state_dict") else {}
    model_kp_active_mask = model_state_dict.get("_kp_active_mask") if isinstance(model_state_dict, dict) else None
    if (
        isinstance(ckpt_kp_active_mask, Tensor)
        and isinstance(model_kp_active_mask, Tensor)
        and ckpt_kp_active_mask.shape != model_kp_active_mask.shape
        and not should_restore_config_keypoint_schema
    ):
        logger.warning(
            "load_pretrain_weights: dropping checkpoint _kp_active_mask with shape %s "
            "because current model expects %s (derived from current keypoint schema).",
            tuple(ckpt_kp_active_mask.shape),
            tuple(model_kp_active_mask.shape),
        )
        checkpoint["model"].pop("_kp_active_mask", None)
    interpolate_position_embeddings(checkpoint["model"], mc.positional_encoding_size)
    incompatible = nn_model.load_state_dict(checkpoint["model"], strict=False)
    _warn_on_partial_load(incompatible, pretrain_weights)

    if should_restore_config_keypoint_schema and hasattr(nn_model, "reinitialize_keypoint_head"):
        logger.warning(
            "load_pretrain_weights: restoring configured keypoint schema %s after checkpoint load.",
            configured_keypoint_schema,
        )
        nn_model.reinitialize_keypoint_head(configured_keypoint_schema)

    # If the user explicitly set a class count larger than the checkpoint,
    # expand/reinitialize the head back to the configured size after load.
    if checkpoint_num_classes < configured_num_classes_plus_bg and user_overrode:
        nn_model.reinitialize_detection_head(configured_num_classes_plus_bg)

    # Only trim back down when loading a larger pretrain checkpoint into a
    # smaller configured task-specific class count.
    if num_classes + 1 < checkpoint_num_classes:
        nn_model.reinitialize_detection_head(num_classes + 1)

    return class_names


def apply_lora(nn_model: LWDETR) -> None:
    """Apply LoRA adapters to the backbone encoder of *nn_model*.

    Replaces ``nn_model.backbone[0].encoder`` in-place with a PEFT-wrapped encoder using DoRA with rank 16 and alpha 16.

    Args:
        nn_model: LWDETR model whose backbone encoder will receive LoRA adapters.

    Raises:
        ImportError: If ``peft`` is not installed.
            Install via the RF-DETR extras, for example::

                pip install "rfdetr[lora]"
                # or
                pip install "rfdetr[train]"
    """
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise ImportError(
            "LoRA requires the 'peft' dependency. "
            "Install it via RF-DETR extras, e.g.: "
            'pip install "rfdetr[lora]" or pip install "rfdetr[train]".'
        ) from exc

    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        use_dora=True,
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "qkv",
            "query",
            "key",
            "value",
            "cls_token",
            "register_tokens",
        ],
    )
    backbone = cast(Backbone, nn_model.backbone[0])
    # peft.get_peft_model() type-hints its first argument as PreTrainedModel, but only actually
    # needs an nn.Module whose named submodules match target_modules; DinoV2 (a plain nn.Module
    # wrapper, not itself a PreTrainedModel) satisfies that at runtime.
    backbone.encoder = get_peft_model(backbone.encoder, lora_config)  # type: ignore[arg-type,assignment]

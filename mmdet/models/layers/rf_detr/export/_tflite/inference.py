# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""TFLite inference helpers for RF-DETR exported models.

These functions handle interpreter creation, image preprocessing, and decoding of detection and segmentation-mask
outputs without requiring PyTorch or the RF-DETR training stack: only ``tflite-runtime`` (or ``tensorflow``), ``numpy``,
``supervision``, and ``Pillow`` are needed at inference time.
"""

from __future__ import annotations

import contextlib
import importlib
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image as PILImage
from supervision import Detections

from mmdet.models.layers.rf_detr.utilities.logger import get_logger

logger = get_logger()

# PILImage.Resampling was introduced in Pillow 9.1; fall back to the legacy constant.
_PIL_BILINEAR = getattr(PILImage, "Resampling", PILImage).BILINEAR

_IMAGENET_MEAN: list[float] = [0.485, 0.456, 0.406]
_IMAGENET_STD: list[float] = [0.229, 0.224, 0.225]


def _create_interpreter(model_path: str | Path) -> Any:
    """Load a TFLite model, allocate tensors, and log I/O shapes.

    Tries ``tflite_runtime`` first (lightweight; preferred on edge devices), then falls back to ``tensorflow.lite``
    (pre-installed on Colab / full TF environments).

    Args:
        model_path: Path to the ``.tflite`` model file.

    Returns:
        An allocated TFLite interpreter ready for inference.
    """
    _Interpreter = None  # noqa: N806
    _tried: list[str] = []
    for _pkg, _attr in (
        ("ai_edge_litert.interpreter", "Interpreter"),
        ("tflite_runtime.interpreter", "Interpreter"),
        ("tensorflow.lite", "Interpreter"),
    ):
        with contextlib.suppress(ImportError):
            _Interpreter = getattr(importlib.import_module(_pkg), _attr)  # noqa: N806
            break
        _tried.append(_pkg.split(".")[0])
    if _Interpreter is None:
        _tried_str = ", ".join(f"'{p}'" for p in _tried)
        raise ImportError(
            f"TFLite inference requires 'ai_edge_litert', 'tflite-runtime', or 'tensorflow' "
            f"(tried: {_tried_str}). "
            "Install one: `pip install ai_edge_litert`  OR  `pip install tflite-runtime`"
        )

    interp = _Interpreter(model_path=str(model_path))
    interp.allocate_tensors()
    inp_det = interp.get_input_details()
    out_det = interp.get_output_details()
    logger.debug("Input  : %s  %s", inp_det[0]["shape"], inp_det[0]["dtype"].__name__)
    for od in out_det:
        logger.debug("Output : %s  name=%s", od["shape"], od.get("name", "<unnamed>"))
    return interp


def _bilinear_resize_half_pixel(src: NDArray[np.float32], out_h: int, out_w: int) -> NDArray[np.float32]:
    """Numpy bilinear resize matching ``F.interpolate(mode="bilinear", align_corners=False)``.

    Half-pixel center convention. Used by ``_decode_masks`` only when ``torch`` is not importable.

    Args:
        src: Source array of shape ``(K, src_h, src_w)``.
        out_h: Target height in pixels.
        out_w: Target width in pixels.

    Returns:
        Float32 array of shape ``(K, out_h, out_w)``.

    Note:
        Replaces ``PIL.Image.resize(BILINEAR)``, which uses a corner-aligned half-pixel convention and
        produced border-pixel discrepancies vs ``F.interpolate``.
    """
    src_h, src_w = src.shape[-2], src.shape[-1]
    src_y = (np.arange(out_h, dtype=np.float32) + 0.5) * (src_h / out_h) - 0.5
    src_x = (np.arange(out_w, dtype=np.float32) + 0.5) * (src_w / out_w) - 0.5
    src_y = np.clip(src_y, 0.0, src_h - 1)
    src_x = np.clip(src_x, 0.0, src_w - 1)
    y0 = np.floor(src_y).astype(np.int64)
    x0 = np.floor(src_x).astype(np.int64)
    y1 = np.minimum(y0 + 1, src_h - 1)
    x1 = np.minimum(x0 + 1, src_w - 1)
    dy = (src_y - y0)[:, None]
    dx = (src_x - x0)[None, :]
    a = src[..., y0[:, None], x0[None, :]]
    b = src[..., y0[:, None], x1[None, :]]
    c = src[..., y1[:, None], x0[None, :]]
    d = src[..., y1[:, None], x1[None, :]]
    out = (1 - dy) * ((1 - dx) * a + dx * b) + dy * ((1 - dx) * c + dx * d)
    return np.asarray(out, dtype=np.float32)


def _decode_masks(mask_logits: NDArray[np.floating[Any]], out_size: tuple[int, int]) -> NDArray[np.bool_]:
    """Upsample mask logits to image size and threshold at zero.

    Matches ``PostProcess.forward``: bilinear upsample with ``align_corners=False`` followed by ``> 0``.
    Uses ``torch.nn.functional.interpolate`` when torch is importable for bit-exact parity, and falls
    back to the pure-NumPy ``_bilinear_resize_half_pixel`` otherwise.

    Args:
        mask_logits: Raw mask logits of shape ``(K, Hm, Wm)``.
        out_size: Target ``(width, height)`` in pixels.

    Returns:
        Boolean mask array of shape ``(K, height, width)``.

    Raises:
        ValueError: If *mask_logits* is not rank-3.

    Note:
        ``out_size`` follows PIL convention ``(width, height)``; the returned array uses
        NumPy/PyTorch convention ``(K, height, width)``.
    """
    if mask_logits.ndim != 3:
        raise ValueError(
            f"_decode_masks expects rank-3 (K, Hm, Wm); got shape {mask_logits.shape}. "
            "This usually means the rank-4 mask-output heuristic in _run_inference matched the wrong tensor."
        )
    width, height = out_size
    if mask_logits.shape[0] == 0:
        return np.zeros((0, height, width), dtype=np.bool_)
    try:
        import torch
        import torch.nn.functional as _F  # noqa: N812

        with torch.no_grad():
            t = torch.from_numpy(mask_logits.astype(np.float32)).unsqueeze(0)
            t = _F.interpolate(t, size=(height, width), mode="bilinear", align_corners=False)
        resized: NDArray[np.float32] = np.asarray(t.squeeze(0).numpy(), dtype=np.float32)
    except ImportError:
        resized = _bilinear_resize_half_pixel(mask_logits.astype(np.float32), height, width)
    return resized > 0.0


def _preprocess_image(
    pil_img: PILImage.Image,
    hw: tuple[int, int],
    channels: int = 3,
) -> NDArray[np.float32]:
    """Resize and ImageNet-normalise an image to match ``RFDETR.predict()``.

    Uses ``torchvision.transforms.functional`` when importable for bit-exact parity, and falls back
    to ``PIL.Image.resize`` with BILINEAR for torch-free deployments.

    Args:
        pil_img: Source PIL image at native resolution.
        hw: Target ``(height, width)`` from the interpreter's input shape.
        channels: Channel count (3 for RGB, 1 for grayscale).

    Returns:
        Float32 array of shape ``(1, height, width, channels)`` in NHWC.

    Note:
        The PIL fallback uses BILINEAR resize, which does not perfectly match PyTorch's ``F.resize``
        (different coordinate conventions). For bit-exact parity with ``RFDETR.predict()``, ensure
        ``torch`` and ``torchvision`` are importable.
    """
    height, width = hw
    pil_mode = "L" if channels == 1 else "RGB"
    pil_rgb = pil_img.convert(pil_mode)

    nchw_float: NDArray[np.float32] | None = None
    try:
        # Match PyTorch.predict() exactly: torchvision to_tensor -> resize -> normalize.
        import torch
        import torchvision.transforms.functional as _F  # noqa: N812

        with torch.no_grad():
            t = _F.to_tensor(pil_rgb)
            t = _F.resize(t, list(hw))
            mean_list = [_IMAGENET_MEAN[i % 3] for i in range(channels)]
            std_list = [_IMAGENET_STD[i % 3] for i in range(channels)]
            t = _F.normalize(t, mean_list, std_list)
        nchw_float = np.asarray(t.unsqueeze(0).cpu().numpy(), dtype=np.float32)
    except ImportError:
        pass

    if nchw_float is not None:
        # NCHW -> NHWC for the TFLite interpreter.
        return np.asarray(nchw_float.transpose(0, 2, 3, 1), dtype=np.float32)

    # Torch-free fallback: PIL BILINEAR. PIL's default is BICUBIC, which diverges from PyTorch.
    arr = np.array(pil_rgb.resize((width, height), _PIL_BILINEAR), dtype=np.float32) / 255.0
    if arr.ndim == 2:  # "L" -> (height, width); TFLite needs (height, width, 1).
        arr = arr[:, :, np.newaxis]

    mean = np.array([_IMAGENET_MEAN[i % 3] for i in range(channels)], dtype=np.float32)
    std = np.array([_IMAGENET_STD[i % 3] for i in range(channels)], dtype=np.float32)

    return ((arr - mean) / std)[np.newaxis]


def _run_inference(
    interp: Any,
    image_path: str | Path,
    threshold: float = 0.3,
) -> tuple[Detections, PILImage.Image]:
    """Preprocess one image, run TFLite inference, and decode detections.

    Reads input shape from the interpreter (NHWC ``float32``), resizes and normalises the image with ImageNet
    statistics, invokes the model, then decodes the ``dets`` / ``labels`` output tensors into a
    :class:`supervision.Detections` object with pixel-space ``xyxy`` boxes. For segmentation exports the ``masks``
    output is also decoded into ``Detections.mask``.

    Args:
        interp: Allocated TFLite interpreter returned by ``_create_interpreter``.
        image_path: Path to the input image (any format supported by Pillow).
        threshold: Confidence threshold; detections below this are discarded.

    Returns:
        A tuple of ``(detections, pil_img)`` where ``detections`` contains pixel-space ``xyxy`` boxes (and ``mask`` for
        segmentation models) and ``pil_img`` is the original PIL image at its original resolution.
    """
    inp_det = interp.get_input_details()
    out_det = interp.get_output_details()
    _, height, width, channels = inp_det[0]["shape"]

    expected_dtype = np.float32
    actual_dtype = inp_det[0]["dtype"]
    if actual_dtype != expected_dtype:
        raise ValueError(
            f"_run_inference only supports float32 input tensors, but model expects {actual_dtype.__name__}. "
            "Export the model with float32 quantization or implement input quantization manually."
        )

    with PILImage.open(image_path) as pil_img:
        inp_tensor = _preprocess_image(pil_img, (int(height), int(width)), int(channels))

    interp.set_tensor(inp_det[0]["index"], inp_tensor)
    interp.invoke()

    # RF-DETR ONNX output names: "dets" = pred_boxes, "labels" = pred_logits.
    # Match by name so the code is robust to onnx2tf output reordering.
    available_output_names = [str(od.get("name", "<unnamed>")) for od in out_det]
    boxes_idx = next((i for i, od in enumerate(out_det) if "dets" in str(od.get("name", ""))), None)
    logits_idx = next((i for i, od in enumerate(out_det) if "labels" in str(od.get("name", ""))), None)
    if boxes_idx is None or logits_idx is None:
        # onnx2tf sometimes renames outputs to generic "Identity", "Identity_N"
        # instead of preserving the original ONNX node names. Fall back to
        # shape-based matching: boxes are the rank-3 tensor with last dim 4,
        # logits the rank-3 tensor with last dim != 4. A rank-4 mask output,
        # if present, is matched separately below.
        logger.debug(
            "Name-based output matching failed (available: %s). Falling back to shape-based matching.",
            available_output_names,
        )
        shape_boxes_candidates = [i for i, od in enumerate(out_det) if len(od["shape"]) == 3 and od["shape"][-1] == 4]
        shape_logits_candidates = [i for i, od in enumerate(out_det) if len(od["shape"]) == 3 and od["shape"][-1] != 4]
        if len(shape_boxes_candidates) == 1 and len(shape_logits_candidates) == 1:
            boxes_idx = shape_boxes_candidates[0]
            logits_idx = shape_logits_candidates[0]
        elif len(out_det) == 2:
            # Ambiguous shapes (e.g. num_classes==3 → logits dim==4 == boxes dim).
            # onnx2tf preserves ONNX output order: index 0 = dets (boxes), index 1 = labels (logits).
            logger.debug("Shape-based matching ambiguous. Using positional order (0=boxes, 1=logits).")
            boxes_idx = 0
            logits_idx = 1
        else:
            available_shapes = [list(od["shape"]) for od in out_det]
            raise ValueError(
                f"Shape-based TFLite output matching failed. Expected exactly one rank-3 tensor with "
                f"last dim == 4 (boxes) and one rank-3 tensor with last dim != 4 (logits). "
                f"Available output shapes: {available_shapes}"
            )
    boxes_cwh = interp.get_tensor(out_det[boxes_idx]["index"])[0]  # (Q, 4) normalized cxcywh

    # Sanity-check: normalized cxcywh boxes must be in [0, 1].  When num_classes==3
    # the logits tensor also has last-dim 4, making shape-based and positional matching
    # ambiguous — onnx2tf may output [labels, dets] rather than [dets, labels].
    # A max > 2.0 or min < -2.0 reliably signals the tensors are swapped (logits routinely
    # reach ±3–10; normalized coords are in [0, 1] by definition).  The min check handles
    # the case where all logits are negative (e.g. max ≈ -2.96) — without it the swap is
    # never triggered and logit values are misinterpreted as box coords.
    if float(boxes_cwh.max()) > 2.0 or float(boxes_cwh.min()) < -2.0:
        logger.debug(
            "Box tensor max=%.2f exceeds [0,1] — swapping boxes/logits assignment "
            "(num_classes==%d likely caused ambiguous positional fallback).",
            float(boxes_cwh.max()),
            interp.get_tensor(out_det[logits_idx]["index"]).shape[-1] - 1,
        )
        boxes_idx, logits_idx = logits_idx, boxes_idx
        boxes_cwh = interp.get_tensor(out_det[boxes_idx]["index"])[0]

    # Drop last logit column: RF-DETR adds +1 to num_classes (no-object slot, criterion.py:323).
    # Keeping it causes class_id == len(class_names) → IndexError at display time.
    logits = interp.get_tensor(out_det[logits_idx]["index"])[0, :, :-1]  # (Q, num_classes)

    # RF-DETR uses per-class sigmoid (not softmax) — mirrors PostProcess.forward in postprocess.py.
    logger.debug(
        "Logits stats: shape=%s min=%.3f max=%.3f mean=%.3f",
        logits.shape,
        float(logits.min()),
        float(logits.max()),
        float(logits.mean()),
    )
    one = np.asarray(1, dtype=logits.dtype)
    scores_all = one / (one + np.exp(-logits.clip(-88, 88)))
    scores = scores_all.max(axis=-1)
    cls = scores_all.argmax(axis=-1)
    logger.debug(
        "Scores stats: min=%.3f max=%.3f — detections above threshold %.2f: %d",
        float(scores.min()),
        float(scores.max()),
        threshold,
        int((scores > threshold).sum()),
    )
    keep = scores > threshold

    cx, cy, bw, bh = boxes_cwh[keep].T
    ow, oh = pil_img.size
    xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
    xyxy *= np.array([ow, oh, ow, oh], dtype=np.float32)

    # Segmentation exports add a rank-4 mask output; decode it when present.
    mask_idx = next((i for i, od in enumerate(out_det) if "masks" in str(od.get("name", ""))), None)
    if mask_idx is None:
        rank4_candidates = [i for i, od in enumerate(out_det) if len(od["shape"]) == 4]
        if len(rank4_candidates) == 1:
            mask_idx = rank4_candidates[0]
        elif len(rank4_candidates) >= 2:
            logger.warning(
                "Ambiguous rank-4 outputs (%d candidates); skipping mask decode. "
                "Name your mask output to contain 'masks' to disambiguate.",
                len(rank4_candidates),
            )
    masks = None
    if mask_idx is not None and keep.any():
        raw_masks = interp.get_tensor(out_det[mask_idx]["index"])[0]  # (Q, Hm, Wm)
        masks = _decode_masks(raw_masks[keep], (ow, oh))

    detections = Detections(xyxy=xyxy, confidence=scores[keep], class_id=cls[keep].astype(int), mask=masks)
    return detections, pil_img

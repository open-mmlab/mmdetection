# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied and modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------
"""This tool provides performance benchmarks by using ONNX Runtime and TensorRT to run inference on a given model with
the COCO validation set.

It offers reliable measurements of inference latency using ONNX Runtime or TensorRT on the device.
"""

import contextlib
import importlib
import json
import os
import os.path as osp
import time
from collections import OrderedDict, namedtuple
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol, cast

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from tqdm.auto import tqdm

try:
    import tensorrt as trt
except ImportError:
    trt = None

try:
    import pycuda.driver as cuda
except ImportError:
    cuda = None

from mmdet.models.layers.rf_detr.utilities.logger import get_logger

logger = get_logger()


class _JsonArgparseCLI(Protocol):
    """Minimal jsonargparse CLI callable interface used by this script."""

    def __call__(self, component: Callable[..., Any]) -> Any:
        """Run jsonargparse CLI for a callable component."""
        ...


def get_image_list(ann_file: str) -> list[dict[str, Any]]:
    with open(ann_file) as fin:
        data = json.load(fin)
    return list(data["images"])


def load_image(file_path: str) -> Image.Image:
    return Image.open(file_path).convert("RGB")


_DEFAULT_INPUT_SIZE = (640, 640)
_DEFAULT_NUM_QUERIES = 300


def _static_dim(value: Any, fallback: int) -> int:
    """Coerce a tensor-shape entry to a positive int, falling back for dynamic/unknown axes.

    ONNX Runtime and TensorRT report dynamic axes as strings (e.g. ``"height"``), ``None``, or ``-1``. Such values
    cannot drive a fixed preprocessing size, so the caller-supplied *fallback* is used instead.

    Args:
        value: A single entry from a model input/output shape.
        fallback: Value to return when *value* is not a concrete positive integer.

    Returns:
        The integer dimension, or *fallback* for dynamic axes.
    """
    try:
        dim = int(value)
    except (TypeError, ValueError):
        return fallback
    return dim if dim > 0 else fallback


def infer_transforms(size: tuple[int, int] = _DEFAULT_INPUT_SIZE) -> Any:
    """Build the benchmark preprocessing pipeline for a given model input size.

    Args:
        size: Target ``(height, width)`` the image is resized to before inference. Defaults to
            :data:`_DEFAULT_INPUT_SIZE` for dynamic-axis models where a static size cannot be read.

    Returns:
        A ``torchvision.transforms.v2.Compose`` that resizes, tensorizes, and normalizes an image.
    """
    from torchvision.transforms.v2 import Compose, Resize, ToDtype, ToImage

    from mmdet.models.layers.rf_detr.datasets.transforms import Normalize

    return Compose(
        [
            Resize(size),
            ToImage(),
            ToDtype(torch.float32, scale=True),
            Normalize(),
        ]
    )


def box_cxcywh_to_xyxy(x: Tensor) -> Tensor:
    x_c, y_c, w, h = x.unbind(-1)
    b = [
        (x_c - 0.5 * w.clamp(min=0.0)),
        (y_c - 0.5 * h.clamp(min=0.0)),
        (x_c + 0.5 * w.clamp(min=0.0)),
        (y_c + 0.5 * h.clamp(min=0.0)),
    ]
    return torch.stack(b, dim=-1)


def post_process(
    outputs: Mapping[str, Tensor], target_sizes: Tensor, num_queries: int = _DEFAULT_NUM_QUERIES
) -> list[dict[str, Tensor]]:
    out_logits, out_bbox = outputs["labels"], outputs["dets"]

    assert len(out_logits) == len(target_sizes)
    assert target_sizes.shape[1] == 2

    prob = out_logits.sigmoid()
    flat_scores = prob.view(out_logits.shape[0], -1)
    # Clamp k to the flattened dimension: when num_queries is a fallback for a dynamic-axis model
    # it may exceed num_queries*num_classes and trigger a topk runtime error.
    k = min(num_queries, flat_scores.shape[1])
    topk_values, topk_indexes = torch.topk(flat_scores, k, dim=1)
    scores = topk_values
    topk_boxes = topk_indexes // out_logits.shape[2]
    labels = topk_indexes % out_logits.shape[2]
    boxes = box_cxcywh_to_xyxy(out_bbox)
    boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

    # and from relative [0, 1] to absolute [0, height] coordinates
    img_h, img_w = target_sizes.unbind(1)
    scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
    boxes = boxes * scale_fct[:, None, :]

    results = [{"scores": score, "labels": label, "boxes": box} for score, label, box in zip(scores, labels, boxes)]

    return results


def infer_onnx(
    sess: Any,
    coco_evaluator: Any,
    time_profile: "TimeProfiler",
    prefix: str,
    img_list: Sequence[dict[str, Any]],
    device: str | torch.device,
    repeats: int = 1,
) -> None:
    input_shape = sess.get_inputs()[0].shape
    # fallback for dynamic-axis models (dynamic H/W report as strings or None)
    input_h = _static_dim(input_shape[2] if len(input_shape) > 3 else None, _DEFAULT_INPUT_SIZE[0])
    input_w = _static_dim(input_shape[3] if len(input_shape) > 3 else None, _DEFAULT_INPUT_SIZE[1])
    output_shape = sess.get_outputs()[0].shape
    num_queries = _static_dim(output_shape[1] if len(output_shape) > 1 else None, _DEFAULT_NUM_QUERIES)
    transforms = infer_transforms((input_h, input_w))

    time_list = []
    for img_dict in tqdm(img_list):
        image = load_image(os.path.join(prefix, img_dict["file_name"]))
        width, height = image.size
        orig_target_sizes = torch.Tensor([height, width])
        image_tensor, _ = transforms(image, None)  # target is None

        samples = image_tensor[None].numpy()

        time_profile.reset()
        with time_profile:
            for _ in range(repeats):
                res = sess.run(None, {"input": samples})
        time_list.append(time_profile.total / repeats)
        outputs = {}
        outputs["labels"] = torch.Tensor(res[1]).to(device)
        outputs["dets"] = torch.Tensor(res[0]).to(device)

        orig_target_sizes = torch.stack([orig_target_sizes], dim=0).to(device)
        results = post_process(outputs, orig_target_sizes, num_queries=num_queries)
        res = {img_dict["id"]: results[0]}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    logger.info(f"Model latency with ONNX Runtime: {1000 * sum(time_list) / len(img_list)}ms")

    # accumulate predictions from all images
    stats = {}
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        stats["coco_eval_bbox"] = coco_evaluator.coco_eval["bbox"].stats.tolist()
        logger.info(stats)


def infer_engine(
    model: "TRTInference",
    coco_evaluator: Any,
    time_profile: "TimeProfiler",
    prefix: str,
    img_list: Sequence[dict[str, Any]],
    device: str | torch.device,
    repeats: int = 1,
) -> None:
    input_shape = list(model.bindings[model.input_names[0]].shape)
    # fallback for dynamic-axis models
    input_h = _static_dim(input_shape[2] if len(input_shape) > 3 else None, _DEFAULT_INPUT_SIZE[0])
    input_w = _static_dim(input_shape[3] if len(input_shape) > 3 else None, _DEFAULT_INPUT_SIZE[1])
    output_shape = list(model.bindings[model.output_names[0]].shape)
    num_queries = _static_dim(output_shape[1] if len(output_shape) > 1 else None, _DEFAULT_NUM_QUERIES)
    transforms = infer_transforms((input_h, input_w))

    time_list = []
    for img_dict in tqdm(img_list):
        image = load_image(os.path.join(prefix, img_dict["file_name"]))
        width, height = image.size
        orig_target_sizes = torch.Tensor([height, width])
        image_tensor, _ = transforms(image, None)  # target is None

        samples = image_tensor[None].to(device)
        _, _, h, w = samples.shape
        # torch.Tensor(np.array([h, w]).reshape((1, 2)).astype(np.float32)).to(device)
        # torch.Tensor(np.array([h / height, w / width]).reshape((1, 2)).astype(np.float32)).to(device)

        time_profile.reset()
        with time_profile:
            for _ in range(repeats):
                outputs = model({"input": samples})

        time_list.append(time_profile.total / repeats)
        orig_target_sizes = torch.stack([orig_target_sizes], dim=0).to(device)
        if coco_evaluator is not None:
            results = post_process(outputs, orig_target_sizes, num_queries=num_queries)
            res = {img_dict["id"]: results[0]}
            coco_evaluator.update(res)

    logger.info(f"Model latency with TensorRT: {1000 * sum(time_list) / len(img_list)}ms")

    # accumulate predictions from all images
    stats = {}
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        stats["coco_eval_bbox"] = coco_evaluator.coco_eval["bbox"].stats.tolist()
        logger.info(stats)


class TRTInference:
    """TensorRT inference engine."""

    def __init__(
        self,
        engine_path: str = "dino.engine",
        device: str | torch.device = "cuda:0",
        sync_mode: bool = False,
        max_batch_size: int = 32,
        verbose: bool = False,
    ) -> None:
        if not trt:
            raise ImportError("TensorRT is not installed. Please install TensorRT to use TRTInference.")

        self.engine_path = engine_path
        self.device = device
        self.sync_mode = sync_mode
        self.max_batch_size = max_batch_size

        self.logger = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger(trt.Logger.INFO)

        self.engine = self.load_engine(engine_path)

        self.context = self.engine.create_execution_context()

        self.bindings = self.get_bindings(self.engine, self.context, self.max_batch_size, self.device)
        self.bindings_addr = OrderedDict((n, v.ptr) for n, v in self.bindings.items())

        self.input_names = self.get_input_names()
        self.output_names = self.get_output_names()
        self.stream = None

        if not self.sync_mode:
            if not cuda:
                raise ImportError(
                    "pycuda is not installed. Please install `pycuda` to use TRTInference with async mode."
                )

            self.stream = cuda.Stream()

        # self.time_profile = TimeProfiler()
        self.time_profile = TimeProfiler()

    def get_dummy_input(self, batch_size: int) -> dict[str, Tensor]:
        blob: dict[str, Tensor] = {}
        for name, binding in self.bindings.items():
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                logger.info(f"make dummy input {name} with shape {binding.shape}")
                blob[name] = torch.rand(batch_size, *binding.shape[1:]).float().to("cuda:0")
        return blob

    def load_engine(self, path: str) -> Any:
        """Load engine."""
        trt.init_libnvinfer_plugins(self.logger, "")
        with open(path, "rb") as f, trt.Runtime(self.logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def get_input_names(self) -> list[str]:
        names: list[str] = []
        for _, name in enumerate(self.engine):
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                names.append(name)
        return names

    def get_output_names(self) -> list[str]:
        names: list[str] = []
        for _, name in enumerate(self.engine):
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                names.append(name)
        return names

    def get_bindings(
        self, engine: Any, context: Any, max_batch_size: int = 32, device: str | torch.device | None = None
    ) -> OrderedDict[str, Any]:
        """Build binddings."""
        Binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))
        bindings = OrderedDict()

        for i, name in enumerate(engine):
            shape = engine.get_tensor_shape(name)
            dtype = trt.nptype(engine.get_tensor_dtype(name))

            if shape[0] == -1:
                raise NotImplementedError

            else:
                data = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
                bindings[name] = Binding(name, dtype, shape, data, data.data_ptr())

        return bindings

    def run_sync(self, blob: Mapping[str, Tensor]) -> dict[str, Tensor]:
        self.bindings_addr.update({n: blob[n].data_ptr() for n in self.input_names})
        self.context.execute_v2(list(self.bindings_addr.values()))
        outputs = {n: self.bindings[n].data for n in self.output_names}
        return outputs

    def run_async(self, blob: Mapping[str, Tensor]) -> dict[str, Tensor]:
        self.bindings_addr.update({n: blob[n].data_ptr() for n in self.input_names})
        bindings_addr = [int(v) for _, v in self.bindings_addr.items()]
        if self.stream is None:
            raise RuntimeError("Async TensorRT inference requires a CUDA stream.")
        self.context.execute_async_v2(bindings=bindings_addr, stream_handle=self.stream.handle)
        outputs = {n: self.bindings[n].data for n in self.output_names}
        self.stream.synchronize()
        return outputs

    def __call__(self, blob: Mapping[str, Tensor]) -> dict[str, Tensor]:
        if self.sync_mode:
            return self.run_sync(blob)
        else:
            return self.run_async(blob)

    def synchronize(self) -> None:
        if self.sync_mode:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            return

        if self.stream is not None:
            self.stream.synchronize()
        elif torch.cuda.is_available():
            torch.cuda.synchronize()

    def speed(self, blob: Mapping[str, Tensor], n: int) -> float:
        self.time_profile.reset()
        with self.time_profile:
            for _ in range(n):
                _ = self(blob)
        return self.time_profile.total / n

    def build_engine(self, onnx_file_path: str, engine_file_path: str, max_batch_size: int = 32) -> Any:
        """Takes an ONNX file and creates a TensorRT engine to run inference with
        http://gitlab.baidu.com/paddle-inference/benchmark/blob/main/backend_trt.py#L57
        """
        explicit_batch_flag = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        with (
            trt.Builder(self.logger) as builder,
            builder.create_network(explicit_batch_flag) as network,
            trt.OnnxParser(network, self.logger) as parser,
            builder.create_builder_config() as config,
        ):
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1024 MiB
            config.set_flag(trt.BuilderFlag.FP16)

            with open(onnx_file_path, "rb") as model:
                if not parser.parse(model.read()):
                    logger.error("ERROR: Failed to parse the ONNX file.")
                    for error in range(parser.num_errors):
                        logger.error(parser.get_error(error))
                    return None

            serialized_engine = builder.build_serialized_network(network, config)
            with open(engine_file_path, "wb") as f:
                f.write(serialized_engine)

            return serialized_engine


class TimeProfiler(contextlib.ContextDecorator):
    def __init__(self) -> None:
        self.total = 0.0
        self.start = 0.0

    def __enter__(self) -> "TimeProfiler":
        self.start = self.time()
        return self

    def __exit__(self, type: Any, value: Any, traceback: Any) -> None:
        self.total += self.time() - self.start

    def reset(self) -> None:
        self.total = 0.0

    def time(self) -> float:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter()


def main(
    path: str,
    coco_path: str = "data/coco",
    device: int = 0,
    run_benchmark: bool = False,
    disable_eval: bool = False,
) -> None:
    """Performance benchmark tool for ONNX/TRT models.

    Args:
        path: Engine file path (.onnx or .engine).
        coco_path: COCO dataset path.
        device: CUDA device index.
        run_benchmark: Repeat inference 10x to measure latency.
        disable_eval: Skip COCO evaluation.
    """
    logger.info(
        {
            "path": path,
            "coco_path": coco_path,
            "device": device,
            "run_benchmark": run_benchmark,
            "disable_eval": disable_eval,
        }
    )
    coco_gt = osp.join(coco_path, "annotations/instances_val2017.json")
    img_list = get_image_list(coco_gt)
    prefix = osp.join(coco_path, "val2017")
    if run_benchmark:
        repeats = 10
        logger.info("Inference for each image will be repeated 10 times ...")
    else:
        repeats = 1

    if not disable_eval:
        from mmdet.models.layers.rf_detr.evaluation.coco_eval import CocoEvaluator

        coco_evaluator = CocoEvaluator(coco_gt, ["bbox"])
    else:
        coco_evaluator = None
    time_profile = TimeProfiler()

    if path.endswith(".onnx"):
        import onnxruntime as nxrun

        sess = nxrun.InferenceSession(path, providers=["CUDAExecutionProvider"])
        infer_onnx(sess, coco_evaluator, time_profile, prefix, img_list, device=f"cuda:{device}", repeats=repeats)
    elif path.endswith(".engine"):
        model = TRTInference(path, sync_mode=True, device=f"cuda:{device}")
        infer_engine(model, coco_evaluator, time_profile, prefix, img_list, device=f"cuda:{device}", repeats=repeats)
    else:
        raise NotImplementedError('Only model file names ending with ".onnx" and ".engine" are supported.')


if __name__ == "__main__":
    jsonargparse = importlib.import_module("jsonargparse")

    cast(_JsonArgparseCLI, getattr(jsonargparse, "CLI"))(main)

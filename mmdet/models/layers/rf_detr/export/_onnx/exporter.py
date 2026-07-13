# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied and modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------
"""ONNX export, simplification, and OnnxOptimizer."""

import inspect
import json
import os
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from os import PathLike
from typing import Any, Protocol, TypeVar, cast

import numpy as np
import torch

from mmdet.models.layers.rf_detr.export._onnx.symbolic import CustomOpSymbolicRegistry
from mmdet.models.layers.rf_detr.utilities.logger import get_logger

_DependencyT = TypeVar("_DependencyT")


class _OnnxModule(Protocol):
    """Minimal ONNX module interface used by this exporter."""

    def load(self, model_path: str, *args: Any, **kwargs: Any) -> Any:
        """Load an ONNX model from disk."""
        ...

    def save(self, model: Any, model_path: str, *args: Any, **kwargs: Any) -> None:
        """Save an ONNX model to disk."""
        ...


class _ShapeInferenceModule(Protocol):
    """Minimal ONNX shape inference interface used by this exporter."""

    def infer_shapes(self, model: Any, *args: Any, **kwargs: Any) -> Any:
        """Infer ONNX graph shapes."""
        ...


class _GraphSurgeonModule(Protocol):
    """Minimal ONNX GraphSurgeon module interface used by the optimizer."""

    def import_onnx(self, model: Any) -> Any:
        """Import an ONNX model into a GraphSurgeon graph."""
        ...

    def export_onnx(self, graph: Any) -> Any:
        """Export a GraphSurgeon graph into an ONNX model."""
        ...


class _GraphSurgeonLogger(Protocol):
    """Minimal GraphSurgeon logger interface used by the optimizer."""

    INFO: Any
    severity: Any

    def info(self, message: str) -> None:
        """Log an informational message."""
        ...

    def verbose(self, message: str) -> None:
        """Log a verbose message."""
        ...


class _FoldConstants(Protocol):
    """Callable Polygraphy constant-folding interface used by the optimizer."""

    def __call__(self, model: Any, *args: Any, **kwargs: Any) -> Any:
        """Fold constants in an ONNX model."""
        ...


onnx: _OnnxModule | None
shape_inference: _ShapeInferenceModule | None
try:
    import onnx as _onnx
    from onnx import shape_inference as _shape_inference
except ImportError:
    onnx = None
    shape_inference = None
else:
    onnx = cast(_OnnxModule, _onnx)
    shape_inference = cast(_ShapeInferenceModule, _shape_inference)

try:
    import onnx_graphsurgeon as gs
    from onnx_graphsurgeon.logger.logger import G_LOGGER
except ImportError:
    gs = None
    G_LOGGER = None

try:
    from polygraphy.backend.onnx.loader import fold_constants
except ImportError:
    fold_constants = None

logger = get_logger()


def _onnx_dependency_error(missing_deps: Sequence[str]) -> ImportError:
    """Build the shared ONNX optional-dependency error."""
    missing_str = ", ".join(missing_deps)
    return ImportError(f"ONNX export dependencies are missing ({missing_str}). Install with: pip install rfdetr[onnx]")


def _require_dependency(dependency: _DependencyT | None, name: str) -> _DependencyT:
    """Return a dependency after narrowing it away from ``None``."""
    if dependency is None:
        raise _onnx_dependency_error([name])
    return dependency


def _require_onnx_optimizer_dependencies() -> tuple[
    _OnnxModule,
    _ShapeInferenceModule,
    _GraphSurgeonModule,
    _GraphSurgeonLogger,
    _FoldConstants,
]:
    """Resolve optional ONNX optimizer dependencies as non-optional typed handles."""
    graphsurgeon = cast(_GraphSurgeonModule | None, gs)
    graphsurgeon_logger = cast(_GraphSurgeonLogger | None, G_LOGGER)
    constant_folder = cast(_FoldConstants | None, fold_constants)
    missing_deps = []
    if onnx is None:
        missing_deps.append("onnx")
    if shape_inference is None:
        missing_deps.append("onnx.shape_inference")
    if graphsurgeon is None or graphsurgeon_logger is None:
        missing_deps.append("onnx_graphsurgeon")
    if constant_folder is None:
        missing_deps.append("polygraphy.backend.onnx.loader.fold_constants")
    if missing_deps:
        raise _onnx_dependency_error(missing_deps)
    return (
        _require_dependency(onnx, "onnx"),
        _require_dependency(shape_inference, "onnx.shape_inference"),
        _require_dependency(graphsurgeon, "onnx_graphsurgeon"),
        _require_dependency(graphsurgeon_logger, "onnx_graphsurgeon"),
        _require_dependency(constant_folder, "polygraphy.backend.onnx.loader.fold_constants"),
    )


# ---------------------------------------------------------------------------
# ONNX export helpers (moved from export/export.py)
# ---------------------------------------------------------------------------


def export_onnx(
    output_dir: str | PathLike[str],
    model: torch.nn.Module,
    input_names: Sequence[str],
    input_tensors: torch.Tensor | Sequence[torch.Tensor],
    output_names: Sequence[str],
    dynamic_axes: Mapping[str, Mapping[int, str]] | None,
    backbone_only: bool = False,
    verbose: bool = True,
    opset_version: int = 17,
    variant_name: str | None = None,
    *,
    notes: object = None,
) -> str:
    """Export a model to ONNX.

    Args:
        output_dir: Directory where the ONNX file will be written.
        model: Model to export.
        input_names: Names of model inputs in ONNX graph.
        input_tensors: Example model input tensor(s) for tracing.
        output_names: Names of model outputs in ONNX graph.
        dynamic_axes: Optional dynamic axis configuration for ONNX export.
        backbone_only: Whether to export backbone-only graph naming.
        verbose: Whether ONNX exporter should emit verbose logs.
        opset_version: ONNX opset version.
        variant_name: Model variant identifier (e.g. ``"rfdetr-medium"``).
            When provided, the exported file is named ``{variant_name}.onnx`` or ``{variant_name}-backbone.onnx`` (when
            ``backbone_only=True``) instead of the generic ``inference_model.onnx`` or ``backbone_model.onnx``.
        notes: Optional user-defined metadata (string, dict, list, or any
            JSON-serialisable value) to embed in the exported ONNX model under the ``"rfdetr_notes"`` metadata property.
            Ignored when ``None``. String values are stored verbatim; all other types are JSON-encoded, so consumers
            must call ``json.loads()`` to recover a dict or list.

    Returns:
        Path to the exported ONNX model.
    """
    if variant_name:
        # Sanitize against path traversal (e.g. "foo/bar" → "bar", "/tmp/x" → "x")
        variant_name = os.path.splitext(os.path.basename(variant_name))[0]
        export_name = f"{variant_name}-backbone" if backbone_only else variant_name
    else:
        export_name = "backbone_model" if backbone_only else "inference_model"
    output_file = os.path.join(output_dir, f"{export_name}.onnx")

    # Prepare model for export
    export_method = getattr(model, "export", None)
    if callable(export_method):
        export_method()

    export_kwargs: dict[str, Any] = {}
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        # Torch 2.10+ may default to the dynamo exporter which requires extra deps
        # (e.g. onnxscript). Use the legacy path for compatibility.
        export_kwargs["dynamo"] = False

    torch.onnx.export(
        model,
        (input_tensors,) if isinstance(input_tensors, torch.Tensor) else tuple(input_tensors),
        output_file,
        input_names=input_names,
        output_names=output_names,
        export_params=True,
        keep_initializers_as_inputs=False,
        do_constant_folding=True,
        verbose=verbose,
        opset_version=opset_version,
        dynamic_axes=dynamic_axes,
        **export_kwargs,
    )

    if notes is not None and onnx is not None:
        # torch.onnx.export writes to disk only; no in-memory handle is available,
        # so we reload and resave to inject metadata (~1-2 s on large models).
        onnx_model = onnx.load(output_file)
        # Strings stored as-is so readers can consume without JSON-decoding;
        # non-strings go through json.dumps to survive the round-trip.
        notes_value = notes if isinstance(notes, str) else json.dumps(notes, allow_nan=False)
        existing = next((p for p in onnx_model.metadata_props if p.key == "rfdetr_notes"), None)
        if existing is not None:
            existing.value = notes_value
        else:
            meta = onnx_model.metadata_props.add()
            meta.key = "rfdetr_notes"
            meta.value = notes_value
        onnx.save(onnx_model, output_file)

    logger.info(f"\nSuccessfully exported ONNX model: {output_file}")
    return output_file


def onnx_simplify(
    onnx_dir: str,
    input_names: Sequence[str],
    input_tensors: torch.Tensor | Sequence[torch.Tensor],
    force: bool = False,
) -> str:
    """Optimize and simplify an ONNX graph.

    Args:
        onnx_dir: Path to the source ONNX model.
        input_names: Input names matching the exported graph.
        input_tensors: Input tensor sample(s) used by the simplifier.
        force: Whether to overwrite an existing simplified model.

    Returns:
        Path to the simplified ONNX model.
    """
    import onnx
    import onnxsim

    sim_onnx_dir = onnx_dir.replace(".onnx", ".sim.onnx")
    if os.path.isfile(sim_onnx_dir) and not force:
        return sim_onnx_dir

    if isinstance(input_tensors, torch.Tensor):
        input_tensors = [input_tensors]

    logger.info(f"start simplify ONNX model: {onnx_dir}")
    opt = OnnxOptimizer(onnx_dir)
    opt.info("Model: original")
    opt.common_opt()
    opt.info("Model: optimized")
    opt.save_onnx(sim_onnx_dir)
    input_dict = {name: tensor.detach().cpu().numpy() for name, tensor in zip(input_names, input_tensors)}
    model_opt, check_ok = onnxsim.simplify(sim_onnx_dir, check_n=3, input_data=input_dict, dynamic_input_shape=False)
    if check_ok:
        onnx.save(model_opt, sim_onnx_dir)
    else:
        raise RuntimeError("Failed to simplify ONNX model.")
    logger.info(f"Successfully simplified ONNX model: {sim_onnx_dir}")
    return sim_onnx_dir


# ---------------------------------------------------------------------------
# OnnxOptimizer (moved from _onnx/optimizer.py)
# ---------------------------------------------------------------------------


class OnnxOptimizer:
    def __init__(self, input: object, severity: object | None = None) -> None:
        onnx_module, shape_inference_module, graphsurgeon, graphsurgeon_logger, constant_folder = (
            _require_onnx_optimizer_dependencies()
        )
        self._gs = graphsurgeon
        self._onnx_logger = graphsurgeon_logger
        self._fold_constants = constant_folder
        self._shape_inference = shape_inference_module
        self._onnx = onnx_module
        if severity is None:
            severity = self._onnx_logger.INFO
        if isinstance(input, (str, PathLike)):
            onnx_graph = self.load_onnx(input)
        else:
            onnx_graph = input
        self.graph = self._gs.import_onnx(onnx_graph)
        self.severity = severity
        self.set_severity(severity)

    def set_severity(self, severity: object) -> None:
        self._onnx_logger.severity = severity

    def load_onnx(self, onnx_path: str | PathLike[str]) -> Any:
        """Load onnx from file."""
        path = os.fspath(onnx_path)
        assert os.path.isfile(path), f"not found onnx file: {path}"
        onnx_graph = self._onnx.load(path)
        self._onnx_logger.info(f"load onnx file: {path}")
        return onnx_graph

    def save_onnx(self, onnx_path: str) -> None:
        onnx_graph = self._gs.export_onnx(self.graph)
        self._onnx_logger.info(f"save onnx file: {onnx_path}")
        self._onnx.save(onnx_graph, onnx_path)

    def info(self, prefix: str = "") -> None:
        self._onnx_logger.verbose(
            f"{prefix} .. {len(self.graph.nodes)} nodes, "
            f"{len(self.graph.tensors().keys())} tensors, "
            f"{len(self.graph.inputs)} inputs, {len(self.graph.outputs)} outputs"
        )

    def cleanup(self, return_onnx: bool = False) -> Any | None:
        self.graph.cleanup().toposort()
        if return_onnx:
            return self._gs.export_onnx(self.graph)
        return None

    def select_outputs(self, keep: Sequence[int], names: Sequence[str] | None = None) -> None:
        self.graph.outputs = [self.graph.outputs[o] for o in keep]
        if names:
            for i, name in enumerate(names):
                self.graph.outputs[i].name = name

    def find_node_input(self, node: Any, name: str | None = None, value: Any = None) -> int:
        index = -1
        for i, inp in enumerate(node.inputs):
            if isinstance(name, str) and inp.name == name or inp == value:
                index = i
        if index < 0:
            raise ValueError(f"not found {name}({value}) in node.inputs")
        return index

    def find_node_output(self, node: Any, name: str | None = None, value: Any = None) -> int:
        index = -1
        for i, inp in enumerate(node.outputs):
            if isinstance(name, str) and inp.name == name or inp == value:
                index = i
        if index < 0:
            raise ValueError(f"not found {name}({value}) in node.outputs")
        return index

    def common_opt(self, return_onnx: bool = False) -> Any | None:
        for fn in CustomOpSymbolicRegistry._OPTIMIZER:
            fn(self)
            self.cleanup()
        onnx_graph = self._fold_constants(self._gs.export_onnx(self.graph), allow_onnxruntime_shape_inference=False)
        if onnx_graph.ByteSize() > 2147483648:
            raise TypeError("ERROR: model size exceeds supported 2GB limit")
        else:
            onnx_graph = self._shape_inference.infer_shapes(onnx_graph)
        self.graph = self._gs.import_onnx(onnx_graph)
        self.cleanup()
        if return_onnx:
            return onnx_graph
        return None

    def resize_fix(self) -> int:
        """This function loops through the graph looking for Resize nodes that uses scales for resize (has 3 inputs).

        It substitutes found Resize with Resize that takes the size of the output tensor instead of scales. It adds
        Shape->Slice->Concat         Shape->Slice----^     subgraph to the graph to extract the shape of the output
        tensor. This fix is required for the dynamic shape support.
        """
        resized_node_count = 0
        for node in self.graph.nodes:
            if node.op == "Resize" and len(node.inputs) == 3:
                name = node.name + "/"

                add_node = node.o().o().i(1)
                div_node = node.i()

                shape_hw_out = gs.Variable(name=name + "shape_hw_out", dtype=np.int64, shape=[4])
                shape_hw = gs.Node(
                    op="Shape", name=name + "shape_hw", inputs=[add_node.outputs[0]], outputs=[shape_hw_out]
                )

                const_zero = gs.Constant(name=name + "const_zero", values=np.array([0], dtype=np.int64))
                const_two = gs.Constant(name=name + "const_two", values=np.array([2], dtype=np.int64))
                const_four = gs.Constant(name=name + "const_four", values=np.array([4], dtype=np.int64))

                slice_hw_out = gs.Variable(name=name + "slice_hw_out", dtype=np.int64, shape=[2])
                slice_hw = gs.Node(
                    op="Slice",
                    name=name + "slice_hw",
                    inputs=[shape_hw_out, const_two, const_four, const_zero],
                    outputs=[slice_hw_out],
                )

                shape_bc_out = gs.Variable(name=name + "shape_bc_out", dtype=np.int64, shape=[2])
                shape_bc = gs.Node(
                    op="Shape", name=name + "shape_bc", inputs=[div_node.outputs[0]], outputs=[shape_bc_out]
                )

                slice_bc_out = gs.Variable(name=name + "slice_bc_out", dtype=np.int64, shape=[2])
                slice_bc = gs.Node(
                    op="Slice",
                    name=name + "slice_bc",
                    inputs=[shape_bc_out, const_zero, const_two, const_zero],
                    outputs=[slice_bc_out],
                )

                concat_bchw_out = gs.Variable(name=name + "concat_bchw_out", dtype=np.int64, shape=[4])
                concat_bchw = gs.Node(
                    op="Concat",
                    name=name + "concat_bchw",
                    attrs={"axis": 0},
                    inputs=[slice_bc_out, slice_hw_out],
                    outputs=[concat_bchw_out],
                )

                none_var = gs.Variable.empty()

                resize_bchw = gs.Node(
                    op="Resize",
                    name=name + "resize_bchw",
                    attrs=node.attrs,
                    inputs=[node.inputs[0], none_var, none_var, concat_bchw_out],
                    outputs=[node.outputs[0]],
                )

                self.graph.nodes.extend([shape_hw, slice_hw, shape_bc, slice_bc, concat_bchw, resize_bchw])

                node.inputs = []
                node.outputs = []

                resized_node_count += 1

        self.cleanup()
        return resized_node_count

    def adjustAddNode(self) -> int:  # noqa: N802
        adjusted_add_node_count = 0
        for node in self.graph.nodes:
            # Change the bias const to the second input to allow Gemm+BiasAdd fusion in TRT.
            if node.op in ["Add"] and isinstance(node.inputs[0], gs.ir.tensor.Constant):
                tensor = node.inputs[1]
                bias = node.inputs[0]
                node.inputs = [tensor, bias]
                adjusted_add_node_count += 1

        self.cleanup()
        return adjusted_add_node_count

    def decompose_instancenorms(self) -> int:
        removed_instance_norm_count = 0
        for node in self.graph.nodes:
            if node.op == "InstanceNormalization":
                name = node.name + "/"
                input_tensor = node.inputs[0]
                output_tensor = node.outputs[0]
                mean_out = gs.Variable(name=name + "mean_out")
                mean_node = gs.Node(
                    op="ReduceMean",
                    name=name + "mean_node",
                    attrs={"axes": [-1]},
                    inputs=[input_tensor],
                    outputs=[mean_out],
                )
                sub_out = gs.Variable(name=name + "sub_out")
                sub_node = gs.Node(
                    op="Sub", name=name + "sub_node", attrs={}, inputs=[input_tensor, mean_out], outputs=[sub_out]
                )
                pow_out = gs.Variable(name=name + "pow_out")
                pow_const = gs.Constant(name=name + "pow_const", values=np.array([2.0], dtype=np.float32))
                pow_node = gs.Node(
                    op="Pow", name=name + "pow_node", attrs={}, inputs=[sub_out, pow_const], outputs=[pow_out]
                )
                mean2_out = gs.Variable(name=name + "mean2_out")
                mean2_node = gs.Node(
                    op="ReduceMean",
                    name=name + "mean2_node",
                    attrs={"axes": [-1]},
                    inputs=[pow_out],
                    outputs=[mean2_out],
                )
                epsilon_out = gs.Variable(name=name + "epsilon_out")
                epsilon_const = gs.Constant(
                    name=name + "epsilon_const", values=np.array([node.attrs["epsilon"]], dtype=np.float32)
                )
                epsilon_node = gs.Node(
                    op="Add",
                    name=name + "epsilon_node",
                    attrs={},
                    inputs=[mean2_out, epsilon_const],
                    outputs=[epsilon_out],
                )
                sqrt_out = gs.Variable(name=name + "sqrt_out")
                sqrt_node = gs.Node(
                    op="Sqrt", name=name + "sqrt_node", attrs={}, inputs=[epsilon_out], outputs=[sqrt_out]
                )
                div_out = gs.Variable(name=name + "div_out")
                div_node = gs.Node(
                    op="Div", name=name + "div_node", attrs={}, inputs=[sub_out, sqrt_out], outputs=[div_out]
                )
                constant_scale = gs.Constant(
                    "InstanceNormScaleV-" + str(removed_instance_norm_count),
                    np.ascontiguousarray(node.inputs[1].inputs[0].attrs["value"].values.reshape(1, 32, 1)),
                )
                constant_bias = gs.Constant(
                    "InstanceBiasV-" + str(removed_instance_norm_count),
                    np.ascontiguousarray(node.inputs[2].inputs[0].attrs["value"].values.reshape(1, 32, 1)),
                )
                mul_out = gs.Variable(name=name + "mul_out")
                mul_node = gs.Node(
                    op="Mul", name=name + "mul_node", attrs={}, inputs=[div_out, constant_scale], outputs=[mul_out]
                )
                add_node = gs.Node(
                    op="Add", name=name + "add_node", attrs={}, inputs=[mul_out, constant_bias], outputs=[output_tensor]
                )
                self.graph.nodes.extend(
                    [mean_node, sub_node, pow_node, mean2_node, epsilon_node, sqrt_node, div_node, mul_node, add_node]
                )
                node.inputs = []
                node.outputs = []
                removed_instance_norm_count += 1

        self.cleanup()
        return removed_instance_norm_count

    def insert_groupnorm_plugin(self) -> int:
        group_norm_plugin_count = 0
        for node in self.graph.nodes:
            if (
                node.op == "Reshape"
                and node.outputs != []
                and node.o().op == "ReduceMean"
                and node.o(1).op == "Sub"
                and node.o().o() == node.o(1)
                and node.o().o().o().o().o().o().o().o().o().o().o().op == "Mul"
                and node.o().o().o().o().o().o().o().o().o().o().o().o().op == "Add"
                and len(node.o().o().o().o().o().o().o().o().inputs[1].values.shape) == 3
            ):
                # "node.outputs != []" is added for VAE

                input_tensor = node.inputs[0]

                gamma_node = node.o().o().o().o().o().o().o().o().o().o().o()
                index = [isinstance(inp, gs.ir.tensor.Constant) for inp in gamma_node.inputs].index(True)
                gamma = np.array(deepcopy(gamma_node.inputs[index].values.tolist()), dtype=np.float32)
                constant_gamma = gs.Constant(
                    "groupNormGamma-" + str(group_norm_plugin_count), np.ascontiguousarray(gamma.reshape(-1))
                )  # MUST use np.ascontiguousarray, or TRT will regard the shape of this Constant as (0) !!!

                beta_node = gamma_node.o()
                index = [isinstance(inp, gs.ir.tensor.Constant) for inp in beta_node.inputs].index(True)
                beta = np.array(deepcopy(beta_node.inputs[index].values.tolist()), dtype=np.float32)
                constant_beta = gs.Constant(
                    "groupNormBeta-" + str(group_norm_plugin_count), np.ascontiguousarray(beta.reshape(-1))
                )

                epsilon = node.o().o().o().o().o().inputs[1].values.tolist()[0]

                if beta_node.o().op == "Sigmoid":  # need Swish
                    use_swish = True
                    last_node = beta_node.o().o()  # Mul node of Swish
                else:
                    use_swish = False
                    last_node = beta_node  # Cast node after Group Norm

                if last_node.o().op == "Cast":
                    last_node = last_node.o()
                input_list = [input_tensor, constant_gamma, constant_beta]
                group_norm_v = gs.Variable(
                    "GroupNormV-" + str(group_norm_plugin_count), np.dtype(np.float16), input_tensor.shape
                )
                group_norm_n = gs.Node(
                    "GroupNorm",
                    "GroupNormN-" + str(group_norm_plugin_count),
                    inputs=input_list,
                    outputs=[group_norm_v],
                    attrs=OrderedDict([("epsilon", epsilon), ("bSwish", int(use_swish))]),
                )
                self.graph.nodes.append(group_norm_n)

                for sub_node in self.graph.nodes:
                    if last_node.outputs[0] in sub_node.inputs:
                        index = sub_node.inputs.index(last_node.outputs[0])
                        sub_node.inputs[index] = group_norm_v
                node.inputs = []
                last_node.outputs = []
                group_norm_plugin_count += 1

        self.cleanup()
        return group_norm_plugin_count

    def insert_layernorm_plugin(self) -> int:
        layer_norm_plugin_count = 0
        for node in self.graph.nodes:
            if (
                node.op == "ReduceMean"
                and node.o().op == "Sub"
                and node.o().inputs[0] == node.inputs[0]
                and node.o().o(0).op == "Pow"
                and node.o().o(1).op == "Div"
                and node.o().o(0).o().op == "ReduceMean"
                and node.o().o(0).o().o().op == "Add"
                and node.o().o(0).o().o().o().op == "Sqrt"
                and node.o().o(0).o().o().o().o().op == "Div"
                and node.o().o(0).o().o().o().o() == node.o().o(1)
                and node.o().o(0).o().o().o().o().o().op == "Mul"
                and node.o().o(0).o().o().o().o().o().o().op == "Add"
                and len(node.o().o(0).o().o().o().o().o().inputs[1].values.shape) == 1
            ):
                if node.i().op == "Add":
                    input_tensor = node.inputs[0]  # CLIP
                else:
                    input_tensor = node.i().inputs[0]  # UNet and VAE

                gamma_node = node.o().o().o().o().o().o().o()
                index = [isinstance(inp, gs.ir.tensor.Constant) for inp in gamma_node.inputs].index(True)
                gamma = np.array(deepcopy(gamma_node.inputs[index].values.tolist()), dtype=np.float32)
                constant_gamma = gs.Constant(
                    "LayerNormGamma-" + str(layer_norm_plugin_count), np.ascontiguousarray(gamma.reshape(-1))
                )  # MUST use np.ascontiguousarray, or TRT will regard the shape of this Constant as (0) !!!

                beta_node = gamma_node.o()
                index = [isinstance(inp, gs.ir.tensor.Constant) for inp in beta_node.inputs].index(True)
                beta = np.array(deepcopy(beta_node.inputs[index].values.tolist()), dtype=np.float32)
                constant_beta = gs.Constant(
                    "LayerNormBeta-" + str(layer_norm_plugin_count), np.ascontiguousarray(beta.reshape(-1))
                )

                input_list = [input_tensor, constant_gamma, constant_beta]
                layer_norm_v = gs.Variable(
                    "LayerNormV-" + str(layer_norm_plugin_count), np.dtype(np.float32), input_tensor.shape
                )
                layer_norm_n = gs.Node(
                    "LayerNorm",
                    "LayerNormN-" + str(layer_norm_plugin_count),
                    inputs=input_list,
                    attrs=OrderedDict([("epsilon", 1.0e-5)]),
                    outputs=[layer_norm_v],
                )
                self.graph.nodes.append(layer_norm_n)
                layer_norm_plugin_count += 1

                if beta_node.outputs[0] in self.graph.outputs:
                    index = self.graph.outputs.index(beta_node.outputs[0])
                    self.graph.outputs[index] = layer_norm_v
                else:
                    if beta_node.o().op == "Cast":
                        last_node = beta_node.o()
                    else:
                        last_node = beta_node
                    for sub_node in self.graph.nodes:
                        if last_node.outputs[0] in sub_node.inputs:
                            index = sub_node.inputs.index(last_node.outputs[0])
                            sub_node.inputs[index] = layer_norm_v
                    last_node.outputs = []

        self.cleanup()
        return layer_norm_plugin_count

    def fuse_kv(self, node_k: Any, node_v: Any, fused_kv_idx: int, heads: int, num_dynamic: int = 0) -> Any:
        # Get weights of K
        weights_k = node_k.inputs[1].values
        # Get weights of V
        weights_v = node_v.inputs[1].values
        # Input number of channels to K and V
        channel_count = weights_k.shape[0]
        # Number of heads
        num_heads = heads
        # Dimension per head
        head_dim = weights_k.shape[1] // num_heads

        # Concat and interleave weights such that the output of fused KV GEMM has [b, s_kv, h, 2, d] shape
        weights_kv = np.dstack(
            [
                weights_k.reshape(channel_count, num_heads, head_dim),
                weights_v.reshape(channel_count, num_heads, head_dim),
            ]
        ).reshape(channel_count, 2 * num_heads * head_dim)

        # K and V have the same input
        input_tensor = node_k.inputs[0]
        # K and V must have the same output which we feed into fmha plugin
        output_tensor_k = node_k.outputs[0]
        # Create tensor
        constant_weights_kv = gs.Constant(f"Weights_KV_{fused_kv_idx}", np.ascontiguousarray(weights_kv))

        # Create fused KV node
        fused_kv_node = gs.Node(
            op="MatMul",
            name=f"MatMul_KV_{fused_kv_idx}",
            inputs=[input_tensor, constant_weights_kv],
            outputs=[output_tensor_k],
        )
        self.graph.nodes.append(fused_kv_node)

        # Connect the output of fused node to the inputs of the nodes after K and V
        node_v.o(num_dynamic).inputs[0] = output_tensor_k
        node_k.o(num_dynamic).inputs[0] = output_tensor_k
        for i in range(0, num_dynamic):
            node_v.o().inputs.clear()
            node_k.o().inputs.clear()

        # Clear inputs and outputs of K and V to ge these nodes cleared
        node_k.outputs.clear()
        node_v.outputs.clear()
        node_k.inputs.clear()
        node_v.inputs.clear()

        self.cleanup()
        return fused_kv_node

    def insert_fmhca(
        self, node_q: Any, node_kv: Any, final_tranpose: Any, mhca_idx: int, heads: int, num_dynamic: int = 0
    ) -> None:
        # Get inputs and outputs for the fMHCA plugin
        # We take an output of reshape that follows the Q GEMM
        output_q = node_q.o(num_dynamic).o().inputs[0]
        output_kv = node_kv.o().inputs[0]
        output_final_tranpose = final_tranpose.outputs[0]

        # Clear the inputs of the nodes that follow the Q and KV GEMM
        # to delete these subgraphs (it will be substituted by fMHCA plugin)
        node_kv.outputs[0].outputs[0].inputs.clear()
        node_kv.outputs[0].outputs[0].inputs.clear()
        node_q.o(num_dynamic).o().inputs.clear()
        for i in range(0, num_dynamic):
            node_q.o(i).o().o(1).inputs.clear()

        weights_kv = node_kv.inputs[1].values
        dims_per_head = weights_kv.shape[1] // (heads * 2)

        # Reshape dims
        shape = gs.Constant(
            f"Shape_KV_{mhca_idx}",
            np.ascontiguousarray(np.array([0, 0, heads, 2, dims_per_head], dtype=np.int64)),
        )

        # Reshape output tensor
        output_reshape = gs.Variable(f"ReshapeKV_{mhca_idx}", np.dtype(np.float16), None)
        # Create fMHA plugin
        reshape = gs.Node(op="Reshape", name=f"Reshape_{mhca_idx}", inputs=[output_kv, shape], outputs=[output_reshape])
        # Insert node
        self.graph.nodes.append(reshape)

        # Create fMHCA plugin
        fmhca = gs.Node(
            op="fMHCA",
            name=f"fMHCA_{mhca_idx}",
            inputs=[output_q, output_reshape],
            outputs=[output_final_tranpose],
        )
        # Insert node
        self.graph.nodes.append(fmhca)

        # Connect input of fMHCA to output of Q GEMM
        node_q.o(num_dynamic).outputs[0] = output_q

        if num_dynamic > 0:
            reshape2_input1_out = gs.Variable(f"Reshape2_fmhca{mhca_idx}_out", np.dtype(np.int64), None)
            reshape2_input1_shape = gs.Node(
                "Shape",
                f"Reshape2_fmhca{mhca_idx}_shape",
                inputs=[node_q.inputs[0]],
                outputs=[reshape2_input1_out],
            )
            self.graph.nodes.append(reshape2_input1_shape)
            final_tranpose.o().inputs[1] = reshape2_input1_out

        # Clear outputs of transpose to get this subgraph cleared
        final_tranpose.outputs.clear()

        self.cleanup()

    def fuse_qkv(
        self, node_q: Any, node_k: Any, node_v: Any, fused_qkv_idx: int, heads: int, num_dynamic: int = 0
    ) -> Any:
        # Get weights of Q
        weights_q = node_q.inputs[1].values
        # Get weights of K
        weights_k = node_k.inputs[1].values
        # Get weights of V
        weights_v = node_v.inputs[1].values

        # Input number of channels to Q, K and V
        channel_count = weights_k.shape[0]
        # Number of heads
        num_heads = heads
        # Hidden dimension per head
        head_dim = weights_k.shape[1] // num_heads

        # Concat and interleave weights such that the output of fused QKV GEMM has [b, s, h, 3, d] shape
        weights_qkv = np.dstack(
            [
                weights_q.reshape(channel_count, num_heads, head_dim),
                weights_k.reshape(channel_count, num_heads, head_dim),
                weights_v.reshape(channel_count, num_heads, head_dim),
            ]
        ).reshape(channel_count, 3 * num_heads * head_dim)

        input_tensor = node_k.inputs[0]  # K and V have the same input
        # Q, K and V must have the same output which we feed into fmha plugin
        output_tensor_k = node_k.outputs[0]
        # Concat and interleave weights such that the output of fused QKV GEMM has [b, s, h, 3, d] shape
        constant_weights_qkv = gs.Constant(f"Weights_QKV_{fused_qkv_idx}", np.ascontiguousarray(weights_qkv))

        # Created a fused node
        fused_qkv_node = gs.Node(
            op="MatMul",
            name=f"MatMul_QKV_{fused_qkv_idx}",
            inputs=[input_tensor, constant_weights_qkv],
            outputs=[output_tensor_k],
        )
        self.graph.nodes.append(fused_qkv_node)

        # Connect the output of the fused node to the inputs of the nodes after Q, K and V
        node_q.o(num_dynamic).inputs[0] = output_tensor_k
        node_k.o(num_dynamic).inputs[0] = output_tensor_k
        node_v.o(num_dynamic).inputs[0] = output_tensor_k
        for i in range(0, num_dynamic):
            node_q.o().inputs.clear()
            node_k.o().inputs.clear()
            node_v.o().inputs.clear()

        # Clear inputs and outputs of Q, K and V to ge these nodes cleared
        node_q.outputs.clear()
        node_k.outputs.clear()
        node_v.outputs.clear()

        node_q.inputs.clear()
        node_k.inputs.clear()
        node_v.inputs.clear()

        self.cleanup()
        return fused_qkv_node

    def insert_fmha(self, node_qkv: Any, final_tranpose: Any, mha_idx: int, heads: int, num_dynamic: int = 0) -> None:
        # Get inputs and outputs for the fMHA plugin
        output_qkv = node_qkv.o().inputs[0]
        output_final_tranpose = final_tranpose.outputs[0]

        # Clear the inputs of the nodes that follow the QKV GEMM
        # to delete these subgraphs (it will be substituted by fMHA plugin)
        node_qkv.outputs[0].outputs[2].inputs.clear()
        node_qkv.outputs[0].outputs[1].inputs.clear()
        node_qkv.outputs[0].outputs[0].inputs.clear()

        weights_qkv = node_qkv.inputs[1].values
        dims_per_head = weights_qkv.shape[1] // (heads * 3)

        # Reshape dims
        shape = gs.Constant(
            f"Shape_QKV_{mha_idx}",
            np.ascontiguousarray(np.array([0, 0, heads, 3, dims_per_head], dtype=np.int64)),
        )

        # Reshape output tensor
        output_shape = gs.Variable(f"ReshapeQKV_{mha_idx}", np.dtype(np.float16), None)
        # Create fMHA plugin
        reshape = gs.Node(op="Reshape", name=f"Reshape_{mha_idx}", inputs=[output_qkv, shape], outputs=[output_shape])
        # Insert node
        self.graph.nodes.append(reshape)

        # Create fMHA plugin
        fmha = gs.Node(op="fMHA_V2", name=f"fMHA_{mha_idx}", inputs=[output_shape], outputs=[output_final_tranpose])
        # Insert node
        self.graph.nodes.append(fmha)

        if num_dynamic > 0:
            reshape2_input1_out = gs.Variable(f"Reshape2_{mha_idx}_out", np.dtype(np.int64), None)
            reshape2_input1_shape = gs.Node(
                "Shape", f"Reshape2_{mha_idx}_shape", inputs=[node_qkv.inputs[0]], outputs=[reshape2_input1_out]
            )
            self.graph.nodes.append(reshape2_input1_shape)
            final_tranpose.o().inputs[1] = reshape2_input1_out

        # Clear outputs of transpose to get this subgraph cleared
        final_tranpose.outputs.clear()

        self.cleanup()

    def mha_mhca_detected(self, node: Any, mha: bool) -> tuple[bool, int, int, Any, Any, Any, Any]:
        # Go from V GEMM down to the S*V MatMul and all way up to K GEMM
        # If we are looking for MHCA inputs of two matmuls (K and V) must be equal.
        # If we are looking for MHA inputs (K and V) must be not equal.
        if (
            node.op == "MatMul"
            and len(node.outputs) == 1
            and (
                (mha and len(node.inputs[0].inputs) > 0 and node.i().op == "Add")
                or (not mha and len(node.inputs[0].inputs) == 0)
            )
        ):
            if node.o().op == "Shape":
                if node.o(1).op == "Shape":
                    num_dynamic_kv = 3 if node.o(2).op == "Shape" else 2
                else:
                    num_dynamic_kv = 1
                # For Cross-Attention, if batch axis is dynamic (in QKV), assume H*W (in Q) is dynamic as well
                num_dynamic_q = num_dynamic_kv if mha else num_dynamic_kv + 1
            else:
                num_dynamic_kv = 0
                num_dynamic_q = 0

            o = node.o(num_dynamic_kv)
            if (
                o.op == "Reshape"
                and o.o().op == "Transpose"
                and o.o().o().op == "Reshape"
                and o.o().o().o().op == "MatMul"
                and o.o().o().o().i(0).op == "Softmax"
                and o.o().o().o().i(1).op == "Reshape"
                and o.o().o().o().i(0).i().op == "Mul"
                and o.o().o().o().i(0).i().i().op == "MatMul"
                and o.o().o().o().i(0).i().i().i(0).op == "Reshape"
                and o.o().o().o().i(0).i().i().i(1).op == "Transpose"
                and o.o().o().o().i(0).i().i().i(1).i().op == "Reshape"
                and o.o().o().o().i(0).i().i().i(1).i().i().op == "Transpose"
                and o.o().o().o().i(0).i().i().i(1).i().i().i().op == "Reshape"
                and o.o().o().o().i(0).i().i().i(1).i().i().i().i().op == "MatMul"
                and node.name != o.o().o().o().i(0).i().i().i(1).i().i().i().i().name
            ):
                # "len(node.outputs) == 1" to make sure we are not in the already fused node
                node_q = o.o().o().o().i(0).i().i().i(0).i().i().i()
                node_k = o.o().o().o().i(0).i().i().i(1).i().i().i().i()
                node_v = node
                final_tranpose = o.o().o().o().o(num_dynamic_q).o()
                # Sanity check to make sure that the graph looks like expected
                if node_q.op == "MatMul" and final_tranpose.op == "Transpose":
                    return True, num_dynamic_q, num_dynamic_kv, node_q, node_k, node_v, final_tranpose
        return False, 0, 0, None, None, None, None

    def fuse_kv_insert_fmhca(self, heads: int, mhca_index: int, sm: int) -> bool:
        nodes = self.graph.nodes
        # Iterate over graph and search for MHCA pattern
        for idx, _ in enumerate(nodes):
            # fMHCA can't be at the 2 last layers of the network. It is a guard from OOB
            if idx + 1 > len(nodes) or idx + 2 > len(nodes):
                continue

            # Get anchor nodes for fusion and fMHCA plugin insertion if the MHCA is detected
            detected, num_dynamic_q, num_dynamic_kv, node_q, node_k, node_v, final_tranpose = self.mha_mhca_detected(
                nodes[idx], mha=False
            )
            if detected:
                assert num_dynamic_q == 0 or num_dynamic_q == num_dynamic_kv + 1
                # Skip the FMHCA plugin for SM75 except for when the dim per head is 40.
                if sm == 75 and node_q.inputs[1].shape[1] // heads == 160:
                    continue
                # Fuse K and V GEMMS
                node_kv = self.fuse_kv(node_k, node_v, mhca_index, heads, num_dynamic_kv)
                # Insert fMHCA plugin
                self.insert_fmhca(node_q, node_kv, final_tranpose, mhca_index, heads, num_dynamic_q)
                return True
        return False

    def fuse_qkv_insert_fmha(self, heads: int, mha_index: int) -> bool:
        nodes = self.graph.nodes
        # Iterate over graph and search for MHA pattern
        for idx, _ in enumerate(nodes):
            # fMHA can't be at the 2 last layers of the network. It is a guard from OOB
            if idx + 1 > len(nodes) or idx + 2 > len(nodes):
                continue

            # Get anchor nodes for fusion and fMHA plugin insertion if the MHA is detected
            detected, num_dynamic_q, num_dynamic_kv, node_q, node_k, node_v, final_tranpose = self.mha_mhca_detected(
                nodes[idx], mha=True
            )
            if detected:
                assert num_dynamic_q == num_dynamic_kv
                # Fuse Q, K and V GEMMS
                node_qkv = self.fuse_qkv(node_q, node_k, node_v, mha_index, heads, num_dynamic_kv)
                # Insert fMHA plugin
                self.insert_fmha(node_qkv, final_tranpose, mha_index, heads, num_dynamic_kv)
                return True
        return False

    def insert_fmhca_plugin(self, num_heads: int, sm: int) -> int:
        mhca_index = 0
        while self.fuse_kv_insert_fmhca(num_heads, mhca_index, sm):
            mhca_index += 1
        return mhca_index

    def insert_fmha_plugin(self, num_heads: int) -> int:
        mha_index = 0
        while self.fuse_qkv_insert_fmha(num_heads, mha_index):
            mha_index += 1
        return mha_index

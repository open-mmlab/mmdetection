# Copyright (C) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import tempfile
from copy import deepcopy as copy

import torch
import torch._C
import torch.jit
import torch.nn as nn
import torch.utils.cpp_extension
import torch.onnx.symbolic_helper as sym_help
from torch.onnx.symbolic_registry import register_op, is_registered_op

from ...ops import RoIAlign


class TracerStub(nn.Module):
    # References to all of the name not listed here will be forwarded to self.inner.
    special = {'inner', 'namespace', 'name', 'qualified_name',
               'num_outputs', 'params', 'verbose', 'force_rebuild'}

    def __init__(self, inner, namespace, name, verbose=False,
                 force_rebuild=True, unique_name=True):
        super().__init__()
        self.inner = inner
        self.namespace = namespace
        self.name = name
        if unique_name:
            self.name = name + '_' + next(tempfile._get_candidate_names()) + 'x'
        self.qualified_name = '{}::{}'.format(self.namespace, self.name)
        self.num_outputs = 1
        self.params = {}
        self.verbose = verbose
        self.force_rebuild = force_rebuild

        # Register symbolic function for ONNX export.
        while not is_registered_op(self.name, self.namespace, 10):
            register_op(self.name, self.symbolic, self.namespace, 10)

    def _flatten(self, args):
        flat_args = []
        for a in args:
            if isinstance(a, (list, tuple)):
                flat_args.extend(self._flatten(a))
            elif isinstance(a, torch.Tensor):
                flat_args.append(a.cpu())
            else:
                print('WARNING! Input argument is neither a Tensor,'
                      'nor a list/tuple of Tensors.')
        return flat_args

    def get_op(self):
        try:
            ns = getattr(torch.ops, self.namespace)
            op = getattr(ns, self.name)
        except RuntimeError:
            op = None
        return op

    @staticmethod
    def as_list(x):
        if not isinstance(x, (tuple, list)):
            return [x, ]
        return x

    def forward(self, *args, **kwargs):
        op = self.get_op()
        flat_args = self._flatten(args)

        with torch.jit._disable_tracing():
            # Run inner module to get actual output values.
            actual_outputs = self.as_list(self.inner(*args, **kwargs))
            self.num_outputs = len(actual_outputs)
            devices = list(actual_output.device for actual_output in actual_outputs)

            if op is None or self.force_rebuild:
                # Generate C++ stub code.
                stub_op_source_template = \
                    self.compose_stub_code(flat_args, self.num_outputs, actual_outputs)
                if self.verbose:
                    print(stub_op_source_template)

                # Compile and register stub.
                torch.utils.cpp_extension.load_inline(
                    name=next(tempfile._get_candidate_names()),
                    cpp_sources=stub_op_source_template,
                    is_python_module=False,
                    verbose=self.verbose,
                )

                # Get just loaded operation.
                op = self.get_op()

            # kwargs are not forwarded to a stub call.
            # Save those as stubs' parameters to make them available for symbolic.
            self.params.update(copy(kwargs))

        # Call operation from generated extension with tracing enabled.
        outputs = self.as_list(op(*flat_args))

        with torch.jit._disable_tracing(), torch.no_grad():
            # Fill output tensors with valid values.
            for actual_output, output in zip(actual_outputs, outputs):
                output[...] = actual_output.cpu()

        # Restore storage device for outputs.
        outputs = list(output.to(device) for output, device in zip(outputs, devices))

        if self.num_outputs == 1:
            outputs = outputs[0]
        return outputs

    def __getattr__(self, item):
        if item in self.special:
            return super().__getattr__(item)
        else:
            # Forward calls to self.inner.
            inner = self.inner
            try:
                v = inner.__getattribute__(item)
            except AttributeError:
                v = inner.__getattr__(item)
            return v

    def symbolic(self, g, *args):
        raise NotImplementedError

    def compose_stub_code(self, args, nout, outs):
        def format_shape(s):
            return str(list(s)).replace('[', '{').replace(']', '}')

        includes = '#include <torch/script.h>'
        usings = 'using torch::Tensor;'
        register = 'static auto registry = torch::RegisterOperators("{}", &stub);'\
            .format(self.qualified_name)
        in_args = ', '.join(['Tensor in_{}'.format(i) for i in range(len(args))])
        return_value_type = 'Tensor'
        if nout > 1:
            return_value_type = 'std::tuple<' + ', '.join(['Tensor'] * nout) + '>'
        outputs = '{' + \
                  ', '.join('torch::zeros({}, torch::requires_grad())'
                            .format(format_shape(out.shape)) for out in outs) +\
                  '}'
        all_at_once = """
        {}
        {}

        {} stub({}) {{
            {} outs {};
            return outs;
        }};

        {}
        """.format(includes, usings, return_value_type, in_args,
                   return_value_type, outputs, register)
        return all_at_once


class AnchorsGridGeneratorStub(TracerStub):
    def __init__(self, inner, namespace='mmdet_custom',
                 name='anchor_grid_generator', **kwargs):
        super().__init__(inner, namespace=namespace, name=name, **kwargs)
        self.inner = lambda x, **kw: inner(x.shape[2:4], **kw)

    def forward(self, featmap, stride=16, device='cpu'):
        # Force `stride` and `device` to be passed in as kwargs.
        return super().forward(featmap, stride=stride, device=device)

    def symbolic(self, g, featmap):
        stride = float(self.params['stride'])
        shift = torch.full(self.params['base_anchors'].shape, - 0.5 * stride, dtype=torch.float32)
        prior_boxes = g.op('Constant', value_t=torch.tensor(self.params['base_anchors'], dtype=torch.float32) + shift)
        # TODO. im_data is not needed actually.
        im_data = g.op('Constant', value_t=torch.zeros([1, 1, 1, 1], dtype=torch.float32))

        return g.op('ExperimentalDetectronPriorGridGenerator',
                    prior_boxes,
                    featmap,
                    im_data,
                    stride_x_f=stride,
                    stride_y_f=stride,
                    h_i=0,
                    w_i=0,
                    flatten_i=1,
                    outputs=self.num_outputs)


class ROIFeatureExtractorStub(TracerStub):

    class Wrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def __call__(self, rois, *feats):
            return self.inner(feats, rois)

        def __getattr__(self, item):
            if item == 'inner':
                return super().__getattr__(item)
            # Forward calls to self.inner.
            inner = self.inner
            try:
                v = inner.__getattribute__(item)
            except AttributeError:
                v = inner.__getattr__(item)
            return v

    def __init__(self, inner, namespace='mmdet_custom',
                 name='roi_feature_extractor', **kwargs):
        super().__init__(self.Wrapper(inner), namespace=namespace, name=name, **kwargs)
        out_size = self.inner.roi_layers[0].out_size[0]
        for roi_layer in self.inner.roi_layers:
            size = roi_layer.out_size
            assert isinstance(roi_layer, RoIAlign)
            assert len(size) == 2
            assert size[0] == size[1]
            assert size[0] == out_size

    def forward(self, feats, rois, roi_scale_factor=None):
        assert roi_scale_factor is None, 'roi_scale_factor is not supported'
        return super().forward(rois, *feats)

    def symbolic(self, g, rois, *feats):
        rois = sym_help._slice_helper(g, rois, axes=[1], starts=[1], ends=[5])
        roi_feats, _ = g.op('ExperimentalDetectronROIFeatureExtractor',
            rois,
            *feats,
            output_size_i=self.inner.roi_layers[0].out_size[0],
            pyramid_scales_i=self.inner.featmap_strides,
            sampling_ratio_i=self.inner.roi_layers[0].sample_num,
            image_id_i=0,
            distribute_rois_between_levels_i=1,
            preserve_rois_order_i=1,
            outputs=2
            )
        return roi_feats


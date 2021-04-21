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

import onnx
import onnxruntime
from onnx import helper, shape_inference

from mmdet.models import build_detector


class ModelONNXRuntime:

    def __init__(self, model_file_path, cfg=None, classes=None):
        self.device = onnxruntime.get_device()
        self.model = onnx.load(model_file_path)
        self.classes = classes
        self.pt_model = None
        if cfg is not None:
            self.pt_model = build_detector(
                cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
            if classes is not None:
                self.pt_model.CLASSES = classes

        self.sess_options = onnxruntime.SessionOptions()
        # self.sess_options.enable_profiling = False

        self.session = onnxruntime.InferenceSession(
            self.model.SerializeToString(), self.sess_options)
        self.input_names = []
        self.output_names = []
        for input in self.session.get_inputs():
            self.input_names.append(input.name)
        for output in self.session.get_outputs():
            self.output_names.append(output.name)

    def show(self, data, result, score_thr=0.3, wait_time=0):
        if self.pt_model is not None:
            self.pt_model.show_result(
                data, result, show=True, score_thr=score_thr, wait_time=wait_time)

    def add_output(self, output_ids):
        if not isinstance(output_ids, (tuple, list, set)):
            output_ids = [
                output_ids,
            ]

        inferred_model = shape_inference.infer_shapes(self.model)
        all_blobs_info = {
            value_info.name: value_info
            for value_info in inferred_model.graph.value_info
        }

        extra_outputs = []
        for output_id in output_ids:
            value_info = all_blobs_info.get(output_id, None)
            if value_info is None:
                print('WARNING! No blob with name {}'.format(output_id))
                extra_outputs.append(
                    helper.make_empty_tensor_value_info(output_id))
            else:
                extra_outputs.append(value_info)

        self.model.graph.output.extend(extra_outputs)
        self.output_names.extend(output_ids)
        self.session = onnxruntime.InferenceSession(
            self.model.SerializeToString(), self.sess_options)

    def unify_inputs(self, inputs):
        if not isinstance(inputs, dict):
            if len(self.input_names) == 1 and not isinstance(inputs, (list, tuple)):
                inputs = [inputs]
            inputs = dict(zip(self.input_names, inputs))
        return inputs

    def __call__(self, inputs, *args, **kwargs):
        inputs = self.unify_inputs(inputs)
        outputs = self.session.run(None, inputs, *args, **kwargs)
        outputs = dict(zip(self.output_names, outputs))
        return outputs

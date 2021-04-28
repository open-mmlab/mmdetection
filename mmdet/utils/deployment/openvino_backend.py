# Copyright (C) 2021 Intel Corporation
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

import logging
import numpy as np
import os.path as osp
import string
import torch
from collections import OrderedDict
from lxml import etree
from openvino.inference_engine import IECore
from scipy.special import softmax

from ...models import build_detector


class Model:
    def __init__(self, model_path, ie=None, device='CPU', cfg=None, classes=None):
        self.logger = logging.getLogger()
        self.logger.info('Reading network from IR...')

        self.ie = IECore() if ie is None else ie
        bin_path = osp.splitext(model_path)[0] + '.bin'
        self.net = self.ie.read_network(model_path, bin_path)

        self.device = None
        self.exec_net = None
        self.to(device)

        self.pt_model = None
        if cfg is not None:
            self.pt_model = build_detector(cfg.model)
            if classes is not None:
                self.pt_model.CLASSES = classes

    def to(self, device):
        if self.device != device:
            self.device = device
            self.exec_net = self.ie.load_network(network=self.net, device_name=self.device, num_requests=1)
        return self

    def unify_inputs(self, inputs):
        if not isinstance(inputs, dict):
            inputs_dict = {next(iter(self.net.input_info)): inputs}
        else:
            inputs_dict = inputs
        return inputs_dict

    def reshape(self, inputs=None, input_shapes=None):
        assert (inputs is None) != (input_shapes is None)
        if input_shapes is None:
            input_shapes = {name: data.shape for name, data in inputs.items()}
        reshape_needed = False
        for input_name, input_shape in input_shapes.items():
            blob_shape = self.net.input_info[input_name].input_data.shape
            if not np.array_equal(input_shape, blob_shape):
                reshape_needed = True
                break
        if reshape_needed:
            self.logger.info(f'reshape net to {input_shapes}')
            self.net.reshape(input_shapes)
            self.exec_net = self.ie.load_network(network=self.net, device_name=self.device, num_requests=1)

    def get(self, outputs, name):
        try:
            key = self.net.get_ov_name_for_tensor(name)
            assert key in outputs, f'"{key}" is not a valid output identifier'
        except KeyError:
            if name not in outputs:
                raise KeyError(f'Failed to identify output "{name}"')
            key = name
        return outputs[key]

    def preprocess(self, inputs):
        return inputs

    def postprocess(self, outputs):
        return outputs

    def __call__(self, inputs):
        inputs = self.unify_inputs(inputs)
        inputs = self.preprocess(inputs)
        self.reshape(inputs=inputs)
        outputs = self.exec_net.infer(inputs)
        outputs = self.postprocess(outputs)
        return outputs

    def show(self, data, result, dataset=None, score_thr=0.3, wait_time=0):
        if self.pt_model is not None:
            self.pt_model.show_result(
                data, result, show=True, score_thr=score_thr, wait_time=wait_time)


class Detector(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        batch_size = self.net.input_info['image'].input_data.shape[0]
        assert batch_size == 1, 'Only batch 1 is supported.'

    def __call__(self, inputs):
        inputs = self.unify_inputs(inputs)
        output = super().__call__(inputs)

        if 'detection_out' in output:
            detection_out = output['detection_out']
            output['labels'] = detection_out[0, 0, :, 1].astype(np.int32)
            output['boxes'] = detection_out[0, 0, :, 3:] * np.tile(inputs['image'].shape[:1:-1], 2)
            output['boxes'] = np.concatenate((output['boxes'], detection_out[0, 0, :, 2:3]), axis=1)
            del output['detection_out']
            return output

        outs = output
        output = {}
        output = {
            'labels': self.get(outs, 'labels'),
            'boxes': self.get(outs, 'boxes')
        }
        valid_detections_mask = output['labels'] >= 0
        output['labels'] = output['labels'][valid_detections_mask]
        output['boxes'] = output['boxes'][valid_detections_mask]
        try:
            output['masks'] = self.get(outs, 'masks')
            output['masks'] = output['masks'][valid_detections_mask]
        except RuntimeError:
            pass

        return output


class MaskTextSpotterOpenVINO(Model):
    def __init__(self, xml_file_path, *args, text_recognition_thr=0.5, **kwargs):
        super().__init__(xml_file_path, *args, **kwargs)

        batch_size = self.net.input_info['image'].input_data.shape[0]
        assert batch_size == 1, 'Only batch 1 is supported.'

        self.n, self.c, self.h, self.w = self.net.input_info['image'].input_data.shape
        assert self.n == 1, 'Only batch 1 is supported.'

        xml_path = xml_file_path.replace('.xml', '_text_recognition_head_encoder.xml')
        self.text_encoder = Model(xml_path)

        xml_path = xml_file_path.replace('.xml', '_text_recognition_head_decoder.xml')
        self.text_decoder = Model(xml_path)
        self.hidden_shape = [v.shape for k, v in self.text_decoder.net.inputs.items() if k == 'prev_hidden'][0]
        self.alphabet = '  ' + string.ascii_lowercase + string.digits
        self.text_recognition_thr = text_recognition_thr


    def __call__(self, inputs, **kwargs):
        inputs = self.unify_inputs(inputs)
        output = super().__call__(inputs)

        output = {k: self.get(output, k) for k in ('boxes', 'labels', 'masks', 'text_features')}

        valid_detections_mask = output['boxes'][:,-1] > 0
        output['labels'] = output['labels'][valid_detections_mask]
        output['boxes'] = output['boxes'][valid_detections_mask]
        output['text_features'] = output['text_features'][valid_detections_mask]

        if 'masks' in output:
            output['masks'] = output['masks'][valid_detections_mask]

        eos = 1
        max_seq_len = 28

        confidences = []
        decoded_texts = []
        distributions = []
        for feature in output['text_features']:
            feature = np.expand_dims(feature, 0)
            feature = self.text_encoder({'input': feature})
            feature = self.text_encoder.get(feature, 'output')
            feature = np.reshape(feature, (feature.shape[0], feature.shape[1], -1))
            feature = np.transpose(feature, (0, 2, 1))

            hidden = np.zeros(self.hidden_shape)
            prev_symbol = np.zeros((1,))

            decoded = ''
            confidence = 1

            distribution = []

            for _ in range(max_seq_len):
                out = self.text_decoder({
                    'prev_symbol': prev_symbol,
                    'prev_hidden': hidden,
                    'encoder_outputs': feature
                })
                softmaxed = softmax(self.text_decoder.get(out, 'output'), axis=1)
                distribution.append(softmaxed[0, 2:])
                softmaxed_max = np.max(softmaxed, axis=1)
                confidence *= softmaxed_max[0]
                prev_symbol = np.argmax(softmaxed, axis=1)

                if prev_symbol == eos:
                    break
                hidden = self.text_decoder.get(out, 'hidden')
                decoded = decoded + self.alphabet[prev_symbol[0]]
            distribution = np.transpose(np.array(distribution))
            distributions.append(distribution)
            confidences.append(confidence)
            decoded_texts.append(decoded if confidence >= self.text_recognition_thr else '')

        output['texts'] = decoded_texts, confidences, distributions

        return output

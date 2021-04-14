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

from collections import OrderedDict
import logging
import string

from lxml import etree
import numpy as np
from openvino.inference_engine import IECore
import os.path as osp
from scipy.special import softmax
import torch

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
            self.pt_model = build_detector(
                cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
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
            print(f'reshape net to {input_shapes}')
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
        print(f'get {name} {key}')
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

        print(list(output.keys()))

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


class ModelOpenVINO:

    def __init__(self,
                 xml_file_path,
                 bin_file_path=None,
                 mapping_file_path=None,
                 device='CPU',
                 required_inputs=None,
                 required_outputs=None,
                 max_num_requests=1,
                 collect_perf_counters=False,
                 cfg=None,
                 classes=None):

        ie = IECore()
        logging.info('Reading network from IR...')
        if bin_file_path is None:
            bin_file_path = osp.splitext(xml_file_path)[0] + '.bin'
        if mapping_file_path is None:
            mapping_file_path = osp.splitext(xml_file_path)[0] + '.mapping'

        self.net = ie.read_network(model=xml_file_path, weights=bin_file_path)

        self.orig_ir_mapping = self.get_mapping(mapping_file_path)
        self.ir_orig_mapping = {v: k for k, v in self.orig_ir_mapping.items()}

        self.net_inputs_mapping = OrderedDict({})
        self.net_outputs_mapping = OrderedDict({})
        self.configure_inputs(required_inputs)
        self.configure_outputs(required_outputs)

        logging.info('Loading network to plugin...')
        self.max_num_requests = max_num_requests
        self.exec_net = ie.load_network(network=self.net, device_name=device, num_requests=max_num_requests)

        self.perf_counters = None
        if collect_perf_counters:
            self.perf_counters = PerformanceCounters()

        self.pt_model = None
        if cfg is not None:
            self.pt_model = build_detector(
                cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
            if classes is not None:
                self.pt_model.CLASSES = classes

    def get_mapping(self, mapping_file_path=None):
        mapping = {}
        if mapping_file_path is not None:
            logging.info('Loading mapping file...')
            root = etree.parse(mapping_file_path).getroot()
            for m in root:
                if m.tag != 'map':
                    continue
                framework = m.find('framework')
                ir = m.find('IR')
                framework_name = framework.get('name')
                ir_name = ir.get('name')
                mapping[framework_name] = ir_name
        return mapping

    def try_add_extra_outputs(self, extra_outputs):
        if extra_outputs is None:
            return
        for extra_output in extra_outputs:
            if extra_output not in self.orig_ir_mapping:
                continue
            ir_name = self.orig_ir_mapping[extra_output]
            try:
                self.net.add_outputs(ir_name)
                logging.warning(f'Added "{extra_output}" output with "{ir_name}" name in IR')
            except RuntimeError as e:
                logging.warning(f'Failed to add "{extra_output}" output with "{ir_name}" name in IR')
                pass

    def configure_inputs(self, required):
        self.net_inputs_mapping = OrderedDict((i, i) for i in self.net.input_info.keys())
        self.check_required(self.net_inputs_mapping.keys(), required)

    def configure_outputs(self, required):
        self.try_add_extra_outputs(required)
        self.net_outputs_mapping = OrderedDict((i, self.ir_orig_mapping[i]) for i in self.net.outputs.keys())
        self.check_required(self.orig_ir_mapping.keys(), required)

    def set_outputs(self, outputs):
        self.check_required(self.orig_ir_mapping.keys(), outputs)
        self.net_outputs_mapping = OrderedDict((self.orig_ir_mapping[i], i) for i in outputs)

    @staticmethod
    def check_required(available, required):
        if required is None:
            return
        for x in required:
            if x not in available:
                raise ValueError(f'Failed to identify data blob with name "{x}"')

    def rename_outputs(self, outputs):
        new_items = []
        for k, v in self.net_outputs_mapping.items():
            if k not in outputs:
                new_items.append([k + '.0', v])
        if new_items:
            for k, v in new_items:
                self.net_outputs_mapping[k] = v

        return {self.net_outputs_mapping[k]: v for k, v in outputs.items() if k in self.net_outputs_mapping}

    def unify_inputs(self, inputs):
        if not isinstance(inputs, dict):
            if len(self.net_inputs_mapping) == 1 and not isinstance(inputs, (list, tuple)):
                inputs = [inputs]
            inputs = {k: v for (k, _), v in zip(self.net_inputs_mapping.items(), inputs)}
        inputs = {self.net_inputs_mapping[k]: v for k, v in inputs.items()}
        return inputs

    def __call__(self, inputs):
        inputs = self.unify_inputs(inputs)
        outputs = self.exec_net.infer(inputs)
        if self.perf_counters:
            perf_counters = self.exec_net.requests[0].get_perf_counts()
            self.perf_counters.update(perf_counters)
        return self.rename_outputs(outputs)

    def print_performance_counters(self):
        if self.perf_counters:
            self.perf_counters.print()

    def show(self, data, result, dataset=None, score_thr=0.3, wait_time=0):
        if self.pt_model is not None:
            self.pt_model.show_result(
                data, result, show=True, score_thr=score_thr, wait_time=wait_time)


class MaskTextSpotterOpenVINO(ModelOpenVINO):
    def __init__(self, xml_file_path, *args, text_recognition_thr=0.5, **kwargs):
        self.with_mask = False
        super().__init__(xml_file_path,
                         *args,
                         required_inputs=('image', ),
                         required_outputs=None,
                         **kwargs)
        self.n, self.c, self.h, self.w = self.net.input_info['image'].input_data.shape
        assert self.n == 1, 'Only batch 1 is supported.'

        xml_path = xml_file_path.replace('.xml', '_text_recognition_head_encoder.xml')
        self.text_encoder = ModelOpenVINO(xml_path, xml_path.replace('.xml', '.bin'))

        xml_path = xml_file_path.replace('.xml', '_text_recognition_head_decoder.xml')
        self.text_decoder = ModelOpenVINO(xml_path, xml_path.replace('.xml', '.bin'))
        self.hidden_shape = [v.shape for k, v in self.text_decoder.net.inputs.items() if k == 'prev_hidden'][0]
        self.alphabet = '  ' + string.ascii_lowercase + string.digits
        self.text_recognition_thr = text_recognition_thr

    def configure_outputs(self, required):
        extra_outputs = ['boxes', 'labels', 'masks', 'text_features']

        for output in extra_outputs:
            if output not in self.orig_ir_mapping and output in self.net.outputs:
                self.orig_ir_mapping[output] = output
            print(self.orig_ir_mapping[output], output)

        self.try_add_extra_outputs(extra_outputs)
        outputs = []

        self.check_required(self.orig_ir_mapping.keys(), ['boxes', 'labels', 'text_features'])
        self.with_detection_output = False
        outputs.extend(['boxes', 'labels', 'text_features'])

        try:
            self.check_required(self.orig_ir_mapping.keys(), ['masks'])
            self.with_mask = True
            outputs.append('masks')
        except ValueError:
            self.with_mask = False

        self.set_outputs(outputs)

    def __call__(self, inputs, **kwargs):
        inputs = self.unify_inputs(inputs)
        data_h, data_w = inputs['image'].shape[-2:]
        inputs['image'] = np.pad(inputs['image'],
                                 ((0, 0), (0, 0), (0, self.h - data_h), (0, self.w - data_w)),
                                 mode='constant')

        output = super().__call__(inputs)

        valid_detections_mask = output['boxes'][:,-1] > 0
        output['labels'] = output['labels'][valid_detections_mask]
        output['boxes'] = output['boxes'][valid_detections_mask]
        output['text_features'] = output['text_features'][valid_detections_mask]

        if 'masks' in output:
            output['masks'] = output['masks'][valid_detections_mask]

        texts = []
        for feature in output['text_features']:
            feature = np.expand_dims(feature, 0)
            feature = self.text_encoder({'input': feature})['output']
            feature = np.reshape(feature, (feature.shape[0], feature.shape[1], -1))
            feature = np.transpose(feature, (0, 2, 1))

            hidden = np.zeros(self.hidden_shape)
            prev_symbol = np.zeros((1,))

            eos = 1
            max_seq_len = 28

            decoded = ''
            confidence = 1

            for _ in range(max_seq_len):
                out = self.text_decoder({
                    'prev_symbol': prev_symbol,
                    'prev_hidden': hidden,
                    'encoder_outputs': feature
                })
                softmaxed = softmax(out['output'], axis=1)
                softmaxed_max = np.max(softmaxed, axis=1)
                confidence *= softmaxed_max[0]
                prev_symbol = np.argmax(softmaxed, axis=1)[0]
                if prev_symbol == eos:
                    break
                hidden = out['hidden']
                decoded = decoded + self.alphabet[prev_symbol]

            texts.append(decoded if confidence >= self.text_recognition_thr else '')

        texts = np.array(texts)
        output['texts'] = texts

        return output

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


import logging
import os.path as osp
from collections import OrderedDict

import numpy as np
import torch
from lxml import etree
from openvino.inference_engine import IECore

from ...models import build_detector


class PerformanceCounters(object):
    def __init__(self):
        self.pc = {}

    def update(self, pc):
        for layer, stats in pc.items():
            if layer not in self.pc:
                self.pc[layer] = dict(layer_type=stats['layer_type'],
                                      exec_type=stats['exec_type'],
                                      status=stats['status'],
                                      real_time=stats['real_time'],
                                      calls=1)
            else:
                self.pc[layer]['real_time'] += stats['real_time']
                self.pc[layer]['calls'] += 1

    def print(self):
        print('Performance counters:')
        print(' '.join(['name', 'layer_type', 'exec_type', 'status', 'real_time(us)']))
        for layer, stats in self.pc.items():
            print('{} {} {} {} {}'.format(layer, stats['layer_type'], stats['exec_type'],
                                          stats['status'], stats['real_time'] / stats['calls']))


class ModelOpenVINO(object):

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

        from openvino.inference_engine import IENetwork

        ie = IECore()
        logging.info('Reading network from IR...')
        if bin_file_path is None:
            bin_file_path = osp.splitext(xml_file_path)[0] + '.bin'
        if mapping_file_path is None:
            mapping_file_path = osp.splitext(xml_file_path)[0] + '.mapping'

        self.net = IENetwork(model=xml_file_path, weights=bin_file_path)

        self.orig_ir_mapping = self.get_mapping(mapping_file_path)
        self.ir_orig_mapping = {v: k for k, v in self.orig_ir_mapping.items()}

        self.net_inputs_mapping = OrderedDict({})
        self.net_outputs_mapping = OrderedDict({})
        self.configure_inputs(required_inputs)
        self.configure_outputs(required_outputs)

        if 'CPU' in device:
            self.check_cpu_support(ie, self.net)

        logging.info('Loading network to plugin...')
        self.max_num_requests = max_num_requests
        self.exec_net = ie.load_network(network=self.net, device_name=device, num_requests=max_num_requests)

        self.perf_counters = None
        if collect_perf_counters:
            self.perf_counters = PerformanceCounters()

        self.pt_model = None
        if cfg is not None:
            self.pt_model = build_detector(
                cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
            if classes is not None:
                self.pt_model.CLASSES = classes

    @staticmethod
    def check_cpu_support(ie, net):
        logging.info('Check that all layers are supported...')
        supported_layers = ie.query_network(net, 'CPU')
        not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
        if len(not_supported_layers) != 0:
            unsupported_info = '\n\t'.join('{} ({} with params {})'.format(layer_id,
                                                                           net.layers[layer_id].type,
                                                                           str(net.layers[layer_id].params))
                                           for layer_id in not_supported_layers)
            logging.warning('Following layers are not supported '
                            'by the CPU plugin:\n\t{}'.format(unsupported_info))
            logging.warning('Please try to specify cpu extensions library path.')
            raise ValueError('Some of the layers are not supported.')

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
                if framework_name != ir_name:
                    # FIXME. This may not be correct for all operations.
                    mapping[framework_name] += '.0'
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
            except RuntimeError:
                pass

    def configure_inputs(self, required):
        self.net_inputs_mapping = OrderedDict((i, i) for i in self.net.inputs.keys())
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
                data, result, dataset=dataset, score_thr=score_thr, wait_time=wait_time)


class DetectorOpenVINO(ModelOpenVINO):
    def __init__(self, *args, with_detection_output=False, **kwargs):
        self.with_detection_output = False
        self.with_mask = False
        super().__init__(*args,
                         required_inputs=('image', ),
                         required_outputs=None,
                         **kwargs)
        self.n, self.c, self.h, self.w = self.net.inputs['image'].shape
        assert self.n == 1, 'Only batch 1 is supported.'

    def configure_outputs(self, required):
        self.try_add_extra_outputs(['boxes', 'labels', 'masks', 'detection_out'])
        outputs = []
        
        try:
            self.check_required(self.orig_ir_mapping.keys(), ['detection_out'])
            self.with_detection_output = True
            outputs.append('detection_out')
        except ValueError:
            self.check_required(self.orig_ir_mapping.keys(), ['boxes', 'labels'])
            self.with_detection_output = False
            outputs.extend(['boxes', 'labels'])

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
        if self.with_detection_output:
            detection_out = output['detection_out']
            output['labels'] = detection_out[0, 0, :, 1].astype(np.int32) - 1
            output['boxes'] = detection_out[0, 0, :, 3:] * np.tile(inputs['image'].shape[2:][::-1], 2)
            output['boxes'] = np.concatenate((output['boxes'], detection_out[0, 0, :, 2:3]), axis=1)
            del output['detection_out']

        valid_detections_mask = output['labels'] >= 0
        output['labels'] = output['labels'][valid_detections_mask]
        output['boxes'] = output['boxes'][valid_detections_mask]
        if 'masks' in output:
            output['masks'] = output['masks'][valid_detections_mask]

        return output

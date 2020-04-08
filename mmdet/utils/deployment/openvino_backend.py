"""
 Copyright (c) 2020 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import logging
import os.path as osp
from collections import OrderedDict

import numpy as np
import torch
from lxml import etree

from mmdet.models import build_detector


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


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return x


class IECore(object):
    def __new__(cls):
        from openvino.inference_engine import IECore as IECore_

        if not hasattr(cls, 'instance'):
            cls.instance = IECore_()
        return cls.instance


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
        self.net = IENetwork(model=xml_file_path, weights=bin_file_path)

        self.mapping = {'labels': 'labels', 'boxes': '23597/Split.0'}
        self.mapping = {}
        self.mapping_file_path = mapping_file_path
        self.net_inputs_mapping = OrderedDict((i, i) for i in self.net.inputs.keys())
        if required_inputs is not None:
            assert set(required_inputs) == set(self.net.inputs.keys())
        self.net_outputs_mapping = self.configure_outputs(self.net, required_outputs)

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

    def get_mapping(self):
        if len(self.mapping) == 0 and self.mapping_file_path is not None:
            logging.info('Loading mapping file...')
            self.mapping = {}
            root = etree.parse(self.mapping_file_path).getroot()
            for m in root:
                if m.tag != 'map':
                    continue
                framework = m.find('framework')
                ir = m.find('IR')
                self.mapping[framework.get('name')] = ir.get('name')
                if framework.get('name') != ir.get('name'):
                    self.mapping[framework.get('name')] += '.0'
        return self.mapping

    def configure_outputs(self, net, required_outputs):
        net_outputs_mapping = OrderedDict()
        if required_outputs is None:
            for o in net.outputs.keys():
                net_outputs_mapping[o] = o
        else:
            for required_output in required_outputs:
                output = self.get_mapping()[required_output]
                try:
                    net.add_outputs(output)
                except RuntimeError:
                    pass
                if output in net.outputs:
                    net_outputs_mapping[output] = required_output
                else:
                    raise ValueError('Failed to identify output "{}"'.format(required_output))
        return net_outputs_mapping

    def rename_outputs(self, outputs):
        return {self.net_outputs_mapping[k]: v for k, v in outputs.items() if k in self.net_outputs_mapping}

    def normalize_inputs(self, inputs):
        if not isinstance(inputs, dict):
            if len(self.net_inputs_mapping) == 1 and not isinstance(inputs, (list, tuple)):
                inputs = [inputs]
            inputs = {k: v for (k, _), v in zip(self.net_inputs_mapping.items(), inputs)}
        inputs = {self.net_inputs_mapping[k]: v for k, v in inputs.items()}
        return inputs

    def __call__(self, inputs):
        inputs = self.normalize_inputs(inputs)
        outputs = self.exec_net.infer(inputs)
        if self.perf_counters:
            perf_counters = self.exec_net.requests[0].get_perf_counts()
            self.perf_counters.update(perf_counters)
        return self.rename_outputs(outputs)

    def __del__(self):
        del self.net
        del self.exec_net

    def print_performance_counters(self):
        if self.perf_counters:
            self.perf_counters.print()

    def show(self, data, result, dataset=None, score_thr=0.3, wait_time=0):
        if self.pt_model is not None:
            self.pt_model.show_result(
                data, result, dataset=dataset, score_thr=score_thr, wait_time=wait_time)


class DetectorOpenVINO(ModelOpenVINO):
    def __init__(self, *args, with_detection_output=False, **kwargs):
        super().__init__(*args,
                         required_inputs=('image', ),
                         required_outputs=('detection_out',) if with_detection_output else ('boxes', 'labels'),
                         **kwargs)

        self.with_detection_output = with_detection_output

        self.n, self.c, self.h, self.w = self.net.inputs['image'].shape
        assert self.n == 1, 'Only batch 1 is supported.'

    def __call__(self, inputs, **kwargs):
        inputs = self.normalize_inputs(inputs)
        output = super().__call__(inputs)
        if self.with_detection_output:
            detection_out = output['detection_out']
            output = {}
            output['labels'] = detection_out[0, 0, :, 1]
            valid_detections_mask = output['labels'] > 0
            output['boxes'] = detection_out[0, 0, :, 3:] * np.tile(inputs['image'].shape[2:][::-1], 2)
            output['boxes'] = np.concatenate((output['boxes'], detection_out[0, 0, :, 2:3]), axis=1)

            output['boxes'] = output['boxes'][valid_detections_mask]
            output['labels'] = output['labels'][valid_detections_mask].astype(np.int32) - 1
        else:
            classes = output['labels']
            valid_detections_mask = classes >= 0
            output['labels'] = classes[valid_detections_mask]
            output['boxes'] = output['boxes'][valid_detections_mask]

        return output

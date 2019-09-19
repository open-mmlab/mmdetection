"""
 Copyright (c) 2019 Intel Corporation

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

import numpy as np
import torch

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


class OpenVINONet(object):

    def __init__(self, xml_file_path, bin_file_path, device='CPU',
                 plugin_dir=None, cpu_extension_lib_path=None, collect_perf_counters=False,
                 cfg=None, classes=None):

        from openvino.inference_engine import IECore, IENetwork

        logging.info('Creating {} plugin...'.format(device))
        self.ie = IECore()
        if cpu_extension_lib_path and 'CPU' in device:
            self.ie.add_extension(cpu_extension_lib_path, 'CPU')

        # Read IR
        logging.info('Reading network from IR...')
        self.net = IENetwork(model=xml_file_path, weights=bin_file_path)

        # self.required_input_keys = {'image'}
        # assert self.required_input_keys == set(self.net.inputs.keys())
        # print(self.net.inputs['image'].shape)
        # required_output_keys = {'boxes', 'labels', 'masks'}
        # if not required_output_keys.issubset(self.net.outputs.keys()):
        #     logging.error('Some of the required outputs {} are not present as actual outputs of the net {}'.format(
        #         list(required_output_keys), list(self.net.outputs.keys())))
        #     raise ValueError

        # self.n, self.c, self.h, self.w = self.net.inputs['image'].shape

        if 'CPU' in device:
            logging.info('Check that all layers are supported...')
            supported_layers = self.ie.query_network(self.net, 'CPU')
            not_supported_layers = [l for l in self.net.layers.keys() if l not in supported_layers]
            if len(not_supported_layers) != 0:
                unsupported_info = '\n\t'.join('{} ({} with params {})'.format(layer_id,
                                                                               self.net.layers[layer_id].type,
                                                                               str(self.net.layers[layer_id].params))
                                               for layer_id in not_supported_layers)
                logging.warning('Following layers are not supported '
                                'by the plugin for specified device {}:'
                                '\n\t{}'.format(self.plugin.device, unsupported_info))
                logging.warning('Please try to specify cpu extensions library path.')
                raise ValueError('Some of the layers are not supported.')

        logging.info('Loading network to plugin...')
        # self.exec_net = self.plugin.load(network=self.net, num_requests=1)
        self.exec_net = self.ie.load_network(network=self.net, device_name=device, num_requests=1)

        self.perf_counters = None
        if collect_perf_counters:
            self.perf_counters = PerformanceCounters()

        if cfg is not None:
            self.pt_model = build_detector(
                cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
            if classes is not None:
                self.pt_model.CLASSES = classes

    def __call__(self, inputs):
        outputs = self.exec_net.infer(inputs)
        if self.perf_counters:
            perf_counters = self.exec_net.requests[0].get_perf_counts()
            self.perf_counters.update(perf_counters)
        return outputs

    def print_performance_counters(self):
        if self.perf_counters:
            self.perf_counters.print()

    def __del__(self):
        del self.net
        del self.exec_net
        del self.ie

    def show(self, data, result, dataset=None, score_thr=0.3, wait_time=0):
        if self.pt_model is not None:
            self.pt_model.show_result(
                data, result, dataset=dataset, score_thr=score_thr, wait_time=wait_time)


class DetectorOpenVINO(OpenVINONet):
    def __init__(self, *args, required_output_keys=('boxes', 'labels'),  **kwargs):
        super().__init__(*args, **kwargs)
        self.required_input_keys = {'image'}
        assert self.required_input_keys == set(self.net.inputs.keys())
        self.output_alias = dict(zip(required_output_keys, ['boxes', 'labels']))
        if not set(required_output_keys).issubset(self.net.outputs.keys()):
            logging.error('Some of the required outputs {} are not present as actual outputs of the net {}'.format(
                list(required_output_keys), list(self.net.outputs.keys())))
            raise ValueError

        self.n, self.c, self.h, self.w = self.net.inputs['image'].shape
        assert self.n == 1, 'Only batch 1 is supported.'

    def __call__(self, inputs, **kwargs):
        if isinstance(inputs, (list, tuple)):
            inputs = dict(zip(self.required_input_keys, inputs))
        # im_data = to_numpy(image[0])
        # if (self.h - im_data.shape[1] < 0) or (self.w - im_data.shape[2] < 0):
        #     raise ValueError('Input image should resolution of {}x{} or less, '
        #                      'got {}x{}.'.format(self.w, self.h, im_data.shape[2], im_data.shape[1]))
        # im_data = np.pad(im_data, ((0, 0),
        #                            (0, self.h - im_data.shape[1]),
        #                            (0, self.w - im_data.shape[2])),
        #                  mode='constant', constant_values=0).reshape(1, self.c, self.h, self.w)
        output = super().__call__(inputs)
        output = {v: output[k] for k, v in self.output_alias.items()}
        classes = output['labels']
        valid_detections_mask = classes >= 0
        output['labels'] = classes[valid_detections_mask]
        output['boxes'] = output['boxes'][valid_detections_mask]
        return output


class MaskRCNNOpenVINO(OpenVINONet):
    def __init__(self, *args, **kwargs):
        super(MaskRCNNOpenVINO, self).__init__(*args, **kwargs)
        self.required_input_keys = {'image'}
        assert self.required_input_keys == set(self.net.inputs.keys())
        required_output_keys = {'boxes', 'labels', 'masks'}
        if not required_output_keys.issubset(self.net.outputs.keys()):
            logging.error('Some of the required outputs {} are not present as actual outputs of the net {}'.format(
                list(required_output_keys), list(self.net.outputs.keys())))
            raise ValueError

        self.n, self.c, self.h, self.w = self.net.inputs['image'].shape
        assert self.n == 1, 'Only batch 1 is supported.'

    def __call__(self, inputs, **kwargs):
        if isinstance(inputs, (list, tuple)):
            inputs = dict(zip(self.required_input_keys, inputs))
        for k, v in inputs.items():
            print(k, v.shape())
        # im_data = to_numpy(image[0])
        # if (self.h - im_data.shape[1] < 0) or (self.w - im_data.shape[2] < 0):
        #     raise ValueError('Input image should resolution of {}x{} or less, '
        #                      'got {}x{}.'.format(self.w, self.h, im_data.shape[2], im_data.shape[1]))
        # im_data = np.pad(im_data, ((0, 0),
        #                            (0, self.h - im_data.shape[1]),
        #                            (0, self.w - im_data.shape[2])),
        #                  mode='constant', constant_values=0).reshape(1, self.c, self.h, self.w)
        output = super().__call__(inputs)

        classes = output['labels']
        valid_detections_mask = classes > 0
        output['labels'] = classes[valid_detections_mask]
        n = len(output['labels'])
        output['boxes'] = output['boxes'][valid_detections_mask]
        output['masks'] = output['masks'][valid_detections_mask][np.arange(n), classes.astype(int).flatten(), ...]
        return output

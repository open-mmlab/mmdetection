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
import tempfile
import types

import torch.nn as nn
from pytorchcv.model_provider import _models
from torch.nn.modules.batchnorm import _BatchNorm

from ..registry import BACKBONES

from mmcv.runner import get_dist_info


def generate_backbones():
    logger = logging.getLogger()

    for model_name, model_getter in _models.items():

        def closure(model_name, model_getter):

            def multioutput_forward(self, x):
                outputs = []
                y = x

                last_stage = max(self.out_indices)
                for i, stage in enumerate(self.features):
                    y = stage(y)
                    s = str(i) + ' ' + str(y.shape)
                    if i in self.out_indices:
                        outputs.append(y)
                        s += '*'
                    if self.verbose:
                        print(s)
                    if i == last_stage:
                        break

                return outputs

            def init_weights(self, pretrained=True):
                pass

            def train(self, mode=True):
                super(self.__class__, self).train(mode)

                for i in range(self.frozen_stages + 1):
                    m = self.features[i]
                    m.eval()
                    for param in m.parameters():
                        param.requires_grad = False

                if mode and self.norm_eval:
                    for m in self.modules():
                        # trick: eval have effect on BatchNorm only
                        if isinstance(m, _BatchNorm):
                            m.eval()

            class custom_model_getter(nn.Module):
                def __init__(self, *args, out_indices=None, frozen_stages=0, norm_eval=False, verbose=False, **kwargs):
                    super().__init__()
                    if 'pretrained' in kwargs and kwargs['pretrained']:
                        rank, _ = get_dist_info()
                        if rank > 0:
                            if 'root' not in kwargs:
                                kwargs['root'] = tempfile.mkdtemp()
                            kwargs['root'] = tempfile.mkdtemp(dir=kwargs['root'])
                            logger.info('Rank: {}, Setting {} as a target location of pretrained models'.format(rank, kwargs['root']))
                    model = model_getter(*args, **kwargs)
                    model.out_indices = out_indices
                    model.frozen_stages = frozen_stages
                    model.norm_eval = norm_eval
                    model.verbose = verbose
                    if hasattr(model, 'features') and isinstance(model.features, nn.Sequential):
                        # Save original forward, just in case.
                        model.forward_single_output = model.forward
                        model.forward = types.MethodType(multioutput_forward, model)
                        model.init_weights = types.MethodType(init_weights, model)
                        model.train = types.MethodType(train, model)
                    else:
                        raise ValueError('Failed to automatically wrap backbone network. '
                                         'Object of type {} has no valid attribute called '
                                         '"features".'.format(model.__class__))
                    self.__dict__.update(model.__dict__)

            custom_model_getter.__name__ = model_name
            return custom_model_getter

        BACKBONES.register_module(closure(model_name, model_getter))


generate_backbones()

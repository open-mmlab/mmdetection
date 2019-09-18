import logging
import os.path as osp
import tempfile
import types

import torch.nn as nn
from pytorchcv.model_provider import _models
from torch.nn.modules.batchnorm import _BatchNorm

from ..registry import BACKBONES


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
                    if i in self.out_indices:
                        outputs.append(y)
                    elif i == last_stage:
                        break

                # print('*' * 20)
                # print(x.shape)
                # for y in outputs:
                #     print(y.shape)
                # print('-' * 20)

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

            def custom_model_getter(*args, out_indices=None, frozen_stages=0, norm_eval=False, **kwargs):
                if 'pretrained' in kwargs and kwargs['pretrained'] and 'root' in kwargs:
                    path = kwargs['root']
                    if not osp.exists(path):
                        logger.warning('{} does not exist, using standard location of pretrained models.'.format(path))
                        del kwargs['root']
                    else:
                        kwargs['root'] = tempfile.mkdtemp(dir=path)
                        logger.info('Setting {} as a target location of pretrained models'.format(kwargs['root']))
                model = model_getter(*args, **kwargs)
                model.out_indices = out_indices
                model.frozen_stages = frozen_stages
                model.norm_eval = norm_eval
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
                return model

            custom_model_getter.__name__ = model_name
            return custom_model_getter

        BACKBONES.register_module(closure(model_name, model_getter))


generate_backbones()

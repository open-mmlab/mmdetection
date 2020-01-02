import json
import os

import torch
from mmcv.cnn import normal_init
from torch import nn

from ..registry import HEADS
from ..utils import bias_init_with_prob
from . import *


@HEADS.register_function
def SampleFreeHead(head, model_key, init_prior_file, loss_cls_scale, **kwargs):
    class SFHead(eval(head)):
        def __init__(self, **kwargs):
            super(SFHead, self).__init__(**kwargs)
            self.prior = load_prior(model_key, init_prior_file)
            self.guided_loss = GuidedLoss(loss_cls_scale)
            self.sampling = False
            # when use down-sample, the avg_factor of bbox loss in rpn stage
            # is num_total_pos + num_total_neg, however if we use sample-free,
            # the avg_factor is num_total_pos, the bbox loss will be larger,
            # so we need to rescale the loss, here we set
            # loss_bbox_avg_factor = 8, the value refer to the Official Repo,
            # https://github.com/ChenJoya/sampling-free/blob/master/maskrcnn_benchmark/modeling/rpn/loss.py#L232
            self.loss_bbox_avg_factor = 8.0 if head == 'RPNHead' else 1.0

        def init_weights(self):
            super().init_weights()
            conv_cls_attr = [attr for attr in dir(self) if 'cls' in attr
                             and isinstance(getattr(self, attr), nn.Conv2d)]
            assert len(conv_cls_attr) == 1
            bias_cls = bias_init_with_prob(self.prior)
            normal_init(
                getattr(self, conv_cls_attr[0]), std=0.01, bias=bias_cls)

        def loss(self, *args, **kwargs):
            loss_dict = super().loss(*args, **kwargs)
            loss_cls_key = [key for key in loss_dict.keys() if 'cls' in key][0]
            loss_bbox_key = [
                key for key in loss_dict.keys() if 'bbox' in key][0]
            loss_cls = loss_dict[loss_cls_key]
            loss_bbox = loss_dict[loss_bbox_key]
            loss_bbox = list(
                map(lambda x: x / self.loss_bbox_avg_factor, loss_bbox))

            loss_cls = self.guided_loss(loss_bbox, loss_cls)
            loss_dict[loss_cls_key] = loss_cls
            loss_dict[loss_bbox_key] = loss_bbox
            return loss_dict

    return SFHead(**kwargs)


# https://github.com/ChenJoya/sampling-free/blob/master/maskrcnn_benchmark/modeling/sampling_free.py
class GuidedLoss(object):
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, loss_bbox, loss_cls):
        if not isinstance(loss_bbox, list):
            loss_bbox, loss_cls = [loss_bbox], [loss_cls]

        loss_bbox_all_levels = torch.tensor(loss_bbox).sum()
        loss_cls_all_levels = torch.tensor(loss_cls).sum()
        r = loss_bbox_all_levels / loss_cls_all_levels

        loss_cls = list(map(lambda x: x * r * self.scale, loss_cls))
        return loss_cls


def load_prior(model_key, filename):
    if not os.path.exists(filename):
        msg = '{} is not existed, please calculate it first'.format(
               filename)
        raise FileNotFoundError(msg)

    with open(filename, 'r') as f:
        model_prior_dict = json.load(f)
        if model_key in model_prior_dict:
            print('Find it. Use it to initialize the model.')
            return model_prior_dict[model_key]
        else:
            msg = '{} is not existed, please calculate it frist'.format(
               model_key)
            raise KeyError(msg)

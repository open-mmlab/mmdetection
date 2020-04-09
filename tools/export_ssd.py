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

# pylint: disable=all
import argparse

import numpy as np
import torch
from mmcv.parallel import MMDataParallel

from mmdet.apis import init_detector
from mmdet.models import detectors
from mmdet.utils.deployment.ssd_export_helpers import (get_proposals, PriorBox,
                                                       PriorBoxClustered, DetectionOutput)


def onnx_export(self, img, img_meta, export_name='', **kwargs):
    self._export_mode = True
    self.img_metas = img_meta
    torch.onnx.export(self, img, export_name, verbose=False, **kwargs)


def forward(self, img, img_meta=[None], return_loss=True, **kwargs): #passing None here is a hack to fool the jit engine
    if self._export_mode:
        return self.forward_export(img)
    if return_loss:
        return self.forward_train(img, img_meta, **kwargs)
    else:
        return self.forward_test(img, img_meta, **kwargs)


def forward_export_detector(self, img):
    x = self.extract_feat(img)
    outs = self.bbox_head(x)
    bbox_result = self.bbox_head.export_forward(*outs, self.test_cfg, True,
                                                self.img_metas, x, img)
    return bbox_result


def export_forward_ssd_head(self, cls_scores, bbox_preds, cfg, rescale,
                            img_metas, feats, img_tensor):
    num_levels = len(cls_scores)

    anchors = []
    for i in range(num_levels):
        if self.anchor_generators[i].clustered:
            anchors.append(PriorBoxClustered.apply(
                self.anchor_generators[i], self.anchor_strides[i],
                feats[i], img_tensor, self.target_stds))
        else:
            anchors.append(PriorBox.apply(self.anchor_generators[i],
                                          self.anchor_strides[i],
                                          feats[i],
                                          img_tensor, self.target_stds))
    anchors = torch.cat(anchors, 2)
    cls_scores, bbox_preds = self._prepare_cls_scores_bbox_preds(cls_scores, bbox_preds)

    return DetectionOutput.apply(cls_scores, bbox_preds, img_metas, cfg,
                                 rescale, anchors, self.cls_out_channels,
                                 self.use_sigmoid_cls, self.target_means,
                                 self.target_stds)


def prepare_cls_scores_bbox_preds_ssd_head(self, cls_scores, bbox_preds):
    scores_list = []
    for o in cls_scores:
        score = o.permute(0, 2, 3, 1).contiguous().view(o.size(0), -1)
        scores_list.append(score)
    cls_scores = torch.cat(scores_list, 1)
    cls_scores = cls_scores.view(cls_scores.size(0), -1, self.num_classes)
    if self.use_sigmoid_cls:
        cls_scores = cls_scores.sigmoid()
    else:
        cls_scores = cls_scores.softmax(-1)
    cls_scores = cls_scores.view(cls_scores.size(0), -1)
    bbox_list = []
    for o in bbox_preds:
        boxes = o.permute(0, 2, 3, 1).contiguous().view(o.size(0), -1)
        bbox_list.append(boxes)
    bbox_preds = torch.cat(bbox_list, 1)
    return cls_scores, bbox_preds


def get_bboxes_ssd_head(self, cls_scores, bbox_preds, img_metas, cfg, rescale=False):
    assert len(cls_scores) == len(bbox_preds)
    num_levels = len(cls_scores)
    mlvl_anchors = [
        self.anchor_generators[i].grid_anchors(cls_scores[i].size()[-2:],
                                               self.anchor_strides[i])
        for i in range(num_levels)
    ]
    mlvl_anchors = torch.cat(mlvl_anchors, 0)
    cls_scores, bbox_preds = self._prepare_cls_scores_bbox_preds(
        cls_scores, bbox_preds)
    bboxes_list = get_proposals(img_metas, cls_scores, bbox_preds,
                                mlvl_anchors, cfg, rescale,
                                self.cls_out_channels,
                                self.use_sigmoid_cls, self.target_means,
                                self.target_stds)


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet onnx exporter for SSD detector')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('output', help='onnx file')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    model = init_detector(args.config, args.checkpoint)
    cfg = model.cfg
    assert getattr(detectors, cfg.model['type']) is detectors.SingleStageDetector
    model = MMDataParallel(model, device_ids=[0])

    batch = torch.FloatTensor(1, 3, cfg.input_size, cfg.input_size).cuda()
    input_shape = (cfg.input_size, cfg.input_size, 3)
    scale = np.array([1, 1, 1, 1], dtype=np.float32)
    data = dict(img=batch, img_meta=[{'img_shape': input_shape, 'scale_factor': scale}])

    model.eval()
    model.module.onnx_export = onnx_export.__get__(model.module)
    model.module.forward = forward.__get__(model.module)
    model.module.forward_export = forward_export_detector.__get__(model.module)
    model.module.bbox_head.export_forward = export_forward_ssd_head.__get__(model.module.bbox_head)
    model.module.bbox_head._prepare_cls_scores_bbox_preds = prepare_cls_scores_bbox_preds_ssd_head.__get__(model.module.bbox_head)
    model.module.bbox_head.get_bboxes = get_bboxes_ssd_head.__get__(model.module.bbox_head)
    model.module.onnx_export(export_name=args.output, input_names=['image'],
                             output_names=['detection_out'], **data)

if __name__ == '__main__':
    main()

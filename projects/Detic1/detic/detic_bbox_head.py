# Copyright (c) OpenMMLab. All rights reserved.
import json
from typing import Optional, Union

import torch
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.models.layers import multiclass_nms
from mmdet.models.roi_heads.bbox_heads import Shared2FCBBoxHead
from mmdet.models.utils import empty_instances
from mmdet.registry import MODELS
from mmdet.structures.bbox import get_box_tensor, scale_boxes


def load_class_freq(path='datasets/metadata/lvis_v1_train_cat_info.json',
                    freq_weight=0.5):
    cat_info = json.load(open(path, 'r'))
    cat_info = torch.tensor(
        [c['image_count'] for c in sorted(cat_info, key=lambda x: x['id'])])
    freq_weight = cat_info.float()**freq_weight
    return freq_weight


def get_fed_loss_inds(gt_classes, num_sample_cats, C, weight=None):
    appeared = torch.unique(gt_classes)  # C'
    prob = appeared.new_ones(C + 1).float()
    prob[-1] = 0
    if len(appeared) < num_sample_cats:
        if weight is not None:
            prob[:C] = weight.float().clone()
        prob[appeared] = 0
        more_appeared = torch.multinomial(
            prob, num_sample_cats - len(appeared), replacement=False)
        appeared = torch.cat([appeared, more_appeared])
    return appeared


@MODELS.register_module()
class DeticBBoxHead(Shared2FCBBoxHead):

    def __init__(self,
                 image_loss_weight: float = 0.1,
                 use_fed_loss: bool = False,
                 cat_freq_path: str = '',
                 fed_loss_freq_weight: float = 0.5,
                 fed_loss_num_cat: int = 50,
                 *args,
                 init_cfg: Optional[Union[dict, ConfigDict]] = None,
                 **kwargs) -> None:
        super().__init__(*args, init_cfg=init_cfg, **kwargs)
        # reconstruct fc_cls and fc_reg since input channels are changed
        assert self.with_cls
        cls_channels = self.num_classes
        cls_predictor_cfg_ = self.cls_predictor_cfg.copy()
        cls_predictor_cfg_.update(
            in_features=self.cls_last_dim, out_features=cls_channels)
        self.fc_cls = MODELS.build(cls_predictor_cfg_)

        self.image_loss_weight = image_loss_weight
        self.use_fed_loss = use_fed_loss
        self.cat_freq_path = cat_freq_path
        self.fed_loss_freq_weight = fed_loss_freq_weight
        self.fed_loss_num_cat = fed_loss_num_cat

        if self.use_fed_loss:
            freq_weight = load_class_freq(cat_freq_path, fed_loss_freq_weight)
            self.register_buffer('freq_weight', freq_weight)
        else:
            self.freq_weight = None

    def _predict_by_feat_single(
            self,
            roi: Tensor,
            cls_score: Tensor,
            bbox_pred: Tensor,
            img_meta: dict,
            rescale: bool = False,
            rcnn_test_cfg: Optional[ConfigDict] = None) -> InstanceData:
        """Transform a single image's features extracted from the head into
        bbox results.

        Args:
            roi (Tensor): Boxes to be transformed. Has shape (num_boxes, 5).
                last dimension 5 arrange as (batch_index, x1, y1, x2, y2).
            cls_score (Tensor): Box scores, has shape
                (num_boxes, num_classes + 1).
            bbox_pred (Tensor): Box energies / deltas.
                has shape (num_boxes, num_classes * 4).
            img_meta (dict): image information.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of Bbox Head.
                Defaults to None

        Returns:
            :obj:`InstanceData`: Detection results of each image\
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        results = InstanceData()
        if roi.shape[0] == 0:
            return empty_instances([img_meta],
                                   roi.device,
                                   task_type='bbox',
                                   instance_results=[results],
                                   box_type=self.predict_box_type,
                                   use_box_type=False,
                                   num_classes=self.num_classes,
                                   score_per_cls=rcnn_test_cfg is None)[0]
        scores = cls_score
        img_shape = img_meta['img_shape']
        num_rois = roi.size(0)

        num_classes = 1 if self.reg_class_agnostic else self.num_classes
        roi = roi.repeat_interleave(num_classes, dim=0)
        bbox_pred = bbox_pred.view(-1, self.bbox_coder.encode_size)
        bboxes = self.bbox_coder.decode(
            roi[..., 1:], bbox_pred, max_shape=img_shape)

        if rescale and bboxes.size(0) > 0:
            assert img_meta.get('scale_factor') is not None
            scale_factor = [1 / s for s in img_meta['scale_factor']]
            bboxes = scale_boxes(bboxes, scale_factor)

        # Get the inside tensor when `bboxes` is a box type
        bboxes = get_box_tensor(bboxes)
        box_dim = bboxes.size(-1)
        bboxes = bboxes.view(num_rois, -1)

        if rcnn_test_cfg is None:
            # This means that it is aug test.
            # It needs to return the raw results without nms.
            results.bboxes = bboxes
            results.scores = scores
        else:
            det_bboxes, det_labels = multiclass_nms(
                bboxes,
                scores,
                rcnn_test_cfg.score_thr,
                rcnn_test_cfg.nms,
                rcnn_test_cfg.max_per_img,
                box_dim=box_dim)
            results.bboxes = det_bboxes[:, :-1]
            results.scores = det_bboxes[:, -1]
            results.labels = det_labels
        return results

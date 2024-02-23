# Copyright (c) OpenMMLab. All rights reserved.

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.registry import MODELS
from mmdet.structures.bbox import bbox_xyxy_to_cxcywh
from mmdet.utils import OptConfigType
from .loss import SetCriterion
from .maskdino_decoder_layers import MaskDINODecoder
from .maskdino_encoder_layers import MaskDINOEncoder


def get_bounding_boxes(mask):
    """
    Returns:
        Boxes: tight bounding boxes around bitmasks.
        If a mask is empty, it's bounding box will be all zero.
    """
    boxes = torch.zeros(
        mask.shape[0], 4, dtype=torch.float32, device=mask.device)
    x_any = torch.any(mask, dim=1)
    y_any = torch.any(mask, dim=2)
    for idx in range(mask.shape[0]):
        x = torch.where(x_any[idx, :])[0]
        y = torch.where(y_any[idx, :])[0]
        if len(x) > 0 and len(y) > 0:
            boxes[idx, :] = torch.as_tensor([x[0], y[0], x[-1] + 1, y[-1] + 1],
                                            dtype=torch.float32)
    return boxes


@MODELS.register_module()
class MaskDINOHead(nn.Module):

    def __init__(
        self,
        num_stuff_classes: int,
        num_things_classes: int,
        encoder: OptConfigType,
        decoder: OptConfigType,
        train_cfg: OptConfigType = None,
        test_cfg: OptConfigType = None,
    ):
        super().__init__()
        self.num_stuff_classes = num_stuff_classes
        self.num_things_classes = num_things_classes
        self.num_classes = num_stuff_classes + num_things_classes
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.pixel_decoder = MaskDINOEncoder(**encoder)
        self.predictor = MaskDINODecoder(**decoder)
        self.criterion = SetCriterion(**train_cfg)

    def forward(self, features, mask=None, targets=None):
        mask_features, transformer_encoder_features, multi_scale_features = \
            self.pixel_decoder.forward_features(features, mask)

        predictions = self.predictor(
            multi_scale_features, mask_features, mask, targets=targets)
        return predictions

    def loss(self, feats, batch_data_samples):
        targets = self.prepare_targets(batch_data_samples)
        outputs, mask_dict = self(
            feats, mask=None,
            targets=targets)  # TODO: deal with key_padding_masks ?
        # bipartite matching-based loss
        losses = self.criterion(outputs, targets, mask_dict)

        for k in list(losses.keys()):
            if k in self.criterion.weight_dict:
                losses[k] *= self.criterion.weight_dict[k]
            else:
                # remove this loss if not specified in `weight_dict`
                losses.pop(k)

        return losses

    def predict(self, feats, batch_data_samples):
        outputs, _ = self(feats)
        mask_cls_results = outputs['pred_logits']
        mask_pred_results = outputs['pred_masks']
        mask_box_results = outputs['pred_boxes']

        # upsample masks
        batch_input_shape = batch_data_samples[0].metainfo['batch_input_shape']
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(batch_input_shape[0], batch_input_shape[1]),
            mode='bilinear',
            align_corners=False)

        return mask_cls_results, mask_pred_results, mask_box_results

    def prepare_targets(self, batch_data_samples):
        h_pad, w_pad = batch_data_samples[
            0].batch_input_shape  # TODO: make a check
        # h, w = data_sample.img_shape[:2]
        h, w = h_pad, w_pad
        new_targets = []
        for data_sample in batch_data_samples:
            if 'bboxes' in data_sample.gt_instances:
                # pad gt
                device = data_sample.gt_instances.bboxes.device
                gt_masks = torch.from_numpy(
                    data_sample.gt_instances.masks.masks).bool().to(device)
                gt_labels = data_sample.gt_instances.labels
                gt_bboxes = data_sample.gt_instances.bboxes

            if 'gt_sem_seg' in data_sample:
                gt_semantic_seg = data_sample.gt_sem_seg.sem_seg
                gt_semantic_seg = gt_semantic_seg.squeeze(0)
                device = gt_semantic_seg.device
                semantic_labels = torch.unique(
                    gt_semantic_seg,
                    sorted=False,
                    return_inverse=False,
                    return_counts=False)
                stuff_masks_list = []
                stuff_labels_list = []
                for label in semantic_labels:
                    if label < self.num_things_classes or label >= self.num_classes:
                        continue
                    stuff_mask = gt_semantic_seg == label
                    stuff_masks_list.append(stuff_mask)
                    stuff_labels_list.append(label)
                if len(stuff_masks_list) > 0:
                    stuff_masks = torch.stack(stuff_masks_list, dim=0)
                    stuff_bboxes = get_bounding_boxes(stuff_masks)
                    stuff_labels = torch.stack(stuff_labels_list, dim=0)

                    if 'bboxes' in data_sample.gt_instances:
                        gt_labels = torch.cat([gt_labels, stuff_labels], dim=0)
                        gt_masks = torch.cat([gt_masks, stuff_masks], dim=0)
                        gt_bboxes = torch.cat([gt_bboxes, stuff_bboxes], dim=0)
                    else:
                        gt_labels = stuff_labels.long()
                        gt_masks = stuff_masks
                        gt_bboxes = stuff_bboxes
                else:
                    gt_masks = torch.zeros((0, h_pad, w_pad),
                                           dtype=torch.bool,
                                           device=device)
                    gt_bboxes = torch.zeros((0, 4),
                                            dtype=torch.float,
                                            device=device)
                    gt_labels = torch.zeros((0, ),
                                            dtype=torch.long,
                                            device=device)

            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad),
                                       dtype=gt_masks.dtype,
                                       device=device)
            padded_masks[:, :gt_masks.shape[1], :gt_masks.shape[2]] = gt_masks
            image_size_xyxy = torch.as_tensor([w, h, w, h],
                                              dtype=torch.float,
                                              device=device)

            new_targets.append({
                'labels':
                gt_labels,
                'masks':
                padded_masks,
                'boxes':
                bbox_xyxy_to_cxcywh(gt_bboxes) / image_size_xyxy
            })

        return new_targets

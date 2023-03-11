# Copyright (c) OpenMMLab. All rights reserved.
import copy
import warnings
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d
from mmcv.ops import point_sample
from mmengine.model import ModuleList, caffe2_xavier_init
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures import SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig, reduce_mean
from mmdet.structures.bbox import bbox_xyxy_to_cxcywh, bbox_cxcywh_to_xyxy
from .maskdino_encoder_layers import MaskDINOEncoder
from .maskdino_decoder_layers import MaskDINODecoder
from .criterion import SetCriterion

from mmdet.utils.memory import AvoidCUDAOOM
from mmengine.structures import InstanceData, PixelData
# from ..layers import Mask2FormerTransformerDecoder, SinePositionalEncoding
# from ..utils import get_uncertain_point_coords_with_randomness
# from .anchor_free_head import AnchorFreeHead
# from .maskformer_head import MaskFormerHead


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

        predictions = self.predictor(multi_scale_features, mask_features,
                                     mask, targets=targets)
        return predictions

    def loss(self, feats, batch_data_samples):
        targets = self.prepare_targets(batch_data_samples)
        outputs, mask_dict = self(feats, mask=None, targets=targets)  # TODO: deal with key_padding_masks ?
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
        outputs, mask_dict = self(feats)
        mask_cls_results = outputs["pred_logits"]
        mask_pred_results = outputs["pred_masks"]
        mask_box_results = outputs["pred_boxes"]

        # upsample masks
        batch_input_shape = batch_data_samples[0].metainfo['batch_input_shape']
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(batch_input_shape[0], batch_input_shape[1]),
            mode='bilinear',
            align_corners=False)

        return mask_cls_results, mask_pred_results, mask_box_results

    def prepare_targets(self, batch_data_samples):
        # h_pad, w_pad = images.tensor.shape[-2:]  # TODO: Here is confusing
        h_pad, w_pad = batch_data_samples[0].batch_input_shape  # TODO: make a check
        new_targets = []
        for data_sample in batch_data_samples:
            # pad gt
            device = data_sample.gt_instances.bboxes.device
            h, w, _ = data_sample.img_shape
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=device)

            gt_masks = torch.from_numpy(data_sample.gt_instances.masks.masks).bool().to(device)
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "labels": data_sample.gt_instances.labels,
                    "masks": padded_masks,
                    "boxes": bbox_xyxy_to_cxcywh(data_sample.gt_instances.bboxes) / image_size_xyxy
                }
            )

            warnings.warn(  # TODO: align the lsj pipeline
                'The lsj for MaskDINO and Mask2Former has not been fully aligned '
                'with COCOPanopticNewBaselineDatasetMapper in original repo')

        return new_targets

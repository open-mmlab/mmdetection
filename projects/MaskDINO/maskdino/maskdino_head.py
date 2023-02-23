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
from mmdet.structures.bbox import bbox_xyxy_to_cxcywh
from .maskdino_encoder_layers import MaskDINOEncoder
from .maskdino_decoder_layers import MaskDINODecoder
from .criterion import SetCriterion
# from ..layers import Mask2FormerTransformerDecoder, SinePositionalEncoding
# from ..utils import get_uncertain_point_coords_with_randomness
# from .anchor_free_head import AnchorFreeHead
# from .maskformer_head import MaskFormerHead


@MODELS.register_module()
class MaskDINOHead(nn.Module):

    def __init__(
        self,
        # input_shape: Dict[str, ShapeSpec],
        num_classes: int,
        encoder: OptConfigType,
        decoder: OptConfigType,
        loss_weight: float = 1.0,
        ignore_value: int = -1,
        *
        # detector
        criterion: OptConfigType,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        # metadata,  # TODO: what to replace this?
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,

        # TODO: move to preprocessor ?
        # pixel_mean: Tuple[float],
        # pixel_std: Tuple[float],

        # # inference  # move to test_cfg for mmdet style
        # semantic_on: bool,
        # panoptic_on: bool,
        # instance_on: bool,
        # test_topk_per_image: int,
        # data_loader: str,
        # pano_temp: float,
        # focus_on_box: bool = False,
        # transform_eval: bool = False,
        # semantic_ce_loss: bool = False,

        # train_cfg and test_cfg for mmdet style
        train_cfg: OptConfigType = None,
        test_cfg: OptConfigType = None,
    ):
        """
        Args:
            input_shape: shapes (channels and stride) of the input features
            num_classes: number of classes to predict
            pixel_decoder: the pixel decoder module
            loss_weight: loss weight
            ignore_value: category id to be ignored during training.
            transformer_predictor: the transformer decoder that makes prediction
            transformer_in_feature: input feature name to the transformer_predictor
        """
        super().__init__()
        # input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        # self.in_features = [k for k, v in input_shape]  # useless
        self.ignore_value = ignore_value
        self.common_stride = 4
        self.loss_weight = loss_weight

        self.criterion = SetCriterion(**train_cfg)

        self.pixel_decoder = MaskDINOEncoder(**encoder)
        self.predictor = MaskDINODecoder(**decoder)

        self.num_classes = num_classes

    def forward(self, features, mask=None, targets=None):
        return self.layers(features, mask, targets=targets)

    def layers(self, features, mask=None, targets=None):
        mask_features, transformer_encoder_features, multi_scale_features = \
            self.pixel_decoder.forward_features(features, mask)

        predictions = self.predictor(multi_scale_features, mask_features,
                                     mask, targets=targets)
        return predictions

    def loss(self, feats, batch_data_samples):
        targets = self.prepare_targets(batch_data_samples)
        outputs, mask_dict = self(feats, mask=None, targets=targets)  # TODO: deal with masks
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
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(images.tensor.shape[-2], images.tensor.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )

        del outputs

        processed_results = []
        for mask_cls_result, mask_pred_result, mask_box_result, input_per_image, image_size in zip(
                mask_cls_results, mask_pred_results, mask_box_results,
                batched_inputs, images.image_sizes
        ):  # image_size is augmented size, not divisible to 32
            height = input_per_image.get("height", image_size[0])  # real size
            width = input_per_image.get("width", image_size[1])
            processed_results.append({})
            new_size = mask_pred_result.shape[
                       -2:]  # padded size (divisible to 32)

            if self.sem_seg_postprocess_before_inference:
                mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                    mask_pred_result, image_size, height, width
                )
                mask_cls_result = mask_cls_result.to(mask_pred_result)
                # mask_box_result = mask_box_result.to(mask_pred_result)
                # mask_box_result = self.box_postprocess(mask_box_result, height, width)

            # semantic segmentation inference
            if self.semantic_on:
                r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result,
                                                               mask_pred_result)
                if not self.sem_seg_postprocess_before_inference:
                    r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size,
                                                               height, width)
                processed_results[-1]["sem_seg"] = r

            # panoptic segmentation inference
            if self.panoptic_on:
                panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(
                    mask_cls_result, mask_pred_result)
                processed_results[-1]["panoptic_seg"] = panoptic_r

            # instance segmentation inference
            if self.instance_on:
                mask_box_result = mask_box_result.to(mask_pred_result)
                height = new_size[0] / image_size[0] * height
                width = new_size[1] / image_size[1] * width
                mask_box_result = self.box_postprocess(mask_box_result, height,
                                                       width)

                instance_r = retry_if_cuda_oom(self.instance_inference)(
                    mask_cls_result, mask_pred_result, mask_box_result)
                processed_results[-1]["instances"] = instance_r

        return processed_results

    def instance_inference(self, mask_cls, mask_pred, mask_box_result):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]
        scores = mask_cls.sigmoid()  # [100, 80]
        labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)  # select 100
        labels_per_image = labels[topk_indices]
        topk_indices = topk_indices // self.sem_seg_head.num_classes
        mask_pred = mask_pred[topk_indices]
        # if this is panoptic segmentation, we only keep the "thing" classes
        if self.panoptic_on:
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                keep[i] = lab in self.metadata.thing_dataset_id_to_contiguous_id.values()
            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]
        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = (mask_pred > 0).float()
        # half mask box half pred box
        mask_box_result = mask_box_result[topk_indices]
        if self.panoptic_on:
            mask_box_result = mask_box_result[keep]
        result.pred_boxes = Boxes(mask_box_result)
        # Uncomment the following to get boxes from masks (this is slow)
        # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

        # calculate average mask prob
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        if self.focus_on_box:
            mask_scores_per_image = 1.0
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image
        return result

    def box_postprocess(self, out_bbox, img_h, img_w):
        # postprocess box height and width
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        scale_fct = torch.tensor([img_w, img_h, img_w, img_h])
        scale_fct = scale_fct.to(out_bbox)
        boxes = boxes * scale_fct
        return boxes

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

            warnings.warn(  # TODO: aligh the lsj pipeline
                'The lsj for MaskDINO and Mask2Former has not been fully aligned '
                'with COCOPanopticNewBaselineDatasetMapper in original repo')

        return new_targets
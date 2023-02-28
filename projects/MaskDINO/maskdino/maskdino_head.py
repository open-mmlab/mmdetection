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
        # input_shape: Dict[str, ShapeSpec],
        num_classes: int,
        num_things_classes: int,
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
        self.num_things_classes = num_things_classes

        self.num_queries = num_queries
        self.object_mask_threshold = object_mask_threshold
        self.overlap_threshold = overlap_threshold
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

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
            # size=(images.tensor.shape[-2], images.tensor.shape[-1]),
            size=batch_data_samples[0].batch_input_shape,  # TODO: make a check
            mode="bilinear",
            align_corners=False,
        )

        del outputs

        processed_results = []
        for mask_cls_result, mask_pred_result, mask_box_result, data_sample in zip(
                mask_cls_results, mask_pred_results, mask_box_results, batch_data_samples
        ):  # image_size is augmented size, not divisible to 32
            image_size = data_sample.ori_shape  # TODO: check this, img_shape raise Error in evaluator
            height = image_size[0]  # real size
            width = image_size[1]
            processed_results.append({})
            new_size = mask_pred_result.shape[-2:]  # padded size (divisible to 32)

            if self.sem_seg_postprocess_before_inference:
                mask_pred_result = self.sem_seg_postprocess(
                    mask_pred_result, image_size, height, width
                )  # TODO: make a check
                mask_cls_result = mask_cls_result.to(mask_pred_result)
                # mask_box_result = mask_box_result.to(mask_pred_result)
                # mask_box_result = self.box_postprocess(mask_box_result, height, width)

            # semantic segmentation inference
            if self.test_cfg.semantic_on:
                raise NotImplementedError

            # panoptic segmentation inference
            if self.test_cfg.panoptic_on:
                panoptic_r, _ = self.panoptic_inference(mask_cls_result, mask_pred_result)
                processed_results[-1]["pan_results"] = panoptic_r

            # instance segmentation inference
            if self.test_cfg.instance_on:
                mask_box_result = mask_box_result.to(mask_pred_result)
                height = new_size[0] / image_size[0] * height
                width = new_size[1] / image_size[1] * width
                mask_box_result = self.box_postprocess(mask_box_result, height,
                                                       width)

                instance_r = self.instance_inference(
                    mask_cls_result, mask_pred_result, mask_box_result)
                processed_results[-1]["ins_results"] = instance_r

        results = self.add_pred_to_datasample(batch_data_samples, processed_results)

        return results

    def semantic_inference(self, mask_cls, mask_pred):
        # if use cross-entropy loss in training, evaluate with softmax
        if self.semantic_ce_loss:
            mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
            mask_pred = mask_pred.sigmoid()
            semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
            return semseg
        # if use focal loss in training, evaluate with sigmoid. As sigmoid is mainly for detection and not sharp
        # enough for semantic and panoptic segmentation, we additionally use use softmax with a temperature to
        # make the score sharper.
        else:
            T = self.test_cfg.pano_temp
            mask_cls = mask_cls.sigmoid()
            if self.test_cfg.transform_eval:
                mask_cls = F.softmax(mask_cls / T, dim=-1)  # already sigmoid
            mask_pred = mask_pred.sigmoid()
            semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
            return semseg

    @AvoidCUDAOOM.retry_if_cuda_oom
    def panoptic_inference(self, mask_cls, mask_pred):
        # As we use focal loss in training, evaluate with sigmoid. As sigmoid is mainly for detection and not sharp
        # enough for semantic and panoptic segmentation, we additionally use use softmax with a temperature to
        # make the score sharper.
        prob = 0.5
        T = self.test_cfg.pano_temp
        scores, labels = mask_cls.sigmoid().max(-1)
        mask_pred = mask_pred.sigmoid()
        keep = labels.ne(self.num_classes) & (scores > self.object_mask_threshold)
        # added process
        if self.test_cfg.transform_eval:
            scores, labels = F.softmax(mask_cls.sigmoid() / T, dim=-1).max(-1)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return PixelData(sem_seg=panoptic_seg[None]), segments_info
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class < self.num_things_classes
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= prob).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= prob)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue

                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )

            return PixelData(sem_seg=panoptic_seg[None]), segments_info

    @AvoidCUDAOOM.retry_if_cuda_oom
    def instance_inference(self, mask_cls, mask_pred, mask_box_result):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]
        scores = mask_cls.sigmoid()  # [100, 80]
        labels = torch.arange(self.num_classes, device=mask_cls.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_cfg.test_topk_per_image, sorted=False)  # select 100
        labels_per_image = labels[topk_indices]
        topk_indices = topk_indices // self.num_classes
        mask_pred = mask_pred[topk_indices]
        # if this is panoptic segmentation, we only keep the "thing" classes
        if self.test_cfg.panoptic_on:
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                # keep[i] = lab in self.metadata.thing_dataset_id_to_contiguous_id.values()
                keep[i] = lab < self.num_things_classes
            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]

        result = InstanceData()
        # mask (before sigmoid)
        result.masks = (mask_pred > 0).float()
        # half mask box half pred box
        mask_box_result = mask_box_result[topk_indices]
        if self.test_cfg.panoptic_on:
            mask_box_result = mask_box_result[keep]
        result.bboxes = mask_box_result
        # Uncomment the following to get boxes from masks (this is slow)
        # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

        # calculate average mask prob
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.masks.flatten(1)).sum(1) / (result.masks.flatten(1).sum(1) + 1e-6)
        if self.test_cfg.focus_on_box:
            mask_scores_per_image = 1.0
        result.scores = scores_per_image * mask_scores_per_image
        result.labels = labels_per_image
        return result

    def box_postprocess(self, out_bbox, img_h, img_w):
        # postprocess box height and width
        boxes = bbox_cxcywh_to_xyxy(out_bbox)
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

    @staticmethod
    @AvoidCUDAOOM.retry_if_cuda_oom
    def sem_seg_postprocess(result, img_size, output_height, output_width):
        """ copied from detectron2
        Return semantic segmentation predictions in the original resolution.

        The input images are often resized when entering semantic segmentor. Moreover, in same
        cases, they also padded inside segmentor to be divisible by maximum network stride.
        As a result, we often need the predictions of the segmentor in a different
        resolution from its inputs.

        Args:
            result (Tensor): semantic segmentation prediction logits. A tensor of shape (C, H, W),
                where C is the number of classes, and H, W are the height and width of the prediction.
            img_size (tuple): image size that segmentor is taking as input.
            output_height, output_width: the desired output resolution.

        Returns:
            semantic segmentation prediction (Tensor): A tensor of the shape
                (C, output_height, output_width) that contains per-pixel soft predictions.
        """
        result = result[:, : img_size[0], : img_size[1]].expand(1, -1, -1, -1)
        result = F.interpolate(
            result, size=(output_height, output_width), mode="bilinear", align_corners=False
        )[0]
        return result

    def add_pred_to_datasample(self, data_samples: SampleList,
                               results_list: List[dict]) -> SampleList:
        """Add predictions to `DetDataSample`.

        Args:
            data_samples (list[:obj:`DetDataSample`], optional): A batch of
                data samples that contain annotations and predictions.
            results_list (List[dict]): Instance segmentation, segmantic
                segmentation and panoptic segmentation results.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the
            input images. Each DetDataSample usually contain
            'pred_instances' and `pred_panoptic_seg`. And the
            ``pred_instances`` usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).

            And the ``pred_panoptic_seg`` contains the following key

                - sem_seg (Tensor): panoptic segmentation mask, has a
                    shape (1, h, w).
        """
        for data_sample, pred_results in zip(data_samples, results_list):
            if 'pan_results' in pred_results:
                data_sample.pred_panoptic_seg = pred_results['pan_results']

            if 'ins_results' in pred_results:
                data_sample.pred_instances = pred_results['ins_results']

            assert 'sem_results' not in pred_results, 'segmantic ' \
                'segmentation results are not supported yet.'

        return data_samples
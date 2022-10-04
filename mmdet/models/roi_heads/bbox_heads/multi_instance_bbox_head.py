# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData
from torch import Tensor, nn

from mmdet.models.layers import multiclass_nms
from mmdet.models.roi_heads.bbox_heads.bbox_head import BBoxHead
from mmdet.models.task_modules.samplers import SamplingResult
from mmdet.models.utils import empty_instances
from mmdet.registry import MODELS


@MODELS.register_module()
class MultiInstanceBBoxHead(BBoxHead):

    def __init__(self,
                 num_instance: int = 2,
                 refine_flag: bool = True,
                 num_shared_convs: int = 0,
                 num_shared_fcs: int = 2,
                 num_cls_convs: int = 0,
                 num_cls_fcs: int = 0,
                 num_reg_convs: int = 0,
                 num_reg_fcs: int = 0,
                 conv_out_channels: int = 256,
                 fc_out_channels: int = 1024,
                 conv_cfg: Optional[Union[dict, ConfigDict]] = None,
                 norm_cfg: Optional[Union[dict, ConfigDict]] = None,
                 init_cfg: Optional[Union[dict, ConfigDict]] = None,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, init_cfg=init_cfg, **kwargs)
        assert (num_shared_convs + num_shared_fcs + num_cls_convs +
                num_cls_fcs + num_reg_convs + num_reg_fcs > 0)
        if num_cls_convs > 0 or num_reg_convs > 0:
            assert num_shared_fcs == 0
        if not self.with_cls:
            assert num_cls_convs == 0 and num_cls_fcs == 0
        if not self.with_reg:
            assert num_reg_convs == 0 and num_reg_fcs == 0
        self.num_instance = num_instance
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.refine_flag = refine_flag
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim
        self.relu = nn.ReLU(inplace=True)

        if self.refine_flag:
            refine_model_cfg = {
                'type': 'Linear',
                'in_features': self.shared_out_channels + 20,
                'out_features': self.shared_out_channels
            }
            self.shared_fcs_ref = MODELS.build(refine_model_cfg)
            self.fc_cls_ref = nn.ModuleList()
            self.fc_reg_ref = nn.ModuleList()

        self.cls_convs = nn.ModuleList()
        self.cls_fcs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.reg_fcs = nn.ModuleList()
        self.cls_last_dim = list()
        self.reg_last_dim = list()
        self.fc_cls = nn.ModuleList()
        self.fc_reg = nn.ModuleList()
        for k in range(self.num_instance):
            # add cls specific branch
            cls_convs, cls_fcs, cls_last_dim = self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)
            self.cls_convs.append(cls_convs)
            self.cls_fcs.append(cls_fcs)
            self.cls_last_dim.append(cls_last_dim)

            # add reg specific branch
            reg_convs, reg_fcs, reg_last_dim = self._add_conv_fc_branch(
                self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)
            self.reg_convs.append(reg_convs)
            self.reg_fcs.append(reg_fcs)
            self.reg_last_dim.append(reg_last_dim)

            if self.num_shared_fcs == 0 and not self.with_avg_pool:
                if self.num_cls_fcs == 0:
                    self.cls_last_dim *= self.roi_feat_area
                if self.num_reg_fcs == 0:
                    self.reg_last_dim *= self.roi_feat_area

            if self.with_cls:
                if self.custom_cls_channels:
                    cls_channels = self.loss_cls.get_cls_channels(
                        self.num_classes)
                else:
                    cls_channels = self.num_classes + 1
                cls_predictor_cfg_ = self.cls_predictor_cfg.copy()  # deepcopy
                cls_predictor_cfg_.update(
                    in_features=self.cls_last_dim[k],
                    out_features=cls_channels)
                self.fc_cls.append(MODELS.build(cls_predictor_cfg_))
                if self.refine_flag:
                    self.fc_cls_ref.append(MODELS.build(cls_predictor_cfg_))

            if self.with_reg:
                out_dim_reg = (4 if self.reg_class_agnostic else 4 *
                               self.num_classes)
                reg_predictor_cfg_ = self.reg_predictor_cfg.copy()
                reg_predictor_cfg_.update(
                    in_features=self.reg_last_dim[k], out_features=out_dim_reg)
                self.fc_reg.append(MODELS.build(reg_predictor_cfg_))
                if self.refine_flag:
                    self.fc_reg_ref.append(MODELS.build(reg_predictor_cfg_))

        # 1.
        if init_cfg is None:
            # when init_cfg is None,
            # It has been set to
            # [[dict(type='Normal', std=0.01, override=dict(name='fc_cls'))],
            #  [dict(type='Normal', std=0.001, override=dict(name='fc_reg'))]
            # after `super(ConvFCBBoxHead, self).__init__()`
            # we only need to append additional configuration
            # for `shared_fcs`, `cls_fcs` and `reg_fcs`
            self.init_cfg += [
                dict(
                    type='Xavier',
                    distribution='uniform',
                    override=[
                        dict(name='shared_fcs'),
                        dict(name='cls_fcs'),
                        dict(name='reg_fcs')
                    ])
            ]

    def _add_conv_fc_branch(self,
                            num_branch_convs: int,
                            num_branch_fcs: int,
                            in_channels: int,
                            is_shared: bool = False) -> tuple:
        """Add shared or separable branch.

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                    or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def forward(self, x: Tuple[Tensor]) -> tuple:
        """Forward features from the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: A tuple of classification scores and bbox prediction.

                - cls_score (Tensor): Classification scores for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_base_priors * num_classes.
                - bbox_pred (Tensor): Box energies / deltas for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_base_priors * 4.
                -cls_score_ref (Tensor): The cls_score after refine model.
                -bbox_pred_ref (Tensor): The bbox_pred after refine model.
        """
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)
            for fc in self.shared_fcs:
                x = self.relu(fc(x))

        x_cls = x
        x_reg = x
        # separate branches
        cls_score = list()
        bbox_pred = list()
        for k in range(self.num_instance):
            for conv in self.cls_convs[k]:
                x_cls = conv(x_cls)
            if x_cls.dim() > 2:
                if self.with_avg_pool:
                    x_cls = self.avg_pool(x_cls)
                x_cls = x_cls.flatten(1)
            for fc in self.cls_fcs[k]:
                x_cls = self.relu(fc(x_cls))

            for conv in self.reg_convs[k]:
                x_reg = conv(x_reg)
            if x_reg.dim() > 2:
                if self.with_avg_pool:
                    x_reg = self.avg_pool(x_reg)
                x_reg = x_reg.flatten(1)
            for fc in self.reg_fcs[k]:
                x_reg = self.relu(fc(x_reg))

            cls_score.append(self.fc_cls[k](x_cls) if self.with_cls else None)
            bbox_pred.append(self.fc_reg[k](x_reg) if self.with_reg else None)

        if self.refine_flag:
            x_ref = x
            cls_score_ref = list()
            bbox_pred_ref = list()
            for k in range(self.num_instance):
                feat_ref = F.softmax(cls_score[k], dim=-1)
                feat_ref = torch.cat((bbox_pred[k], feat_ref[:, 1][:, None]),
                                     dim=1).repeat(1, 4)
                feat_ref = torch.cat((x_ref, feat_ref), dim=1)
                feat_ref = F.relu_(self.shared_fcs_ref(feat_ref))

                cls_score_ref.append(self.fc_cls_ref[k](feat_ref))
                bbox_pred_ref.append(self.fc_reg_ref[k](feat_ref))

            cls_score = torch.cat(cls_score, dim=1)
            bbox_pred = torch.cat(bbox_pred, dim=1)
            cls_score_ref = torch.cat(cls_score_ref, dim=1)
            bbox_pred_ref = torch.cat(bbox_pred_ref, dim=1)
            return cls_score, bbox_pred, cls_score_ref, bbox_pred_ref

        cls_score = torch.cat(cls_score, dim=1)
        bbox_pred = torch.cat(bbox_pred, dim=1)

        return cls_score, bbox_pred

    def get_targets(self,
                    sampling_results: List[SamplingResult],
                    rcnn_train_cfg: ConfigDict,
                    concat: bool = True) -> tuple:
        """Calculate the ground truth for all samples in a batch according to
        the sampling_results.

        Almost the same as the implementation in bbox_head, we passed
        additional parameters pos_inds_list and neg_inds_list to
        `_get_targets_single` function.

        Args:
            sampling_results (List[obj:SamplingResult]): Assign results of
                all images in a batch after sampling.
            rcnn_train_cfg (obj:ConfigDict): `train_cfg` of RCNN.
            concat (bool): Whether to concatenate the results of all
                the images in a single batch.

        Returns:
            Tuple[Tensor]: Ground truth for proposals in a single image.
            Containing the following list of Tensors:

            - labels (list[Tensor],Tensor): Gt_labels for all
                proposals in a batch, each tensor in list has
                shape (num_proposals,) when `concat=False`, otherwise
                just a single tensor has shape (num_all_proposals,).
            - label_weights (list[Tensor]): Labels_weights for
                all proposals in a batch, each tensor in list has
                shape (num_proposals,) when `concat=False`, otherwise
                just a single tensor has shape (num_all_proposals,).
            - bbox_targets (list[Tensor],Tensor): Regression target
                for all proposals in a batch, each tensor in list
                has shape (num_proposals, 4) when `concat=False`,
                otherwise just a single tensor has shape
                (num_all_proposals, 4), the last dimension 4 represents
                [tl_x, tl_y, br_x, br_y].
            - bbox_weights (list[tensor],Tensor): Regression weights for
                all proposals in a batch, each tensor in list has shape
                (num_proposals, 4) when `concat=False`, otherwise just a
                single tensor has shape (num_all_proposals, 4).
        """
        labels = []
        bbox_targets = []
        bbox_weights = []
        label_weights = []
        for i in range(len(sampling_results)):
            sample_bboxes = torch.cat([
                sampling_results[i].pos_gt_bboxes,
                sampling_results[i].neg_gt_bboxes
            ])
            sample_priors = sampling_results[i].priors
            sample_priors = sample_priors.repeat(1, self.num_instance).reshape(
                -1, 4)
            sample_bboxes = sample_bboxes.reshape(-1, 4)

            if not self.reg_decoded_bbox:
                _bbox_targets = self.bbox_coder.encode(sample_priors,
                                                       sample_bboxes)
            else:
                _bbox_targets = sample_priors
            _bbox_targets = _bbox_targets.reshape(-1, self.num_instance * 4)
            _bbox_weights = torch.ones(_bbox_targets.shape)
            _labels = torch.cat([
                sampling_results[i].pos_gt_labels,
                sampling_results[i].neg_gt_labels
            ])
            _labels_weights = torch.ones(_labels.shape)

            bbox_targets.append(_bbox_targets)
            bbox_weights.append(_bbox_weights)
            labels.append(_labels)
            label_weights.append(_labels_weights)

        if concat:
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            bbox_targets = torch.cat(bbox_targets, 0)
            bbox_weights = torch.cat(bbox_weights, 0)
        return labels, label_weights, bbox_targets, bbox_weights

    def loss(self,
             cls_score: Tensor,
             bbox_pred: Tensor,
             rois: Tensor,
             labels: Tensor,
             label_weights: Tensor,
             bbox_targets: Tensor,
             bbox_weights: Tensor,
             reduction_override: Optional[str] = None) -> dict:
        """Calculate the loss based on the network predictions and targets.

        Args:
            cls_score (Tensor): Classification prediction
                results of all class, has shape
                (batch_size * num_proposals_single_image, num_classes)
            bbox_pred (Tensor): Regression prediction results,
                has shape
                (batch_size * num_proposals_single_image, 4), the last
                dimension 4 represents [tl_x, tl_y, br_x, br_y].
            rois (Tensor): RoIs with the shape
                (batch_size * num_proposals_single_image, 5) where the first
                column indicates batch id of each RoI.
            labels (Tensor): Gt_labels for all proposals in a batch, has
                shape (batch_size * num_proposals_single_image, ).
            label_weights (Tensor): Labels_weights for all proposals in a
                batch, has shape (batch_size * num_proposals_single_image, ).
            bbox_targets (Tensor): Regression target for all proposals in a
                batch, has shape (batch_size * num_proposals_single_image, 4),
                the last dimension 4 represents [tl_x, tl_y, br_x, br_y].
            bbox_weights (Tensor): Regression weights for all proposals in a
                batch, has shape (batch_size * num_proposals_single_image, 4).
            reduction_override (str, optional): The reduction
                method used to override the original reduction
                method of the loss. Options are "none",
                "mean" and "sum". Defaults to None,

        Returns:
            dict: A dictionary of loss.
        """

        losses = dict()
        loss0 = self.emd_loss_softmax(bbox_pred[:, 0:4], cls_score[:, 0:2],
                                      bbox_pred[:, 4:8], cls_score[:, 2:4],
                                      bbox_targets, labels)
        loss1 = self.emd_loss_softmax(bbox_pred[:, 4:8], cls_score[:, 2:4],
                                      bbox_pred[:, 0:4], cls_score[:, 0:2],
                                      bbox_targets, labels)
        loss = torch.cat([loss0, loss1], dim=1)
        _, min_indices = loss.min(dim=1)
        loss_emd = loss[torch.arange(loss.shape[0]), min_indices]
        loss_emd = loss_emd.mean()
        losses['loss_rcnn_emd'] = loss_emd
        return losses

    def emd_loss_softmax(self, p_b0, p_s0, p_b1, p_s1, targets, labels):
        # reshape
        pred_delta = torch.cat([p_b0, p_b1], dim=1).reshape(-1, p_b0.shape[-1])
        pred_score = torch.cat([p_s0, p_s1], dim=1).reshape(-1, p_s0.shape[-1])
        targets = targets.reshape(-1, 4)
        labels = labels.long().flatten()

        # cons masks
        valid_masks = labels >= 0
        fg_masks = labels > 0

        # multiple class
        pred_delta = pred_delta.reshape(-1, self.num_classes, 4)
        fg_gt_classes = labels[fg_masks]
        pred_delta = pred_delta[fg_masks, fg_gt_classes - 1, :]

        # loss for regression
        loss_bbox = self.loss_bbox(pred_delta, targets[fg_masks])
        loss_bbox = torch.sum(loss_bbox, dim=1)

        # loss for classification
        labels = labels * valid_masks
        loss_cls = self.loss_cls(pred_score, labels)

        loss_cls[fg_masks] = loss_cls[fg_masks] + loss_bbox
        loss = loss_cls.reshape(-1, 2).sum(axis=1)
        return loss.reshape(-1, 1)

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
            bbox_pred (Tensor): Box energies / deltas. has shape
                (num_boxes, num_classes * 4).
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

        cls_score = cls_score.reshape(-1, self.num_classes + 1)
        bbox_pred = bbox_pred.reshape(-1, 4)
        roi = roi.repeat_interleave(self.num_instance, dim=0)

        results = InstanceData()
        if roi.shape[0] == 0:
            return empty_instances([img_meta],
                                   roi.device,
                                   task_type='bbox',
                                   instance_results=[results])[0]

        # some loss (Seesaw loss..) may have custom activation
        if self.custom_cls_channels:
            scores = self.loss_cls.get_activation(cls_score)
        else:
            scores = F.softmax(
                cls_score, dim=-1) if cls_score is not None else None

        img_shape = img_meta['img_shape']
        # bbox_pred would be None in some detector when with_reg is False,
        # e.g. Grid R-CNN.
        if bbox_pred is not None:
            bboxes = self.bbox_coder.decode(
                roi[..., 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = roi[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])

        if rescale and bboxes.size(0) > 0:
            assert img_meta.get('scale_factor') is not None
            scale_factor = bboxes.new_tensor(img_meta['scale_factor']).repeat(
                (1, 2))
            bboxes = (bboxes.view(bboxes.size(0), -1, 4) / scale_factor).view(
                bboxes.size()[0], -1)

        if rcnn_test_cfg is None:
            # This means that it is aug test.
            # It needs to return the raw results without nms.
            results.bboxes = bboxes
            results.scores = scores
        else:
            # If self.num_classes is one, the NMS in CrowdDet is used.
            if self.num_classes > 1:
                det_bboxes, det_labels = multiclass_nms(
                    bboxes, scores, rcnn_test_cfg.score_thr, rcnn_test_cfg.nms,
                    rcnn_test_cfg.max_per_img)
                results.bboxes = det_bboxes[:, :4]
                results.scores = det_bboxes[:, -1]
                results.labels = det_labels
            else:
                nms_input = np.zeros([bboxes.shape[0], 7])
                nms_input[:, :4] = bboxes.cpu().detach().numpy()
                nms_input[:, 4] = scores[:, 1].cpu().detach().numpy()
                nms_input[:, 5] = np.zeros([
                    bboxes.shape[0],
                ])
                nms_input[:, 6] = np.tile(
                    np.arange(bboxes.shape[0] / self.num_instance)[:, None],
                    (1, self.num_instance)).reshape(-1, 1)[:, 0]
                keep = nms_input[:, 4] > rcnn_test_cfg.score_thr
                nms_input = nms_input[keep]
                keep = self.set_cpu_nms(nms_input,
                                        rcnn_test_cfg.nms['iou_threshold'])
                nms_result = nms_input[keep]
                results.bboxes = torch.from_numpy(nms_result[:, :4]).to(
                    bboxes.device)
                results.scores = torch.from_numpy(nms_result[:, 4]).to(
                    bboxes.device)
                results.labels = torch.from_numpy(nms_result[:, 5]).to(
                    bboxes.device)

        return results

    @staticmethod
    def set_cpu_nms(pred_boxes, iou_threshold):
        """NMS for multi-instance prediction. Please refer to
        https://github.com/Purkialo/CrowdDet for more details.

        Args:
            pred_boxes (ndarray): shape (n, 7). [n, 0:4] denotes predict
                bboxes, [:, 4:5] is  the score and label for each prediction
                box, and [:, 6] indicates the prediction box comes from
                which propose box.
            iou_threshold (float): IoU threshold to be considered as
                conflicted.

        Returns:
            keep (ndarray): The location of the valid bboxes.
        """

        def _overlap(det_boxes, base, others):
            eps = 1e-8
            x1_basement = det_boxes[base, 0]
            y1_basement = det_boxes[base, 1]
            x2_basement = det_boxes[base, 2]
            y2_basement = det_boxes[base, 3]

            x1_others = det_boxes[others, 0]
            y1_others = det_boxes[others, 1]
            x2_others = det_boxes[others, 2]
            y2_others = det_boxes[others, 3]

            areas_basement = (x2_basement - x1_basement) * (
                y2_basement - y1_basement)
            areas_others = (x2_others - x1_others) * (y2_others - y1_others)
            xx1 = np.maximum(x1_basement, x1_others)
            yy1 = np.maximum(y1_basement, y1_others)
            xx2 = np.minimum(x2_basement, x2_others)
            yy2 = np.minimum(y2_basement, y2_others)
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            ovr = inter / (areas_basement + areas_others - inter + eps)
            return ovr

        scores = pred_boxes[:, 4]
        order = np.argsort(-scores)
        dets = pred_boxes[order]

        numbers = dets[:, -1]
        keep = np.ones(len(dets)) == 1
        ruler = np.arange(len(dets))
        while ruler.size > 0:
            basement = ruler[0]
            ruler = ruler[1:]
            num = numbers[basement]
            # calculate the body overlap
            overlap = _overlap(dets[:, :4], basement, ruler)
            indices = np.where(overlap > iou_threshold)[0]
            loc = np.where(numbers[ruler][indices] == num)[0]
            # the mask won't change in the step
            mask = keep[ruler[indices][loc]]
            keep[ruler[indices]] = False
            keep[ruler[indices][loc][mask]] = True
            ruler[~keep[ruler]] = -1
            ruler = ruler[ruler > 0]
        keep = keep[np.argsort(order)]
        return keep

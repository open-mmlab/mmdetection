# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F

from mmdet.core import build_assigner, build_sampler
from mmdet.core.bbox.samplers import MaskPseudoSampler
from mmdet.models.builder import HEADS, build_head
from mmdet.models.roi_heads import BaseRoIHead


@HEADS.register_module()
class KernelIterHead(BaseRoIHead):
    """Knernel iterative head.

    Args:
        num_stages (int): Number of stages. Defaults to 3.
        stage_loss_weights (tuple): Loss weights for each stage.
            Defaults to (1, 1, 1).
        proposal_feature_channel (int): Number of channels of proposal feature
            maps. Defaults to 256.
        num_proposals (int): Number of proposals. Defaults to 100.
        num_things_classes (int): Number of things classes. Defaults to 80.
        num_stuff_classes (int): Number of stuff classes. Defaults to 53.
        mask_assign_out_stride (int): Output strides of mask for assigning.
            Defaults to 4.
        mask_head (list[:obj:`mmcv.Config`]): Config of mask head.
            Defaults to None.
        train_cfg (:obj:`mmcv.Config` or None): Training Config.
            Defaults to None.
        test_cfg (:obj:`mmcv.Config` or None): Training Config.
            Defaults to None.
    """

    def __init__(self,
                 num_stages=3,
                 stage_loss_weights=(1, 1, 1),
                 proposal_feature_channel=256,
                 num_proposals=100,
                 num_things_classes=80,
                 num_stuff_classes=53,
                 mask_assign_out_stride=4,
                 mask_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 **kwargs):
        assert mask_head is not None
        assert len(stage_loss_weights) == num_stages
        self.num_stages = num_stages
        self.stage_loss_weights = stage_loss_weights

        self.proposal_feature_channel = proposal_feature_channel
        self.num_proposals = num_proposals
        self.num_things_classes = num_things_classes
        self.num_stuff_classes = num_stuff_classes
        self.num_classes = num_things_classes + num_stuff_classes
        self.mask_assign_out_stride = mask_assign_out_stride

        super().__init__(
            mask_head=mask_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            **kwargs)

        # train_cfg would be None when run the test.py
        if train_cfg is not None:
            for stage in range(num_stages):
                assert isinstance(
                    self.mask_sampler[stage], MaskPseudoSampler), \
                    'Sparse Mask only support `MaskPseudoSampler`'

    def init_bbox_head(self, mask_roi_extractor, mask_head):
        """Initialize box head and box roi extractor.

        Args:
            mask_roi_extractor (dict): Config of box roi extractor.
            mask_head (dict): Config of box in box head.
        """
        # abstract method must be implemented here.
        pass

    def init_mask_head(self, mask_roi_extractor, mask_head):
        """Initialize mask head and mask roi extractor.

        Args:
            mask_roi_extractor (dict): Config of mask roi extractor.
            mask_head (dict): Config of mask in mask head.
        """
        self.mask_head = nn.ModuleList()
        if not isinstance(mask_head, list):
            mask_head = [mask_head for _ in range(self.num_stages)]
        assert len(mask_head) == self.num_stages
        for head in mask_head:
            self.mask_head.append(build_head(head))

    def init_assigner_sampler(self):
        """Initialize assigner and sampler for each stage."""
        self.mask_assigner = []
        self.mask_sampler = []
        if self.train_cfg is not None:
            for idx, rcnn_train_cfg in enumerate(self.train_cfg):
                self.mask_assigner.append(
                    build_assigner(rcnn_train_cfg.assigner))
                self.current_stage = idx
                self.mask_sampler.append(
                    build_sampler(rcnn_train_cfg.sampler, context=self))

    def init_weights(self):
        """Initialize weights."""
        for i in range(self.num_stages):
            self.mask_head[i].init_weights()

    def _mask_forward(self, stage, x, object_feats, mask_preds):
        """Mask head forward function used in both training and testing."""
        mask_head = self.mask_head[stage]
        cls_scores, mask_preds, object_feats = mask_head(
            x, object_feats, mask_preds)

        if mask_head.mask_upsample_stride > 1 and (stage == self.num_stages - 1
                                                   or self.training):
            scaled_mask_preds = F.interpolate(
                mask_preds,
                scale_factor=mask_head.mask_upsample_stride,
                align_corners=False,
                mode='bilinear')
        else:
            scaled_mask_preds = mask_preds

        mask_results = dict(
            cls_scores=cls_scores,
            mask_preds=mask_preds,
            scaled_mask_preds=scaled_mask_preds,
            object_feats=object_feats)

        return mask_results

    def forward_train(self, x, proposal_feats, mask_preds, img_metas, gt_masks,
                      gt_labels, gt_sem_seg, gt_sem_cls):
        """
        Args:
            x (Tensor): Feature map, shape (batch_size, c, h, w).
            proposal_feats (Tensor): Proposal features, shape (batch_size,
                n, c, kernel_size, kernel_size). n is (num_proposals +
                num_stuff_classes) for panoptic segmentation, num_proposals for
                instance segmentation.
            mask_preds (Tensor): Mask logits with shape (batch_size, n, h, w).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_masks (list[Tensor]): Each element of list is ground truth masks
                of instances, shape (num_instances, h, w).
            gt_labels (list[Tensor]): Each element of list is ground truth
                labels of instances, shape (num_instances,).
            gt_sem_seg (list[Tensor]): Each element of list is ground truth
                masks of stuff, shape (num_stuff, h, w). It is list[None]
                for instance segmentation.
            gt_sem_cls (list[Tensor]): Each element of list is ground truth
                labels of stuff, shape (num_stuff,). Is is list[None] for
                instance segmentation.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        num_imgs = len(img_metas)
        if self.mask_head[0].mask_upsample_stride > 1:
            prev_mask_preds = F.interpolate(
                mask_preds.detach(),
                scale_factor=self.mask_head[0].mask_upsample_stride,
                mode='bilinear',
                align_corners=False)
        else:
            prev_mask_preds = mask_preds.detach()

        prev_cls_score = [None] * num_imgs

        object_feats = proposal_feats
        all_stage_loss = {}
        all_stage_mask_results = []
        for stage in range(self.num_stages):
            mask_results = self._mask_forward(stage, x, object_feats,
                                              mask_preds)
            all_stage_mask_results.append(mask_results)
            mask_preds = mask_results['mask_preds']
            scaled_mask_preds = mask_results['scaled_mask_preds']
            cls_scores = mask_results['cls_scores']
            object_feats = mask_results['object_feats']

            sampling_results = []
            assign_results = []
            for i in range(num_imgs):
                mask_for_assign = prev_mask_preds[i][:self.num_proposals]
                if prev_cls_score[i] is not None:
                    cls_for_assign = prev_cls_score[
                        i][:self.num_proposals, :self.num_things_classes]
                else:
                    cls_for_assign = None
                assign_result = self.mask_assigner[stage].assign(
                    cls_for_assign, mask_for_assign, gt_labels[i], gt_masks[i],
                    img_metas[i])
                assign_results.append(assign_result)
                sampling_result = self.mask_sampler[stage].sample(
                    assign_results[i], scaled_mask_preds[i], gt_masks[i])
                sampling_results.append(sampling_result)

            mask_targets = self.mask_head[stage].get_targets(
                sampling_results,
                self.train_cfg[stage],
                gt_sem_seg=gt_sem_seg,
                gt_sem_cls=gt_sem_cls)

            single_stage_loss = self.mask_head[stage].loss(
                cls_scores, scaled_mask_preds, *mask_targets)
            for key, value in single_stage_loss.items():
                all_stage_loss[f's{stage}_{key}'] = value * \
                    self.stage_loss_weights[stage]

            prev_mask_preds = scaled_mask_preds.detach()
            prev_cls_score = cls_scores.detach()

        return all_stage_loss

    def simple_test(self, x, proposal_feats, mask_preds):
        """
        Args:
            x (Tensor): Feature map, shape (batch_size, c, h, w).
            proposal_feats (Tensor): Proposal features, shape (batch_size,
                n, c, kernel_size, kernel_size). n is (num_proposals +
                num_stuff_classes) for panoptic segmentation, num_proposals for
                instance segmentation.
            mask_preds (Tensor): Mask logits with shape (batch_size, n, h, w).

        Returns:
            tuple[Tensor]:

            - cls_scores (Tensor): Classification scores, shape (batch_size,
              n, cls_channels). n is num_proposals + num_stuff_classes for
              panoptic segmentation, num_proposals for instance segmentation.
            - scaled_mask_preds (Tensor): Mask scores, shape (batch_size,
              n, h, w).
        """
        object_feats = proposal_feats
        for stage in range(self.num_stages):
            mask_results = self._mask_forward(stage, x, object_feats,
                                              mask_preds)
            object_feats = mask_results['object_feats']
            cls_scores = mask_results['cls_scores']
            mask_preds = mask_results['mask_preds']
            scaled_mask_preds = mask_results['scaled_mask_preds']

        if self.mask_head[-1].loss_cls.use_sigmoid:
            cls_scores = cls_scores.sigmoid()
        else:
            cls_scores = cls_scores.softmax(-1)[..., :-1]

        scaled_mask_preds = scaled_mask_preds.sigmoid()

        return cls_scores, scaled_mask_preds

    def aug_test(self, features, proposal_list, img_metas, rescale=False):
        raise NotImplementedError('K-Net does not support `aug_test`')

    def forward_dummy(self, x, proposal_boxes, proposal_feats, img_metas):
        """Dummy forward function when do the flops computing."""
        all_stage_mask_results = []
        num_imgs = len(img_metas)
        num_proposals = proposal_feats.size(1)
        c, h, w = x.shape[-3:]
        mask_preds = proposal_feats.bmm(x.view(num_imgs, c, -1)).view(
            num_imgs, num_proposals, h, w)
        object_feats = proposal_feats
        for stage in range(self.num_stages):
            mask_results = self._mask_forward(stage, x, object_feats,
                                              mask_preds, img_metas)
            all_stage_mask_results.append(mask_results)
        return all_stage_mask_results

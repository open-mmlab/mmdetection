# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F
from mmcv.runner import force_fp32

from ..builder import HEADS
from ..losses import smooth_l1_loss
from .ascend_anchor_head import AscendAnchorHead
from .ssd_head import SSDHead


@HEADS.register_module()
class AscendSSDHead(SSDHead, AscendAnchorHead):
    """Ascend SSD head used in https://arxiv.org/abs/1512.02325.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        stacked_convs (int): Number of conv layers in cls and reg tower.
            Default: 0.
        feat_channels (int): Number of hidden channels when stacked_convs
            > 0. Default: 256.
        use_depthwise (bool): Whether to use DepthwiseSeparableConv.
            Default: False.
        conv_cfg (dict): Dictionary to construct and config conv layer.
            Default: None.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: None.
        act_cfg (dict): Dictionary to construct and config activation layer.
            Default: None.
        anchor_generator (dict): Config dict for anchor generator
        bbox_coder (dict): Config of bounding box coder.
        reg_decoded_bbox (bool): If true, the regression loss would be
            applied directly on decoded bounding boxes, converting both
            the predicted boxes and regression targets to absolute
            coordinates format. Default False. It should be `True` when
            using `IoULoss`, `GIoULoss`, or `DIoULoss` in the bbox head.
        train_cfg (dict): Training config of anchor head.
        test_cfg (dict): Testing config of anchor head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """  # noqa: W605

    def __init__(self,
                 num_classes=80,
                 in_channels=(512, 1024, 512, 256, 256, 256),
                 stacked_convs=0,
                 feat_channels=256,
                 use_depthwise=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 anchor_generator=dict(
                     type='SSDAnchorGenerator',
                     scale_major=False,
                     input_size=300,
                     strides=[8, 16, 32, 64, 100, 300],
                     ratios=([2], [2, 3], [2, 3], [2, 3], [2], [2]),
                     basesize_ratio_range=(0.1, 0.9)),
                 bbox_coder=dict(
                     type='DeltaXYWHBBoxCoder',
                     clip_border=True,
                     target_means=[.0, .0, .0, .0],
                     target_stds=[1.0, 1.0, 1.0, 1.0],
                 ),
                 reg_decoded_bbox=False,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=dict(
                     type='Xavier',
                     layer='Conv2d',
                     distribution='uniform',
                     bias=0)):
        super(AscendSSDHead, self).__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            stacked_convs=stacked_convs,
            feat_channels=feat_channels,
            use_depthwise=use_depthwise,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            anchor_generator=anchor_generator,
            bbox_coder=bbox_coder,
            reg_decoded_bbox=reg_decoded_bbox,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg)
        assert self.reg_decoded_bbox is False, \
            'reg_decoded_bbox only support False now.'

    def get_static_anchors(self, featmap_sizes, img_metas, device='cuda'):
        """Get static anchors according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.
            device (torch.device | str): Device for returned tensors

        Returns:
            tuple:
                anchor_list (list[Tensor]): Anchors of each image.
                valid_flag_list (list[Tensor]): Valid flags of each image.
        """
        if not hasattr(self, 'static_anchors') or \
                not hasattr(self, 'static_valid_flags'):
            static_anchors, static_valid_flags = self.get_anchors(
                featmap_sizes, img_metas, device)
            self.static_anchors = static_anchors
            self.static_valid_flags = static_valid_flags
        return self.static_anchors, self.static_valid_flags

    def get_targets(self,
                    anchor_list,
                    valid_flag_list,
                    gt_bboxes_list,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    label_channels=1,
                    unmap_outputs=True,
                    return_sampling_results=False,
                    return_level=True):
        """Compute regression and classification targets for anchors in
        multiple images.

        Args:
            anchor_list (list[list[Tensor]]): Multi level anchors of each
                image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, 4).
            valid_flag_list (list[list[Tensor]]): Multi level valid flags of
                each image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, )
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
            img_metas (list[dict]): Meta info of each image.
            gt_bboxes_ignore_list (list[Tensor]): Ground truth bboxes to be
                ignored.
            gt_labels_list (list[Tensor]): Ground truth labels of each box.
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.
            return_sampling_results (bool): Whether to return the result of
                sample.
            return_level (bool): Whether to map outputs back to the levels
                of feature map sizes.
        Returns:
            tuple: Usually returns a tuple containing learning targets.

                - labels_list (list[Tensor]): Labels of each level.
                - label_weights_list (list[Tensor]): Label weights of each
                  level.
                - bbox_targets_list (list[Tensor]): BBox targets of each level.
                - bbox_weights_list (list[Tensor]): BBox weights of each level.
                - num_total_pos (int): Number of positive samples in all
                  images.
                - num_total_neg (int): Number of negative samples in all
                  images.

            additional_returns: This function enables user-defined returns from
                `self._get_targets_single`. These returns are currently refined
                to properties at each feature map (i.e. having HxW dimension).
                The results will be concatenated after the end
        """
        return AscendAnchorHead.get_targets(
            self,
            anchor_list,
            valid_flag_list,
            gt_bboxes_list,
            img_metas,
            gt_bboxes_ignore_list,
            gt_labels_list,
            label_channels,
            unmap_outputs,
            return_sampling_results,
            return_level,
        )

    def batch_loss(self, batch_cls_score, batch_bbox_pred, batch_anchor,
                   batch_labels, batch_label_weights, batch_bbox_targets,
                   batch_bbox_weights, batch_pos_mask, batch_neg_mask,
                   num_total_samples):
        """Compute loss of all images.

        Args:
            batch_cls_score (Tensor): Box scores for all image
                Has shape (num_imgs, num_total_anchors, num_classes).
            batch_bbox_pred (Tensor): Box energies / deltas for all image
                level with shape (num_imgs, num_total_anchors, 4).
            batch_anchor (Tensor): Box reference for all image with shape
                (num_imgs, num_total_anchors, 4).
            batch_labels (Tensor): Labels of all anchors with shape
                (num_imgs, num_total_anchors,).
            batch_label_weights (Tensor): Label weights of all anchor with
                shape (num_imgs, num_total_anchors,)
            batch_bbox_targets (Tensor): BBox regression targets of all anchor
                weight shape (num_imgs, num_total_anchors, 4).
            batch_bbox_weights (Tensor): BBox regression loss weights of
                all anchor with shape (num_imgs, num_total_anchors, 4).
            batch_pos_mask (Tensor): Positive samples mask in all images.
            batch_neg_mask (Tensor): negative samples mask in all images.
            num_total_samples (int): If sampling, num total samples equal to
                the number of total anchors; Otherwise, it is the number of
                positive anchors.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        num_images, num_anchors, _ = batch_anchor.size()

        batch_loss_cls_all = F.cross_entropy(
            batch_cls_score.view((-1, self.cls_out_channels)),
            batch_labels.view(-1),
            reduction='none').view(
                batch_label_weights.size()) * batch_label_weights
        # # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        batch_num_pos_samples = torch.sum(batch_pos_mask, dim=1)
        batch_num_neg_samples = \
            self.train_cfg.neg_pos_ratio * batch_num_pos_samples

        batch_num_neg_samples_max = torch.sum(batch_neg_mask, dim=1)
        batch_num_neg_samples = torch.min(batch_num_neg_samples,
                                          batch_num_neg_samples_max)

        batch_topk_loss_cls_neg, _ = torch.topk(
            batch_loss_cls_all * batch_neg_mask, k=num_anchors, dim=1)
        batch_loss_cls_pos = torch.sum(
            batch_loss_cls_all * batch_pos_mask, dim=1)

        anchor_index = torch.arange(
            end=num_anchors, dtype=torch.float,
            device=batch_anchor.device).view((1, -1))
        topk_loss_neg_mask = (anchor_index < batch_num_neg_samples.view(
            -1, 1)).float()

        batch_loss_cls_neg = torch.sum(
            batch_topk_loss_cls_neg * topk_loss_neg_mask, dim=1)
        loss_cls = \
            (batch_loss_cls_pos + batch_loss_cls_neg) / num_total_samples

        if self.reg_decoded_bbox:
            # TODO: support self.reg_decoded_bbox is True
            raise RuntimeError

        loss_bbox_all = smooth_l1_loss(
            batch_bbox_pred,
            batch_bbox_targets,
            batch_bbox_weights,
            reduction='none',
            beta=self.train_cfg.smoothl1_beta,
            avg_factor=num_total_samples)
        eps = torch.finfo(torch.float32).eps

        sum_dim = (i for i in range(1, len(loss_bbox_all.size())))
        loss_bbox = loss_bbox_all.sum(tuple(sum_dim)) / (
            num_total_samples + eps)
        return loss_cls[None], loss_bbox

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            gt_bboxes (list[Tensor]): each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=1,
            unmap_outputs=True,
            return_level=False)
        if cls_reg_targets is None:
            return None

        (batch_labels, batch_label_weights, batch_bbox_targets,
         batch_bbox_weights, batch_pos_mask, batch_neg_mask, sampling_result,
         num_total_pos, num_total_neg, batch_anchors) = cls_reg_targets

        num_imgs = len(img_metas)
        batch_cls_score = torch.cat([
            s.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.cls_out_channels)
            for s in cls_scores
        ], 1)

        batch_bbox_pred = torch.cat([
            b.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4) for b in bbox_preds
        ], -2)

        batch_losses_cls, batch_losses_bbox = self.batch_loss(
            batch_cls_score, batch_bbox_pred, batch_anchors, batch_labels,
            batch_label_weights, batch_bbox_targets, batch_bbox_weights,
            batch_pos_mask, batch_neg_mask, num_total_pos)
        losses_cls = [
            batch_losses_cls[:, index_imgs] for index_imgs in range(num_imgs)
        ]
        losses_bbox = [losses_bbox for losses_bbox in batch_losses_bbox]
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)

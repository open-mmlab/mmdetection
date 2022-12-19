import torch
from ..builder import HEADS
from .anchor_head import AnchorHead
from ...core.bbox.assigners import AscendMaxIoUAssigner
from ...core.bbox.samplers import PseudoSampler
from ...utils import set_index, images_to_levels, generate_max_gt_nums


@HEADS.register_module()
class AscendAnchorHead(AnchorHead):
    """Anchor-based head (RPN, RetinaNet, SSD, etc.).

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of hidden channels. Used in child classes.
        anchor_generator (dict): Config dict for anchor generator
        bbox_coder (dict): Config of bounding box coder.
        reg_decoded_bbox (bool): If true, the regression loss would be
            applied directly on decoded bounding boxes, converting both
            the predicted boxes and regression targets to absolute
            coordinates format. Default False. It should be `True` when
            using `IoULoss`, `GIoULoss`, or `DIoULoss` in the bbox head.
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        train_cfg (dict): Training config of anchor head.
        test_cfg (dict): Testing config of anchor head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """  # noqa: W605

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 anchor_generator=dict(
                     type='AnchorGenerator',
                     scales=[8, 16, 32],
                     ratios=[0.5, 1.0, 2.0],
                     strides=[4, 8, 16, 32, 64]),
                 bbox_coder=dict(
                     type='DeltaXYWHBBoxCoder',
                     clip_border=True,
                     target_means=(.0, .0, .0, .0),
                     target_stds=(1.0, 1.0, 1.0, 1.0)),
                 reg_decoded_bbox=False,
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=dict(type='Normal', layer='Conv2d', std=0.01)):
        super(AscendAnchorHead, self).__init__(
            num_classes, in_channels, feat_channels, anchor_generator,
            bbox_coder, reg_decoded_bbox, loss_cls, loss_bbox,
            train_cfg, test_cfg, init_cfg)

    def _get_concat_gt_bboxes(self, gt_bboxes_list, num_images, gt_nums, device, max_gt_labels):
        if not hasattr(self, 'concat_gt_bboxes'):
            self.concat_gt_bboxes = {}
        if not hasattr(self, 'min_anchor'):
            self.min_anchor = (-1354, -1344)
        if gt_bboxes_list is None:
            concat_gt_bboxes = None
        else:
            if self.concat_gt_bboxes.get(max_gt_labels) is None:
                concat_gt_bboxes = torch.zeros((num_images, max_gt_labels, 4),
                                               dtype=gt_bboxes_list[0].dtype,
                                               device=device)
                concat_gt_bboxes[:, :, :2] = self.min_anchor[0]
                concat_gt_bboxes[:, :, 2:] = self.min_anchor[1]
                self.concat_gt_bboxes[max_gt_labels] = concat_gt_bboxes.clone()
            else:
                concat_gt_bboxes = self.concat_gt_bboxes.get(max_gt_labels).clone()
            for index_imgs, gt_bboxes in enumerate(gt_bboxes_list):
                concat_gt_bboxes[index_imgs, :gt_nums[index_imgs]] = gt_bboxes
        return concat_gt_bboxes

    def _get_concat_gt_bboxes_ignore(self, gt_bboxes_ignore_list, num_images, gt_nums, device):
        if gt_bboxes_ignore_list is None:
            concat_gt_bboxes_ignore = None
        else:
            raise RuntimeError("gt_bboxes_ignore not support yet")
        return concat_gt_bboxes_ignore

    def _get_concat_gt_labels(self, gt_labels_list, num_images, gt_nums, device, max_gt_labels):
        if gt_labels_list is None:
            concat_gt_labels = None
        else:
            concat_gt_labels = torch.zeros((num_images, max_gt_labels),
                                           dtype=gt_labels_list[0].dtype,
                                           device=device)
            for index_imgs, gt_labels in enumerate(gt_labels_list):
                concat_gt_labels[index_imgs, :gt_nums[index_imgs]] = gt_labels

        return concat_gt_labels

    def _get_targets_concat(self,
                            concat_anchors,
                            concat_valid_flags,
                            concat_gt_bboxes,
                            concat_gt_bboxes_ignore,
                            concat_gt_labels,
                            img_metas,
                            label_channels=1,
                            unmap_outputs=True):
        num_imgs, num_anchors, _ = concat_anchors.size()
        # assign gt and sample concat_anchors
        assign_result = self.assigner.assign(
            concat_anchors, concat_gt_bboxes, concat_gt_bboxes_ignore,
            None if self.sampling else concat_gt_labels,
            concat_bboxes_ignore_mask=concat_valid_flags)

        sampling_result = None
        concat_pos_mask = assign_result.concat_pos_mask
        concat_neg_mask = assign_result.concat_neg_mask
        concat_anchor_gt_indes = assign_result.concat_anchor_gt_indes
        concat_anchor_gt_labels = assign_result.concat_anchor_gt_labels

        concat_anchor_gt_bboxes = torch.zeros(concat_anchors.size(),
                                              dtype=concat_anchors.dtype,
                                              device=concat_anchors.device)
        for index_imgs in range(num_imgs):
            concat_anchor_gt_bboxes[index_imgs] = torch.index_select(concat_gt_bboxes[index_imgs], 0,
                                                                     concat_anchor_gt_indes[index_imgs])

        concat_bbox_targets = torch.zeros_like(concat_anchors)
        concat_bbox_weights = torch.zeros_like(concat_anchors)
        concat_labels = concat_anchors.new_full((num_imgs, num_anchors), self.num_classes, dtype=torch.int)
        concat_label_weights = concat_anchors.new_zeros((num_imgs, num_anchors), dtype=torch.float)

        if not self.reg_decoded_bbox:
            concat_pos_bbox_targets = self.bbox_coder.encode(concat_anchors, concat_anchor_gt_bboxes)
        else:
            concat_pos_bbox_targets = concat_anchor_gt_bboxes

        concat_bbox_targets = set_index(concat_bbox_targets, concat_pos_mask.unsqueeze(2), concat_pos_bbox_targets)
        concat_bbox_weights = set_index(concat_bbox_weights, concat_pos_mask.unsqueeze(2), 1.0)
        if concat_gt_labels is None:
            concat_labels = set_index(concat_labels, concat_pos_mask, 0.0)
        else:
            concat_labels = set_index(concat_labels, concat_pos_mask, concat_anchor_gt_labels)
        if self.train_cfg.pos_weight <= 0:
            concat_label_weights = set_index(concat_label_weights, concat_pos_mask, 1.0)
        else:
            concat_label_weights = set_index(concat_label_weights, concat_pos_mask, self.train_cfg.pos_weight)
        concat_label_weights = set_index(concat_label_weights, concat_neg_mask, 1.0)
        return (concat_labels, concat_label_weights, concat_bbox_targets,
                concat_bbox_weights, concat_pos_mask,
                concat_neg_mask, sampling_result)

    # concat target
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
        assert gt_bboxes_ignore_list is None
        assert unmap_outputs is True
        assert return_sampling_results is False
        assert self.train_cfg.allowed_border < 0
        assert isinstance(self.assigner, AscendMaxIoUAssigner)
        assert isinstance(self.sampler, PseudoSampler)
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        device = anchor_list[0][0].device
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]

        concat_anchor_list = []
        concat_valid_flag_list = []
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            concat_anchor_list.append(torch.cat(anchor_list[i]))
            concat_valid_flag_list.append(torch.cat(valid_flag_list[i]))
        concat_anchors = torch.cat(
            [torch.unsqueeze(anchor, 0) for anchor in concat_anchor_list], 0
        )
        concat_valid_flags = torch.cat(
            [torch.unsqueeze(concat_valid_flag, 0) for concat_valid_flag in concat_valid_flag_list], 0
        )

        gt_nums = [len(gt_bbox) for gt_bbox in gt_bboxes_list]
        max_gt_nums = generate_max_gt_nums(gt_nums)
        concat_gt_bboxes = self._get_concat_gt_bboxes(gt_bboxes_list, num_imgs, gt_nums, device, max_gt_nums)
        concat_gt_bboxes_ignore = self._get_concat_gt_bboxes_ignore(gt_bboxes_ignore_list, num_imgs, gt_nums, device)
        concat_gt_labels = self._get_concat_gt_labels(gt_labels_list, num_imgs, gt_nums, device, max_gt_nums)

        results = self._get_targets_concat(
            concat_anchors,
            concat_valid_flags,
            concat_gt_bboxes,
            concat_gt_bboxes_ignore,
            concat_gt_labels,
            img_metas,
            label_channels=label_channels,
            unmap_outputs=unmap_outputs)

        (concat_labels, concat_label_weights, concat_bbox_targets,
         concat_bbox_weights, concat_pos_mask, concat_neg_mask, sampling_result) = results[:7]
        rest_results = list(results[7:])  # user-added return values

        # sampled anchors of all images
        min_num = torch.ones((num_imgs,), dtype=concat_pos_mask.dtype, device=concat_pos_mask.device)
        num_total_pos = torch.sum(torch.max(torch.sum(concat_pos_mask, dim=1), min_num))
        num_total_neg = torch.sum(torch.max(torch.sum(concat_neg_mask, dim=1), min_num))
        if return_level is True:
            labels_list = images_to_levels(concat_labels, num_level_anchors)
            label_weights_list = images_to_levels(concat_label_weights, num_level_anchors)
            bbox_targets_list = images_to_levels(concat_bbox_targets, num_level_anchors)
            bbox_weights_list = images_to_levels(concat_bbox_weights, num_level_anchors)
            res = (labels_list, label_weights_list, bbox_targets_list,
                   bbox_weights_list, num_total_pos, num_total_neg)
            if return_sampling_results:
                res = res + (sampling_result,)
            for i, r in enumerate(rest_results):  # user-added return values
                rest_results[i] = images_to_levels(r, num_level_anchors)

            return res + tuple(rest_results)
        else:
            res = (concat_labels, concat_label_weights, concat_bbox_targets, concat_bbox_weights, concat_pos_mask,
                   concat_neg_mask, sampling_result, num_total_pos, num_total_neg, concat_anchors)
            return res


# Modified from https://github.com/facebookresearch/detectron2/tree/master/projects/PointRend/point_head/point_head.py  # noqa

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, normal_init

from mmdet.models.builder import HEADS, build_loss
from mmdet.ops import point_sample, rel_roi_point2rel_img_point


@HEADS.register_module()
class PointHead(nn.Module):
    """A mask point head use in PointRend.

    ``PointHead`` use shared multi-layer perceptron (equivalent to
    nn.Conv1d) to predict the logit of input points. The fine-grained feature
    and coarse feature will be concatenate together for predication.

    Args:
        num_fcs (int): Number of fc layers in the head. Default: 3.
        in_channels (int): Number of input channels. Default: 256.
        fc_channels (int): Number of fc channels. Default: 256.
        num_classes (int): Number of classes for logits. Default: 80.
        class_agnostic (bool): Whether use class agnostic classification.
            If so, the output channels of logits will be 1. Default: False.
        coarse_pred_each_layer (bool): Whether concatenate coarse feature with
            the output of each fc layer. Default: True.
        conv_cfg (dict|None): Dictionary to construct and config conv layer.
            Default: dict(type='Conv1d'))
        norm_cfg (dict|None): Dictionary to construct and config norm layer.
            Default: None.
        loss_point (dict): Dictionary to construct and config loss layer of
            point head. Default: dict(type='CrossEntropyLoss', use_mask=True,
            loss_weight=1.0).
    """

    def __init__(self,
                 num_fcs=3,
                 in_channels=256,
                 fc_channels=256,
                 num_classes=80,
                 class_agnostic=False,
                 coarse_pred_each_layer=True,
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=None,
                 loss_point=dict(
                     type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)):
        super(PointHead, self).__init__()
        self.num_fcs = num_fcs
        self.in_channels = in_channels
        self.fc_channles = fc_channels
        self.num_classes = num_classes
        self.class_agnostic = class_agnostic
        self.coarse_pred_each_layer = coarse_pred_each_layer
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.loss_point = build_loss(loss_point)

        fc_in_channels = in_channels + num_classes
        self.fcs = nn.ModuleList()
        for k in range(num_fcs):
            fc = ConvModule(
                fc_in_channels,
                fc_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg)
            self.fcs.append(fc)
            fc_in_channels = fc_channels
            fc_in_channels += num_classes if self.coarse_pred_each_layer else 0

        out_channels = 1 if self.class_agnostic else self.num_classes
        self.fc_logits = nn.Conv1d(
            fc_in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()

    def init_weights(self):
        normal_init(self.fc_logits, std=0.001)

    def forward(self, fine_grained_feats, coarse_feats):
        x = torch.cat([fine_grained_feats, coarse_feats], dim=1)
        for fc in self.fcs:
            x = self.relu(fc(x))
            if self.coarse_pred_each_layer:
                x = torch.cat((x, coarse_feats), dim=1)
        return self.fc_logits(x)

    def get_targets(self, rois, rel_roi_points, sampling_results, gt_masks,
                    cfg):
        num_imgs = len(sampling_results)
        rois_list = []
        rel_roi_points_list = []
        for batch_ind in range(num_imgs):
            inds = (rois[:, 0] == batch_ind)
            rois_list.append(rois[inds])
            rel_roi_points_list.append(rel_roi_points[inds])
        pos_assigned_gt_inds_list = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]
        cfg_list = [cfg for _ in range(num_imgs)]

        point_targets = map(self._get_target_single, rois_list,
                            rel_roi_points_list, pos_assigned_gt_inds_list,
                            gt_masks, cfg_list)
        point_targets = list(point_targets)

        if len(point_targets) > 0:
            point_targets = torch.cat(point_targets)

        return point_targets

    def _get_target_single(self, rois, rel_roi_points, pos_assigned_gt_inds,
                           gt_masks, cfg):
        num_pos = rois.size(0)
        num_points = cfg.num_points
        if num_pos > 0:
            gt_masks_th = (
                gt_masks.to_tensor(rois.dtype, rois.device).index_select(
                    0, pos_assigned_gt_inds))
            gt_masks_th = gt_masks_th.unsqueeze(1)
            rel_img_points = rel_roi_point2rel_img_point(
                rois, rel_roi_points, gt_masks_th.shape[2:])
            point_targets = point_sample(gt_masks_th,
                                         rel_img_points).squeeze(1)
        else:
            point_targets = rois.new_zeros((0, num_points))
        return point_targets

    def loss(self, point_pred, point_targets, labels):
        loss = dict()
        if self.class_agnostic:
            loss_point = self.loss_point(point_pred, point_targets,
                                         torch.zeros_like(labels))
        else:
            loss_point = self.loss_point(point_pred, point_targets, labels)
        loss['loss_point'] = loss_point
        return loss

    def _get_uncertainty(self, mask_pred, labels):
        """Estimate uncertainty based on pred logits

        We estimate uncertainty as L1 distance between 0.0 and the logit
        prediction in 'mask_pred' for the foreground class in `classes`.

        Args:
            mask_pred (Tensor): mask predication logits, shape (R, C, H, W),
                where R is the total number of predicted masks in all images
                and C is the number of foreground classes.

            labels (list[Tensor]): A list of length R that contains either
                predicted or ground truth class for each predicted mask.

        Returns:
            scores (Tensor): Uncertainty scores with the most uncertain
                locations having the highest uncertainty score,
                shape (R, 1, H, W)
        """
        if mask_pred.shape[1] == 1:
            gt_class_logits = mask_pred.clone()
        else:
            inds = torch.arange(mask_pred.shape[0], device=mask_pred.device)
            gt_class_logits = mask_pred[inds, labels].unsqueeze(1)
        return -torch.abs(gt_class_logits)

    def get_roi_rel_points_train(self, mask_pred, labels, cfg):
        """
        Sample points in [0, 1] x [0, 1] coordinate space based on their
        uncertainty. The uncertainties are calculated for each point using
        '_get_uncertainty()' function that takes point's logit prediction as
        input.

        Args:
            mask_pred (Tensor): A tensor of shape (N, C, mask_height,
                mask_width) for class-specific or class-agnostic prediction.
            labels (list): The ground truth class for each instance.
            cfg (dict): Training config of point head.

        Returns:
            point_coords (Tensor): A tensor of shape (N, P, 2) that contains
                the coordinates of P sampled points.
        """
        num_points = cfg.num_points
        oversample_ratio = cfg.oversample_ratio
        importance_sample_ratio = cfg.importance_sample_ratio
        assert oversample_ratio >= 1
        assert 0 <= importance_sample_ratio <= 1
        batch_size = mask_pred.shape[0]
        num_sampled = int(num_points * oversample_ratio)
        point_coords = torch.rand(
            batch_size, num_sampled, 2, device=mask_pred.device)
        point_logits = point_sample(mask_pred, point_coords)
        # It is crucial to calculate uncertainty based on the sampled
        # prediction value for the points. Calculating uncertainties of the
        # coarse predictions first and sampling them for points leads to
        # incorrect results.  To illustrate this: assume uncertainty func(
        # logits)=-abs(logits), a sampled point between two coarse
        # predictions with -1 and 1 logits has 0 logits, and therefore 0
        # uncertainty value. However, if we calculate uncertainties for the
        # coarse predictions first, both will have -1 uncertainty,
        # and sampled point will get -1 uncertainty.
        point_uncertainties = self._get_uncertainty(point_logits, labels)
        num_uncertain_points = int(importance_sample_ratio * num_points)
        num_random_points = num_points - num_uncertain_points
        idx = torch.topk(
            point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
        shift = num_sampled * torch.arange(
            batch_size, dtype=torch.long, device=mask_pred.device)
        idx += shift[:, None]
        point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(
            batch_size, num_uncertain_points, 2)
        if num_random_points > 0:
            rand_roi_coords = torch.rand(
                batch_size, num_random_points, 2, device=mask_pred.device)
            point_coords = torch.cat((point_coords, rand_roi_coords), dim=1)
        return point_coords

    def get_roi_rel_points_test(self, mask_pred, pred_label, cfg):
        """
        Find `num_points` most uncertain points from `uncertainty_map` grid.

        Args:
            mask_pred (Tensor): A tensor of shape (N, C, mask_height,
                mask_width) for class-specific or class-agnostic prediction.
            pred_label (list): The predication class for each instance.
            cfg (dict): Testing config of point head.

        Returns:
            point_indices (Tensor): A tensor of shape (N, P) that contains
                indices from [0, mask_height x mask_width) of the most
                uncertain points.
            point_coords (Tensor): A tensor of shape (N, P, 2) that contains
                [0, 1] x [0, 1] normalized coordinates of the most uncertain
                points from the mask_height x mask_width grid .
            """
        num_points = cfg.subdivision_num_points
        uncertainty_map = self._get_uncertainty(mask_pred, pred_label)
        num_rois, _, mask_height, mask_width = uncertainty_map.shape
        h_step = 1.0 / mask_height
        w_step = 1.0 / mask_width

        uncertainty_map = uncertainty_map.view(num_rois,
                                               mask_height * mask_width)
        num_points = min(mask_height * mask_width, num_points)
        point_indices = uncertainty_map.topk(num_points, dim=1)[1]
        point_coords = uncertainty_map.new_zeros(num_rois, num_points, 2)
        point_coords[:, :, 0] = w_step / 2.0 + (point_indices %
                                                mask_width).float() * w_step
        point_coords[:, :, 1] = h_step / 2.0 + (point_indices //
                                                mask_width).float() * h_step
        return point_indices, point_coords

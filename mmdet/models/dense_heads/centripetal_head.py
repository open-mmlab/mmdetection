# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import ConvModule, normal_init
from mmcv.ops import DeformConv2d

from mmdet.core import multi_apply
from ..builder import HEADS, build_loss
from .corner_head import CornerHead


@HEADS.register_module()
class CentripetalHead(CornerHead):
    """Head of CentripetalNet: Pursuing High-quality Keypoint Pairs for Object
    Detection.

    CentripetalHead inherits from :class:`CornerHead`. It removes the
    embedding branch and adds guiding shift and centripetal shift branches.
    More details can be found in the `paper
    <https://arxiv.org/abs/2003.09119>`_ .

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        num_feat_levels (int): Levels of feature from the previous module. 2
            for HourglassNet-104 and 1 for HourglassNet-52. HourglassNet-104
            outputs the final feature and intermediate supervision feature and
            HourglassNet-52 only outputs the final feature. Default: 2.
        corner_emb_channels (int): Channel of embedding vector. Default: 1.
        train_cfg (dict | None): Training config. Useless in CornerHead,
            but we keep this variable for SingleStageDetector. Default: None.
        test_cfg (dict | None): Testing config of CornerHead. Default: None.
        loss_heatmap (dict | None): Config of corner heatmap loss. Default:
            GaussianFocalLoss.
        loss_embedding (dict | None): Config of corner embedding loss. Default:
            AssociativeEmbeddingLoss.
        loss_offset (dict | None): Config of corner offset loss. Default:
            SmoothL1Loss.
        loss_guiding_shift (dict): Config of guiding shift loss. Default:
            SmoothL1Loss.
        loss_centripetal_shift (dict): Config of centripetal shift loss.
            Default: SmoothL1Loss.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 *args,
                 centripetal_shift_channels=2,
                 guiding_shift_channels=2,
                 feat_adaption_conv_kernel=3,
                 loss_guiding_shift=dict(
                     type='SmoothL1Loss', beta=1.0, loss_weight=0.05),
                 loss_centripetal_shift=dict(
                     type='SmoothL1Loss', beta=1.0, loss_weight=1),
                 init_cfg=None,
                 **kwargs):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
                                 'behavior, init_cfg is not allowed to be set'
        assert centripetal_shift_channels == 2, (
            'CentripetalHead only support centripetal_shift_channels == 2')
        self.centripetal_shift_channels = centripetal_shift_channels
        assert guiding_shift_channels == 2, (
            'CentripetalHead only support guiding_shift_channels == 2')
        self.guiding_shift_channels = guiding_shift_channels
        self.feat_adaption_conv_kernel = feat_adaption_conv_kernel
        super(CentripetalHead, self).__init__(
            *args, init_cfg=init_cfg, **kwargs)
        self.loss_guiding_shift = build_loss(loss_guiding_shift)
        self.loss_centripetal_shift = build_loss(loss_centripetal_shift)

    def _init_centripetal_layers(self):
        """Initialize centripetal layers.

        Including feature adaption deform convs (feat_adaption), deform offset
        prediction convs (dcn_off), guiding shift (guiding_shift) and
        centripetal shift ( centripetal_shift). Each branch has two parts:
        prefix `tl_` for top-left and `br_` for bottom-right.
        """
        self.tl_feat_adaption = nn.ModuleList()
        self.br_feat_adaption = nn.ModuleList()
        self.tl_dcn_offset = nn.ModuleList()
        self.br_dcn_offset = nn.ModuleList()
        self.tl_guiding_shift = nn.ModuleList()
        self.br_guiding_shift = nn.ModuleList()
        self.tl_centripetal_shift = nn.ModuleList()
        self.br_centripetal_shift = nn.ModuleList()

        for _ in range(self.num_feat_levels):
            self.tl_feat_adaption.append(
                DeformConv2d(self.in_channels, self.in_channels,
                             self.feat_adaption_conv_kernel, 1, 1))
            self.br_feat_adaption.append(
                DeformConv2d(self.in_channels, self.in_channels,
                             self.feat_adaption_conv_kernel, 1, 1))

            self.tl_guiding_shift.append(
                self._make_layers(
                    out_channels=self.guiding_shift_channels,
                    in_channels=self.in_channels))
            self.br_guiding_shift.append(
                self._make_layers(
                    out_channels=self.guiding_shift_channels,
                    in_channels=self.in_channels))

            self.tl_dcn_offset.append(
                ConvModule(
                    self.guiding_shift_channels,
                    self.feat_adaption_conv_kernel**2 *
                    self.guiding_shift_channels,
                    1,
                    bias=False,
                    act_cfg=None))
            self.br_dcn_offset.append(
                ConvModule(
                    self.guiding_shift_channels,
                    self.feat_adaption_conv_kernel**2 *
                    self.guiding_shift_channels,
                    1,
                    bias=False,
                    act_cfg=None))

            self.tl_centripetal_shift.append(
                self._make_layers(
                    out_channels=self.centripetal_shift_channels,
                    in_channels=self.in_channels))
            self.br_centripetal_shift.append(
                self._make_layers(
                    out_channels=self.centripetal_shift_channels,
                    in_channels=self.in_channels))

    def _init_layers(self):
        """Initialize layers for CentripetalHead.

        Including two parts: CornerHead layers and CentripetalHead layers
        """
        super()._init_layers()  # using _init_layers in CornerHead
        self._init_centripetal_layers()

    def init_weights(self):
        super(CentripetalHead, self).init_weights()
        for i in range(self.num_feat_levels):
            normal_init(self.tl_feat_adaption[i], std=0.01)
            normal_init(self.br_feat_adaption[i], std=0.01)
            normal_init(self.tl_dcn_offset[i].conv, std=0.1)
            normal_init(self.br_dcn_offset[i].conv, std=0.1)
            _ = [x.conv.reset_parameters() for x in self.tl_guiding_shift[i]]
            _ = [x.conv.reset_parameters() for x in self.br_guiding_shift[i]]
            _ = [
                x.conv.reset_parameters() for x in self.tl_centripetal_shift[i]
            ]
            _ = [
                x.conv.reset_parameters() for x in self.br_centripetal_shift[i]
            ]

    def forward_single(self, x, lvl_ind):
        """Forward feature of a single level.

        Args:
            x (Tensor): Feature of a single level.
            lvl_ind (int): Level index of current feature.

        Returns:
            tuple[Tensor]: A tuple of CentripetalHead's output for current
            feature level. Containing the following Tensors:

                - tl_heat (Tensor): Predicted top-left corner heatmap.
                - br_heat (Tensor): Predicted bottom-right corner heatmap.
                - tl_off (Tensor): Predicted top-left offset heatmap.
                - br_off (Tensor): Predicted bottom-right offset heatmap.
                - tl_guiding_shift (Tensor): Predicted top-left guiding shift
                  heatmap.
                - br_guiding_shift (Tensor): Predicted bottom-right guiding
                  shift heatmap.
                - tl_centripetal_shift (Tensor): Predicted top-left centripetal
                  shift heatmap.
                - br_centripetal_shift (Tensor): Predicted bottom-right
                  centripetal shift heatmap.
        """
        tl_heat, br_heat, _, _, tl_off, br_off, tl_pool, br_pool = super(
        ).forward_single(
            x, lvl_ind, return_pool=True)

        tl_guiding_shift = self.tl_guiding_shift[lvl_ind](tl_pool)
        br_guiding_shift = self.br_guiding_shift[lvl_ind](br_pool)

        tl_dcn_offset = self.tl_dcn_offset[lvl_ind](tl_guiding_shift.detach())
        br_dcn_offset = self.br_dcn_offset[lvl_ind](br_guiding_shift.detach())

        tl_feat_adaption = self.tl_feat_adaption[lvl_ind](tl_pool,
                                                          tl_dcn_offset)
        br_feat_adaption = self.br_feat_adaption[lvl_ind](br_pool,
                                                          br_dcn_offset)

        tl_centripetal_shift = self.tl_centripetal_shift[lvl_ind](
            tl_feat_adaption)
        br_centripetal_shift = self.br_centripetal_shift[lvl_ind](
            br_feat_adaption)

        result_list = [
            tl_heat, br_heat, tl_off, br_off, tl_guiding_shift,
            br_guiding_shift, tl_centripetal_shift, br_centripetal_shift
        ]
        return result_list

    def loss(self,
             tl_heats,
             br_heats,
             tl_offs,
             br_offs,
             tl_guiding_shifts,
             br_guiding_shifts,
             tl_centripetal_shifts,
             br_centripetal_shifts,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            tl_heats (list[Tensor]): Top-left corner heatmaps for each level
                with shape (N, num_classes, H, W).
            br_heats (list[Tensor]): Bottom-right corner heatmaps for each
                level with shape (N, num_classes, H, W).
            tl_offs (list[Tensor]): Top-left corner offsets for each level
                with shape (N, corner_offset_channels, H, W).
            br_offs (list[Tensor]): Bottom-right corner offsets for each level
                with shape (N, corner_offset_channels, H, W).
            tl_guiding_shifts (list[Tensor]): Top-left guiding shifts for each
                level with shape (N, guiding_shift_channels, H, W).
            br_guiding_shifts (list[Tensor]): Bottom-right guiding shifts for
                each level with shape (N, guiding_shift_channels, H, W).
            tl_centripetal_shifts (list[Tensor]): Top-left centripetal shifts
                for each level with shape (N, centripetal_shift_channels, H,
                W).
            br_centripetal_shifts (list[Tensor]): Bottom-right centripetal
                shifts for each level with shape (N,
                centripetal_shift_channels, H, W).
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [left, top, right, bottom] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (list[Tensor] | None): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components. Containing the
            following losses:

                - det_loss (list[Tensor]): Corner keypoint losses of all
                  feature levels.
                - off_loss (list[Tensor]): Corner offset losses of all feature
                  levels.
                - guiding_loss (list[Tensor]): Guiding shift losses of all
                  feature levels.
                - centripetal_loss (list[Tensor]): Centripetal shift losses of
                  all feature levels.
        """
        targets = self.get_targets(
            gt_bboxes,
            gt_labels,
            tl_heats[-1].shape,
            img_metas[0]['pad_shape'],
            with_corner_emb=self.with_corner_emb,
            with_guiding_shift=True,
            with_centripetal_shift=True)
        mlvl_targets = [targets for _ in range(self.num_feat_levels)]
        [det_losses, off_losses, guiding_losses, centripetal_losses
         ] = multi_apply(self.loss_single, tl_heats, br_heats, tl_offs,
                         br_offs, tl_guiding_shifts, br_guiding_shifts,
                         tl_centripetal_shifts, br_centripetal_shifts,
                         mlvl_targets)
        loss_dict = dict(
            det_loss=det_losses,
            off_loss=off_losses,
            guiding_loss=guiding_losses,
            centripetal_loss=centripetal_losses)
        return loss_dict

    def loss_single(self, tl_hmp, br_hmp, tl_off, br_off, tl_guiding_shift,
                    br_guiding_shift, tl_centripetal_shift,
                    br_centripetal_shift, targets):
        """Compute losses for single level.

        Args:
            tl_hmp (Tensor): Top-left corner heatmap for current level with
                shape (N, num_classes, H, W).
            br_hmp (Tensor): Bottom-right corner heatmap for current level with
                shape (N, num_classes, H, W).
            tl_off (Tensor): Top-left corner offset for current level with
                shape (N, corner_offset_channels, H, W).
            br_off (Tensor): Bottom-right corner offset for current level with
                shape (N, corner_offset_channels, H, W).
            tl_guiding_shift (Tensor): Top-left guiding shift for current level
                with shape (N, guiding_shift_channels, H, W).
            br_guiding_shift (Tensor): Bottom-right guiding shift for current
                level with shape (N, guiding_shift_channels, H, W).
            tl_centripetal_shift (Tensor): Top-left centripetal shift for
                current level with shape (N, centripetal_shift_channels, H, W).
            br_centripetal_shift (Tensor): Bottom-right centripetal shift for
                current level with shape (N, centripetal_shift_channels, H, W).
            targets (dict): Corner target generated by `get_targets`.

        Returns:
            tuple[torch.Tensor]: Losses of the head's differnet branches
            containing the following losses:

                - det_loss (Tensor): Corner keypoint loss.
                - off_loss (Tensor): Corner offset loss.
                - guiding_loss (Tensor): Guiding shift loss.
                - centripetal_loss (Tensor): Centripetal shift loss.
        """
        targets['corner_embedding'] = None

        det_loss, _, _, off_loss = super().loss_single(tl_hmp, br_hmp, None,
                                                       None, tl_off, br_off,
                                                       targets)

        gt_tl_guiding_shift = targets['topleft_guiding_shift']
        gt_br_guiding_shift = targets['bottomright_guiding_shift']
        gt_tl_centripetal_shift = targets['topleft_centripetal_shift']
        gt_br_centripetal_shift = targets['bottomright_centripetal_shift']

        gt_tl_heatmap = targets['topleft_heatmap']
        gt_br_heatmap = targets['bottomright_heatmap']
        # We only compute the offset loss at the real corner position.
        # The value of real corner would be 1 in heatmap ground truth.
        # The mask is computed in class agnostic mode and its shape is
        # batch * 1 * width * height.
        tl_mask = gt_tl_heatmap.eq(1).sum(1).gt(0).unsqueeze(1).type_as(
            gt_tl_heatmap)
        br_mask = gt_br_heatmap.eq(1).sum(1).gt(0).unsqueeze(1).type_as(
            gt_br_heatmap)

        # Guiding shift loss
        tl_guiding_loss = self.loss_guiding_shift(
            tl_guiding_shift,
            gt_tl_guiding_shift,
            tl_mask,
            avg_factor=tl_mask.sum())
        br_guiding_loss = self.loss_guiding_shift(
            br_guiding_shift,
            gt_br_guiding_shift,
            br_mask,
            avg_factor=br_mask.sum())
        guiding_loss = (tl_guiding_loss + br_guiding_loss) / 2.0
        # Centripetal shift loss
        tl_centripetal_loss = self.loss_centripetal_shift(
            tl_centripetal_shift,
            gt_tl_centripetal_shift,
            tl_mask,
            avg_factor=tl_mask.sum())
        br_centripetal_loss = self.loss_centripetal_shift(
            br_centripetal_shift,
            gt_br_centripetal_shift,
            br_mask,
            avg_factor=br_mask.sum())
        centripetal_loss = (tl_centripetal_loss + br_centripetal_loss) / 2.0

        return det_loss, off_loss, guiding_loss, centripetal_loss

    def get_bboxes(self,
                   tl_heats,
                   br_heats,
                   tl_offs,
                   br_offs,
                   tl_guiding_shifts,
                   br_guiding_shifts,
                   tl_centripetal_shifts,
                   br_centripetal_shifts,
                   img_metas,
                   rescale=False,
                   with_nms=True):
        """Transform network output for a batch into bbox predictions.

        Args:
            tl_heats (list[Tensor]): Top-left corner heatmaps for each level
                with shape (N, num_classes, H, W).
            br_heats (list[Tensor]): Bottom-right corner heatmaps for each
                level with shape (N, num_classes, H, W).
            tl_offs (list[Tensor]): Top-left corner offsets for each level
                with shape (N, corner_offset_channels, H, W).
            br_offs (list[Tensor]): Bottom-right corner offsets for each level
                with shape (N, corner_offset_channels, H, W).
            tl_guiding_shifts (list[Tensor]): Top-left guiding shifts for each
                level with shape (N, guiding_shift_channels, H, W). Useless in
                this function, we keep this arg because it's the raw output
                from CentripetalHead.
            br_guiding_shifts (list[Tensor]): Bottom-right guiding shifts for
                each level with shape (N, guiding_shift_channels, H, W).
                Useless in this function, we keep this arg because it's the
                raw output from CentripetalHead.
            tl_centripetal_shifts (list[Tensor]): Top-left centripetal shifts
                for each level with shape (N, centripetal_shift_channels, H,
                W).
            br_centripetal_shifts (list[Tensor]): Bottom-right centripetal
                shifts for each level with shape (N,
                centripetal_shift_channels, H, W).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.
        """
        assert tl_heats[-1].shape[0] == br_heats[-1].shape[0] == len(img_metas)
        result_list = []
        for img_id in range(len(img_metas)):
            result_list.append(
                self._get_bboxes_single(
                    tl_heats[-1][img_id:img_id + 1, :],
                    br_heats[-1][img_id:img_id + 1, :],
                    tl_offs[-1][img_id:img_id + 1, :],
                    br_offs[-1][img_id:img_id + 1, :],
                    img_metas[img_id],
                    tl_emb=None,
                    br_emb=None,
                    tl_centripetal_shift=tl_centripetal_shifts[-1][
                        img_id:img_id + 1, :],
                    br_centripetal_shift=br_centripetal_shifts[-1][
                        img_id:img_id + 1, :],
                    rescale=rescale,
                    with_nms=with_nms))

        return result_list

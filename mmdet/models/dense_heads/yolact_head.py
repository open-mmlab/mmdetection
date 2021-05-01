import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, ModuleList, force_fp32

from mmdet.core import build_sampler, fast_nms, images_to_levels, multi_apply
from ..builder import HEADS, build_loss
from .anchor_head import AnchorHead


@HEADS.register_module()
class YOLACTHead(AnchorHead):
    """YOLACT box head used in https://arxiv.org/abs/1904.02689.

    Note that YOLACT head is a light version of RetinaNet head.
    Four differences are described as follows:

    1. YOLACT box head has three-times fewer anchors.
    2. YOLACT box head shares the convs for box and cls branches.
    3. YOLACT box head uses OHEM instead of Focal loss.
    4. YOLACT box head predicts a set of mask coefficients for each box.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        anchor_generator (dict): Config dict for anchor generator
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        num_head_convs (int): Number of the conv layers shared by
            box and cls branches.
        num_protos (int): Number of the mask coefficients.
        use_ohem (bool): If true, ``loss_single_OHEM`` will be used for
            cls loss calculation. If false, ``loss_single`` will be used.
        conv_cfg (dict): Dictionary to construct and config conv layer.
        norm_cfg (dict): Dictionary to construct and config norm layer.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 anchor_generator=dict(
                     type='AnchorGenerator',
                     octave_base_scale=3,
                     scales_per_octave=1,
                     ratios=[0.5, 1.0, 2.0],
                     strides=[8, 16, 32, 64, 128]),
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     reduction='none',
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='SmoothL1Loss', beta=1.0, loss_weight=1.5),
                 num_head_convs=1,
                 num_protos=32,
                 use_ohem=True,
                 conv_cfg=None,
                 norm_cfg=None,
                 init_cfg=dict(
                     type='Xavier',
                     distribution='uniform',
                     bias=0,
                     layer='Conv2d'),
                 **kwargs):
        self.num_head_convs = num_head_convs
        self.num_protos = num_protos
        self.use_ohem = use_ohem
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        super(YOLACTHead, self).__init__(
            num_classes,
            in_channels,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            anchor_generator=anchor_generator,
            init_cfg=init_cfg,
            **kwargs)
        if self.use_ohem:
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
            self.sampling = False

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.head_convs = ModuleList()
        for i in range(self.num_head_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.head_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.conv_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            3,
            padding=1)
        self.conv_reg = nn.Conv2d(
            self.feat_channels, self.num_anchors * 4, 3, padding=1)
        self.conv_coeff = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.num_protos,
            3,
            padding=1)

    def forward_single(self, x):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level \
                    the channels number is num_anchors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale \
                    level, the channels number is num_anchors * 4.
                coeff_pred (Tensor): Mask coefficients for a single scale \
                    level, the channels number is num_anchors * num_protos.
        """
        for head_conv in self.head_convs:
            x = head_conv(x)
        cls_score = self.conv_cls(x)
        bbox_pred = self.conv_reg(x)
        coeff_pred = self.conv_coeff(x).tanh()
        return cls_score, bbox_pred, coeff_pred

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """A combination of the func:``AnchorHead.loss`` and
        func:``SSDHead.loss``.

        When ``self.use_ohem == True``, it functions like ``SSDHead.loss``,
        otherwise, it follows ``AnchorHead.loss``. Besides, it additionally
        returns ``sampling_results``.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss. Default: None

        Returns:
            tuple:
                dict[str, Tensor]: A dictionary of loss components.
                List[:obj:``SamplingResult``]: Sampler results for each image.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            unmap_outputs=not self.use_ohem,
            return_sampling_results=True)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg, sampling_results) = cls_reg_targets

        if self.use_ohem:
            num_images = len(img_metas)
            all_cls_scores = torch.cat([
                s.permute(0, 2, 3, 1).reshape(
                    num_images, -1, self.cls_out_channels) for s in cls_scores
            ], 1)
            all_labels = torch.cat(labels_list, -1).view(num_images, -1)
            all_label_weights = torch.cat(label_weights_list,
                                          -1).view(num_images, -1)
            all_bbox_preds = torch.cat([
                b.permute(0, 2, 3, 1).reshape(num_images, -1, 4)
                for b in bbox_preds
            ], -2)
            all_bbox_targets = torch.cat(bbox_targets_list,
                                         -2).view(num_images, -1, 4)
            all_bbox_weights = torch.cat(bbox_weights_list,
                                         -2).view(num_images, -1, 4)

            # concat all level anchors to a single tensor
            all_anchors = []
            for i in range(num_images):
                all_anchors.append(torch.cat(anchor_list[i]))

            # check NaN and Inf
            assert torch.isfinite(all_cls_scores).all().item(), \
                'classification scores become infinite or NaN!'
            assert torch.isfinite(all_bbox_preds).all().item(), \
                'bbox predications become infinite or NaN!'

            losses_cls, losses_bbox = multi_apply(
                self.loss_single_OHEM,
                all_cls_scores,
                all_bbox_preds,
                all_anchors,
                all_labels,
                all_label_weights,
                all_bbox_targets,
                all_bbox_weights,
                num_total_samples=num_total_pos)
        else:
            num_total_samples = (
                num_total_pos +
                num_total_neg if self.sampling else num_total_pos)

            # anchor number of multi levels
            num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
            # concat all level anchors and flags to a single tensor
            concat_anchor_list = []
            for i in range(len(anchor_list)):
                concat_anchor_list.append(torch.cat(anchor_list[i]))
            all_anchor_list = images_to_levels(concat_anchor_list,
                                               num_level_anchors)
            losses_cls, losses_bbox = multi_apply(
                self.loss_single,
                cls_scores,
                bbox_preds,
                all_anchor_list,
                labels_list,
                label_weights_list,
                bbox_targets_list,
                bbox_weights_list,
                num_total_samples=num_total_samples)

        return dict(
            loss_cls=losses_cls, loss_bbox=losses_bbox), sampling_results

    def loss_single_OHEM(self, cls_score, bbox_pred, anchors, labels,
                         label_weights, bbox_targets, bbox_weights,
                         num_total_samples):
        """"See func:``SSDHead.loss``."""
        loss_cls_all = self.loss_cls(cls_score, labels, label_weights)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        pos_inds = ((labels >= 0) & (labels < self.num_classes)).nonzero(
            as_tuple=False).reshape(-1)
        neg_inds = (labels == self.num_classes).nonzero(
            as_tuple=False).view(-1)

        num_pos_samples = pos_inds.size(0)
        if num_pos_samples == 0:
            num_neg_samples = neg_inds.size(0)
        else:
            num_neg_samples = self.train_cfg.neg_pos_ratio * num_pos_samples
            if num_neg_samples > neg_inds.size(0):
                num_neg_samples = neg_inds.size(0)
        topk_loss_cls_neg, _ = loss_cls_all[neg_inds].topk(num_neg_samples)
        loss_cls_pos = loss_cls_all[pos_inds].sum()
        loss_cls_neg = topk_loss_cls_neg.sum()
        loss_cls = (loss_cls_pos + loss_cls_neg) / num_total_samples
        if self.reg_decoded_bbox:
            # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
            # is applied directly on the decoded bounding boxes, it
            # decodes the already encoded coordinates to absolute format.
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)
        loss_bbox = self.loss_bbox(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_samples)
        return loss_cls[None], loss_bbox

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'coeff_preds'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   coeff_preds,
                   img_metas,
                   cfg=None,
                   rescale=False):
        """"Similiar to func:``AnchorHead.get_bboxes``, but additionally
        processes coeff_preds.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                with shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            coeff_preds (list[Tensor]): Mask coefficients for each scale
                level with shape (N, num_anchors * num_protos, H, W)
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Returns:
            list[tuple[Tensor, Tensor, Tensor]]: Each item in result_list is
                a 3-tuple. The first item is an (n, 5) tensor, where the
                first 4 columns are bounding box positions
                (tl_x, tl_y, br_x, br_y) and the 5-th column is a score
                between 0 and 1. The second item is an (n,) tensor where each
                item is the predicted class label of the corresponding box.
                The third item is an (n, num_protos) tensor where each item
                is the predicted mask coefficients of instance inside the
                corresponding box.
        """
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes, device=device)

        det_bboxes = []
        det_labels = []
        det_coeffs = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            coeff_pred_list = [
                coeff_preds[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            bbox_res = self._get_bboxes_single(cls_score_list, bbox_pred_list,
                                               coeff_pred_list, mlvl_anchors,
                                               img_shape, scale_factor, cfg,
                                               rescale)
            det_bboxes.append(bbox_res[0])
            det_labels.append(bbox_res[1])
            det_coeffs.append(bbox_res[2])
        return det_bboxes, det_labels, det_coeffs

    def _get_bboxes_single(self,
                           cls_score_list,
                           bbox_pred_list,
                           coeff_preds_list,
                           mlvl_anchors,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False):
        """"Similiar to func:``AnchorHead._get_bboxes_single``, but
        additionally processes coeff_preds_list and uses fast NMS instead of
        traditional NMS.

        Args:
            cls_score_list (list[Tensor]): Box scores for a single scale level
                Has shape (num_anchors * num_classes, H, W).
            bbox_pred_list (list[Tensor]): Box energies / deltas for a single
                scale level with shape (num_anchors * 4, H, W).
            coeff_preds_list (list[Tensor]): Mask coefficients for a single
                scale level with shape (num_anchors * num_protos, H, W).
            mlvl_anchors (list[Tensor]): Box reference for a single scale level
                with shape (num_total_anchors, 4).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (ndarray): Scale factor of the image arange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.

        Returns:
            tuple[Tensor, Tensor, Tensor]: The first item is an (n, 5) tensor,
                where the first 4 columns are bounding box positions
                (tl_x, tl_y, br_x, br_y) and the 5-th column is a score between
                0 and 1. The second item is an (n,) tensor where each item is
                the predicted class label of the corresponding box. The third
                item is an (n, num_protos) tensor where each item is the
                predicted mask coefficients of instance inside the
                corresponding box.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_score_list) == len(bbox_pred_list) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_coeffs = []
        for cls_score, bbox_pred, coeff_pred, anchors in \
                zip(cls_score_list, bbox_pred_list,
                    coeff_preds_list, mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            coeff_pred = coeff_pred.permute(1, 2,
                                            0).reshape(-1, self.num_protos)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                # Get maximum scores for foreground classes.
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)
                else:
                    # remind that we set FG labels to [0, num_class-1]
                    # since mmdet v2.0
                    # BG cat_id: num_class
                    max_scores, _ = scores[:, :-1].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                coeff_pred = coeff_pred[topk_inds, :]
            bboxes = self.bbox_coder.decode(
                anchors, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_coeffs.append(coeff_pred)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        mlvl_coeffs = torch.cat(mlvl_coeffs)
        if self.use_sigmoid_cls:
            # Add a dummy background class to the backend when using sigmoid
            # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
            # BG cat_id: num_class
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        det_bboxes, det_labels, det_coeffs = fast_nms(mlvl_bboxes, mlvl_scores,
                                                      mlvl_coeffs,
                                                      cfg.score_thr,
                                                      cfg.iou_thr, cfg.top_k,
                                                      cfg.max_per_img)
        return det_bboxes, det_labels, det_coeffs


@HEADS.register_module()
class YOLACTSegmHead(BaseModule):
    """YOLACT segmentation head used in https://arxiv.org/abs/1904.02689.

    Apply a semantic segmentation loss on feature space using layers that are
    only evaluated during training to increase performance with no speed
    penalty.

    Args:
        in_channels (int): Number of channels in the input feature map.
        num_classes (int): Number of categories excluding the background
            category.
        loss_segm (dict): Config of semantic segmentation loss.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 num_classes,
                 in_channels=256,
                 loss_segm=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 init_cfg=dict(
                     type='Xavier',
                     distribution='uniform',
                     override=dict(name='segm_conv'))):
        super(YOLACTSegmHead, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.loss_segm = build_loss(loss_segm)
        self._init_layers()
        self.fp16_enabled = False

    def _init_layers(self):
        """Initialize layers of the head."""
        self.segm_conv = nn.Conv2d(
            self.in_channels, self.num_classes, kernel_size=1)

    def forward(self, x):
        """Forward feature from the upstream network.

        Args:
            x (Tensor): Feature from the upstream network, which is
                a 4D-tensor.

        Returns:
            Tensor: Predicted semantic segmentation map with shape
                (N, num_classes, H, W).
        """
        return self.segm_conv(x)

    @force_fp32(apply_to=('segm_pred', ))
    def loss(self, segm_pred, gt_masks, gt_labels):
        """Compute loss of the head.

        Args:
            segm_pred (list[Tensor]): Predicted semantic segmentation map
                with shape (N, num_classes, H, W).
            gt_masks (list[Tensor]): Ground truth masks for each image with
                the same shape of the input image.
            gt_labels (list[Tensor]): Class indices corresponding to each box.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        loss_segm = []
        num_imgs, num_classes, mask_h, mask_w = segm_pred.size()
        for idx in range(num_imgs):
            cur_segm_pred = segm_pred[idx]
            cur_gt_masks = gt_masks[idx].float()
            cur_gt_labels = gt_labels[idx]
            segm_targets = self.get_targets(cur_segm_pred, cur_gt_masks,
                                            cur_gt_labels)
            if segm_targets is None:
                loss = self.loss_segm(cur_segm_pred,
                                      torch.zeros_like(cur_segm_pred),
                                      torch.zeros_like(cur_segm_pred))
            else:
                loss = self.loss_segm(
                    cur_segm_pred,
                    segm_targets,
                    avg_factor=num_imgs * mask_h * mask_w)
            loss_segm.append(loss)
        return dict(loss_segm=loss_segm)

    def get_targets(self, segm_pred, gt_masks, gt_labels):
        """Compute semantic segmentation targets for each image.

        Args:
            segm_pred (Tensor): Predicted semantic segmentation map
                with shape (num_classes, H, W).
            gt_masks (Tensor): Ground truth masks for each image with
                the same shape of the input image.
            gt_labels (Tensor): Class indices corresponding to each box.

        Returns:
            Tensor: Semantic segmentation targets with shape
                (num_classes, H, W).
        """
        if gt_masks.size(0) == 0:
            return None
        num_classes, mask_h, mask_w = segm_pred.size()
        with torch.no_grad():
            downsampled_masks = F.interpolate(
                gt_masks.unsqueeze(0), (mask_h, mask_w),
                mode='bilinear',
                align_corners=False).squeeze(0)
            downsampled_masks = downsampled_masks.gt(0.5).float()
            segm_targets = torch.zeros_like(segm_pred, requires_grad=False)
            for obj_idx in range(downsampled_masks.size(0)):
                segm_targets[gt_labels[obj_idx] - 1] = torch.max(
                    segm_targets[gt_labels[obj_idx] - 1],
                    downsampled_masks[obj_idx])
            return segm_targets


@HEADS.register_module()
class YOLACTProtonet(BaseModule):
    """YOLACT mask head used in https://arxiv.org/abs/1904.02689.

    This head outputs the mask prototypes for YOLACT.

    Args:
        in_channels (int): Number of channels in the input feature map.
        proto_channels (tuple[int]): Output channels of protonet convs.
        proto_kernel_sizes (tuple[int]): Kernel sizes of protonet convs.
        include_last_relu (Bool): If keep the last relu of protonet.
        num_protos (int): Number of prototypes.
        num_classes (int): Number of categories excluding the background
            category.
        loss_mask_weight (float): Reweight the mask loss by this factor.
        max_masks_to_train (int): Maximum number of masks to train for
            each image.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 num_classes,
                 in_channels=256,
                 proto_channels=(256, 256, 256, None, 256, 32),
                 proto_kernel_sizes=(3, 3, 3, -2, 3, 1),
                 include_last_relu=True,
                 num_protos=32,
                 loss_mask_weight=1.0,
                 max_masks_to_train=100,
                 init_cfg=dict(
                     type='Xavier',
                     distribution='uniform',
                     override=dict(name='protonet'))):
        super(YOLACTProtonet, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.proto_channels = proto_channels
        self.proto_kernel_sizes = proto_kernel_sizes
        self.include_last_relu = include_last_relu
        self.protonet = self._init_layers()

        self.loss_mask_weight = loss_mask_weight
        self.num_protos = num_protos
        self.num_classes = num_classes
        self.max_masks_to_train = max_masks_to_train
        self.fp16_enabled = False

    def _init_layers(self):
        """A helper function to take a config setting and turn it into a
        network."""
        # Possible patterns:
        # ( 256, 3) -> conv
        # ( 256,-2) -> deconv
        # (None,-2) -> bilinear interpolate
        in_channels = self.in_channels
        protonets = ModuleList()
        for num_channels, kernel_size in zip(self.proto_channels,
                                             self.proto_kernel_sizes):
            if kernel_size > 0:
                layer = nn.Conv2d(
                    in_channels,
                    num_channels,
                    kernel_size,
                    padding=kernel_size // 2)
            else:
                if num_channels is None:
                    layer = InterpolateModule(
                        scale_factor=-kernel_size,
                        mode='bilinear',
                        align_corners=False)
                else:
                    layer = nn.ConvTranspose2d(
                        in_channels,
                        num_channels,
                        -kernel_size,
                        padding=kernel_size // 2)
            protonets.append(layer)
            protonets.append(nn.ReLU(inplace=True))
            in_channels = num_channels if num_channels is not None \
                else in_channels
        if not self.include_last_relu:
            protonets = protonets[:-1]
        return nn.Sequential(*protonets)

    def forward(self, x, coeff_pred, bboxes, img_meta, sampling_results=None):
        """Forward feature from the upstream network to get prototypes and
        linearly combine the prototypes, using masks coefficients, into
        instance masks. Finally, crop the instance masks with given bboxes.

        Args:
            x (Tensor): Feature from the upstream network, which is
                a 4D-tensor.
            coeff_pred (list[Tensor]): Mask coefficients for each scale
                level with shape (N, num_anchors * num_protos, H, W).
            bboxes (list[Tensor]): Box used for cropping with shape
                (N, num_anchors * 4, H, W). During training, they are
                ground truth boxes. During testing, they are predicted
                boxes.
            img_meta (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            sampling_results (List[:obj:``SamplingResult``]): Sampler results
                for each image.

        Returns:
            list[Tensor]: Predicted instance segmentation masks.
        """
        prototypes = self.protonet(x)
        prototypes = prototypes.permute(0, 2, 3, 1).contiguous()

        num_imgs = x.size(0)
        # Training state
        if self.training:
            coeff_pred_list = []
            for coeff_pred_per_level in coeff_pred:
                coeff_pred_per_level = \
                    coeff_pred_per_level.permute(
                        0, 2, 3, 1).reshape(num_imgs, -1, self.num_protos)
                coeff_pred_list.append(coeff_pred_per_level)
            coeff_pred = torch.cat(coeff_pred_list, dim=1)

        mask_pred_list = []
        for idx in range(num_imgs):
            cur_prototypes = prototypes[idx]
            cur_coeff_pred = coeff_pred[idx]
            cur_bboxes = bboxes[idx]
            cur_img_meta = img_meta[idx]

            # Testing state
            if not self.training:
                bboxes_for_cropping = cur_bboxes
            else:
                cur_sampling_results = sampling_results[idx]
                pos_assigned_gt_inds = \
                    cur_sampling_results.pos_assigned_gt_inds
                bboxes_for_cropping = cur_bboxes[pos_assigned_gt_inds].clone()
                pos_inds = cur_sampling_results.pos_inds
                cur_coeff_pred = cur_coeff_pred[pos_inds]

            # Linearly combine the prototypes with the mask coefficients
            mask_pred = cur_prototypes @ cur_coeff_pred.t()
            mask_pred = torch.sigmoid(mask_pred)

            h, w = cur_img_meta['img_shape'][:2]
            bboxes_for_cropping[:, 0] /= w
            bboxes_for_cropping[:, 1] /= h
            bboxes_for_cropping[:, 2] /= w
            bboxes_for_cropping[:, 3] /= h

            mask_pred = self.crop(mask_pred, bboxes_for_cropping)
            mask_pred = mask_pred.permute(2, 0, 1).contiguous()
            mask_pred_list.append(mask_pred)
        return mask_pred_list

    @force_fp32(apply_to=('mask_pred', ))
    def loss(self, mask_pred, gt_masks, gt_bboxes, img_meta, sampling_results):
        """Compute loss of the head.

        Args:
            mask_pred (list[Tensor]): Predicted prototypes with shape
                (num_classes, H, W).
            gt_masks (list[Tensor]): Ground truth masks for each image with
                the same shape of the input image.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            img_meta (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            sampling_results (List[:obj:``SamplingResult``]): Sampler results
                for each image.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        loss_mask = []
        num_imgs = len(mask_pred)
        total_pos = 0
        for idx in range(num_imgs):
            cur_mask_pred = mask_pred[idx]
            cur_gt_masks = gt_masks[idx].float()
            cur_gt_bboxes = gt_bboxes[idx]
            cur_img_meta = img_meta[idx]
            cur_sampling_results = sampling_results[idx]

            pos_assigned_gt_inds = cur_sampling_results.pos_assigned_gt_inds
            num_pos = pos_assigned_gt_inds.size(0)
            # Since we're producing (near) full image masks,
            # it'd take too much vram to backprop on every single mask.
            # Thus we select only a subset.
            if num_pos > self.max_masks_to_train:
                perm = torch.randperm(num_pos)
                select = perm[:self.max_masks_to_train]
                cur_mask_pred = cur_mask_pred[select]
                pos_assigned_gt_inds = pos_assigned_gt_inds[select]
                num_pos = self.max_masks_to_train
            total_pos += num_pos

            gt_bboxes_for_reweight = cur_gt_bboxes[pos_assigned_gt_inds]

            mask_targets = self.get_targets(cur_mask_pred, cur_gt_masks,
                                            pos_assigned_gt_inds)
            if num_pos == 0:
                loss = cur_mask_pred.sum() * 0.
            elif mask_targets is None:
                loss = F.binary_cross_entropy(cur_mask_pred,
                                              torch.zeros_like(cur_mask_pred),
                                              torch.zeros_like(cur_mask_pred))
            else:
                cur_mask_pred = torch.clamp(cur_mask_pred, 0, 1)
                loss = F.binary_cross_entropy(
                    cur_mask_pred, mask_targets,
                    reduction='none') * self.loss_mask_weight

                h, w = cur_img_meta['img_shape'][:2]
                gt_bboxes_width = (gt_bboxes_for_reweight[:, 2] -
                                   gt_bboxes_for_reweight[:, 0]) / w
                gt_bboxes_height = (gt_bboxes_for_reweight[:, 3] -
                                    gt_bboxes_for_reweight[:, 1]) / h
                loss = loss.mean(dim=(1,
                                      2)) / gt_bboxes_width / gt_bboxes_height
                loss = torch.sum(loss)
            loss_mask.append(loss)

        if total_pos == 0:
            total_pos += 1  # avoid nan
        loss_mask = [x / total_pos for x in loss_mask]

        return dict(loss_mask=loss_mask)

    def get_targets(self, mask_pred, gt_masks, pos_assigned_gt_inds):
        """Compute instance segmentation targets for each image.

        Args:
            mask_pred (Tensor): Predicted prototypes with shape
                (num_classes, H, W).
            gt_masks (Tensor): Ground truth masks for each image with
                the same shape of the input image.
            pos_assigned_gt_inds (Tensor): GT indices of the corresponding
                positive samples.
        Returns:
            Tensor: Instance segmentation targets with shape
                (num_instances, H, W).
        """
        if gt_masks.size(0) == 0:
            return None
        mask_h, mask_w = mask_pred.shape[-2:]
        gt_masks = F.interpolate(
            gt_masks.unsqueeze(0), (mask_h, mask_w),
            mode='bilinear',
            align_corners=False).squeeze(0)
        gt_masks = gt_masks.gt(0.5).float()
        mask_targets = gt_masks[pos_assigned_gt_inds]
        return mask_targets

    def get_seg_masks(self, mask_pred, label_pred, img_meta, rescale):
        """Resize, binarize, and format the instance mask predictions.

        Args:
            mask_pred (Tensor): shape (N, H, W).
            label_pred (Tensor): shape (N, ).
            img_meta (dict): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            rescale (bool): If rescale is False, then returned masks will
                fit the scale of imgs[0].
        Returns:
            list[ndarray]: Mask predictions grouped by their predicted classes.
        """
        ori_shape = img_meta['ori_shape']
        scale_factor = img_meta['scale_factor']
        if rescale:
            img_h, img_w = ori_shape[:2]
        else:
            img_h = np.round(ori_shape[0] * scale_factor[1]).astype(np.int32)
            img_w = np.round(ori_shape[1] * scale_factor[0]).astype(np.int32)

        cls_segms = [[] for _ in range(self.num_classes)]
        if mask_pred.size(0) == 0:
            return cls_segms

        mask_pred = F.interpolate(
            mask_pred.unsqueeze(0), (img_h, img_w),
            mode='bilinear',
            align_corners=False).squeeze(0) > 0.5
        mask_pred = mask_pred.cpu().numpy().astype(np.uint8)

        for m, l in zip(mask_pred, label_pred):
            cls_segms[l].append(m)
        return cls_segms

    def crop(self, masks, boxes, padding=1):
        """Crop predicted masks by zeroing out everything not in the predicted
        bbox.

        Args:
            masks (Tensor): shape [H, W, N].
            boxes (Tensor): bbox coords in relative point form with
                shape [N, 4].

        Return:
            Tensor: The cropped masks.
        """
        h, w, n = masks.size()
        x1, x2 = self.sanitize_coordinates(
            boxes[:, 0], boxes[:, 2], w, padding, cast=False)
        y1, y2 = self.sanitize_coordinates(
            boxes[:, 1], boxes[:, 3], h, padding, cast=False)

        rows = torch.arange(
            w, device=masks.device, dtype=x1.dtype).view(1, -1,
                                                         1).expand(h, w, n)
        cols = torch.arange(
            h, device=masks.device, dtype=x1.dtype).view(-1, 1,
                                                         1).expand(h, w, n)

        masks_left = rows >= x1.view(1, 1, -1)
        masks_right = rows < x2.view(1, 1, -1)
        masks_up = cols >= y1.view(1, 1, -1)
        masks_down = cols < y2.view(1, 1, -1)

        crop_mask = masks_left * masks_right * masks_up * masks_down

        return masks * crop_mask.float()

    def sanitize_coordinates(self, x1, x2, img_size, padding=0, cast=True):
        """Sanitizes the input coordinates so that x1 < x2, x1 != x2, x1 >= 0,
        and x2 <= image_size. Also converts from relative to absolute
        coordinates and casts the results to long tensors.

        Warning: this does things in-place behind the scenes so
        copy if necessary.

        Args:
            _x1 (Tensor): shape (N, ).
            _x2 (Tensor): shape (N, ).
            img_size (int): Size of the input image.
            padding (int): x1 >= padding, x2 <= image_size-padding.
            cast (bool): If cast is false, the result won't be cast to longs.

        Returns:
            tuple:
                x1 (Tensor): Sanitized _x1.
                x2 (Tensor): Sanitized _x2.
        """
        x1 = x1 * img_size
        x2 = x2 * img_size
        if cast:
            x1 = x1.long()
            x2 = x2.long()
        x1 = torch.min(x1, x2)
        x2 = torch.max(x1, x2)
        x1 = torch.clamp(x1 - padding, min=0)
        x2 = torch.clamp(x2 + padding, max=img_size)
        return x1, x2


class InterpolateModule(BaseModule):
    """This is a module version of F.interpolate.

    Any arguments you give it just get passed along for the ride.
    """

    def __init__(self, *args, init_cfg=None, **kwargs):
        super().__init__(init_cfg)

        self.args = args
        self.kwargs = kwargs

    def forward(self, x):
        """Forward features from the upstream network."""
        return F.interpolate(x, *self.args, **self.kwargs)

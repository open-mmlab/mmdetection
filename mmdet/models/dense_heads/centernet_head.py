import torch
import torch.nn as nn
from mmcv.cnn import normal_init
from mmcv.ops import batched_nms
from mmcv.runner import force_fp32

from mmdet.core.bbox import bbox_mapping_back
from mmdet.models import HEADS, build_loss
from mmdet.models.utils import gaussian_radius, gen_gaussian_target
from ..utils.gaussian_target import get_local_maximum, get_topk_from_heatmap
from .base_dense_head import BaseDenseHead


@HEADS.register_module()
class CenterNetHead(BaseDenseHead):
    """Objects as Points Head. CenterHead use center_point to indicate object's
    position. Paper link <https://arxiv.org/abs/1904.07850>

    Args:
        in_channel (int): Number of channel in the input feature map.
        feat_channel (int): Number of channel in the intermediate feature map.
        num_classes (int): Number of categories excluding the background
            category.
        loss_center_heatmap (dict | None): Config of center heatmap loss.
            Default: GaussianFocalLoss.
        loss_wh (dict | None): Config of wh loss. Default: L1Loss.
        loss_offset (dict | None): Config of offset loss. Default: L1Loss.
        train_cfg (dict | None): Training config. Useless in CenterNet,
            but we keep this variable for SingleStageDetector. Default: None.
        test_cfg (dict | None): Testing config of CenterNet. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 in_channel,
                 feat_channel,
                 num_classes,
                 loss_center_heatmap=dict(
                     type='GaussianFocalLoss', loss_weight=1.0),
                 loss_wh=dict(type='L1Loss', loss_weight=0.1),
                 loss_offset=dict(type='L1Loss', loss_weight=1.0),
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super(CenterNetHead, self).__init__(init_cfg)
        self.num_classes = num_classes
        self.heatmap_head = self._build_head(
            in_channel,
            feat_channel,
            num_classes,
        )
        self.wh_head = self._build_head(in_channel, feat_channel, 2)
        self.offset_head = self._build_head(in_channel, feat_channel, 2)

        self.loss_center_heatmap = build_loss(loss_center_heatmap)
        self.loss_wh = build_loss(loss_wh)
        self.loss_offset = build_loss(loss_offset)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def _build_head(self, in_channel, feat_channel, out_channel):
        """Build head for each branch."""
        layer = nn.Sequential(
            nn.Conv2d(in_channel, feat_channel, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channel, out_channel, kernel_size=1))
        return layer

    def init_weights(self):
        """Initialize weights of the head."""
        self.heatmap_head[-1].bias.data.fill_(-2.19)
        for head in [self.wh_head, self.offset_head]:
            for m in head.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.001)

    def forward(self, feats):
        """Forward features. Notice CenterNet head don't use FPN.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor. The length must be 1.

        Returns:
            center_heatmap_preds (Tensor): center predict heatmaps, the
               channels number is num_classes.
            wh_preds (Tensor): wh predicts, the channels number is 2.
            offset_preds (Tensor): offset predicts, the channels number is 2.
        """
        assert len(feats) == 1
        feat = feats[-1]
        center_heatmap_preds = self.heatmap_head(feat).sigmoid()
        wh_preds = self.wh_head(feat)
        offset_preds = self.offset_head(feat)
        return center_heatmap_preds, wh_preds, offset_preds

    @force_fp32(apply_to=('center_heatmap_preds', 'wh_preds', 'offset_preds'))
    def loss(self,
             center_heatmap_preds,
             wh_preds,
             offset_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            center_heatmap_preds (Tensor): center predict heatmaps,
               shape (B, num_classes, H, W).
            wh_preds (Tensor): wh predicts, shape (B, 2, H, W),
            offset_preds (Tensor): offset predicts, shape (B, 2, H, W),
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss. Default: None

        Returns:
            dict[str, Tensor]: which has components below:
                - loss_center_heatmap (Tensor): loss of center heatmap.
                - loss_wh (Tensor): loss of hw heatmap
                - loss_offset (Tensor): loss of offset heatmap.
        """
        target_result, avg_factor = self.get_targets(
            gt_bboxes, gt_labels, center_heatmap_preds.shape,
            img_metas[0]['pad_shape'])

        center_heatmap_targets = target_result['center_heatmap_targets']
        wh_targets = target_result['wh_targets']
        offset_targets = target_result['offset_targets']
        wh_offset_target_weights = target_result['wh_offset_target_weights']

        loss_center_heatmap = self.loss_center_heatmap(
            center_heatmap_preds,
            center_heatmap_targets,
            avg_factor=avg_factor / 2.)
        loss_wh = self.loss_wh(
            wh_preds,
            wh_targets,
            wh_offset_target_weights,
            avg_factor=avg_factor)
        loss_offset = self.loss_offset(
            offset_preds,
            offset_targets,
            wh_offset_target_weights,
            avg_factor=avg_factor)
        return dict(
            loss_center_heatmap=loss_center_heatmap,
            loss_wh=loss_wh,
            loss_offset=loss_offset)

    def get_targets(self, gt_bboxes, gt_labels, feat_shape, img_shape):
        """Compute regression and classification targets in multiple images.

        Args:
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box.
            feat_shape (list[int]): feature map shape with value [B, _, H, W]
            img_shape (list[int]): image shape in [h, w] format.

        Returns:
            tuple[dict,float]: The float value is mean avg_factor, the dict has
               components below:
               - center_heatmap_targets (Tensor): targets of center heatmap, \
                   shape (B, num_classes, H, W).
               - wh_targets (Tensor): targets of wh predict, shape \
                   (B, 2, H, W).
               - offset_targets (Tensor): targets of offset predict, shape \
                   (B, 2, H, W).
               - wh_offset_target_weights (Tensor): weights of wh and offset \
                   predict, shape (B, 2, H, W).
        """
        img_h, img_w = img_shape[:2]
        bs, _, feat_h, feat_w = feat_shape
        center_heatmap_targets = gt_bboxes[-1].new_zeros(
            [bs, self.num_classes, feat_h, feat_w])
        wh_targets = gt_bboxes[-1].new_zeros([bs, 2, feat_h, feat_w])
        offset_targets = gt_bboxes[-1].new_zeros([bs, 2, feat_h, feat_w])
        wh_offset_target_weights = gt_bboxes[-1].new_zeros(
            [bs, 2, feat_h, feat_w])

        for i in range(bs):
            ratio_w = feat_w / img_w
            ratio_h = feat_h / img_h
            gt_bbox = gt_bboxes[i]
            gt_label = gt_labels[i]
            center_x = (gt_bbox[:, [0]] + gt_bbox[:, [2]]) * ratio_w / 2
            center_y = (gt_bbox[:, [1]] + gt_bbox[:, [3]]) * ratio_h / 2
            gt_centers = torch.cat((center_x, center_y), dim=1)

            for j, ct in enumerate(gt_centers):
                ctx_int, cty_int = ct.int()
                ctx, cty = ct
                scale_box_h = (gt_bbox[j][3] - gt_bbox[j][1]) * ratio_h
                scale_box_w = (gt_bbox[j][2] - gt_bbox[j][0]) * ratio_w
                radius = gaussian_radius([scale_box_h, scale_box_w],
                                         min_overlap=0.3)
                radius = max(0, int(radius))
                ind = gt_label[j]
                gen_gaussian_target(center_heatmap_targets[i, ind],
                                    [ctx_int, cty_int], radius)

                wh_targets[i, 0, cty_int, ctx_int] = scale_box_w
                wh_targets[i, 1, cty_int, ctx_int] = scale_box_h

                offset_targets[i, 0, cty_int, ctx_int] = ctx - ctx_int
                offset_targets[i, 1, cty_int, ctx_int] = cty - cty_int

                wh_offset_target_weights[i, :, cty_int, ctx_int] = 1

        avg_factor = max(1, wh_offset_target_weights.eq(1).sum())
        target_result = dict(
            center_heatmap_targets=center_heatmap_targets,
            wh_targets=wh_targets,
            offset_targets=offset_targets,
            wh_offset_target_weights=wh_offset_target_weights)
        return target_result, avg_factor

    def get_bboxes(self,
                   center_hm,
                   wh_hm,
                   offset_hm,
                   img_metas,
                   rescale=False,
                   with_nms=False):
        """
            Args:
                center_hm (tensor): shape (B, num_class, H, W),
                wh_hm (tensor): shape (B, num_class, H, W),
                offset_hm (tensor): shape (B, num_class, H, W),
                img_metas (list[Dict]): img_meta of each image.
                rescale (bool): If True, return boxes in original image space.
                    Default: False.
                with_nms (bool): use nms before return bboxes.

            Return:
                list[tuple[Tensor, Tensor]]: Each item in result_list is
                    2-tuple.The first item is an (n, 5) tensor, where the first
                    4 columns are bounding box positions (tl_x, tl_y, br_x,
                    br_y) and the 5-th column is a score between 0 and 1.
                    The second item is a (n,) tensor where each item is the
                    predicted class label of the corresponding box.
        """
        result_list = []
        bs, _, feat_h, feat_w = center_hm.shape
        if bs == 2:
            center_hm = (center_hm[0:1] + self.flip_tensor(center_hm[1:2])) / 2
            wh_hm = (wh_hm[0:1] + self.flip_tensor(wh_hm[1:2])) / 2
            offset_hm = (offset_hm[0:1] + self.flip_tensor(offset_hm[1:2])) / 2
            img_metas = img_metas[0]
        result_list.append(
            self._get_bboxes_single(
                center_hm[:1, ...],
                wh_hm[:1, ...],
                offset_hm[:1, ...],
                img_metas[0],
                rescale=rescale,
                with_nms=with_nms))
        return result_list

    def _get_bboxes_single(self,
                           center_hm,
                           wh_hm,
                           offset_hm,
                           img_meta,
                           rescale=False,
                           with_nms=False):
        """
        Args:
            center_hm (tensor): shape (B, num_class, H, W),
            wh_hm (tensor): shape (B, num_class, H, W),
            offset_hm (tensor): shape (B, num_class, H, W),
            img_metas (dict): img_meta of each image.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): use nms before return bboxes.
        Returns:
            tuple[torch.Tensor]: Decoded output of CornerHead, containing the
            following Tensors:
            - bboxes (Tensor): Coords of each box.
            - clses (Tensor): Categories of each box.
        """
        _, _, feat_h, feat_w = center_hm.shape
        x_off = img_meta['border'][2]
        y_off = img_meta['border'][0]
        img_shape = img_meta['img_shape']
        scale_factor = img_meta['scale_factor']
        flip = img_meta['flip']
        ratio_w = feat_w / img_meta['pad_shape'][1]
        ratio_h = feat_h / img_meta['pad_shape'][0]
        # 1. get topK center points
        center_hm = get_local_maximum(center_hm)
        scores, index, clses, cy, cx = get_topk_from_heatmap(
            center_hm, self.test_cfg['topK'])
        wh = wh_hm.permute(0, 2, 3, 1).view(-1, 2)[index[0]]
        offset = offset_hm.permute(0, 2, 3, 1).view(-1, 2)[index[0]]
        # 2. recover to bboxes
        labels = clses.squeeze(0)
        cx = (cx + offset[:, 0]) / ratio_w
        cy = (cy + offset[:, 1]) / ratio_h
        wh[:, [0]] /= ratio_w
        wh[:, [1]] /= ratio_h
        x1 = (cx - wh[:, 0] / 2).view(-1, 1)
        y1 = (cy - wh[:, 1] / 2).view(-1, 1)
        x2 = (cx + wh[:, 0] / 2).view(-1, 1)
        y2 = (cy + wh[:, 1] / 2).view(-1, 1)
        bboxes = torch.cat([x1, y1, x2, y2], dim=1)
        # 4. add border
        scale_factor = torch.from_numpy(scale_factor)
        bboxes = bbox_mapping_back(bboxes, img_shape, scale_factor, flip)
        bboxes[:, [0, 2]] -= x_off
        bboxes[:, [1, 3]] -= y_off
        scores = scores.view(-1, 1).clone().contiguous()
        detections = torch.cat([bboxes, scores], dim=1)
        return detections, labels

    def flip_tensor(self, tensor):
        return torch.flip(tensor, [3])

    def _bboxes_nms(self, bboxes, labels, cfg):
        if labels.numel() == 0:
            return bboxes, labels
        scores = bboxes[:, -1].contiguous()
        out_bboxes, keep = batched_nms(bboxes[:, :4], scores, labels,
                                       cfg['nms_cfg'])
        out_labels = labels[keep]

        if len(out_bboxes) > 0:
            idx = torch.argsort(out_bboxes[:, -1], descending=True)
            idx = idx[:cfg['max_per_img']]
            out_bboxes = out_bboxes[idx]
            out_labels = out_labels[idx]

        return out_bboxes, out_labels

import torch
import torch.nn as nn
from mmcv.cnn import normal_init
from mmcv.ops import batched_nms

from mmdet.core.bbox import bbox_mapping_back
from mmdet.models import HEADS, build_loss
from mmdet.models.utils import gaussian_radius, gen_gaussian_target
from ..utils.gaussian_target import get_local_maximum, get_topk_from_heatmap
from .base_dense_head import BaseDenseHead


@HEADS.register_module()
class CenterNetHead(BaseDenseHead):
    """Objects as PointsModule. CenterHead use center_point to indicate
    object's position. Paper link <https://arxiv.org/abs/1904.07850>

    Args:
        in_channels (int): Input channels of module.
        feat_channels (int): Feat channels of module.
        num_classes (int): detect class number.
        loss_heatmap (dict): build center_loss
        loss_wh (dict): build wh_loss.
        loss_offset (dict): build offset_loss
        train_cfg (dict): train_cfg
        test_cfg (dict): test_cfg
    """

    def __init__(self,
                 in_channels,
                 feat_channels,
                 num_classes,
                 loss_heatmap=dict(type='GaussianFocalLoss', loss_weight=1.0),
                 loss_wh=dict(type='L1Loss', loss_weight=0.1),
                 loss_offset=dict(type='L1Loss', loss_weight=1.0),
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super(CenterNetHead, self).__init__(init_cfg)
        self.num_classes = num_classes
        self.heatmap_head = self._build_head(
            in_channels,
            feat_channels,
            num_classes,
        )
        self.wh_head = self._build_head(in_channels, feat_channels, 2)
        self.offset_head = self._build_head(in_channels, feat_channels, 2)

        self.loss_heatmap = build_loss(loss_heatmap)
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

    def forward(self, x):
        """Forward Tensor.

        Notice centernet head don't use FPN.
        """
        x = x[-1]
        center_out = self.heatmap_head(x).sigmoid()
        wh_out = self.wh_head(x)
        offset_out = self.offset_head(x)
        return center_out, wh_out, offset_out

    def get_bboxes_center(self, bboxes, ratio_w, ratio_h, toInt):
        assert isinstance(bboxes, torch.Tensor)
        center_x = (bboxes[:, [0]] + bboxes[:, [2]]) * ratio_w / 2
        center_y = (bboxes[:, [1]] + bboxes[:, [3]]) * ratio_h / 2
        centers = torch.cat((center_x, center_y), dim=1)
        if toInt:
            return centers.int()
        else:
            return centers

    def get_center_target(self, gt_bboxes, gt_labels, img_metas, feat_shape):
        """
            Args:
                gt_bboxes (list[Tensor]): gt bboxes of each image.
                gt_labels (list[Tensor]): gt labels of each image.
                img_metas (list[Dict]): img_meta of each image.

            Return:
                dict(str, Tensor): which has components below:
                    - target (Tensor): center heatmap.
                    - weight (Tensor): loss_weight.
                    - avg_factor (int): number of positive sample.
        """
        bs, _, feat_h, feat_w = feat_shape
        gt_heatmap = gt_bboxes[-1].new_zeros(
            [bs, self.num_classes, feat_h, feat_w], dtype=torch.float32)
        lw_heatmap = gt_bboxes[-1].new_ones(
            [bs, self.num_classes, feat_h, feat_w], dtype=torch.float32)
        pad_w = max([x['pad_shape'][1] for x in img_metas])
        pad_h = max([x['pad_shape'][0] for x in img_metas])
        for i in range(bs):
            ratio_w = feat_w / pad_w
            ratio_h = feat_h / pad_h
            gt_bbox = gt_bboxes[i]
            gt_label = gt_labels[i]
            gt_centers = self.get_bboxes_center(
                gt_bbox[:, :4], ratio_w, ratio_h, toInt=True)
            for j, ct in enumerate(gt_centers):
                ctx, cty = ct
                scale_box_h = (gt_bbox[j][3] - gt_bbox[j][1]) * ratio_h
                scale_box_w = (gt_bbox[j][2] - gt_bbox[j][0]) * ratio_w
                radius = gaussian_radius([scale_box_h, scale_box_w],
                                         min_overlap=0.3)
                radius = max(0, int(radius))
                ind = gt_label[j]
                gen_gaussian_target(gt_heatmap[i, ind], [ctx, cty], radius)

        avg_factor = max(1, gt_heatmap.eq(1).sum())
        return dict(
            target=gt_heatmap, weight=lw_heatmap, avg_factor=avg_factor)

    def get_offset_target(self, gt_bboxes, img_metas, feat_shape):
        """
            Args:
                gt_bboxes (list[Tensor]): gt bboxes of each image.
                img_metas (list[Dict]): img_meta of each image.

            Return:
                dict(str, Tensor): which has components below:
                    - target (Tensor): offset heatmap, notice just assign
                    positive sample in object's center.
                    - weight (Tensor): loss_weight.
                    - avg_factor (int): number of positive sample.
        """
        bs, _, feat_h, feat_w = feat_shape

        gt_heatmap = gt_bboxes[-1].new_zeros([bs, 2, feat_h, feat_w],
                                             dtype=torch.float32)
        lw_heatmap = gt_bboxes[-1].new_zeros([bs, 2, feat_h, feat_w],
                                             dtype=torch.float32)

        pad_w = max([x['pad_shape'][1] for x in img_metas])
        pad_h = max([x['pad_shape'][0] for x in img_metas])
        for i in range(bs):
            ratio_w = feat_w / pad_w
            ratio_h = feat_h / pad_h
            gt_bbox = gt_bboxes[i]
            gt_centers = self.get_bboxes_center(gt_bbox[:, :4], ratio_w,
                                                ratio_h, False)
            for j, ct in enumerate(gt_centers):
                ctx_int, cty_int = ct.int()
                ctx, cty = ct
                gt_heatmap[i, 0, cty_int, ctx_int] = ctx - ctx_int
                gt_heatmap[i, 1, cty_int, ctx_int] = cty - cty_int
                lw_heatmap[i, :, cty_int, ctx_int] = 1

        avg_factor = max(1, lw_heatmap.eq(1).sum())
        return dict(
            target=gt_heatmap, weight=lw_heatmap, avg_factor=avg_factor)

    def get_wh_target(self, gt_bboxes, img_metas, feat_shape):
        """
        Args:
            gt_bboxes (list[Tensor]): gt bboxes of each image.
            img_metas (list[Dict]): img_meta of each image.

        Return:
            dict(str, Tensor): which has components below:
                - target (Tensor): wh heatmap, notice just assign positive
                        sample in object's center.
                - weight (Tensor): loss_weight.
                - avg_factor (int): number of positive sample.
        """
        bs, _, feat_h, feat_w = feat_shape

        gt_heatmap = gt_bboxes[-1].new_zeros([bs, 2, feat_h, feat_w],
                                             dtype=torch.float32)
        lw_heatmap = gt_bboxes[-1].new_zeros([bs, 2, feat_h, feat_w],
                                             dtype=torch.float32)
        pad_w = max([x['pad_shape'][1] for x in img_metas])
        pad_h = max([x['pad_shape'][0] for x in img_metas])
        for i in range(bs):
            ratio_w = feat_w / pad_w
            ratio_h = feat_h / pad_h
            gt_bbox = gt_bboxes[i]
            gt_centers = self.get_bboxes_center(gt_bbox[:, :4], ratio_w,
                                                ratio_h, True)
            # 1. assign wh heatmap in input_level
            for j, ct in enumerate(gt_centers):
                ctx, cty = ct
                scale_box_h = (gt_bbox[j][3] - gt_bbox[j][1]) * ratio_h
                scale_box_w = (gt_bbox[j][2] - gt_bbox[j][0]) * ratio_w
                gt_heatmap[i, 0, cty, ctx] = scale_box_w
                gt_heatmap[i, 1, cty, ctx] = scale_box_h
                lw_heatmap[i, :, cty, ctx] = 1

        avg_factor = max(1, lw_heatmap.eq(1).sum())
        return dict(
            target=gt_heatmap, weight=lw_heatmap, avg_factor=avg_factor)

    def get_targets(self, gt_bboxes, gt_labels, img_metas, feat_shape):
        center_target = self.get_center_target(gt_bboxes, gt_labels, img_metas,
                                               feat_shape)
        wh_target = self.get_wh_target(gt_bboxes, img_metas, feat_shape)
        offset_target = self.get_offset_target(gt_bboxes, img_metas,
                                               feat_shape)
        return center_target, wh_target, offset_target

    def loss(self,
             center_heatmap,
             wh_preds,
             offset_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """
        Args:
            center_heatmap (tensor): shape (B, num_class, H, W),
            wh_preds (tensor): shape (B, num_class, H, W),
            offset_preds (tensor): shape (B, num_class, H, W),
            gt_bboxes (list[Tensor]): gt bboxes of each image.
            gt_labels (list[Tensor]): gt labels of each image.
            img_metas (list[Dict]): img_meta of each image.
            gt_bboxes_ignore (list[Tensor]): ignore box of each image.

        Return:
            dict(str, Tensor): which has components below:
                - loss_heatmap (Tensor): loss of center heatmap.
                - loss_wh (Tensor): loss of hw heatmap
                - loss_offset (Tensor): loss of offset heatmap.
        """
        feat_shape = center_heatmap.shape
        center_target, wh_target, offset_target = self.get_targets(
            gt_bboxes, gt_labels, img_metas, feat_shape)
        loss_heatmap = self.loss_heatmap(center_heatmap, **center_target)
        loss_wh = self.loss_wh(wh_preds, **wh_target)
        loss_offset = self.loss_offset(offset_preds, **offset_target)
        return dict(
            loss_heatmap=loss_heatmap,
            loss_wh=loss_wh,
            loss_offset=loss_offset)

    def flip_tensor(self, tensor):
        return torch.flip(tensor, [3])

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

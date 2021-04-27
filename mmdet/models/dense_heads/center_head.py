import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init
from mmcv.ops import batched_nms

from mmdet.core.bbox import bbox_mapping_back
from mmdet.models import HEADS, build_loss
from mmdet.models.utils import gaussian_radius, gen_gaussian_target
from .base_dense_head import BaseDenseHead


@HEADS.register_module()
class CenterHead(BaseDenseHead):
    """Objects as PointsModule. CenterHead use center_point to indicate
    object's position. Paper link <https://arxiv.org/abs/1904.07850>

    Args:
        in_channels (int): Input channels of module.
        feat_channels (int): Feat channels of module.
        num_classes (int): detect class number.
        min_overlap (float): min_overlap for center heatmap.
        loss_center (dict): build center_loss
        loss_wh (dict): build wh_loss.
        loss_offset (dict): build offset_loss
        train_cfg (dict): train_cfg
        test_cfg (dict): test_cfg
    """

    def __init__(self,
                 in_channels=160,
                 feat_channels=256,
                 num_classes=1,
                 min_overlap=0.3,
                 loss_center=dict(type='GaussianFocalLoss', loss_weight=1.0),
                 loss_wh=dict(type='L1Loss', loss_weight=0.1),
                 loss_offset=dict(type='L1Loss', loss_weight=1.0),
                 train_cfg=None,
                 test_cfg=None):
        super(CenterHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_chanels = feat_channels
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.min_overlap = min_overlap
        self.loss_center = build_loss(loss_center)
        self.loss_wh = build_loss(loss_wh)
        self.loss_offset = build_loss(loss_offset)
        self.center_head = self.build_head(
            self.in_channels,
            self.feat_chanels,
            self.num_classes,
        )
        self.wh_head = self.build_head(self.in_channels, self.feat_chanels, 2)
        self.offset_head = self.build_head(self.in_channels, self.feat_chanels,
                                           2)

    def init_weights(self):
        """Initialize weights of the head."""
        for m in self.center_head.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.center_head[-1].bias.data.fill_(-2.19)
        for m in self.offset_head.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        for m in self.wh_head.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)

    def build_head(self, in_channel, feat_channel, out_channel):
        """Build head for each branch."""
        layer = nn.Sequential(
            nn.Conv2d(in_channel, feat_channel, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channel, out_channel, kernel_size=1))
        return layer

    def forward(self, x):
        """Forward Tensor.Notice centerhead don't use FPN."""
        if isinstance(x, list) or isinstance(x, tuple):
            x = x[0]
        self.feat_shape = x.shape
        center_out = self.center_head(x).sigmoid()
        wh_out = self.wh_head(x)
        offset_out = self.offset_head(x).sigmoid()
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

    def get_center_target(self, gt_bboxes, gt_labels, img_metas,
                          gt_bboxes_ignore, *args):
        """
            Args:
                gt_bboxes (list[Tensor]): gt bboxes of each image.
                gt_labels (list[Tensor]): gt labels of each image.
                img_metas (list[Dict]): img_meta of each image.
                gt_bboxes_ignore (list[Tensor]): ignore box of each image.

            Return:
                dict(str, Tensor): which has components below:
                    - target (Tensor): center heatmap.
                    - weight (Tensor): loss_weight.
                    - avg_factor (int): number of positive sample.
        """
        bs, _, feat_h, feat_w = self.feat_shape

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
                                         min_overlap=self.min_overlap)
                radius = max(0, int(radius))
                ind = gt_label[j]
                gen_gaussian_target(gt_heatmap[i, ind], [ctx, cty], radius)

        avg_factor = max(1, gt_heatmap.eq(1).sum())
        return dict(
            target=gt_heatmap, weight=lw_heatmap, avg_factor=avg_factor)

    def get_offset_target(self,
                          gt_bboxes,
                          gt_labels,
                          img_metas,
                          gt_bboxes_ignore=None):
        """
            Args:
                gt_bboxes (list[Tensor]): gt bboxes of each image.
                gt_labels (list[Tensor]): gt labels of each image.
                img_metas (list[Dict]): img_meta of each image.
                gt_bboxes_ignore (list[Tensor]): ignore box of each image.

            Return:
                dict(str, Tensor): which has components below:
                    - target (Tensor): offset heatmap, notice just assign
                    positive sample in object's center.
                    - weight (Tensor): loss_weight.
                    - avg_factor (int): number of positive sample.
        """
        bs, _, feat_h, feat_w = self.feat_shape

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

    def get_wh_target(self,
                      gt_bboxes,
                      gt_labels,
                      img_metas,
                      gt_bboxes_ignore=None):
        """
            Args:
                gt_bboxes (list[Tensor]): gt bboxes of each image.
                gt_labels (list[Tensor]): gt labels of each image.
                img_metas (list[Dict]): img_meta of each image.
                gt_bboxes_ignore (list[Tensor]): ignore box of each image.

            Return:
                dict(str, Tensor): which has components below:
                    - target (Tensor): wh heatmap, notice just assign positive
                        sample in object's center.
                    - weight (Tensor): loss_weight.
                    - avg_factor (int): number of positive sample.
        """
        bs, _, feat_h, feat_w = self.feat_shape

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

    def get_targets(self, *args, **kwargs):
        """generate targets."""
        center_target = self.get_center_target(*args, **kwargs)
        wh_target = self.get_wh_target(*args, **kwargs)
        offset_target = self.get_offset_target(*args, **kwargs)
        return center_target, wh_target, offset_target

    def loss(self,
             center_hm,
             wh_hm,
             offset_hm,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """
            Args:
                center_hm (tensor): shape (B, num_class, H, W),
                wh_hm (tensor): shape (B, num_class, H, W),
                offset_hm (tensor): shape (B, num_class, H, W),
                gt_bboxes (list[Tensor]): gt bboxes of each image.
                gt_labels (list[Tensor]): gt labels of each image.
                img_metas (list[Dict]): img_meta of each image.
                gt_bboxes_ignore (list[Tensor]): ignore box of each image.

            Return:
                dict(str, Tensor): which has components below:
                    - loss_center (Tensor): loss of center heatmap.
                    - loss_wh (Tensor): loss of hw heatmap
                    - loss_offset (Tensor): loss of offset heatmap.
        """
        targets = self.get_targets(
            gt_bboxes=gt_bboxes,
            gt_labels=gt_labels,
            img_metas=img_metas,
            gt_bboxes_ignore=gt_bboxes_ignore)
        center_target, wh_target, offset_target = targets
        loss_center = self.loss_center(center_hm, **center_target)
        loss_wh = self.loss_wh(wh_hm, **wh_target)
        loss_offset = self.loss_offset(offset_hm, **offset_target)
        return dict(
            loss_center=loss_center, loss_wh=loss_wh, loss_offset=loss_offset)

    def _local_maximum(self, heat, kernel=3):
        """Extract local maximum pixel with given kernal.

        Args:
            heat (Tensor): Target heatmap.
            kernel (int): Kernel size of max pooling. Default: 3.

        Returns:
            heat (Tensor): A heatmap where local maximum pixels maintain its
                own value and other positions are 0.
        """
        pad = (kernel - 1) // 2
        hmax = F.max_pool2d(heat, kernel, stride=1, padding=pad)
        keep = (hmax == heat).float()
        return heat * keep

    def _topk(self, scores, k=100):
        """Get top k positions from heatmap.

        Args:
            scores (Tensor): Target heatmap with shape
                [batch, num_classes, height, width].
            k (int): Target number. Default: 20.

        Returns:
            tuple[torch.Tensor]: Scores, indexes, categories and coords of
                topk keypoint. Containing following Tensors:

            - topk_scores (Tensor): Max scores of each topk keypoint.
            - topk_inds (Tensor): Indexes of each topk keypoint.
            - topk_clses (Tensor): Categories of each topk keypoint.
            - topk_ys (Tensor): Y-coord of each topk keypoint.
            - topk_xs (Tensor): X-coord of each topk keypoint.
        """
        batch, _, height, width = scores.size()
        topk_scores, topk_inds = torch.topk(scores.view(batch, -1), k)
        topk_clses = topk_inds // (height * width)
        topk_inds = topk_inds % (height * width)
        topk_ys = topk_inds // width
        topk_xs = (topk_inds % width).int().float()
        return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs

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
        bs, _, feat_h, feat_w = self.feat_shape
        assert bs == 1
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
                           with_nms=True):
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
        _, _, feat_h, feat_w = self.feat_shape
        x_off = img_meta['border'][2]
        y_off = img_meta['border'][0]
        img_shape = img_meta['img_shape']
        scale_factor = img_meta['scale_factor']
        flip = img_meta['flip']
        ratio_w = feat_w / img_meta['pad_shape'][1]
        ratio_h = feat_h / img_meta['pad_shape'][0]
        # import cv2
        # import numpy as np
        # hm = center_hm.clone()[0].cpu().sum(axis=0).numpy()
        # hm = (hm * 255).astype(np.uint8)
        # # import pdb;pdb.set_trace()
        # hm = cv2.applyColorMap(hm, cv2.COLORMAP_HOT)
        # cv2.imshow('sadasd', hm)
        # cv2.waitKey()
        # 1. get topK center points
        center_hm = self._local_maximum(center_hm)
        scores, index, clses, cy, cx = self._topk(center_hm,
                                                  self.test_cfg['topK'])
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
        if not rescale:
            scale_factor = torch.ones(4)
        bboxes = bbox_mapping_back(bboxes, img_shape, scale_factor, flip)
        bboxes[:, [0, 2]] -= x_off
        bboxes[:, [1, 3]] -= y_off
        scores = scores.view(-1, 1).clone().contiguous()
        detections = torch.cat([bboxes, scores], dim=1)
        if with_nms:
            detections, labels = self._bboxes_nms(detections, labels,
                                                  self.test_cfg)
            # detections, labels = multiclass_nms(detections[:, :4],
            #                                     detections[:])
            # print('after', len(detections))
            # import cv2
            # img = cv2.imread(img_meta['filename'])
            # for box in detections:
            #     x1, y1, x2, y2 = box[:4].cpu().numpy().astype(int)
            #     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # cv2.imshow('img', img)
            # cv2.waitKey()
            return detections, labels
        return detections, labels

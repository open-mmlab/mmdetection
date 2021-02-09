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
class CTDetHead(BaseDenseHead):
    """ Feature Map Head
    Args:
        in_channels (int) : Number of channels in the input feature map.
        feat_channels (int): Number of channels of the feature map.
        shared_convs_cfg (list of str): shared convs.
        branch_cfg_list (list): branch for each cls_task.
    """

    def __init__(self,
                 in_channels=160,
                 feat_channels=256,
                 num_classes=1,
                 bias_value=-2.19,
                 choose_ind=0,
                 loss_center=None,
                 loss_wh=None,
                 loss_offset=None,
                 train_cfg=None,
                 test_cfg=None):
        super(CTDetHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_chanels = feat_channels
        self.bias_value = bias_value
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        # hard code
        self.choose_ind = choose_ind

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
        self.center_head[-1].bias.data.fill_(self.bias_value)
        for m in self.offset_head.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        for m in self.wh_head.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)

    def build_head(self, in_channel, feat_channel, out_channel):
        layer = nn.Sequential(
            nn.Conv2d(in_channel, feat_channel, kernel_size=3, padding=1),
            nn.ReLU(), nn.Conv2d(feat_channel, out_channel, kernel_size=1))
        return layer

    def forward(self, x):
        if isinstance(x, list) or isinstance(x, tuple):
            x = x[self.choose_ind]
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
                          gt_bboxes_ignore):
        """
            Args:
                gt_bboxes: list of Tensor, gt bboxes of each image.
                gt_labels: list of Tensor, gt labels of each image.
                img_metas: list of dict, img_meta of each image.
                gt_bboxes_ignore: list of Tensor or None, ignore box of each
                image.

            Return:
                gt_center_cls: Tensor (N * self.cls_channels * h * w),
                               ground truth of each class
                lw_center_cls: Tensor (N * self.cls_channels * h * w),
                               label weight of each class
        """
        # 1. get target output shape
        bs, _, feat_h, feat_w = self.feat_shape
        gt_heatmap = torch.zeros([bs, self.num_classes, feat_h, feat_w],
                                 dtype=torch.float32).cuda()
        lw_heatmap = torch.ones([bs, self.num_classes, feat_h, feat_w],
                                dtype=torch.float32).cuda()
        for i in range(bs):
            ratio_w = feat_w / img_metas[i]['pad_shape'][1]
            ratio_h = feat_h / img_metas[i]['pad_shape'][0]
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
        # 2. use gt ignore compute label weight map
        # if gt_bboxes_ignore is not None and len(gt_bboxes_ignore) > 0:
        #     gt_bboxes_ignore[:, [0, 2]] *= ratio_w
        #     gt_bboxes_ignore[:, [1, 3]] *= ratio_h
        #     for i, box in enumerate(gt_bboxes_ignore):
        #         x1, x2 = (box[[0, 2]] * ratio_w).int()
        #         y1, y2 = (box[[1, 3]] * ratio_h).int()
        #         lw_heatmap[:, y1:y2, x1:x2] = 0
        # return gt_heatmap, lw_heatmap, len(gt_centers)
        # list to tensor

        avg_factor = max(1, gt_heatmap.eq(1).sum())
        return dict(
            target=gt_heatmap, weight=lw_heatmap, avg_factor=avg_factor)

    def get_offset_target(self, gt_bboxes, gt_labels, img_metas,
                          gt_bboxes_ignore):
        """
            Args:
                gt_bboxes: list of Tensor, gt bboxes of each image.
                gt_labels: list of Tensor, gt labels of each image.
                img_metas: list of dict, img_meta of each image.
                gt_bboxes_ignore: list of Tensor or None, ignore box of each
                image.

            Return:
                gt_center_cls: Tensor (N * self.cls_channels * h * w),
                               ground truth of each class
                lw_center_cls: Tensor (N * self.cls_channels * h * w),
                               label weight of each class
        """
        # 1. get target output shape
        bs, _, feat_h, feat_w = self.feat_shape

        gt_heatmap = torch.zeros([bs, 2, feat_h, feat_w],
                                 dtype=torch.float32).cuda()
        lw_heatmap = torch.zeros([bs, 2, feat_h, feat_w],
                                 dtype=torch.float32).cuda()
        for i in range(bs):
            ratio_w = feat_w / img_metas[i]['pad_shape'][1]
            ratio_h = feat_h / img_metas[i]['pad_shape'][0]
            gt_bbox = gt_bboxes[i]
            gt_centers = self.get_bboxes_center(gt_bbox[:, :4], ratio_w,
                                                ratio_h, False)
            # 1. assign gt_bboxes_center heatmap
            for j, ct in enumerate(gt_centers):
                ctx_int, cty_int = ct.int()
                ctx, cty = ct
                gt_heatmap[i, 0, cty_int, ctx_int] = ctx - ctx_int
                gt_heatmap[i, 1, cty_int, ctx_int] = cty - cty_int
                lw_heatmap[i, :, cty_int, ctx_int] = 1

        avg_factor = max(1, lw_heatmap.eq(1).sum() // 2)
        return dict(
            target=gt_heatmap, weight=lw_heatmap, avg_factor=avg_factor)

    def get_wh_target(self, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore):
        """
            Args:
                gt_bboxes: list of Tensor, gt bboxes of each image.
                gt_labels: list of Tensor, gt labels of each image.
                img_metas: list of dict, img_meta of each image.
                gt_bboxes_ignore: list of Tensor or None, ignore box of each

            Return:
                gt_center_cls: Tensor (N * self.cls_channels * h * w),
                               ground truth of each class
                lw_center_cls: Tensor (N * self.cls_channels * h * w),
                               label weight of each class
        """

        # 1. get target output shape
        bs, _, feat_h, feat_w = self.feat_shape

        gt_heatmap = torch.zeros([bs, 2, feat_h, feat_w],
                                 dtype=torch.float32).cuda()
        lw_heatmap = torch.zeros([bs, 2, feat_h, feat_w],
                                 dtype=torch.float32).cuda()
        for i in range(bs):
            ratio_w = feat_w / img_metas[i]['pad_shape'][1]
            ratio_h = feat_h / img_metas[i]['pad_shape'][0]
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

        avg_factor = max(1, lw_heatmap.eq(1).sum() // 2)
        return dict(
            target=gt_heatmap, weight=lw_heatmap, avg_factor=avg_factor)

    def get_targets(self, *args, **kwargs):
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
                   with_nms=True):
        """
        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class labelof the
                corresponding box.
        """
        result_list = []
        bs, _, feat_h, feat_w = self.feat_shape
        assert bs == 1
        result_list.append(
            self._get_bboxes_single(center_hm[:1, ...], wh_hm[:1, ...],
                                    offset_hm[:1, ...], img_metas[0], rescale,
                                    self.test_cfg.topK))
        return result_list

    def _bboxes_nms(self, bboxes, labels, cfg):
        if labels.numel() == 0:
            return bboxes, labels
        scores = bboxes[:, -1].contiguous()
        out_bboxes, keep = batched_nms(bboxes[:, :4], scores, labels,
                                       cfg.nms_cfg)
        out_labels = labels[keep]

        if len(out_bboxes) > 0:
            idx = torch.argsort(out_bboxes[:, -1], descending=True)
            idx = idx[:cfg.max_per_img]
            out_bboxes = out_bboxes[idx]
            out_labels = out_labels[idx]

        return out_bboxes, out_labels

    def _get_bboxes_single(self,
                           center_hm,
                           wh_hm,
                           offset_hm,
                           img_meta,
                           rescale,
                           topK=100):
        """
        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class labelof the
                corresponding box.
        """
        _, _, feat_h, feat_w = self.feat_shape
        x_off = img_meta['border'][2]
        y_off = img_meta['border'][0]
        # print(x_off, y_off)
        img_shape = img_meta['img_shape']
        scale_factor = img_meta['scale_factor']
        flip = img_meta['flip']
        ratio_w = feat_w / img_meta['pad_shape'][1]
        ratio_h = feat_h / img_meta['pad_shape'][0]
        # 1. get topK score offset wh
        center_hm = self._local_maximum(center_hm)
        scores, index, clses, cy, cx = self._topk(center_hm, topK)
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
        bboxes[:, [0, 2]] -= x_off
        bboxes[:, [1, 3]] -= y_off
        bboxes = bbox_mapping_back(bboxes, img_shape, scale_factor, flip)
        scores = scores.view(-1, 1).clone().contiguous()
        # if img_meta['flip'] == True:
        # import cv2
        # image = cv2.imread(img_meta['filename'])
        # for box in bboxes[:50]:
        #     x1, y1, x2, y2 = box.int()
        #     cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
        # cv2.imshow('image', image)
        # cv2.waitKey()
        detections = torch.cat([bboxes, scores], dim=1)
        return detections, labels

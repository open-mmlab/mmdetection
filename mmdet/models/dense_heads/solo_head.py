import mmcv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmdet.core import matrix_nms, multi_apply, points_nms
from mmdet.core.results.results import InstanceResults
from mmdet.models.builder import HEADS, build_loss
from .base_mask_head import BaseMaskHead


def center_of_mass(mask):
    h, w = mask.shape
    grid_h = torch.arange(h, device=mask.device)[:, None]
    grid_w = torch.arange(w, device=mask.device)
    normalizer = mask.sum().float().clamp(min=1e-6)
    center_h = (mask * grid_h).sum() / normalizer
    center_w = (mask * grid_w).sum() / normalizer
    return center_h, center_w


@HEADS.register_module()
class SOLOHead(BaseMaskHead):
    """SOLO mask head used in  https://arxiv.org/abs/1912.04488.

    Note that although SOLO head is single-stage instance segmentors,
    it still uses gt_bbox for calculation while getting target, but it
    does not use gt_bbox when calculating loss.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of hidden channels. Used in child classes.
            Default: 256
        stacked_convs (int): Number of stacking convs of the head.
            Default: 4
        strides (tuple): Downsample factor of each feature map.
        scale_ranges (tuple[tuple[int, int]]): Area range of multiple
            level mask.
        sigma (float): Constant scale factor to control the center region.
        num_grids (list): Divided image into a uniform grids, each feature map
            has a different grid value. The number of output channels is
            grid ** 2. Default: [40, 36, 24, 16, 12]
        cls_down_index (int): The index of downsample operation in
            classification branch. Default: 0.
        loss_mask (dict): Config of mask loss.
        loss_cls (dict): Config of classification loss.
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: norm_cfg=dict(type='GN', num_groups=32,
                                   requires_grad=True).
        train_cfg (dict): Training config of head.
        test_cfg (dict): Testing config of head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(
        self,
        num_classes,
        in_channels,
        feat_channels=256,
        stacked_convs=4,
        strides=(4, 8, 16, 32, 64),
        scale_ranges=((8, 32), (16, 64), (32, 128), (64, 256), (128, 512)),
        sigma=0.2,
        num_grids=[40, 36, 24, 16, 12],
        cls_down_index=0,
        loss_mask=None,
        loss_cls=None,
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        train_cfg=None,
        test_cfg=None,
        init_cfg=[
            dict(type='Normal', layer='Conv2d', std=0.01),
            dict(
                type='Normal',
                std=0.01,
                bias_prob=0.01,
                override=dict(name='conv_mask_list')),
            dict(
                type='Normal',
                std=0.01,
                bias_prob=0.01,
                override=dict(name='conv_cls'))
        ],
    ):
        super(SOLOHead, self).__init__(init_cfg)
        self.num_classes = num_classes
        self.cls_out_channels = self.num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        # number of FPN feats
        self.num_levels = len(strides)
        self.scale_ranges = scale_ranges
        self.sigma = sigma
        self.num_grids = num_grids
        self.cls_down_index = cls_down_index
        self.loss_cls = build_loss(loss_cls)
        self.loss_mask = build_loss(loss_mask)
        self.norm_cfg = norm_cfg
        self.init_cfg = init_cfg
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self._init_layers()

    def _init_layers(self):
        self.mask_convs = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels + 2 if i == 0 else self.feat_channels
            self.mask_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))
        self.conv_mask_list = nn.ModuleList()
        for num_grid in self.num_grids:
            self.conv_mask_list.append(
                nn.Conv2d(self.feat_channels, num_grid**2, 1))

        self.conv_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)

    def resize_feats(self, feats):

        return (F.interpolate(feats[0], scale_factor=0.5,
                              mode='bilinear'), feats[1], feats[2], feats[3],
                F.interpolate(
                    feats[4], size=feats[3].shape[-2:], mode='bilinear'))

    def forward(self, feats):

        assert len(feats) == self.num_levels
        feats = self.resize_feats(feats)
        mask_preds = []
        cls_preds = []
        for i in range(self.num_levels):
            x = feats[i]
            mask_feat = x
            cls_feat = x
            # generate and concat the coordinate
            x_range = torch.linspace(
                -1, 1, mask_feat.shape[-1], device=mask_feat.device)
            y_range = torch.linspace(
                -1, 1, mask_feat.shape[-2], device=mask_feat.device)
            y, x = torch.meshgrid(y_range, x_range)
            y = y.expand([mask_feat.shape[0], 1, -1, -1])
            x = x.expand([mask_feat.shape[0], 1, -1, -1])
            coord_feat = torch.cat([x, y], 1)
            mask_feat = torch.cat([mask_feat, coord_feat], 1)

            for mask_layer in (self.mask_convs):
                mask_feat = mask_layer(mask_feat)

            mask_feat = F.interpolate(
                mask_feat, scale_factor=2, mode='bilinear')
            mask_pred = self.conv_mask_list[i](mask_feat)

            # cls branch
            for j, cls_layer in enumerate(self.cls_convs):
                if j == self.cls_down_index:
                    num_grid = self.num_grids[i]
                    cls_feat = F.interpolate(
                        cls_feat, size=num_grid, mode='bilinear')
                cls_feat = cls_layer(cls_feat)

            cls_pred = self.conv_cls(cls_feat)
            if not self.training:
                feat_wh = feats[0].size()[-2:]
                upsampled_size = (feat_wh[0] * 2, feat_wh[1] * 2)
                mask_pred = F.interpolate(
                    mask_pred.sigmoid(), size=upsampled_size, mode='bilinear')
                cls_pred = points_nms(
                    cls_pred.sigmoid(), kernel=2).permute(0, 2, 3, 1)
            mask_preds.append(mask_pred)
            cls_preds.append(cls_pred)
        return mask_preds, cls_preds

    def loss(self,
             mask_preds,
             cls_preds,
             gt_labels,
             gt_masks,
             img_metas,
             gt_bboxes=None,
             gt_bboxes_ignore=None,
             **kwargs):

        featmap_sizes = [featmap.size()[-2:] for featmap in mask_preds]
        ins_label_list, cate_label_list, ins_ind_label_list = multi_apply(
            self._get_targets_single,
            gt_bboxes,
            gt_labels,
            gt_masks,
            featmap_sizes=featmap_sizes)

        ins_labels = [[] for _ in range(len(mask_preds))]
        ins_pred_temp = [[] for _ in range(len(mask_preds))]
        ins_ind_labels = [[] for _ in range(len(mask_preds))]
        cate_labels = [[] for _ in range(len(mask_preds))]
        for i in range(len(ins_label_list)):
            assert len(mask_preds) == len(ins_label_list[i])
            for j in range(len(ins_label_list[i])):
                ins_labels[j].append(
                    ins_label_list[i][j][ins_ind_label_list[i][j], ...])
                ins_pred_temp[j].append(mask_preds[j][i,
                                                      ins_ind_label_list[i][j],
                                                      ...])
                ins_ind_labels[j].append(ins_ind_label_list[i][j].flatten())
                cate_labels[j].append(cate_label_list[i][j].flatten())

        cate_pred_temp = []
        for i in range(len(ins_labels)):
            ins_labels[i] = torch.cat(ins_labels[i], dim=0)
            mask_preds[i] = torch.cat(ins_pred_temp[i], dim=0)
            ins_ind_labels[i] = torch.cat(ins_ind_labels[i], dim=0)
            cate_labels[i] = torch.cat(cate_labels[i], dim=0)
            cate_pred_temp.append(cls_preds[i].permute(0, 2, 3, 1).reshape(
                -1, self.cls_out_channels))
        cls_preds = cate_pred_temp
        # ins
        flatten_ins_ind_labels = torch.cat(ins_ind_labels)
        num_ins = flatten_ins_ind_labels.sum()

        # dice loss
        loss_mask = []
        for pred, target in zip(mask_preds, ins_labels):
            if pred.size()[0] == 0:
                # make sure can get grad
                loss_mask.append(pred.sum().unsqueeze(0))
                continue
            loss_mask.append(self.loss_mask(pred, target))
        if num_ins > 0:
            loss_mask = torch.cat(loss_mask).sum() / num_ins
        else:
            loss_mask = torch.cat(loss_mask).mean()

        # cate
        flatten_cate_labels = torch.cat(cate_labels)
        flatten_cls_preds = torch.cat(cls_preds)
        loss_cls = self.loss_cls(
            flatten_cls_preds, flatten_cate_labels, avg_factor=num_ins + 1)
        return dict(loss_ins=loss_mask, loss_cate=loss_cls)

    def _get_targets_single(self,
                            gt_bboxes,
                            gt_labels,
                            gt_masks,
                            featmap_sizes=None):
        """Compute targets for predictions of single image.

        Args:
            gt_bboxes (Tensor): Ground truth bbox of each instance,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth label of each instance,
                shape (num_gts,).
            gt_masks (Tensor): Ground truth mask of each instance,
                shape (num_gts, h, w).
            featmap_sizes (list[:obj:`torch.size`]): Size of each
                feature map from feature pyramid, each element
                means (feat_h, feat_w). Default: None.

        Returns:
            Tuple: Usually returns a tuple containing targets for predictions.

                - mlvl_mask_targets (list[Tensor]): Each element represent
                    the binary mask targets for all points in this
                    level, has shape (num_grid**2, out_h, out_w)
                - mlvl_labels (list[Tensor]): Each element is
                    classification labels for all
                    points in this level, has shape
                    (num_grid, num_grid)
                - mlvl_pos_masks (list[Tensor]): Each element is
                    a `BoolTensor` to represent whether the
                    corresponding point in single level
                    is positive, has shape (num_grid **2)
        """
        device = gt_labels[0].device
        # ins
        gt_areas = torch.sqrt((gt_bboxes[:, 2] - gt_bboxes[:, 0]) *
                              (gt_bboxes[:, 3] - gt_bboxes[:, 1]))

        mlvl_mask_targets = []
        mlvl_labels = []
        mlvl_pos_masks = []
        for (lower_bound, upper_bound), stride, featmap_size, num_grid \
                in zip(self.scale_ranges, self.strides,
                       featmap_sizes, self.num_grids):

            mask_target = torch.zeros(
                [num_grid**2, featmap_size[0], featmap_size[1]],
                dtype=torch.uint8,
                device=device)
            # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
            labels = torch.zeros([num_grid, num_grid],
                                 dtype=torch.int64,
                                 device=device) + self.num_classes
            pos_mask = torch.zeros([num_grid**2],
                                   dtype=torch.bool,
                                   device=device)

            gt_inds = ((gt_areas >= lower_bound) &
                       (gt_areas <= upper_bound)).nonzero().flatten()
            if len(gt_inds) == 0:
                mlvl_mask_targets.append(mask_target)
                mlvl_labels.append(labels)
                mlvl_pos_masks.append(pos_mask)
                continue
            hit_gt_bboxes = gt_bboxes[gt_inds]
            hit_gt_labels = gt_labels[gt_inds]
            hit_gt_masks = gt_masks[gt_inds, ...]

            pos_w_ranges = 0.5 * (hit_gt_bboxes[:, 2] -
                                  hit_gt_bboxes[:, 0]) * self.sigma
            pos_h_ranges = 0.5 * (hit_gt_bboxes[:, 3] -
                                  hit_gt_bboxes[:, 1]) * self.sigma

            # mass center
            valid_mask_flags = hit_gt_masks.sum(dim=-1).sum(dim=-1) > 0
            output_stride = stride / 2

            for gt_mask, gt_label, pos_h_range, pos_w_range, \
                valid_mask_flag in \
                    zip(hit_gt_masks, hit_gt_labels, pos_h_ranges,
                        pos_w_ranges, valid_mask_flags):
                if not valid_mask_flag:
                    continue
                upsampled_size = (featmap_sizes[0][0] * 4,
                                  featmap_sizes[0][1] * 4)
                center_h, center_w = center_of_mass(gt_mask)

                coord_w = int(
                    (center_w / upsampled_size[1]) // (1. / num_grid))
                coord_h = int(
                    (center_h / upsampled_size[0]) // (1. / num_grid))

                # left, top, right, down
                top_box = max(
                    0,
                    int(((center_h - pos_h_range) / upsampled_size[0]) //
                        (1. / num_grid)))
                down_box = min(
                    num_grid - 1,
                    int(((center_h + pos_h_range) / upsampled_size[0]) //
                        (1. / num_grid)))
                left_box = max(
                    0,
                    int(((center_w - pos_w_range) / upsampled_size[1]) //
                        (1. / num_grid)))
                right_box = min(
                    num_grid - 1,
                    int(((center_w + pos_w_range) / upsampled_size[1]) //
                        (1. / num_grid)))

                top = max(top_box, coord_h - 1)
                down = min(down_box, coord_h + 1)
                left = max(coord_w - 1, left_box)
                right = min(right_box, coord_w + 1)

                labels[top:(down + 1), left:(right + 1)] = gt_label
                # ins
                gt_mask = np.uint8(gt_mask.cpu().numpy())
                gt_mask = mmcv.imrescale(gt_mask, scale=1. / output_stride)
                gt_mask = torch.from_numpy(gt_mask).to(device=device)

                for i in range(top, down + 1):
                    for j in range(left, right + 1):
                        index = int(i * num_grid + j)
                        mask_target[index, :gt_mask.shape[0], :gt_mask.
                                    shape[1]] = gt_mask
                        pos_mask[index] = True
            mlvl_mask_targets.append(mask_target)
            mlvl_labels.append(labels)
            mlvl_pos_masks.append(pos_mask)
        return mlvl_mask_targets, mlvl_labels, mlvl_pos_masks

    def get_masks(self,
                  seg_preds,
                  cate_preds,
                  img_metas,
                  rescale=None,
                  **kwargs):
        assert len(seg_preds) == len(cate_preds)
        num_levels = len(cate_preds)

        results_list = []
        for img_id in range(len(img_metas)):
            cate_pred_list = [
                cate_preds[i][img_id].view(-1, self.cls_out_channels)
                for i in range(num_levels)
            ]
            seg_pred_list = [seg_preds[i][img_id] for i in range(num_levels)]

            cate_pred_list = torch.cat(cate_pred_list, dim=0)
            seg_pred_list = torch.cat(seg_pred_list, dim=0)

            results = self._get_masks_single(
                cate_pred_list, seg_pred_list, img_meta=img_metas[img_id])
            results_list.append(results)

        return results_list

    def _get_masks_single(self, cate_preds, seg_preds, img_meta, cfg=None):

        if cfg is None:
            cfg = self.test_cfg
        assert len(cate_preds) == len(seg_preds)
        processed_results = InstanceResults(img_meta)

        featmap_size = seg_preds.size()[-2:]
        img_shape = img_meta['img_shape']
        ori_shape = img_meta['ori_shape']
        # overall info.
        h, w, _ = img_shape
        # TODO remove hard code 4 ?
        upsampled_size_out = (featmap_size[0] * 4, featmap_size[1] * 4)

        # process.
        inds = (cate_preds > cfg.score_thr)

        # category scores.
        cate_scores = cate_preds[inds]
        if len(cate_scores) == 0:
            processed_results.scores = cate_scores
            processed_results.masks = cate_scores.new_zeros(0, *ori_shape[:2])
            processed_results.labels = cate_scores.new_ones(0)
            return processed_results
        # category labels.
        inds = inds.nonzero()
        cate_labels = inds[:, 1]

        # strides.
        size_trans = cate_labels.new_tensor(self.num_grids).pow(2).cumsum(0)
        strides = cate_scores.new_ones(size_trans[-1])
        n_stage = len(self.num_grids)
        strides[:size_trans[0]] *= self.strides[0]
        for ind_ in range(1, n_stage):
            strides[size_trans[ind_ -
                               1]:size_trans[ind_]] *= self.strides[ind_]
        strides = strides[inds[:, 0]]

        # masks.
        seg_preds = seg_preds[inds[:, 0]]
        seg_masks = seg_preds > cfg.mask_thr
        sum_masks = seg_masks.sum((1, 2)).float()

        # filter.
        keep = sum_masks > strides
        if keep.sum() == 0:
            processed_results.scores = cate_scores.new_ones(0)
            processed_results.masks = cate_scores.new_zeros(0, *ori_shape[:2])
            processed_results.labels = cate_scores.new_ones(0)
            return processed_results

        seg_masks = seg_masks[keep, ...]
        seg_preds = seg_preds[keep, ...]
        sum_masks = sum_masks[keep]
        cate_scores = cate_scores[keep]
        cate_labels = cate_labels[keep]

        # maskness.
        seg_scores = (seg_preds * seg_masks.float()).sum((1, 2)) / sum_masks
        cate_scores *= seg_scores

        # sort and keep top nms_pre
        sort_inds = torch.argsort(cate_scores, descending=True)
        if len(sort_inds) > cfg.nms_pre:
            sort_inds = sort_inds[:cfg.nms_pre]
        seg_masks = seg_masks[sort_inds, :, :]
        seg_preds = seg_preds[sort_inds, :, :]
        sum_masks = sum_masks[sort_inds]
        cate_scores = cate_scores[sort_inds]
        cate_labels = cate_labels[sort_inds]

        # Matrix NMS
        cate_scores = matrix_nms(
            seg_masks,
            cate_labels,
            cate_scores,
            kernel=cfg.kernel,
            sigma=cfg.sigma,
            sum_masks=sum_masks)

        # filter.
        keep = cate_scores >= cfg.update_thr
        if keep.sum() == 0:
            processed_results.scores = cate_scores.new_ones(0)
            processed_results.masks = cate_scores.new_zeros(0, *ori_shape[:2])
            processed_results.labels = cate_scores.new_ones(0)
            return processed_results
        seg_preds = seg_preds[keep, :, :]
        cate_scores = cate_scores[keep]
        cate_labels = cate_labels[keep]

        # sort and keep top_k
        sort_inds = torch.argsort(cate_scores, descending=True)
        if len(sort_inds) > cfg.max_per_img:
            sort_inds = sort_inds[:cfg.max_per_img]
        seg_preds = seg_preds[sort_inds, :, :]
        cate_scores = cate_scores[sort_inds]
        cate_labels = cate_labels[sort_inds]

        seg_preds = F.interpolate(
            seg_preds.unsqueeze(0), size=upsampled_size_out,
            mode='bilinear')[:, :, :h, :w]
        seg_masks = F.interpolate(
            seg_preds, size=ori_shape[:2], mode='bilinear').squeeze(0)
        seg_masks = seg_masks > cfg.mask_thr

        processed_results.masks = seg_masks
        processed_results.labels = cate_labels
        processed_results.scores = cate_scores

        return processed_results


@HEADS.register_module()
class DecoupledSOLOHead(SOLOHead):

    def __init__(
        self,
        num_classes,
        in_channels,
        feat_channels=256,
        stacked_convs=4,
        strides=(4, 8, 16, 32, 64),
        scale_ranges=((8, 32), (16, 64), (32, 128), (64, 256), (128, 512)),
        sigma=0.2,
        num_grids=None,
        cls_down_index=0,
        loss_mask=None,
        loss_cls=None,
        norm_cfg=None,
        train_cfg=None,
        test_cfg=None,
        init_cfg=[
            dict(type='Normal', layer='Conv2d', std=0.01),
            dict(
                type='Normal',
                std=0.01,
                bias_prob=0.01,
                override=dict(name='dsolo_ins_list_x')),
            dict(
                type='Normal',
                std=0.01,
                bias_prob=0.01,
                override=dict(name='dsolo_ins_list_y')),
            dict(
                type='Normal',
                std=0.01,
                bias_prob=0.01,
                override=dict(name='dsolo_cate'))
        ],
    ):
        super(DecoupledSOLOHead, self).__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            feat_channels=feat_channels,
            stacked_convs=stacked_convs,
            strides=strides,
            scale_ranges=scale_ranges,
            sigma=sigma,
            num_grids=num_grids,
            cls_down_index=cls_down_index,
            loss_mask=loss_mask,
            loss_cls=loss_cls,
            norm_cfg=norm_cfg,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg)

    def _init_layers(self):
        self.ins_convs_x = nn.ModuleList()
        self.ins_convs_y = nn.ModuleList()
        self.cate_convs = nn.ModuleList()

        for i in range(self.stacked_convs):
            chn = self.in_channels + 1 if i == 0 else self.feat_channels
            self.ins_convs_x.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))
            self.ins_convs_y.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))

            chn = self.in_channels if i == 0 else self.feat_channels
            self.cate_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))

        self.dsolo_ins_list_x = nn.ModuleList()
        self.dsolo_ins_list_y = nn.ModuleList()
        for seg_num_grid in self.num_grids:
            self.dsolo_ins_list_x.append(
                nn.Conv2d(self.feat_channels, seg_num_grid, 3, padding=1))
            self.dsolo_ins_list_y.append(
                nn.Conv2d(self.feat_channels, seg_num_grid, 3, padding=1))
        self.dsolo_cate = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)

    def forward(self, feats):
        assert len(feats) == self.num_levels
        new_feats = self.resize_feats(feats)
        featmap_sizes = [featmap.size()[-2:] for featmap in new_feats]
        upsampled_size = (featmap_sizes[0][0] * 2, featmap_sizes[0][1] * 2)

        ins_pred_x_maps = []
        ins_pred_y_maps = []
        cate_pred_maps = []
        for i in range(self.num_levels):
            x = new_feats[i]
            ins_feat = x
            cate_feat = x
            # ins branch
            # concat coord
            x_range = torch.linspace(
                -1, 1, ins_feat.shape[-1], device=ins_feat.device)
            y_range = torch.linspace(
                -1, 1, ins_feat.shape[-2], device=ins_feat.device)
            y, x = torch.meshgrid(y_range, x_range)
            y = y.expand([ins_feat.shape[0], 1, -1, -1])
            x = x.expand([ins_feat.shape[0], 1, -1, -1])
            ins_feat_x = torch.cat([ins_feat, x], 1)
            ins_feat_y = torch.cat([ins_feat, y], 1)

            for ins_layer_x, ins_layer_y in \
                    zip(self.ins_convs_x, self.ins_convs_y):
                ins_feat_x = ins_layer_x(ins_feat_x)
                ins_feat_y = ins_layer_y(ins_feat_y)

            ins_feat_x = F.interpolate(
                ins_feat_x, scale_factor=2, mode='bilinear')
            ins_feat_y = F.interpolate(
                ins_feat_y, scale_factor=2, mode='bilinear')

            ins_pred_x = self.dsolo_ins_list_x[i](ins_feat_x)
            ins_pred_y = self.dsolo_ins_list_y[i](ins_feat_y)

            # cate branch
            for j, cate_layer in enumerate(self.cate_convs):
                if j == self.cls_down_index:
                    seg_num_grid = self.num_grids[i]
                    cate_feat = F.interpolate(
                        cate_feat, size=seg_num_grid, mode='bilinear')
                cate_feat = cate_layer(cate_feat)

            cate_pred = self.dsolo_cate(cate_feat)
            if not self.training:
                ins_pred_x = F.interpolate(
                    ins_pred_x.sigmoid(), size=upsampled_size, mode='bilinear')
                ins_pred_y = F.interpolate(
                    ins_pred_y.sigmoid(), size=upsampled_size, mode='bilinear')
                cate_pred = points_nms(
                    cate_pred.sigmoid(), kernel=2).permute(0, 2, 3, 1)
            ins_pred_x_maps.append(ins_pred_x)
            ins_pred_y_maps.append(ins_pred_y)
            cate_pred_maps.append(cate_pred)
        return ins_pred_x_maps, ins_pred_y_maps, cate_pred_maps

    def loss(self,
             ins_preds_x,
             ins_preds_y,
             cate_preds,
             gt_labels,
             gt_masks,
             img_metas,
             gt_bboxes=None,
             gt_bboxes_ignore=None,
             **kwargs):
        featmap_sizes = [featmap.size()[-2:] for featmap in ins_preds_x]

        ins_label_list, cate_label_list, \
            ins_ind_label_list, ins_ind_label_list_xy = \
            multi_apply(self.solo_target_single,
                        gt_bboxes,
                        gt_labels,
                        gt_masks,
                        featmap_sizes=featmap_sizes)

        # ins
        ins_labels = [[] for _ in range(len(ins_preds_x))]
        ins_preds_x_final = [[] for _ in range(len(ins_preds_x))]
        ins_preds_y_final = [[] for _ in range(len(ins_preds_x))]
        cate_labels = [[] for _ in range(len(ins_preds_x))]
        for i in range(len(ins_label_list)):
            assert len(ins_preds_x) == len(ins_label_list[i])
            for j in range(len(ins_label_list[i])):
                ins_labels[j].append(
                    ins_label_list[i][j][ins_ind_label_list[i][j], ...])
                ins_preds_x_final[j].append(
                    ins_preds_x[j][i, ins_ind_label_list_xy[i][j][:, 1], ...])
                ins_preds_y_final[j].append(
                    ins_preds_y[j][i, ins_ind_label_list_xy[i][j][:, 0], ...])
                cate_labels[j].append(cate_label_list[i][j].flatten())

        cate_pred_temp = []
        for i in range(len(ins_labels)):
            ins_labels[i] = torch.cat(ins_labels[i], dim=0)
            ins_preds_x_final[i] = torch.cat(ins_preds_x_final[i], dim=0)
            ins_preds_y_final[i] = torch.cat(ins_preds_y_final[i], dim=0)
            cate_labels[i] = torch.cat(cate_labels[i], dim=0)
            cate_pred_temp.append(cate_preds[i].permute(0, 2, 3, 1).reshape(
                -1, self.cls_out_channels))
        cate_preds = cate_pred_temp

        num_ins = 0.
        # dice loss
        loss_mask = []
        for pred_x, pred_y, target in \
                zip(ins_preds_x_final, ins_preds_y_final, ins_labels):
            mask_n = pred_x.size(0)
            if mask_n == 0:
                # make sure can get grad
                loss_mask.append((pred_x.sum() + pred_y.sum()).unsqueeze(0))
                continue
            num_ins += mask_n
            loss_mask.append(self.loss_mask((pred_x, pred_y), target))
        if num_ins > 0:
            loss_mask = torch.cat(loss_mask).sum() / num_ins
        else:
            loss_mask = torch.cat(loss_mask).mean()

        # cate
        flatten_cate_labels = torch.cat(cate_labels)
        flatten_cate_preds = torch.cat(cate_preds)

        loss_cls = self.loss_cls(
            flatten_cate_preds, flatten_cate_labels, avg_factor=num_ins + 1)
        return dict(loss_mask=loss_mask, loss_cate=loss_cls)

    def solo_target_single(self,
                           gt_bboxes_raw,
                           gt_labels_raw,
                           gt_masks_raw,
                           featmap_sizes=None):

        device = gt_labels_raw[0].device
        # ins
        gt_areas = torch.sqrt((gt_bboxes_raw[:, 2] - gt_bboxes_raw[:, 0]) *
                              (gt_bboxes_raw[:, 3] - gt_bboxes_raw[:, 1]))
        ins_label_list = []
        cate_label_list = []
        ins_ind_label_list = []
        ins_ind_label_list_xy = []
        for (lower_bound, upper_bound), stride, featmap_size, num_grid \
                in zip(self.scale_ranges, self.strides,
                       featmap_sizes, self.num_grids):

            ins_label = torch.zeros(
                [num_grid**2, featmap_size[0], featmap_size[1]],
                dtype=torch.uint8,
                device=device)
            # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
            cate_label = torch.zeros([num_grid, num_grid],
                                     dtype=torch.int64,
                                     device=device) + self.num_classes
            ins_ind_label = torch.zeros([num_grid**2],
                                        dtype=torch.bool,
                                        device=device)

            hit_indices = ((gt_areas >= lower_bound) &
                           (gt_areas <= upper_bound)).nonzero().flatten()

            if len(hit_indices) == 0:
                ins_label = torch.zeros([1, featmap_size[0], featmap_size[1]],
                                        dtype=torch.uint8,
                                        device=device)
                ins_label_list.append(ins_label)
                cate_label_list.append(cate_label)
                ins_ind_label = torch.zeros([1],
                                            dtype=torch.bool,
                                            device=device)
                ins_ind_label_list.append(ins_ind_label)
                ins_ind_label_list_xy.append(
                    (cate_label - self.num_classes).nonzero())
                continue
            gt_bboxes = gt_bboxes_raw[hit_indices]
            gt_labels = gt_labels_raw[hit_indices]
            gt_masks = gt_masks_raw[hit_indices, ...]

            half_ws = 0.5 * (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * self.sigma
            half_hs = 0.5 * (gt_bboxes[:, 3] - gt_bboxes[:, 1]) * self.sigma

            output_stride = stride / 2

            for seg_mask, gt_label, half_h, half_w in \
                    zip(gt_masks, gt_labels, half_hs, half_ws):

                if seg_mask.sum() < 10:
                    continue
                # mass center

                upsampled_size = (featmap_sizes[0][0] * 4,
                                  featmap_sizes[0][1] * 4)
                center_h, center_w = center_of_mass(seg_mask)

                coord_w = int(
                    (center_w / upsampled_size[1]) // (1. / num_grid))
                coord_h = int(
                    (center_h / upsampled_size[0]) // (1. / num_grid))

                # left, top, right, down
                top_box = max(
                    0,
                    int(((center_h - half_h) / upsampled_size[0]) //
                        (1. / num_grid)))
                down_box = min(
                    num_grid - 1,
                    int(((center_h + half_h) / upsampled_size[0]) //
                        (1. / num_grid)))
                left_box = max(
                    0,
                    int(((center_w - half_w) / upsampled_size[1]) //
                        (1. / num_grid)))
                right_box = min(
                    num_grid - 1,
                    int(((center_w + half_w) / upsampled_size[1]) //
                        (1. / num_grid)))

                top = max(top_box, coord_h - 1)
                down = min(down_box, coord_h + 1)
                left = max(coord_w - 1, left_box)
                right = min(right_box, coord_w + 1)

                # squared
                cate_label[top:(down + 1), left:(right + 1)] = gt_label
                # ins
                seg_mask = np.uint8(seg_mask.cpu().numpy())
                seg_mask = mmcv.imrescale(seg_mask, scale=1. / output_stride)
                seg_mask = torch.from_numpy(seg_mask).to(device=device)

                for i in range(top, down + 1):
                    for j in range(left, right + 1):
                        label = int(i * num_grid + j)
                        ins_label[label, :seg_mask.shape[0], :seg_mask.
                                  shape[1]] = seg_mask
                        ins_ind_label[label] = True

            ins_label = ins_label[ins_ind_label]
            ins_label_list.append(ins_label)

            cate_label_list.append(cate_label)

            ins_ind_label = ins_ind_label[ins_ind_label]
            ins_ind_label_list.append(ins_ind_label)

            ins_ind_label_list_xy.append(
                (cate_label - self.num_classes).nonzero())
        return ins_label_list, cate_label_list, \
            ins_ind_label_list, ins_ind_label_list_xy

    def get_masks(self,
                  seg_preds_x,
                  seg_preds_y,
                  cate_preds,
                  img_metas,
                  rescale=None,
                  **kwargs):
        assert len(seg_preds_x) == len(cate_preds)
        num_levels = len(cate_preds)

        results_list = []
        for img_id in range(len(img_metas)):
            cate_pred_list = [
                cate_preds[i][img_id].view(-1, self.cls_out_channels).detach()
                for i in range(num_levels)
            ]
            seg_pred_list_x = [
                seg_preds_x[i][img_id] for i in range(num_levels)
            ]
            seg_pred_list_y = [
                seg_preds_y[i][img_id] for i in range(num_levels)
            ]

            cate_pred_list = torch.cat(cate_pred_list, dim=0)
            seg_pred_list_x = torch.cat(seg_pred_list_x, dim=0)
            seg_pred_list_y = torch.cat(seg_pred_list_y, dim=0)

            results = self.get_masks_single(
                cate_pred_list,
                seg_pred_list_x,
                seg_pred_list_y,
                img_meta=img_metas[img_id],
                cfg=self.test_cfg)
            results_list.append(results)
        return results_list

    def get_masks_single(self, cate_preds, seg_preds_x, seg_preds_y, img_meta,
                         cfg):
        cfg = self.test_cfg if cfg is None else cfg
        # overall info.
        img_shape = img_meta['img_shape']
        ori_shape = img_meta['ori_shape']
        processed_results = InstanceResults(img_meta)

        h, w, _ = img_shape
        featmap_size = seg_preds_x.size()[-2:]
        upsampled_size_out = (featmap_size[0] * 4, featmap_size[1] * 4)

        # trans trans_diff.
        trans_size = torch.Tensor(self.num_grids).pow(2).cumsum(0).long()
        trans_diff = torch.ones(
            trans_size[-1].item(), device=cate_preds.device).long()
        num_grids = torch.ones(
            trans_size[-1].item(), device=cate_preds.device).long()
        seg_size = torch.Tensor(self.num_grids).cumsum(0).long()
        seg_diff = torch.ones(
            trans_size[-1].item(), device=cate_preds.device).long()
        strides = torch.ones(trans_size[-1].item(), device=cate_preds.device)

        n_stage = len(self.num_grids)
        trans_diff[:trans_size[0]] *= 0
        seg_diff[:trans_size[0]] *= 0
        num_grids[:trans_size[0]] *= self.num_grids[0]
        strides[:trans_size[0]] *= self.strides[0]

        for ind_ in range(1, n_stage):
            trans_diff[trans_size[ind_ - 1]:trans_size[ind_]] *= \
                trans_size[ind_ - 1]
            seg_diff[trans_size[ind_ - 1]:trans_size[ind_]] *= \
                seg_size[ind_ - 1]
            num_grids[trans_size[ind_ - 1]:trans_size[ind_]] *= \
                self.num_grids[ind_]
            strides[trans_size[ind_ - 1]:trans_size[ind_]] *= \
                self.strides[ind_]

        # process.
        inds = (cate_preds > cfg.score_thr)
        cate_scores = cate_preds[inds]

        inds = inds.nonzero()
        trans_diff = torch.index_select(trans_diff, dim=0, index=inds[:, 0])
        seg_diff = torch.index_select(seg_diff, dim=0, index=inds[:, 0])
        num_grids = torch.index_select(num_grids, dim=0, index=inds[:, 0])
        strides = torch.index_select(strides, dim=0, index=inds[:, 0])

        y_inds = (inds[:, 0] - trans_diff) // num_grids
        x_inds = (inds[:, 0] - trans_diff) % num_grids
        y_inds += seg_diff
        x_inds += seg_diff

        cate_labels = inds[:, 1]
        seg_masks_soft = seg_preds_x[x_inds, ...] * seg_preds_y[y_inds, ...]
        seg_masks = seg_masks_soft > cfg.mask_thr
        sum_masks = seg_masks.sum((1, 2)).float()
        keep = sum_masks > strides

        seg_masks_soft = seg_masks_soft[keep, ...]
        seg_masks = seg_masks[keep, ...]
        cate_scores = cate_scores[keep]
        sum_masks = sum_masks[keep]
        cate_labels = cate_labels[keep]
        # maskness
        seg_score = (seg_masks_soft * seg_masks.float()).sum(
            (1, 2)) / sum_masks
        cate_scores *= seg_score

        # TODO proecess this case
        if len(cate_scores) == 0:
            processed_results.scores = cate_scores.new_ones(0)
            processed_results.masks = cate_scores.new_zeros(0, *ori_shape[:2])
            processed_results.labels = cate_scores.new_ones(0)
            return processed_results

        # sort and keep top nms_pre
        sort_inds = torch.argsort(cate_scores, descending=True)
        if len(sort_inds) > cfg.nms_pre:
            sort_inds = sort_inds[:cfg.nms_pre]
        seg_masks_soft = seg_masks_soft[sort_inds, :, :]
        seg_masks = seg_masks[sort_inds, :, :]
        cate_scores = cate_scores[sort_inds]
        sum_masks = sum_masks[sort_inds]
        cate_labels = cate_labels[sort_inds]

        # Matrix NMS
        cate_scores = matrix_nms(
            seg_masks,
            cate_labels,
            cate_scores,
            kernel=cfg.kernel,
            sigma=cfg.sigma,
            sum_masks=sum_masks)

        keep = cate_scores >= cfg.update_thr
        if keep.sum() == 0:
            processed_results.scores = cate_scores.new_ones(0)
            processed_results.masks = cate_scores.new_zeros(0, *ori_shape[:2])
            processed_results.labels = cate_scores.new_ones(0)
            return processed_results
        seg_masks_soft = seg_masks_soft[keep, :, :]
        cate_scores = cate_scores[keep]
        cate_labels = cate_labels[keep]
        # sort and keep top_k
        sort_inds = torch.argsort(cate_scores, descending=True)
        if len(sort_inds) > cfg.max_per_img:
            sort_inds = sort_inds[:cfg.max_per_img]
        seg_masks_soft = seg_masks_soft[sort_inds, :, :]
        cate_scores = cate_scores[sort_inds]
        cate_labels = cate_labels[sort_inds]

        seg_masks_soft = F.interpolate(
            seg_masks_soft.unsqueeze(0),
            size=upsampled_size_out,
            mode='bilinear')[:, :, :h, :w]
        seg_masks = F.interpolate(
            seg_masks_soft, size=ori_shape[:2], mode='bilinear').squeeze(0)
        seg_masks = seg_masks > cfg.mask_thr

        processed_results.masks = seg_masks
        processed_results.labels = cate_labels
        processed_results.scores = cate_scores

        return processed_results

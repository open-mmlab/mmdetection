import os
from scipy import ndimage
import mmcv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.core import bbox2result, multi_apply
from mmcv.cnn import ConvModule, build_upsample_layer, bias_init_with_prob
from .base_dense_head import BaseDenseHead
from mmdet.core import matrix_nms
from mmdet.models.builder import HEADS, build_loss


def points_nms(heat, kernel=2):
    # kernel must be 2
    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=1)
    keep = (hmax[:, :, :-1, :-1] == heat).float()
    return heat * keep


@HEADS.register_module()
class SOLOHead(BaseDenseHead):
    def __init__(self,
                 num_classes,
                 in_channels,
                 seg_feat_channels=256,
                 stacked_convs=4,
                 strides=(4, 8, 16, 32, 64),
                 base_edge_list=(16, 32, 64, 128, 256),
                 scale_ranges=((8, 32), (16, 64), (32, 128), (64, 256), (128, 512)),
                 sigma=0.2,
                 num_grids=None,
                 cate_down_pos=0,
                 with_deform=False,
                 loss_mask=None,
                 loss_cls=None,
                 conv_cfg=None,
                 norm_cfg=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=[dict(type='Normal', std=0.01, override=dict(name='ins_convs')),
                           dict(type='Normal', std=0.01, override=dict(name='cate_convs')),
                           dict(type='Normal', std=0.01, bias_prob=0.01, override=dict(name='solo_ins_list')),
                           dict(type='Normal', std=0.01, bias_prob=0.01, override=dict(name='solo_cate')),
                           ]
                 ):
        super(SOLOHead, self).__init__(init_cfg)
        self.num_classes = num_classes
        self.cate_out_channels = self.num_classes
        self.in_channels = in_channels
        self.seg_feat_channels = seg_feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.base_edge_list = base_edge_list
        self.scale_ranges = scale_ranges
        self.sigma = sigma
        self.seg_num_grids = num_grids
        self.cate_down_pos = cate_down_pos
        self.with_deform = with_deform
        self.loss_cls = build_loss(loss_cls)
        self.loss_mask = build_loss(loss_mask)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.init_cfg = init_cfg
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self._init_layers()

    def _init_layers(self):
        self.ins_convs = nn.ModuleList()
        # 被调用多次的模块，是使用同一组 parameters 的，也就是它们的参数是完全一样的，无论你之后怎么更新
        self.cate_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels + 2 if i == 0 else self.seg_feat_channels
            self.ins_convs.append(
                ConvModule(
                    chn,
                    self.seg_feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))
            chn = self.in_channels if i == 0 else self.seg_feat_channels
            self.cate_convs.append(
                ConvModule(
                    chn,
                    self.seg_feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))
        self.solo_ins_list = nn.ModuleList()
        for seg_num_grid in self.seg_num_grids:
            self.solo_ins_list.append(
                nn.Conv2d(
                    self.seg_feat_channels, seg_num_grid ** 2, 1))

        self.solo_cate = nn.Conv2d(
            self.seg_feat_channels, self.cate_out_channels, 3, padding=1)

    def get_bboxes(self, **kwargs):
        """Transform network output for a batch into bbox predictions."""
        pass

    @property
    def num_levels(self):
        return len(self.strides)

    def split_feats(self, feats):
        return (F.interpolate(feats[0], scale_factor=0.5, mode='bilinear'),
                feats[1],
                feats[2],
                feats[3],
                F.interpolate(feats[4], size=feats[3].shape[-2:], mode='bilinear'))

    def forward(self, feats, eval=False):
        assert len(feats) == self.num_levels
        new_feats = self.split_feats(feats)
        featmap_sizes = [featmap.size()[-2:] for featmap in new_feats]
        upsampled_size = (featmap_sizes[0][0] * 2, featmap_sizes[0][1] * 2)

        ins_pred_maps = []
        cate_pred_maps = []
        for i in range(self.num_levels):
            x = new_feats[i]
            ins_feat = x
            cate_feat = x
            # ins branch
            # concat coord
            x_range = torch.linspace(-1, 1, ins_feat.shape[-1], device=ins_feat.device)
            y_range = torch.linspace(-1, 1, ins_feat.shape[-2], device=ins_feat.device)
            y, x = torch.meshgrid(y_range, x_range)
            y = y.expand([ins_feat.shape[0], 1, -1, -1])
            x = x.expand([ins_feat.shape[0], 1, -1, -1])
            coord_feat = torch.cat([x, y], 1)
            ins_feat = torch.cat([ins_feat, coord_feat], 1)

            for _, ins_layer in enumerate(self.ins_convs):
                ins_feat = ins_layer(ins_feat)

            ins_feat = F.interpolate(ins_feat, scale_factor=2, mode='bilinear')
            ins_pred = self.solo_ins_list[i](ins_feat)

            # cate branch
            for j, cate_layer in enumerate(self.cate_convs):
                if j == self.cate_down_pos:
                    seg_num_grid = self.seg_num_grids[i]
                    cate_feat = F.interpolate(cate_feat, size=seg_num_grid, mode='bilinear')
                cate_feat = cate_layer(cate_feat)

            cate_pred = self.solo_cate(cate_feat)
            if eval:
                ins_pred = F.interpolate(ins_pred.sigmoid(), size=upsampled_size, mode='bilinear')
                cate_pred = points_nms(cate_pred.sigmoid(), kernel=2).permute(0, 2, 3, 1)
            ins_pred_maps.append(ins_pred)
            cate_pred_maps.append(cate_pred)
        return ins_pred_maps, cate_pred_maps

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      gt_mask=None,
                      proposal_cfg=None,
                      **kwargs):
        outs = self(x)
        if gt_labels and gt_mask is None:
            loss_inputs = outs + (gt_bboxes, img_metas)
        elif gt_labels and gt_mask is not None:
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas, gt_mask)
        elif gt_labels is not None and gt_mask is None:
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(*outs, img_metas, cfg=proposal_cfg)
            return losses, proposal_list

    def loss(self,
             ins_preds,
             cate_preds,
             gt_bbox_list,
             gt_label_list,
             img_metas,
             gt_mask_list,
             cfg=None,
             gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in
                         ins_preds]
        ins_label_list, cate_label_list, ins_ind_label_list = multi_apply(
            self.solo_target_single,
            gt_bbox_list,
            gt_label_list,
            gt_mask_list,
            featmap_sizes=featmap_sizes)

        # ins
        ins_labels = [torch.cat([ins_labels_level_img[ins_ind_labels_level_img, ...]
                      for ins_labels_level_img, ins_ind_labels_level_img in
                      zip(ins_labels_level, ins_ind_labels_level)], 0)
                      for ins_labels_level, ins_ind_labels_level in
                      zip(zip(*ins_label_list), zip(*ins_ind_label_list))]

        ins_preds = [torch.cat([ins_preds_level_img[ins_ind_labels_level_img, ...]
                     for ins_preds_level_img, ins_ind_labels_level_img in
                     zip(ins_preds_level, ins_ind_labels_level)], 0)
                     for ins_preds_level, ins_ind_labels_level in
                     zip(ins_preds, zip(*ins_ind_label_list))]

        ins_ind_labels = [torch.cat([ins_ind_labels_level_img.flatten()
                          for ins_ind_labels_level_img in ins_ind_labels_level])
                          for ins_ind_labels_level in zip(*ins_ind_label_list)
        ]
        flatten_ins_ind_labels = torch.cat(ins_ind_labels)

        num_ins = flatten_ins_ind_labels.sum()

        # dice loss
        loss_mask = []
        for input, target in zip(ins_preds, ins_labels):
            if input.size()[0] == 0:
                continue
            loss_mask.append(self.loss_mask(input, target))
        loss_mask = torch.cat(loss_mask).mean()

        # cate
        cate_labels = [
            torch.cat([cate_labels_level_img.flatten()
                       for cate_labels_level_img in cate_labels_level])
            for cate_labels_level in zip(*cate_label_list)
        ]
        flatten_cate_labels = torch.cat(cate_labels)

        cate_preds = [
            cate_pred.permute(0, 2, 3, 1).reshape(-1, self.cate_out_channels)
            for cate_pred in cate_preds
        ]
        flatten_cate_preds = torch.cat(cate_preds)

        loss_cls = self.loss_cls(flatten_cate_preds, flatten_cate_labels, avg_factor=num_ins + 1)
        return dict(
            loss_ins=loss_mask,
            loss_cate=loss_cls)

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
        for (lower_bound, upper_bound), stride, featmap_size, num_grid \
                in zip(self.scale_ranges, self.strides,
                       featmap_sizes, self.seg_num_grids):

            ins_label = torch.zeros([num_grid ** 2,
                                     featmap_size[0], featmap_size[1]],
                                    dtype=torch.uint8, device=device)
            # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
            cate_label = torch.zeros([num_grid, num_grid],
                                     dtype=torch.int64,
                                     device=device) + self.num_classes
            ins_ind_label = torch.zeros([num_grid ** 2],
                                        dtype=torch.bool, device=device)

            hit_indices = ((gt_areas >= lower_bound) &
                           (gt_areas <= upper_bound)).nonzero().flatten()
            if len(hit_indices) == 0:
                ins_label_list.append(ins_label)
                cate_label_list.append(cate_label)
                ins_ind_label_list.append(ins_ind_label)
                continue
            gt_bboxes = gt_bboxes_raw[hit_indices]
            gt_labels = gt_labels_raw[hit_indices]
            gt_masks = gt_masks_raw.masks[hit_indices.cpu().numpy(), ...]

            half_ws = 0.5 * (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * self.sigma
            half_hs = 0.5 * (gt_bboxes[:, 3] - gt_bboxes[:, 1]) * self.sigma

            # mass center
            gt_masks_pt = torch.from_numpy(gt_masks).to(device=device)
            valid_mask_flags = gt_masks_pt.sum(dim=-1).sum(dim=-1) > 0

            output_stride = stride / 2

            for seg_mask, gt_label, half_h, half_w, valid_mask_flag in\
                    zip(gt_masks, gt_labels, half_hs, half_ws, valid_mask_flags):
                if not valid_mask_flag:
                    continue
                upsampled_size = (featmap_sizes[0][0] * 4, featmap_sizes[0][1] * 4)

                center_h, center_w = ndimage.measurements.center_of_mass(seg_mask)

                coord_w = int((center_w / upsampled_size[1]) // (1. / num_grid))  # 落在哪个格子
                coord_h = int((center_h / upsampled_size[0]) // (1. / num_grid))

                # left, top, right, down
                top_box = max(0, int(((center_h - half_h) / upsampled_size[0]) // (1. / num_grid)))
                down_box = min(num_grid - 1, int(((center_h + half_h) / upsampled_size[0]) // (1. / num_grid)))
                left_box = max(0, int(((center_w - half_w) / upsampled_size[1]) // (1. / num_grid)))
                right_box = min(num_grid - 1, int(((center_w + half_w) / upsampled_size[1]) // (1. / num_grid)))

                top = max(top_box, coord_h - 1)
                down = min(down_box, coord_h + 1)
                left = max(coord_w - 1, left_box)
                right = min(right_box, coord_w + 1)

                cate_label[top:(down + 1), left:(right + 1)] = gt_label
                # ins
                seg_mask = mmcv.imrescale(seg_mask, scale=1. / output_stride)
                seg_mask = torch.from_numpy(seg_mask).to(device=device)
                for i in range(top, down + 1):
                    for j in range(left, right + 1):
                        label = int(i * num_grid + j)
                        ins_label[label, :seg_mask.shape[0], :seg_mask.shape[1]] = seg_mask
                        ins_ind_label[label] = True
            ins_label_list.append(ins_label)
            cate_label_list.append(cate_label)
            ins_ind_label_list.append(ins_ind_label)
        return ins_label_list, cate_label_list, ins_ind_label_list

    def get_seg(self, seg_preds, cate_preds, img_metas, cfg, rescale=None):
        assert len(seg_preds) == len(cate_preds)
        num_levels = len(cate_preds)
        featmap_size = seg_preds[0].size()[-2:]

        bbox_result_list = []
        segm_result_list = []
        for img_id in range(len(img_metas)):
            cate_pred_list = [
                cate_preds[i][img_id].view(-1, self.cate_out_channels).detach() for i in range(num_levels)
            ]
            seg_pred_list = [
                seg_preds[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            ori_shape = img_metas[img_id]['ori_shape']

            cate_pred_list = torch.cat(cate_pred_list, dim=0)
            seg_pred_list = torch.cat(seg_pred_list, dim=0)

            result = self.get_seg_single(cate_pred_list, seg_pred_list,
                                         featmap_size, img_shape, ori_shape, scale_factor, cfg, rescale)
            bbox_result, segm_result = self.segm2result(result)
            bbox_result_list.append(bbox_result)
            segm_result_list.append(segm_result)
        return bbox_result_list, segm_result_list

    def segm2result(self, result):
        if result is None:
            bbox_result = [np.zeros((0, 5), dtype=np.float32) for i in
                           range(self.num_classes)]
            # BG is not included in num_classes
            segm_result = [[] for _ in range(self.num_classes)]
        else:
            bbox_result = [np.zeros((0, 5), dtype=np.float32) for i in
                           range(self.num_classes)]
            segm_result = [[] for _ in range(self.num_classes)]
            seg_pred = result[0].cpu().numpy()
            cate_label = result[1].cpu().numpy()
            cate_score = result[2].cpu().numpy()
            num_ins = seg_pred.shape[0]
            # fake bboxes
            bboxes = np.zeros((num_ins, 5), dtype=np.float32)
            bboxes[:, -1] = cate_score
            bbox_result = [bboxes[cate_label == i, :] for i in
                           range(self.num_classes)]
            for idx in range(num_ins):
                segm_result[cate_label[idx]].append(seg_pred[idx])
        return bbox_result, segm_result

    def get_seg_single(self,
                       cate_preds,
                       seg_preds,
                       featmap_size,
                       img_shape,
                       ori_shape,
                       scale_factor,
                       cfg,
                       rescale=False, debug=False):
        assert len(cate_preds) == len(seg_preds)

        # overall info.
        h, w, _ = img_shape
        upsampled_size_out = (featmap_size[0] * 4, featmap_size[1] * 4)

        # process.
        inds = (cate_preds > cfg.score_thr)
        # category scores.
        cate_scores = cate_preds[inds]
        if len(cate_scores) == 0:
            return None
        # category labels.
        inds = inds.nonzero()
        cate_labels = inds[:, 1]

        # strides.
        size_trans = cate_labels.new_tensor(self.seg_num_grids).pow(2).cumsum(0)
        strides = cate_scores.new_ones(size_trans[-1])
        n_stage = len(self.seg_num_grids)
        strides[:size_trans[0]] *= self.strides[0]
        for ind_ in range(1, n_stage):
            strides[size_trans[ind_ - 1]:size_trans[ind_]] *= self.strides[ind_]
        strides = strides[inds[:, 0]]

        # masks.
        seg_preds = seg_preds[inds[:, 0]]
        seg_masks = seg_preds > cfg.mask_thr
        sum_masks = seg_masks.sum((1, 2)).float()

        # filter.
        keep = sum_masks > strides
        if keep.sum() == 0:
            return None

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
        cate_scores = matrix_nms(seg_masks, cate_labels, cate_scores,
                                 kernel=cfg.kernel, sigma=cfg.sigma, sum_masks=sum_masks)

        # filter.
        keep = cate_scores >= cfg.update_thr
        if keep.sum() == 0:
            return None
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

        seg_preds = F.interpolate(seg_preds.unsqueeze(0),
                                  size=upsampled_size_out,
                                  mode='bilinear')[:, :, :h, :w]
        seg_masks = F.interpolate(seg_preds,
                                  size=ori_shape[:2],
                                  mode='bilinear').squeeze(0)
        seg_masks = seg_masks > cfg.mask_thr
        return seg_masks, cate_labels, cate_scores
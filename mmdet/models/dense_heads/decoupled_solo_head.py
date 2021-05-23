import mmcv
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import ModuleList
from scipy import ndimage

from mmdet.core import matrix_nms, multi_apply, points_nms
from mmdet.models.builder import HEADS
from .solo_head import SOLOHead


@HEADS.register_module()
class DecoupledSOLOHead(SOLOHead):

    def __init__(
        self,
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
        init_cfg=dict(type='Normal', layer='Conv2d', std=0.01,
                      bias_prob=0.01)):
        super(DecoupledSOLOHead, self).__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            seg_feat_channels=seg_feat_channels,
            stacked_convs=stacked_convs,
            strides=strides,
            base_edge_list=base_edge_list,
            scale_ranges=scale_ranges,
            sigma=sigma,
            num_grids=num_grids,
            cate_down_pos=cate_down_pos,
            with_deform=with_deform,
            loss_mask=loss_mask,
            loss_cls=loss_cls,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg)

    def _init_layers(self):
        self.ins_convs_x = ModuleList(
            init_cfg=dict(type='Normal', layer='Conv2d', std=0.01))
        self.ins_convs_y = ModuleList(
            init_cfg=dict(type='Normal', layer='Conv2d', std=0.01))
        self.cate_convs = ModuleList(
            init_cfg=dict(type='Normal', layer='Conv2d', std=0.01))

        for i in range(self.stacked_convs):
            chn = self.in_channels + 1 if i == 0 else self.seg_feat_channels
            self.ins_convs_x.append(
                ConvModule(
                    chn,
                    self.seg_feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))
            self.ins_convs_y.append(
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

        self.dsolo_ins_list_x = ModuleList(
            init_cfg=dict(
                type='Normal', layer='Conv2d', std=0.01, bias_prob=0.01))
        self.dsolo_ins_list_y = ModuleList(
            init_cfg=dict(
                type='Normal', layer='Conv2d', std=0.01, bias_prob=0.01))
        for seg_num_grid in self.seg_num_grids:
            self.dsolo_ins_list_x.append(
                nn.Conv2d(self.seg_feat_channels, seg_num_grid, 3, padding=1))
            self.dsolo_ins_list_y.append(
                nn.Conv2d(self.seg_feat_channels, seg_num_grid, 3, padding=1))
        self.dsolo_cate = nn.Conv2d(
            self.seg_feat_channels, self.cate_out_channels, 3, padding=1)

    def forward(self, feats):
        assert len(feats) == self.num_levels
        new_feats = self.split_feats(feats)
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
                if j == self.cate_down_pos:
                    seg_num_grid = self.seg_num_grids[i]
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
             gt_bbox_list,
             gt_label_list,
             img_metas,
             gt_mask_list,
             cfg=None,
             gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in ins_preds_x]

        ins_label_list, cate_label_list, \
            ins_ind_label_list, ins_ind_label_list_xy = \
            multi_apply(self.solo_target_single,
                        gt_bbox_list,
                        gt_label_list,
                        gt_mask_list,
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
                -1, self.cate_out_channels))
        cate_preds = cate_pred_temp

        num_ins = 0.
        # dice loss
        loss_mask = []
        for pred_x, pred_y, target in \
                zip(ins_preds_x_final, ins_preds_y_final, ins_labels):
            mask_n = pred_x.size(0)
            if mask_n == 0:
                continue
            num_ins += mask_n
            loss_mask.append(self.loss_mask((pred_x, pred_y), target))
        if num_ins > 0:
            loss_mask = torch.cat(loss_mask).mean()
        else:
            loss_mask = ins_preds_x_final[0].sum() * 0

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
                       featmap_sizes, self.seg_num_grids):

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
            gt_masks = gt_masks_raw[hit_indices.cpu().numpy(), ...]

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
                center_h, center_w = ndimage.measurements.center_of_mass(
                    seg_mask)

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
                seg_mask = mmcv.imrescale(seg_mask, scale=1. / output_stride)
                seg_mask = torch.Tensor(seg_mask)
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

    def get_seg(self,
                seg_preds_x,
                seg_preds_y,
                cate_preds,
                img_metas,
                cfg,
                rescale=None):
        assert len(seg_preds_x) == len(cate_preds)
        num_levels = len(cate_preds)
        featmap_size = seg_preds_x[0].size()[-2:]

        bbox_result_list = []
        segm_result_list = []
        for img_id in range(len(img_metas)):
            cate_pred_list = [
                cate_preds[i][img_id].view(-1,
                                           self.cate_out_channels).detach()
                for i in range(num_levels)
            ]
            seg_pred_list_x = [
                seg_preds_x[i][img_id].detach() for i in range(num_levels)
            ]
            seg_pred_list_y = [
                seg_preds_y[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            ori_shape = img_metas[img_id]['ori_shape']

            cate_pred_list = torch.cat(cate_pred_list, dim=0)
            seg_pred_list_x = torch.cat(seg_pred_list_x, dim=0)
            seg_pred_list_y = torch.cat(seg_pred_list_y, dim=0)

            result = self.get_seg_single(cate_pred_list, seg_pred_list_x,
                                         seg_pred_list_y, featmap_size,
                                         img_shape, ori_shape, scale_factor,
                                         cfg, rescale)
            bbox_result, segm_result = self.segm2result(result)
            bbox_result_list.append(bbox_result)
            segm_result_list.append(segm_result)
        return bbox_result_list, segm_result_list

    def get_seg_single(self,
                       cate_preds,
                       seg_preds_x,
                       seg_preds_y,
                       featmap_size,
                       img_shape,
                       ori_shape,
                       scale_factor,
                       cfg,
                       rescale=False):
        cfg = self.test_cfg if cfg is None else cfg
        # overall info.
        h, w, _ = img_shape
        upsampled_size_out = (featmap_size[0] * 4, featmap_size[1] * 4)

        # trans trans_diff.
        trans_size = torch.Tensor(self.seg_num_grids).pow(2).cumsum(0).long()
        trans_diff = torch.ones(
            trans_size[-1].item(), device=cate_preds.device).long()
        num_grids = torch.ones(
            trans_size[-1].item(), device=cate_preds.device).long()
        seg_size = torch.Tensor(self.seg_num_grids).cumsum(0).long()
        seg_diff = torch.ones(
            trans_size[-1].item(), device=cate_preds.device).long()
        strides = torch.ones(trans_size[-1].item(), device=cate_preds.device)

        n_stage = len(self.seg_num_grids)
        trans_diff[:trans_size[0]] *= 0
        seg_diff[:trans_size[0]] *= 0
        num_grids[:trans_size[0]] *= self.seg_num_grids[0]
        strides[:trans_size[0]] *= self.strides[0]

        for ind_ in range(1, n_stage):
            trans_diff[trans_size[ind_ - 1]:trans_size[ind_]] *= \
                trans_size[ind_ - 1]
            seg_diff[trans_size[ind_ - 1]:trans_size[ind_]] *= \
                seg_size[ind_ - 1]
            num_grids[trans_size[ind_ - 1]:trans_size[ind_]] *= \
                self.seg_num_grids[ind_]
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

        if len(cate_scores) == 0:
            return None

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
        return seg_masks, cate_labels, cate_scores

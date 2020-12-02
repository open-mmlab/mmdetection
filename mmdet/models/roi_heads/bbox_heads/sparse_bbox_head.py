import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import bias_init_with_prob
from mmcv.runner import force_fp32

from mmdet.core import multi_apply
from mmdet.models.builder import HEADS, build_loss
from ... import accuracy
from .bbox_head import BBoxHead


# TODO need this _DEFAULT_SCALE_CLAMP = math.log(100000.0 / 16) ??
@HEADS.register_module()
class SparseBBoxHead(BBoxHead):

    def __init__(self,
                 *args,
                 n_head=8,
                 dropout=0.1,
                 loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                 dynamicconv_cfg=dict(
                     hidden_dim=256,
                     dim_dynamic=64,
                     num_dynamic=2,
                     pooler_resolution=7),
                 num_cls_fcs=1,
                 num_reg_fcs=3,
                 share_fc_channels=2048,
                 **kwargs):
        kwargs.pop('roi_feat_size', None)
        super().__init__(*args, roi_feat_size=1, **kwargs)
        self.loss_iou = build_loss(loss_iou)
        self.dropout = dropout
        self.n_head = n_head
        self.self_attn = nn.MultiheadAttention(
            self.in_channels, n_head, dropout=dropout)
        dynamicconv_cfg = dynamicconv_cfg.copy()
        self.inst_interact = DynamicConv(**dynamicconv_cfg)

        self.linear1 = nn.Linear(self.in_channels, share_fc_channels)
        self.linear2 = nn.Linear(share_fc_channels, self.in_channels)

        cls_module = list()
        for _ in range(num_cls_fcs):
            cls_module.append(
                nn.Linear(self.in_channels, self.in_channels, False))
            cls_module.append(nn.LayerNorm(self.in_channels))
            cls_module.append(nn.ReLU(inplace=True))
        self.cls_module = nn.ModuleList(cls_module)

        reg_module = list()
        for _ in range(num_reg_fcs):
            reg_module.append(
                nn.Linear(self.in_channels, self.in_channels, False))
            reg_module.append(nn.LayerNorm(self.in_channels))
            reg_module.append(nn.ReLU(inplace=True))
        self.reg_module = nn.ModuleList(reg_module)

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(self.in_channels)
        self.norm2 = nn.LayerNorm(self.in_channels)
        self.norm3 = nn.LayerNorm(self.in_channels)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def init_weights(self):
        # conv layers are already initialized by ConvModule
        # init all parameters.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            bias_cls = bias_init_with_prob(0.01)
            if p.shape[-1] == self.num_classes:
                nn.init.constant_(p, bias_cls)

    def forward(self, roi_features, pro_features):
        # shared part
        batch_size = pro_features.size(0)
        num_anchors = pro_features.size(1)
        roi_features = roi_features.reshape(
            roi_features.size(0), roi_features.size(1), -1).permute(2, 0, 1)
        temp_pro_features = self.self_attn(
            pro_features, pro_features, value=pro_features)[0]
        pro_features = pro_features + self.dropout1(temp_pro_features)
        pro_features = self.norm1(pro_features)

        pro_features = pro_features.reshape(1, -1, pro_features.size(-1))

        temp_pro_features = self.inst_interact(pro_features, roi_features)
        pro_features = pro_features + self.dropout2(temp_pro_features)
        obj_features = self.norm2(pro_features)

        temp_pro_features = self.linear2(
            self.dropout(F.relu((self.linear1(obj_features)))))

        obj_features = obj_features + self.dropout3(temp_pro_features)
        obj_features = self.norm3(obj_features)

        fc_feature = obj_features.transpose(0, 1).reshape(
            batch_size * num_anchors, -1)
        cls_feature = fc_feature.clone()
        reg_feature = fc_feature.clone()
        for cls_layer in self.cls_module:
            cls_feature = cls_layer(cls_feature)
        for reg_layer in self.reg_module:
            reg_feature = reg_layer(reg_feature)
        cls_score = self.fc_cls(cls_feature)
        bbox_pred = self.fc_reg(reg_feature)

        return cls_score, bbox_pred, obj_features

    def _get_target_single(self, pos_inds, neg_inds, pos_bboxes, neg_bboxes,
                           pos_gt_bboxes, pos_gt_labels, cfg):
        num_pos = pos_bboxes.size(0)
        num_neg = neg_bboxes.size(0)
        num_samples = num_pos + num_neg

        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        labels = pos_bboxes.new_full((num_samples, ),
                                     self.num_classes,
                                     dtype=torch.long)
        label_weights = pos_bboxes.new_zeros(num_samples)
        bbox_targets = pos_bboxes.new_zeros(num_samples, 4)
        bbox_weights = pos_bboxes.new_zeros(num_samples, 4)
        if num_pos > 0:
            # TODO re_design _get_target_single
            #  and _get_targets in bbox_head
            labels[pos_inds] = pos_gt_labels
            pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            label_weights[pos_inds] = pos_weight
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    pos_bboxes, pos_gt_bboxes)
            else:
                pos_bbox_targets = pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1
        if num_neg > 0:
            label_weights[neg_inds] = 1.0

        assert label_weights.sum() == 100
        return labels, label_weights, bbox_targets, bbox_weights

    # TODO pr to redisign bbox_head
    def get_targets(self,
                    sampling_results,
                    gt_bboxes,
                    gt_labels,
                    rcnn_train_cfg,
                    concat=True):

        pos_inds_list = [res.pos_inds for res in sampling_results]
        neg_inds_list = [res.neg_inds for res in sampling_results]
        pos_bboxes_list = [res.pos_bboxes for res in sampling_results]
        neg_bboxes_list = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]
        labels, label_weights, bbox_targets, bbox_weights = multi_apply(
            self._get_target_single,
            pos_inds_list,
            neg_inds_list,
            pos_bboxes_list,
            neg_bboxes_list,
            pos_gt_bboxes_list,
            pos_gt_labels_list,
            cfg=rcnn_train_cfg)
        if concat:
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            bbox_targets = torch.cat(bbox_targets, 0)
            bbox_weights = torch.cat(bbox_weights, 0)
        return labels, label_weights, bbox_targets, bbox_weights

    # TODO redesign roi forward_train and remove this function
    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self,
             cls_score,
             bbox_pred,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             imgs_whwh,
             reduction_override=None):
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                losses['loss_cls'] = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                losses['acc'] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                num_pos = pos_inds.sum()
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.reshape(
                        bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.reshape(
                        bbox_pred.size(0), -1,
                        4)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred / imgs_whwh,
                    bbox_targets[pos_inds.type(torch.bool)] / imgs_whwh,
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=num_pos,
                )
                losses['loss_iou'] = self.loss_iou(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=num_pos,
                )
            else:
                losses['loss_bbox'] = bbox_pred.sum() * 0
                losses['loss_iou'] = bbox_pred.sum() * 0
        return losses


class DynamicConv(nn.Module):

    def __init__(self,
                 hidden_dim=256,
                 dim_dynamic=64,
                 num_dynamic=2,
                 pooler_resolution=7):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.dim_dynamic = dim_dynamic
        self.num_dynamic = num_dynamic
        self.num_params = self.hidden_dim * self.dim_dynamic
        self.dynamic_layer = nn.Linear(self.hidden_dim,
                                       self.num_dynamic * self.num_params)
        self.norm1 = nn.LayerNorm(self.dim_dynamic)
        self.norm2 = nn.LayerNorm(self.hidden_dim)
        self.activation = nn.ReLU(inplace=True)
        num_output = self.hidden_dim * pooler_resolution**2
        self.out_layer = nn.Linear(num_output, self.hidden_dim)
        self.norm3 = nn.LayerNorm(self.hidden_dim)

    def forward(self, pro_features, roi_features):
        '''
        pro_features: (1,  N * nr_boxes, self.d_model)
        roi_features: (49, N * nr_boxes, self.d_model)
        '''
        features = roi_features.permute(1, 0, 2)
        parameters = self.dynamic_layer(pro_features).permute(1, 0, 2)

        param1 = parameters[:, :, :self.num_params].reshape(
            -1, self.hidden_dim, self.dim_dynamic)
        param2 = parameters[:, :,
                            self.num_params:].reshape(-1, self.dim_dynamic,
                                                      self.hidden_dim)

        features = torch.bmm(features, param1)
        features = self.norm1(features)
        features = self.activation(features)

        features = torch.bmm(features, param2)
        features = self.norm2(features)
        features = self.activation(features)

        features = features.flatten(1)
        features = self.out_layer(features)
        features = self.norm3(features)
        features = self.activation(features)
        return features

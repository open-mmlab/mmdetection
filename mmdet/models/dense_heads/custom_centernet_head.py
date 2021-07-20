import torch
import torch.nn as nn
import math
from mmcv.cnn import bias_init_with_prob, normal_init
from mmcv.ops import batched_nms
from mmcv.runner import force_fp32

from mmdet.core import multi_apply
from mmdet.models import HEADS, build_loss
from mmdet.models.utils import gaussian_radius, gen_gaussian_target
from ..utils.gaussian_target import (get_local_maximum, get_topk_from_heatmap,
                                     transpose_and_gather_feat)
from .base_dense_head import BaseDenseHead
from .dense_test_mixins import BBoxTestMixin

#added by mmz
from typing import List
import torch.distributed as dist
from torch.nn import functional as F
from .centernet_head import CenterNetHead



class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


def get_norm(norm, out_channels):
    """
    Args:
        norm (str or callable): either one of BN, SyncBN, FrozenBN, GN;
            or a callable that takes a channel number and returns
            the normalization layer as a nn.Module.

    Returns:
        nn.Module or None: the normalization layer
    """
    if norm is None:
        return None
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            # "BN": BatchNorm2d,
            # # Fixed in https://github.com/pytorch/pytorch/pull/36382
            # "SyncBN": NaiveSyncBatchNorm if env.TORCH_VERSION <= (1, 5) else nn.SyncBatchNorm,
            # "FrozenBN": FrozenBatchNorm2d,
            "GN": lambda channels: nn.GroupNorm(32, channels),
            # for debugging:
            "nnSyncBN": nn.SyncBatchNorm,
            # "naiveSyncBN": NaiveSyncBatchNorm,
        }[norm]
    return norm(out_channels)


@HEADS.register_module()
class CustomCenterNetHead(BaseDenseHead, BBoxTestMixin):
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
        super(CustomCenterNetHead, self).__init__(init_cfg)
        self.num_classes = num_classes



        self.loss_center_heatmap = build_loss(loss_center_heatmap)
        self.loss_wh = build_loss(loss_wh)
        self.loss_offset = build_loss(loss_offset)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False

        self.out_kernel = 3
        norm = "GN"
        self.only_proposal = True


        ##########  initialize the   1.<cls_tower>   2.<bbox_tower>   3.<share_tower>[no]  4.<bbox_pred>  5.<agn_hm>  6.<cls_logits>[no]

        ######################################################## origin #########################
        # self.heatmap_head = self._build_head(in_channel, feat_channel,
        #                                      num_classes)
        # self.wh_head = self._build_head(in_channel, feat_channel, 2)
        # self.offset_head = self._build_head(in_channel, feat_channel, 2)


        ########################################################## lq #########################
        # head_configs = {"cls": (cfg.MODEL.CENTERNET.NUM_CLS_CONVS \
        #                         if not self.only_proposal else 0,
        #                         cfg.MODEL.CENTERNET.USE_DEFORMABLE),
        #                 "bbox": (cfg.MODEL.CENTERNET.NUM_BOX_CONVS,
        #                          cfg.MODEL.CENTERNET.USE_DEFORMABLE),
        #                 "share": (cfg.MODEL.CENTERNET.NUM_SHARE_CONVS,
        #                           cfg.MODEL.CENTERNET.USE_DEFORMABLE)}

        head_configs = {"cls": (4,False),
                        "bbox": (4,False),
                        "share": (0,False)}

        #############  centernet2 , channels from ["p3", "p4", "p5", "p6", "p7"]'s channels, config
        # in_channels = [s.channels for s in input_shape]
        # assert len(set(in_channels)) == 1, \
        #     "Each level must have the same channel!"
        # in_channels = in_channels[0]

        channels = {
            'cls': in_channel,
            'bbox': in_channel,
            'share': in_channel,
        }

        ##### initialize the     1.<cls_tower>    2.<bbox_tower>     3.<share_tower>
        for head in head_configs:
            tower = []
            num_convs, use_deformable = head_configs[head]
            channel = channels[head]
            for i in range(num_convs):
                conv_func = nn.Conv2d
                tower.append(conv_func(
                        in_channel if i == 0 else channel,
                        channel,
                        kernel_size=3, stride=1,
                        padding=1, bias=True
                ))
                if norm == 'GN' and channel % 32 != 0:
                    tower.append(nn.GroupNorm(25, channel))
                elif norm != '':
                    # print("please add get_norm function")
                    tower.append(get_norm(norm, channel))
                tower.append(nn.ReLU())
            self.add_module('{}_tower'.format(head),
                            nn.Sequential(*tower))

        ### initialize the    <bbox_pred>
        self.bbox_pred = nn.Conv2d(
            in_channel, 4, kernel_size=self.out_kernel,
            stride=1, padding=self.out_kernel // 2
        )

        ### initialize the     <scales>
        self.scales = nn.ModuleList(
            [Scale(init_value=1.0)])
        # self.scales = nn.ModuleList(
        #     [Scale(init_value=1.0) for _ in input_shape])


        ### initialize the     <agn_hm>
        self.agn_hm = nn.Conv2d(
            in_channel, 1, kernel_size=self.out_kernel,
            stride=1, padding=self.out_kernel // 2
        )

        ### initialize the <cls_logits>, config assigns it to false !
        if not self.only_proposal:
            cls_kernel_size = self.out_kernel
            self.cls_logits = nn.Conv2d(
                in_channel, self.num_classes,
                kernel_size=cls_kernel_size,
                stride=1,
                padding=cls_kernel_size // 2,
            )



    def _build_head(self, in_channel, feat_channel, out_channel):
        """Build head for each branch."""
        layer = nn.Sequential(
            nn.Conv2d(in_channel, feat_channel, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channel, out_channel, kernel_size=1))
        return layer


    def init_weights(self):
        """Initialize weights of the head."""
        ###################### origin ######################
        # bias_init = bias_init_with_prob(0.1)
        # self.heatmap_head[-1].bias.data.fill_(bias_init)
        # for head in [self.wh_head, self.offset_head]:
        #     for m in head.modules():
        #         if isinstance(m, nn.Conv2d):
        #             normal_init(m, std=0.001)

        ########################  lq #####################################

        ### initialize the    1.<cls_tower>   2.<bbox_tower>    3.<share_tower>   4.<bbox_pred>
        for modules in [
            self.cls_tower, self.bbox_tower,
            self.share_tower,
            self.bbox_pred,
        ]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        torch.nn.init.constant_(self.bbox_pred.bias, 8.)

        ### initialize the    <agn_hm>
        #prior_prob = cfg.MODEL.CENTERNET.PRIOR_PROB --> 0.01 in config
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)

        torch.nn.init.constant_(self.agn_hm.bias, bias_value)
        torch.nn.init.normal_(self.agn_hm.weight, std=0.01)

        ##############  if <cls_logits>
        if not self.only_proposal:
            torch.nn.init.constant_(self.cls_logits.bias, bias_value)
            torch.nn.init.normal_(self.cls_logits.weight, std=0.01)



    def forward(self, feats):
        """Forward features. Notice CenterNet head does not use FPN.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            cls (List[Tensor]): cls predict for
                all levels, the channels number is num_classes.
            bbox_reg (List[Tensor]): bbox_reg predicts for all levels, the channels
                number is 4.
            agn_hms (List[Tensor]): agn_hms predicts for all levels, the
               channels number is 1.
        """
        return multi_apply(self.forward_single, feats)



    def forward_single(self, feat):
        """Forward feature of a single level.

        Args:
            feat (Tensor): Feature of a single level.

        Returns:
            cls(Tensor): cls predicts, the channels number is class number: 80  # not used
            bbox_reg(Tensor): reg predicts, the channels number is 4
            agn_hms (Tensor): center predict heatmaps, the channels number is 1

        """

        feat = self.share_tower(feat)       # not used
        cls_tower = self.cls_tower(feat)    # not used
        bbox_tower = self.bbox_tower(feat)

        print("cls_tower:",cls_tower.size(), bbox_tower.size())
        if not self.only_proposal:
            clss = self.cls_logits(cls_tower)
        else:
            clss = None
        agn_hms = self.agn_hm(bbox_tower)
        reg = self.bbox_pred(bbox_tower)
        reg = self.scales[0](reg)
        # reg = self.scales[l](reg)
        bbox_reg = F.relu(reg)
        print("bbox_reg",bbox_reg.size(), agn_hms.size())
        return clss, bbox_reg, agn_hms


    # @force_fp32(apply_to=('center_heatmap_preds', 'wh_preds', 'offset_preds'))
    # def loss(self,
    #          center_heatmap_preds,
    #          wh_preds,
    #          offset_preds,
    #          gt_bboxes,
    #          gt_labels,
    #          img_metas,
    #          gt_bboxes_ignore=None):
    #     """Compute losses of the head.

    #     Args:
    #         center_heatmap_preds (list[Tensor]): center predict heatmaps for
    #            all levels with shape (B, num_classes, H, W).
    #         wh_preds (list[Tensor]): wh predicts for all levels with
    #            shape (B, 2, H, W).
    #         offset_preds (list[Tensor]): offset predicts for all levels
    #            with shape (B, 2, H, W).
    #         gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
    #             shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
    #         gt_labels (list[Tensor]): class indices corresponding to each box.
    #         img_metas (list[dict]): Meta information of each image, e.g.,
    #             image size, scaling factor, etc.
    #         gt_bboxes_ignore (None | list[Tensor]): specify which bounding
    #             boxes can be ignored when computing the loss. Default: None

    #     Returns:
    #         dict[str, Tensor]: which has components below:
    #             - loss_center_heatmap (Tensor): loss of center heatmap.
    #             - loss_wh (Tensor): loss of hw heatmap
    #             - loss_offset (Tensor): loss of offset heatmap.
    #     """
    #     assert len(center_heatmap_preds) == len(wh_preds) == len(
    #         offset_preds) == 1
    #     center_heatmap_pred = center_heatmap_preds[0]
    #     wh_pred = wh_preds[0]
    #     offset_pred = offset_preds[0]

    #     target_result, avg_factor = self.get_targets(gt_bboxes, gt_labels,
    #                                                  center_heatmap_pred.shape,
    #                                                  img_metas[0]['pad_shape'])

    #     center_heatmap_target = target_result['center_heatmap_target']
    #     wh_target = target_result['wh_target']
    #     offset_target = target_result['offset_target']
    #     wh_offset_target_weight = target_result['wh_offset_target_weight']

    #     # Since the channel of wh_target and offset_target is 2, the avg_factor
    #     # of loss_center_heatmap is always 1/2 of loss_wh and loss_offset.
    #     loss_center_heatmap = self.loss_center_heatmap(
    #         center_heatmap_pred, center_heatmap_target, avg_factor=avg_factor)
    #     loss_wh = self.loss_wh(
    #         wh_pred,
    #         wh_target,
    #         wh_offset_target_weight,
    #         avg_factor=avg_factor * 2)
    #     loss_offset = self.loss_offset(
    #         offset_pred,
    #         offset_target,
    #         wh_offset_target_weight,
    #         avg_factor=avg_factor * 2)
    #     return dict(
    #         loss_center_heatmap=loss_center_heatmap,
    #         loss_wh=loss_wh,
    #         loss_offset=loss_offset)

    def loss(self,
             clss_per_level,
             reg_pred_per_level,
             agn_hm_pred_per_level,
             gt_bboxes,
             img_metas,             
             gt_bboxes_ignore=None):
        """Compute losses of the dense head.

        Args:
            agn_hm_pred_per_level (list[Tensor]): center predict heatmaps for
               all levels with shape (B, 1, H, W).
            reg_pred_per_level (list[Tensor]): reg predicts for all levels with
               shape (B, 4, H, W).
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss. Default: None

        Returns:
            dict[str, Tensor]: which has components below:
                - loss_centernet_loc (Tensor): loss of center heatmap.
                - loss_centernet_agn_pos (Tensor): loss of 
                - loss_centernet_agn_neg (Tensor): loss of .
        """
        grids = self.compute_grids(agn_hm_pred_per_level)
        shapes_per_level = grids[0].new_tensor(
                    [(x.shape[2], x.shape[3]) for x in reg_pred_per_level])        
        pos_inds, reg_targets, flattened_hms = \
            self._get_ground_truth(
                grids, shapes_per_level, gt_bboxes)
        logits_pred, reg_pred, agn_hm_pred = self._flatten_outputs(
            clss_per_level, reg_pred_per_level, agn_hm_pred_per_level)
        losses = self.compute_losses(
            pos_inds, reg_targets, flattened_hms,
            logits_pred, reg_pred, agn_hm_pred)

        return losses

    def compute_losses(self, pos_inds, reg_targets, flattened_hms,
        logits_pred, reg_pred, agn_hm_pred):
        '''
        Inputs:
            pos_inds: N
            reg_targets: M x 4
            flattened_hms: M x C
            logits_pred: M x C
            reg_pred: M x 4
            agn_hm_pred: M x 1 or None
            N: number of positive locations in all images
            M: number of pixels from all FPN levels
            C: number of classes
        '''
        assert (torch.isfinite(reg_pred).all().item())
        num_pos_local = pos_inds.numel()
        # num_gpus = dist.get_world_size()
        num_gpus = 1
        total_num_pos = self.reduce_sum(
            pos_inds.new_tensor([num_pos_local])).item()
        num_pos_avg = max(total_num_pos / num_gpus, 1.0)

        losses = {}
        # if not self.only_proposal:
        #     pos_loss, neg_loss = heatmap_focal_loss_jit(
        #         logits_pred, flattened_hms, pos_inds, labels,
        #         alpha=self.hm_focal_alpha, 
        #         beta=self.hm_focal_beta, 
        #         gamma=self.loss_gamma, 
        #         reduction='sum',
        #         sigmoid_clamp=self.sigmoid_clamp,
        #         ignore_high_fp=self.ignore_high_fp,
        #     )
        #     pos_loss = self.pos_weight * pos_loss / num_pos_avg
        #     neg_loss = self.neg_weight * neg_loss / num_pos_avg
        #     losses['loss_centernet_pos'] = pos_loss
        #     losses['loss_centernet_neg'] = neg_loss
        
        reg_inds = torch.nonzero(reg_targets.max(dim=1)[0] >= 0).squeeze(1)
        reg_pred = reg_pred[reg_inds]
        reg_targets_pos = reg_targets[reg_inds]
        reg_weight_map = flattened_hms.max(dim=1)[0]
        reg_weight_map = reg_weight_map[reg_inds]
        # reg_weight_map = reg_weight_map * 0 + 1 \
        #     if self.not_norm_reg else reg_weight_map
        reg_weight_map = reg_weight_map * 0 + 1
        reg_norm = max(self.reduce_sum(reg_weight_map.sum()).item() / num_gpus, 1)
        # reg_loss = self.reg_weight * self.iou_loss(
        #     reg_pred, reg_targets_pos, reg_weight_map,
        #     reduction='sum') / reg_norm
        reg_loss = 1.0 * self.my_iou_loss(
            reg_pred, reg_targets_pos, reg_weight_map,
            reduction='sum') / reg_norm
        losses['loss_centernet_loc'] = reg_loss

        # if self.with_agn_hm:
        if True:
            cat_agn_heatmap = flattened_hms.max(dim=1)[0] # M
            agn_pos_loss, agn_neg_loss = self.binary_heatmap_focal_loss(
                agn_hm_pred, cat_agn_heatmap, pos_inds,
                alpha=0.25, 
                beta=4, 
                gamma=2.0,
                sigmoid_clamp=0.0001,
                ignore_high_fp=0.85,
            )
            # agn_pos_loss = self.pos_weight * agn_pos_loss / num_pos_avg
            # agn_neg_loss = self.neg_weight * agn_neg_loss / num_pos_avg
            agn_pos_loss = 0.5 * agn_pos_loss / num_pos_avg
            agn_neg_loss = 0.5 * agn_neg_loss / num_pos_avg
            losses['loss_centernet_agn_pos'] = agn_pos_loss
            losses['loss_centernet_agn_neg'] = agn_neg_loss
    
        # if self.debug:
        #     print('losses', losses)
        #     print('total_num_pos', total_num_pos)
        return losses        

    def compute_grids(self, agn_hm_pred_per_level):
        grids = []
        strides = [8, 16, 32, 64, 128]
        for level, agn_hm_pred in enumerate(agn_hm_pred_per_level):
            h, w = agn_hm_pred.size()[-2:]
            shifts_x = torch.arange(
                0, w * strides[level], 
                step = strides[level],
                dtype = torch.float32, device=agn_hm_pred.device)
            shifts_y = torch.arange(
                0, h * strides[level], 
                step = strides[level],
                dtype = torch.float32, device=agn_hm_pred.device)
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            grids_per_level = torch.stack((shift_x, shift_y), dim=1) + \
                strides[level] // 2
            grids.append(grids_per_level)
        return grids

    def _get_ground_truth(self, grids, shapes_per_level, gt_bboxes):
        '''
        Input:
            grids: list of tensors [(hl x wl, 2)]_l
            shapes_per_level: list of tuples L x 2:
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

        Retuen:
            pos_inds: N
            reg_targets: M x 4
            flattened_hms: M x C or M x 1
            N: number of objects in all images
            M: number of pixels from all FPN levels
        '''

        # get positive pixel index

        INF = 100000000
        strides = [8, 16, 32, 64, 128]
        num_classes=80
        sizes_of_interest=[[0, 80], [64, 160], [128, 320], [256, 640], [512, 10000000]]
        only_proposal=True
        hm_min_overlap=0.8
        delta=(1-hm_min_overlap)/(1+hm_min_overlap)
        min_radius=4

        pos_inds = self._get_label_inds(gt_bboxes, shapes_per_level) 

        heatmap_channels = num_classes
        L = len(grids)
        num_loc_list = [len(loc) for loc in grids]
        strides = torch.cat([
            shapes_per_level.new_ones(num_loc_list[l]) * strides[l] \
            for l in range(L)]).float() # M
        reg_size_ranges = torch.cat([
            shapes_per_level.new_tensor(sizes_of_interest[l]).float().view(
            1, 2).expand(num_loc_list[l], 2) for l in range(L)]) # M x 2
        grids = torch.cat(grids, dim=0) # M x 2
        M = grids.shape[0]

        reg_targets = []
        flattened_hms = []
        for i in range(len(gt_bboxes)): # images
            boxes = gt_bboxes[i] # N x 4
            # area = gt_instances[i].gt_boxes.area() # N
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            # gt_classes = gt_labels[i] # N in [0, self.num_classes]

            N = boxes.shape[0]
            if N == 0:
                reg_targets.append(grids.new_zeros((M, 4)) - INF)
                flattened_hms.append(
                    grids.new_zeros((
                        M, 1 if only_proposal else heatmap_channels)))
                continue
            
            l = grids[:, 0].view(M, 1) - boxes[:, 0].view(1, N) # M x N
            t = grids[:, 1].view(M, 1) - boxes[:, 1].view(1, N) # M x N
            r = boxes[:, 2].view(1, N) - grids[:, 0].view(M, 1) # M x N
            b = boxes[:, 3].view(1, N) - grids[:, 1].view(M, 1) # M x N
            reg_target = torch.stack([l, t, r, b], dim=2) # M x N x 4

            centers = ((boxes[:, [0, 1]] + boxes[:, [2, 3]]) / 2) # N x 2
            centers_expanded = centers.view(1, N, 2).expand(M, N, 2) # M x N x 2
            strides_expanded = strides.view(M, 1, 1).expand(M, N, 2)
            centers_discret = ((centers_expanded / strides_expanded).int() * \
                strides_expanded).float() + strides_expanded / 2 # M x N x 2
            
            is_peak = (((grids.view(M, 1, 2).expand(M, N, 2) - \
                centers_discret) ** 2).sum(dim=2) == 0) # M x N
            is_in_boxes = reg_target.min(dim=2)[0] > 0 # M x N
            is_center3x3 = self.get_center3x3(
                grids, centers, strides) & is_in_boxes # M x N
            is_cared_in_the_level = self.assign_reg_fpn(
                reg_target, reg_size_ranges) # M x N
            reg_mask = is_center3x3 & is_cared_in_the_level # M x N

            dist2 = ((grids.view(M, 1, 2).expand(M, N, 2) - \
                centers_expanded) ** 2).sum(dim=2) # M x N
            dist2[is_peak] = 0
            radius2 = delta ** 2 * 2 * area # N
            radius2 = torch.clamp(
                radius2, min=min_radius ** 2)
            weighted_dist2 = dist2 / radius2.view(1, N).expand(M, N) # M x N            
            reg_target = self._get_reg_targets(
                reg_target, weighted_dist2.clone(), reg_mask, area) # M x 4

            if only_proposal:
                flattened_hm = self._create_agn_heatmaps_from_dist(
                    weighted_dist2.clone()) # M x 1
            # else:
            #     flattened_hm = self._create_heatmaps_from_dist(
            #         weighted_dist2.clone(), gt_classes, 
            #         channels=heatmap_channels) # M x C

            reg_targets.append(reg_target)
            flattened_hms.append(flattened_hm)
        
        # transpose im first training_targets to level first ones
        # reg_targets = self._transpose(reg_targets, num_loc_list)

    #     This function is used to transpose image first training targets to 
    #         level first ones
        for im_i in range(len(reg_targets)):
            reg_targets[im_i] = torch.split(
                reg_targets[im_i], num_loc_list, dim=0)

        targets_level_first = []
        for targets_per_level in zip(*reg_targets):
            targets_level_first.append(
                torch.cat(targets_per_level, dim=0))
        reg_targets = targets_level_first


        # flattened_hms = self._transpose(flattened_hms, num_loc_list)
        for im_i in range(len(flattened_hms)):
            flattened_hms[im_i] = torch.split(
                flattened_hms[im_i], num_loc_list, dim=0)

        hms_level_first = []
        for hms_per_level in zip(*flattened_hms):
            hms_level_first.append(
                torch.cat(hms_per_level, dim=0))
        flattened_hms = hms_level_first

        for l in range(len(reg_targets)):
            reg_targets[l] = reg_targets[l] / float(strides[l])
        reg_targets = torch.cat([x for x in reg_targets], dim=0) # MB x 4
        flattened_hms = torch.cat([x for x in flattened_hms], dim=0) # MB x C
        
        return pos_inds, reg_targets, flattened_hms        

    def _get_label_inds(self, gt_bboxes, shapes_per_level):
        '''
        Inputs:
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            shapes_per_level: L x 2 [(h_l, w_l)]_L
        Returns:
            pos_inds: N'
        '''
        strides_temp = [8, 16, 32, 64, 128]
        pos_inds = []
        # labels = []
        L = len(strides_temp)
        B = len(gt_bboxes)
        shapes_per_level = shapes_per_level.long()
        loc_per_level = (shapes_per_level[:, 0] * shapes_per_level[:, 1]).long() # L
        level_bases = []
        s = 0
        for l in range(L):
            level_bases.append(s)
            s = s + B * loc_per_level[l]
        level_bases = shapes_per_level.new_tensor(level_bases).long() # L
        strides_default = shapes_per_level.new_tensor(strides_temp).float() # L
        for im_i in range(B):
            #targets_per_im = gt_instances[im_i]
            bboxes = gt_bboxes[im_i] # n x 4
            n = bboxes.shape[0]
            centers = ((bboxes[:, [0, 1]] + bboxes[:, [2, 3]]) / 2) # n x 2
            centers = centers.view(n, 1, 2).expand(n, L, 2)
            strides = strides_default.view(1, L, 1).expand(n, L, 2)
            centers_inds = (centers / strides).long() # n x L x 2
            Ws = shapes_per_level[:, 1].view(1, L).expand(n, L)
            pos_ind = level_bases.view(1, L).expand(n, L) + \
                       im_i * loc_per_level.view(1, L).expand(n, L) + \
                       centers_inds[:, :, 1] * Ws + \
                       centers_inds[:, :, 0] # n x L
            is_cared_in_the_level = self.assign_fpn_level(bboxes)
            pos_ind = pos_ind[is_cared_in_the_level].view(-1)
            # label = gt_labels.view(
            #     n, 1).expand(n, L)[is_cared_in_the_level].view(-1)

            pos_inds.append(pos_ind) # n'
            # labels.append(label) # n'
        pos_inds = torch.cat(pos_inds, dim=0).long()
        # labels = torch.cat(labels, dim=0)
        return pos_inds # N

    def assign_fpn_level(self, boxes):
        '''
        Inputs:
            boxes: n x 4
            size_ranges: L x 2
        Return:
            is_cared_in_the_level: n x L
        '''

        sizes_of_interest=[[0, 80], [64, 160], [128, 320], [256, 640], [512, 10000000]]
        size_ranges = boxes.new_tensor(
            sizes_of_interest).view(len(sizes_of_interest), 2) # L x 2
        crit = ((boxes[:, 2:] - boxes[:, :2]) **2).sum(dim=1) ** 0.5 / 2 # n
        n, L = crit.shape[0], size_ranges.shape[0]
        crit = crit.view(n, 1).expand(n, L)
        size_ranges_expand = size_ranges.view(1, L, 2).expand(n, L, 2)
        is_cared_in_the_level = (crit >= size_ranges_expand[:, :, 0]) & \
            (crit <= size_ranges_expand[:, :, 1])
        return is_cared_in_the_level

    # def _transpose(self, training_targets, num_loc_list):
    #     '''
    #     This function is used to transpose image first training targets to 
    #         level first ones
    #     :return: level first training targets
    #     '''
    #     for im_i in range(len(training_targets)):
    #         training_targets[im_i] = torch.split(
    #             training_targets[im_i], num_loc_list, dim=0)

    #     targets_level_first = []
    #     for targets_per_level in zip(*training_targets):
    #         targets_level_first.append(
    #             torch.cat(targets_per_level, dim=0))
    #     return targets_level_first

    # def cat(tensors: List[torch.Tensor], dim: int = 0):
    #     """
    #     Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    #     """
    #     assert isinstance(tensors, (list, tuple))
    #     if len(tensors) == 1:
    #         return tensors[0]
    #     return torch.cat(tensors, dim)

    def get_center3x3(self, locations, centers, strides):
        '''
        Inputs:
            locations: M x 2
            centers: N x 2
            strides: M
        '''
        M, N = locations.shape[0], centers.shape[0]
        locations_expanded = locations.view(M, 1, 2).expand(M, N, 2) # M x N x 2
        centers_expanded = centers.view(1, N, 2).expand(M, N, 2) # M x N x 2
        strides_expanded = strides.view(M, 1, 1).expand(M, N, 2) # M x N
        centers_discret = ((centers_expanded / strides_expanded).int() * \
            strides_expanded).float() + strides_expanded / 2 # M x N x 2
        dist_x = (locations_expanded[:, :, 0] - centers_discret[:, :, 0]).abs()
        dist_y = (locations_expanded[:, :, 1] - centers_discret[:, :, 1]).abs()
        return (dist_x <= strides_expanded[:, :, 0]) & \
            (dist_y <= strides_expanded[:, :, 0])

    def _get_reg_targets(self, reg_targets, dist, mask, area):
        '''
          reg_targets (M x N x 4): long tensor
          dist (M x N)
          is_*: M x N
        '''
        INF=100000000

        dist[mask == 0] = INF * 1.0
        min_dist, min_inds = dist.min(dim=1) # M
        reg_targets_per_im = reg_targets[
            range(len(reg_targets)), min_inds] # M x N x 4 --> M x 4
        reg_targets_per_im[min_dist == INF] = - INF
        return reg_targets_per_im

    def assign_reg_fpn(self, reg_targets_per_im, size_ranges):
        '''
        TODO (Xingyi): merge it with assign_fpn_level
        Inputs:
            reg_targets_per_im: M x N x 4
            size_ranges: M x 2
        '''
        crit = ((reg_targets_per_im[:, :, :2] + \
            reg_targets_per_im[:, :, 2:])**2).sum(dim=2) ** 0.5 / 2 # M x N
        is_cared_in_the_level = (crit >= size_ranges[:, [0]]) & \
            (crit <= size_ranges[:, [1]])
        return is_cared_in_the_level

    def _create_agn_heatmaps_from_dist(self, dist):
        '''
        TODO (Xingyi): merge it with _create_heatmaps_from_dist
        dist: M x N
        return:
          heatmaps: M x 1
        '''
        heatmaps = dist.new_zeros((dist.shape[0], 1))
        heatmaps[:, 0] = torch.exp(-dist.min(dim=1)[0])
        zeros = heatmaps < 1e-4
        heatmaps[zeros] = 0
        return heatmaps


    def _flatten_outputs(self, clss, reg_pred, agn_hm_pred):
        # Reshape: (N, F, Hl, Wl) -> (N, Hl, Wl, F) -> (sum_l N*Hl*Wl, F)
        with_agn_hm=True

        clss = torch.cat([x.permute(0, 2, 3, 1).reshape(-1, x.shape[1]) \
            for x in clss], 0) if clss[0] is not None else None
        reg_pred = torch.cat(
            [x.permute(0, 2, 3, 1).reshape(-1, 4) for x in reg_pred], 0)            
        agn_hm_pred = torch.cat([x.permute(0, 2, 3, 1).reshape(-1) \
            for x in agn_hm_pred], 0) if with_agn_hm else None
        return clss, reg_pred, agn_hm_pred

    def reduce_sum(self, tensor):
        # world_size = dist.get_world_size()
        num_gpus = 1
        world_size = num_gpus
        if world_size < 2:
            return tensor
        tensor = tensor.clone()
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        return tensor

    def my_iou_loss(self, pred, target, weight=None, reduction='sum'):
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]

        target_aera = (target_left + target_right) * \
                      (target_top + target_bottom)
        pred_aera = (pred_left + pred_right) * \
                    (pred_top + pred_bottom)

        w_intersect = torch.min(pred_left, target_left) + \
                      torch.min(pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) + \
                      torch.min(pred_top, target_top)

        g_w_intersect = torch.max(pred_left, target_left) + \
                        torch.max(pred_right, target_right)
        g_h_intersect = torch.max(pred_bottom, target_bottom) + \
                        torch.max(pred_top, target_top)
        ac_uion = g_w_intersect * g_h_intersect

        area_intersect = w_intersect * h_intersect
        area_union = target_aera + pred_aera - area_intersect

        ious = (area_intersect + 1.0) / (area_union + 1.0)
        gious = ious - (ac_uion - area_union) / ac_uion
        # if self.loc_loss_type == 'iou':
        #     losses = -torch.log(ious)
        # elif self.loc_loss_type == 'linear_iou':
        #     losses = 1 - ious
        # elif self.loc_loss_type == 'giou':
        #     losses = 1 - gious
        # else:
        #     raise NotImplementedError
        losses = 1 - gious
        # if weight is not None:
        #     losses = losses * weight
        # else:
        #     losses = losses

        losses = losses * weight

        if reduction == 'sum':
            return losses.sum()
        elif reduction == 'batch':
            return losses.sum(dim=[1])
        elif reduction == 'none':
            return losses
        else:
            raise NotImplementedError    

    def binary_heatmap_focal_loss(
        self,
        inputs,
        targets,
        pos_inds,
        alpha: float = -1,
        beta: float = 4,
        gamma: float = 2,
        sigmoid_clamp: float = 1e-4,
        ignore_high_fp: float = -1.,
        ):
        """
        Args:
            inputs:  (sum_l N*Hl*Wl,)
            targets: (sum_l N*Hl*Wl,)
            pos_inds: N
        Returns:
            Loss tensor with the reduction option applied.
        """
        pred = torch.clamp(inputs.sigmoid_(), min=sigmoid_clamp, max=1-sigmoid_clamp)
        neg_weights = torch.pow(1 - targets, beta)
        pos_pred = pred[pos_inds] # N
        pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, gamma)
        neg_loss = torch.log(1 - pred) * torch.pow(pred, gamma) * neg_weights
        if ignore_high_fp > 0:
            not_high_fp = (pred < ignore_high_fp).float()
            neg_loss = not_high_fp * neg_loss

        pos_loss = - pos_loss.sum()
        neg_loss = - neg_loss.sum()

        if alpha >= 0:
            pos_loss = alpha * pos_loss
            neg_loss = (1 - alpha) * neg_loss

        return pos_loss, neg_loss

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
               - center_heatmap_target (Tensor): targets of center heatmap, \
                   shape (B, num_classes, H, W).
               - wh_target (Tensor): targets of wh predict, shape \
                   (B, 2, H, W).
               - offset_target (Tensor): targets of offset predict, shape \
                   (B, 2, H, W).
               - wh_offset_target_weight (Tensor): weights of wh and offset \
                   predict, shape (B, 2, H, W).
        """
        img_h, img_w = img_shape[:2]
        bs, _, feat_h, feat_w = feat_shape

        width_ratio = float(feat_w / img_w)
        height_ratio = float(feat_h / img_h)

        center_heatmap_target = gt_bboxes[-1].new_zeros(
            [bs, self.num_classes, feat_h, feat_w])
        wh_target = gt_bboxes[-1].new_zeros([bs, 2, feat_h, feat_w])
        offset_target = gt_bboxes[-1].new_zeros([bs, 2, feat_h, feat_w])
        wh_offset_target_weight = gt_bboxes[-1].new_zeros(
            [bs, 2, feat_h, feat_w])

        for batch_id in range(bs):
            gt_bbox = gt_bboxes[batch_id]
            gt_label = gt_labels[batch_id]
            center_x = (gt_bbox[:, [0]] + gt_bbox[:, [2]]) * width_ratio / 2
            center_y = (gt_bbox[:, [1]] + gt_bbox[:, [3]]) * height_ratio / 2
            gt_centers = torch.cat((center_x, center_y), dim=1)

            for j, ct in enumerate(gt_centers):
                ctx_int, cty_int = ct.int()
                ctx, cty = ct
                scale_box_h = (gt_bbox[j][3] - gt_bbox[j][1]) * height_ratio
                scale_box_w = (gt_bbox[j][2] - gt_bbox[j][0]) * width_ratio
                radius = gaussian_radius([scale_box_h, scale_box_w],
                                         min_overlap=0.3)
                radius = max(0, int(radius))
                ind = gt_label[j]
                gen_gaussian_target(center_heatmap_target[batch_id, ind],
                                    [ctx_int, cty_int], radius)

                wh_target[batch_id, 0, cty_int, ctx_int] = scale_box_w
                wh_target[batch_id, 1, cty_int, ctx_int] = scale_box_h

                offset_target[batch_id, 0, cty_int, ctx_int] = ctx - ctx_int
                offset_target[batch_id, 1, cty_int, ctx_int] = cty - cty_int

                wh_offset_target_weight[batch_id, :, cty_int, ctx_int] = 1

        avg_factor = max(1, center_heatmap_target.eq(1).sum())
        target_result = dict(
            center_heatmap_target=center_heatmap_target,
            wh_target=wh_target,
            offset_target=offset_target,
            wh_offset_target_weight=wh_offset_target_weight)
        return target_result, avg_factor

    def get_bboxes(self,
                   center_heatmap_preds,
                   wh_preds,
                   offset_preds,
                   img_metas,
                   rescale=True,
                   with_nms=False):
        """Transform network output for a batch into bbox predictions.

        Args:
            center_heatmap_preds (list[Tensor]): center predict heatmaps for
                all levels with shape (B, num_classes, H, W).
            wh_preds (list[Tensor]): wh predicts for all levels with
                shape (B, 2, H, W).
            offset_preds (list[Tensor]): offset predicts for all levels
                with shape (B, 2, H, W).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            rescale (bool): If True, return boxes in original image space.
                Default: True.
            with_nms (bool): If True, do nms before return boxes.
                Default: False.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where 5 represent
                (tl_x, tl_y, br_x, br_y, score) and the score between 0 and 1.
                The shape of the second tensor in the tuple is (n,), and
                each element represents the class label of the corresponding
                box.
        """
        assert len(center_heatmap_preds) == len(wh_preds) == len(
            offset_preds) == 1
        scale_factors = [img_meta['scale_factor'] for img_meta in img_metas]
        border_pixs = [img_meta['border'] for img_meta in img_metas]

        batch_det_bboxes, batch_labels = self.decode_heatmap(
            center_heatmap_preds[0],
            wh_preds[0],
            offset_preds[0],
            img_metas[0]['batch_input_shape'],
            k=self.test_cfg.topk,
            kernel=self.test_cfg.local_maximum_kernel)

        batch_border = batch_det_bboxes.new_tensor(
            border_pixs)[:, [2, 0, 2, 0]].unsqueeze(1)
        batch_det_bboxes[..., :4] -= batch_border

        if rescale:
            batch_det_bboxes[..., :4] /= batch_det_bboxes.new_tensor(
                scale_factors).unsqueeze(1)

        if with_nms:
            det_results = []
            for (det_bboxes, det_labels) in zip(batch_det_bboxes,
                                                batch_labels):
                det_bbox, det_label = self._bboxes_nms(det_bboxes, det_labels,
                                                       self.test_cfg)
                det_results.append(tuple([det_bbox, det_label]))
        else:
            det_results = [
                tuple(bs) for bs in zip(batch_det_bboxes, batch_labels)
            ]
        return det_results

    def decode_heatmap(self,
                       center_heatmap_pred,
                       wh_pred,
                       offset_pred,
                       img_shape,
                       k=100,
                       kernel=3):
        """Transform outputs into detections raw bbox prediction.

        Args:
            center_heatmap_pred (Tensor): center predict heatmap,
               shape (B, num_classes, H, W).
            wh_pred (Tensor): wh predict, shape (B, 2, H, W).
            offset_pred (Tensor): offset predict, shape (B, 2, H, W).
            img_shape (list[int]): image shape in [h, w] format.
            k (int): Get top k center keypoints from heatmap. Default 100.
            kernel (int): Max pooling kernel for extract local maximum pixels.
               Default 3.

        Returns:
            tuple[torch.Tensor]: Decoded output of CenterNetHead, containing
               the following Tensors:

              - batch_bboxes (Tensor): Coords of each box with shape (B, k, 5)
              - batch_topk_labels (Tensor): Categories of each box with \
                  shape (B, k)
        """
        height, width = center_heatmap_pred.shape[2:]
        inp_h, inp_w = img_shape

        center_heatmap_pred = get_local_maximum(
            center_heatmap_pred, kernel=kernel)

        *batch_dets, topk_ys, topk_xs = get_topk_from_heatmap(
            center_heatmap_pred, k=k)
        batch_scores, batch_index, batch_topk_labels = batch_dets

        wh = transpose_and_gather_feat(wh_pred, batch_index)
        offset = transpose_and_gather_feat(offset_pred, batch_index)
        topk_xs = topk_xs + offset[..., 0]
        topk_ys = topk_ys + offset[..., 1]
        tl_x = (topk_xs - wh[..., 0] / 2) * (inp_w / width)
        tl_y = (topk_ys - wh[..., 1] / 2) * (inp_h / height)
        br_x = (topk_xs + wh[..., 0] / 2) * (inp_w / width)
        br_y = (topk_ys + wh[..., 1] / 2) * (inp_h / height)

        batch_bboxes = torch.stack([tl_x, tl_y, br_x, br_y], dim=2)
        batch_bboxes = torch.cat((batch_bboxes, batch_scores[..., None]),
                                 dim=-1)
        return batch_bboxes, batch_topk_labels

    def _bboxes_nms(self, bboxes, labels, cfg):
        if labels.numel() == 0:
            return bboxes, labels

        out_bboxes, keep = batched_nms(bboxes[:, :4], bboxes[:, -1], labels,
                                       cfg.nms_cfg)
        out_labels = labels[keep]

        if len(out_bboxes) > 0:
            idx = torch.argsort(out_bboxes[:, -1], descending=True)
            idx = idx[:cfg.max_per_img]
            out_bboxes = out_bboxes[idx]
            out_labels = out_labels[idx]

        return out_bboxes, out_labels


# if __name__ == "__main__":
#     centernet2_test = CustomCenterNetHead(in_channel=50, feat_channel=25)
#     centernet2_test.init_weights()
#     print(centernet2_test)
#     feature = torch.randn((16,120,120,16))
#
#
#     centernet2_test([feature, feature, feature])
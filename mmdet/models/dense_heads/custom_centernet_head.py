import torch
import torch.nn as nn
import math
from mmcv.cnn import bias_init_with_prob, normal_init
from mmcv.ops import batched_nms
from mmcv.runner import force_fp32

from mmdet.core import multi_apply
from mmdet.core import distance2bbox
from mmdet.models import HEADS, build_loss
from mmdet.models.utils import gaussian_radius, gen_gaussian_target
from ..utils.gaussian_target import (get_local_maximum, get_topk_from_heatmap,
                                     transpose_and_gather_feat)
# from mmdet.models.utils import _transpose
from .base_dense_head import BaseDenseHead
from .dense_test_mixins import BBoxTestMixin

# added by mmz
from typing import List
import torch.distributed as dist
from torch.nn import functional as F
from .centernet_head import CenterNetHead


INF = 100000000


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
            # "SyncBN": NaiveSyncBatchNorm if env.TORCH_VERSION <= (1, 5) 
            # else nn.SyncBatchNorm,
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
                 num_classes,
                 num_features,
                 num_cls_convs,
                 num_box_convs,
                 num_share_convs,
                 use_deformable,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super(CustomCenterNetHead, self).__init__(init_cfg)
        self.out_kernel = 3
        self.norm = "GN"
        self.only_proposal = True

        self.num_classes = num_classes
        self.strides = [8, 16, 32, 64, 128]
        self.hm_min_overlap = 0.8
        self.delta = (1-self.hm_min_overlap)/(1+self.hm_min_overlap)
        self.sizes_of_interest = [[0, 80], [64, 160], [128, 320], [256, 640], [512, 10000000]]
        self.min_radius = 4
        self.with_agn_hm = True
        self.pos_weight = 0.5
        self.neg_weight = 0.5
        self.not_norm_reg = True
        self.reg_weight = 1.0

        self.hm_focal_alpha = 0.25
        self.hm_focal_beta = 4
        self.loss_gamma = 2.0
        self.sigmoid_clamp = 0.0001
        self.ignore_high_fp = 0.85

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.fp16_enabled = False
        self.more_pos = False
        self.center_nms = False
        self.pre_nms_topk_train = 4000
        self.pre_nms_topk_test = 1000
        self.nms_thresh_train = 0.9
        self.nms_thresh_test = 0.9
        self.post_nms_topk_train = 2000
        self.post_nms_topk_test = 256
        self.more_pos_topk = 9
        self.score_thresh = 0.0001
        self.not_nms = False

        head_configs = {"cls": (4, False),
                        "bbox": (4, False),
                        "share": (0, False)}

<<<<<<< HEAD
        # centernet2, channels from ["p3", "p4", "p5", "p6", "p7"]
        # in_channels = [s.channels for s in input_shape]
        # assert len(set(in_channels)) == 1, \
        #     "Each level must have the same channel!"
        # in_channels = in_channels[0]
=======
        head_configs = {"cls": (num_cls_convs,False),
                        "bbox": (num_box_convs,False),
                        "share": (num_share_convs,False)}
>>>>>>> c98a98393bd8f1a98ad34d3956774e335f006d19

        channels = {
            'cls': in_channel,
            'bbox': in_channel,
            'share': in_channel,
        }

<<<<<<< HEAD
        # initialize the 1.<cls_tower> 2.<bbox_tower> 3.<share_tower>
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

        # initialize the    <bbox_pred>
        self.bbox_pred = nn.Conv2d(
            in_channel, 4, kernel_size=self.out_kernel,
            stride=1, padding=self.out_kernel // 2
        )

        # initialize the     <scales>
        self.scales = nn.ModuleList(
            [Scale(init_value=1.0) for _ in range(num_features)])

        # initialize the     <agn_hm>
        self.agn_hm = nn.Conv2d(
            in_channel, 1, kernel_size=self.out_kernel,
            stride=1, padding=self.out_kernel // 2
        )
=======
        self._build_tower(head_configs, channels)               # init  1.<cls_tower>    2.<bbox_tower>     3.<share_tower>

        self.bbox_pred = self._build_head(in_channel, 4)

        self.scales = nn.ModuleList(                                # init   <scales>
            [Scale(init_value=1.0) for _ in range(num_features)])

        self.agn_hm = self._build_head(in_channel, 1)
>>>>>>> c98a98393bd8f1a98ad34d3956774e335f006d19

        # initialize the <cls_logits>, config assigns it to false !
        if not self.only_proposal:
            cls_kernel_size = self.out_kernel
            self.cls_logits = nn.Conv2d(
                in_channel, self.num_classes,
                kernel_size=cls_kernel_size,
                stride=1,
                padding=cls_kernel_size // 2,
            )

<<<<<<< HEAD
    def init_weights(self):
        """Initialize weights of the head."""
        # bias_init = bias_init_with_prob(0.1)
        # self.heatmap_head[-1].bias.data.fill_(bias_init)
        # for head in [self.wh_head, self.offset_head]:
        #     for m in head.modules():
        #         if isinstance(m, nn.Conv2d):
        #             normal_init(m, std=0.001)

        #  lq ##
=======

    def _build_head(self, in_channel, out_channel):
        """Build head for each branch."""
        layer = nn.Conv2d(
            in_channel, out_channel, kernel_size=self.out_kernel,
            stride=1, padding=self.out_kernel // 2
        )
        return layer


    def _build_tower(self, head_configs, channels):
        ##### init the     1.<cls_tower>    2.<bbox_tower>     3.<share_tower>
        for head in head_configs:
            tower = []
            num_convs, use_deformable = head_configs[head]
            channel = channels[head]
            for i in range(num_convs):
                conv_func = nn.Conv2d
                tower.append(conv_func(
                        channel,
                        channel,
                        kernel_size=3, stride=1,
                        padding=1, bias=True
                ))
                if self.norm == 'GN' and channel % 32 != 0:
                    tower.append(nn.GroupNorm(25, channel))
                elif self.norm != '':
                    # print("please add get_norm function")
                    tower.append(get_norm(self.norm, channel))
                tower.append(nn.ReLU())
            self.add_module('{}_tower'.format(head),
                            nn.Sequential(*tower))



    def init_weights(self):
        """Initialize weights of the head."""
>>>>>>> c98a98393bd8f1a98ad34d3956774e335f006d19

        # initialize the    1.<cls_tower>   2.<bbox_tower>    3.<share_tower>   4.<bbox_pred>
        for modules in [
            self.cls_tower, self.bbox_tower,
            self.share_tower,
            self.bbox_pred,
        ]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)

        torch.nn.init.constant_(self.bbox_pred.bias, 8.)

        # initialize the    <agn_hm>
        # prior_prob = cfg.MODEL.CENTERNET.PRIOR_PROB --> 0.01 in config
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)

        torch.nn.init.constant_(self.agn_hm.bias, bias_value)
        torch.nn.init.normal_(self.agn_hm.weight, std=0.01)

        # if <cls_logits>
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
            bbox_reg (List[Tensor]): bbox_reg predicts for all levels, 
                the channels number is 4.
            agn_hms (List[Tensor]): agn_hms predicts for all levels, 
                the channels number is 1.
        """
        return multi_apply(self.forward_single, feats, 
                                [i for i in range(len(feats))])

    def forward_single(self, feat, i):
        """Forward feature of a single level.

        Args:
            feat (Tensor): Feature of a single level.

        Returns:
            cls(Tensor): cls predicts, the channels number is class number: 80
            bbox_reg(Tensor): reg predicts, the channels number is 4
            agn_hms (Tensor): center predict heatmaps, the channels number is 1

        """
        # feat = self.share_tower(feat)
        # cls_tower = self.cls_tower(feat)
        # bbox_tower = self.bbox_tower(feat)
        # if not self.only_proposal:
        #     clss = self.cls_logits(cls_tower)
        # else:
        #     clss = None

        # if self.with_agn_hm:
        #     agn_hms = self.agn_hm(bbox_tower)
        # else:
        #     agn_hms = None
        # reg = self.bbox_pred(bbox_tower)
        # reg = self.scales(reg)
        # return clss, reg, agn_hms
        feat = self.share_tower(feat)       # not used
        cls_tower = self.cls_tower(feat)    # not used
        bbox_tower = self.bbox_tower(feat)
        # print("cls_tower:",cls_tower.size(), bbox_tower.size())
        if not self.only_proposal:
            clss = self.cls_logits(cls_tower)
        else:
            clss = None
        agn_hms = self.agn_hm(bbox_tower)
        reg = self.bbox_pred(bbox_tower)
        reg = self.scales[i](reg)
        # reg = self.scales[l](reg)
        bbox_reg = F.relu(reg)
        # print("bbox_reg",bbox_reg.size(), agn_hms.size())
        return clss, bbox_reg, agn_hms

    def loss(self,
             clss_per_level,
             reg_pred_per_level,
             agn_hm_pred_per_level,
             gt_bboxes,
             gt_labels,
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
        num_gpus = self.get_world_size()
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
        reg_weight_map = reg_weight_map * 0 + 1 \
            if self.not_norm_reg else reg_weight_map
        reg_norm = \
            max(self.reduce_sum(reg_weight_map.sum()).item() / num_gpus, 1)
        reg_loss = self.reg_weight * self.my_iou_loss(
            reg_pred, reg_targets_pos, reg_weight_map,
            reduction='sum') / reg_norm
        losses['loss_centernet_loc'] = reg_loss

        if self.with_agn_hm:
            cat_agn_heatmap = flattened_hms.max(dim=1)[0]  # M
            agn_pos_loss, agn_neg_loss = self.binary_heatmap_focal_loss(
                agn_hm_pred, cat_agn_heatmap, pos_inds,
                alpha=self.hm_focal_alpha, 
                beta=self.hm_focal_beta, 
                gamma=self.loss_gamma,
                sigmoid_clamp=self.sigmoid_clamp,
                ignore_high_fp=self.ignore_high_fp,
            )
            agn_pos_loss = self.pos_weight * agn_pos_loss / num_pos_avg
            agn_neg_loss = self.neg_weight * agn_neg_loss / num_pos_avg
            losses['loss_centernet_agn_pos'] = agn_pos_loss
            losses['loss_centernet_agn_neg'] = agn_neg_loss
    
        # if self.debug:
        #     print('losses', losses)
        #     print('total_num_pos', total_num_pos)
        return losses        

    def compute_grids(self, agn_hm_pred_per_level):
        grids = []
        for level, agn_hm_pred in enumerate(agn_hm_pred_per_level):
            h, w = agn_hm_pred.size()[-2:]
            shifts_x = torch.arange(
                0, w * self.strides[level],
                step=self.strides[level],
                dtype=torch.float32, device=agn_hm_pred.device)
            shifts_y = torch.arange(
                0, h * self.strides[level],
                step=self.strides[level],
                dtype=torch.float32, device=agn_hm_pred.device)
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            grids_per_level = torch.stack((shift_x, shift_y), dim=1) + \
                self.strides[level] // 2
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
        pos_inds = self._get_label_inds(gt_bboxes, shapes_per_level) 

        heatmap_channels = self.num_classes
        L = len(grids)
        num_loc_list = [len(loc) for loc in grids]
        strides = torch.cat([
            shapes_per_level.new_ones(num_loc_list[l_]) * self.strides[l_]
            for l_ in range(L)]).float()  # M
        reg_size_ranges = \
            torch.cat([shapes_per_level.new_tensor(self.sizes_of_interest[l_]).float().view(1, 2).expand(num_loc_list[l_], 2)
                        for l_ in range(L)])  # M x 2
        grids = torch.cat(grids, dim=0)  # M x 2
        M = grids.shape[0]

        reg_targets = []
        flattened_hms = []
        for i in range(len(gt_bboxes)):  # images
            boxes = gt_bboxes[i]  # N x 4
            # area = gt_instances[i].gt_boxes.area() # N
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            # gt_classes = gt_labels[i] # N in [0, self.num_classes]

            N = boxes.shape[0]
            if N == 0:
                reg_targets.append(grids.new_zeros((M, 4)) - self.INF)
                flattened_hms.append(
                    grids.new_zeros((
                        M, 1 if self.only_proposal else heatmap_channels)))
                continue

            l = grids[:, 0].view(M, 1) - boxes[:, 0].view(1, N)  # M x N
            t = grids[:, 1].view(M, 1) - boxes[:, 1].view(1, N)  # M x N
            r = boxes[:, 2].view(1, N) - grids[:, 0].view(M, 1)  # M x N
            b = boxes[:, 3].view(1, N) - grids[:, 1].view(M, 1)  # M x N
            reg_target = torch.stack([l, t, r, b], dim=2)  # M x N x 4

            centers = ((boxes[:, [0, 1]] + boxes[:, [2, 3]]) / 2)  # N x 2
            centers_expanded = centers.view(1, N, 2).expand(M, N, 2)  # M x N x 2
            strides_expanded = strides.view(M, 1, 1).expand(M, N, 2)
            centers_discret = ((centers_expanded / strides_expanded).int() * 
                strides_expanded).float() + strides_expanded / 2  # M x N x 2
            
            is_peak = (((grids.view(M, 1, 2).expand(M, N, 2) - 
                centers_discret) ** 2).sum(dim=2) == 0)  # M x N
            is_in_boxes = reg_target.min(dim=2)[0] > 0  # M x N
            is_center3x3 = self.get_center3x3(
                grids, centers, strides) & is_in_boxes  # M x N
            is_cared_in_the_level = self.assign_reg_fpn(
                reg_target, reg_size_ranges)  # M x N
            reg_mask = is_center3x3 & is_cared_in_the_level  # M x N

            dist2 = ((grids.view(M, 1, 2).expand(M, N, 2) - centers_expanded) ** 2).sum(dim=2)  # M x N
            dist2[is_peak] = 0
            radius2 = self.delta ** 2 * 2 * area  # N
            radius2 = torch.clamp(
                radius2, min=self.min_radius ** 2)
            weighted_dist2 = dist2 / radius2.view(1, N).expand(M, N)  # M x N            
            reg_target = self._get_reg_targets(
                reg_target, weighted_dist2.clone(), reg_mask, area)  # M x 4

            if self.only_proposal:
                flattened_hm = self._create_agn_heatmaps_from_dist(
                    weighted_dist2.clone())  # M x 1
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

        for i in range(len(reg_targets)):
            reg_targets[i] = reg_targets[i] / float(self.strides[i])
        reg_targets = torch.cat([x for x in reg_targets], dim=0)  # MB x 4
        flattened_hms = torch.cat([x for x in flattened_hms], dim=0)  # MB x C
        
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
        pos_inds = []
        # labels = []
        L = len(self.strides)
        B = len(gt_bboxes)
        shapes_per_level = shapes_per_level.long()
        loc_per_level = (shapes_per_level[:, 0] * shapes_per_level[:, 1]).long() # L
        level_bases = []
        s = 0
        for l in range(L):
            level_bases.append(s)
            s = s + B * loc_per_level[l]
        level_bases = shapes_per_level.new_tensor(level_bases).long()  # L
        strides_default = shapes_per_level.new_tensor(self.strides).float()  # L
        for im_i in range(B):
            # targets_per_im = gt_instances[im_i]
            bboxes = gt_bboxes[im_i]  # n x 4
            n = bboxes.shape[0]
            centers = ((bboxes[:, [0, 1]] + bboxes[:, [2, 3]]) / 2)  # n x 2
            centers = centers.view(n, 1, 2).expand(n, L, 2)
            strides = strides_default.view(1, L, 1).expand(n, L, 2)
            centers_inds = (centers / strides).long()  # n x L x 2
            Ws = shapes_per_level[:, 1].view(1, L).expand(n, L)
            pos_ind = level_bases.view(1, L).expand(n, L) + \
                    im_i * loc_per_level.view(1, L).expand(n, L) + \
                    centers_inds[:, :, 1] * Ws + \
                    centers_inds[:, :, 0]  # n x L
            is_cared_in_the_level = self.assign_fpn_level(bboxes)
            pos_ind = pos_ind[is_cared_in_the_level].view(-1)
            # label = gt_labels.view(
            #     n, 1).expand(n, L)[is_cared_in_the_level].view(-1)

            pos_inds.append(pos_ind)  # n'
            # labels.append(label)  # n'
        pos_inds = torch.cat(pos_inds, dim=0).long()
        # labels = torch.cat(labels, dim=0)
        return pos_inds  # N

    def assign_fpn_level(self, boxes):
        '''
        Inputs:
            boxes: n x 4
            size_ranges: L x 2
        Return:
            is_cared_in_the_level: n x L
        '''
        size_ranges = boxes.new_tensor(
            self.sizes_of_interest).view(len(self.sizes_of_interest), 2) # L x 2
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
        dist[mask == 0] = INF * 1.0
        min_dist, min_inds = dist.min(dim=1)  # M
        reg_targets_per_im = reg_targets[
            range(len(reg_targets)), min_inds]  # M x N x 4 --> M x 4
        reg_targets_per_im[min_dist == INF] = - INF
        return reg_targets_per_im

    def assign_reg_fpn(self, reg_targets_per_im, size_ranges):
        '''
        TODO (Xingyi): merge it with assign_fpn_level
        Inputs:
            reg_targets_per_im: M x N x 4
            size_ranges: M x 2
        '''
        crit = ((reg_targets_per_im[:, :, :2] + 
            reg_targets_per_im[:, :, 2:])**2).sum(dim=2) ** 0.5 / 2  # M x N
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


        clss = torch.cat([x.permute(0, 2, 3, 1).reshape(-1, x.shape[1]) \
            for x in clss], 0) if clss[0] is not None else None
        reg_pred = torch.cat(
            [x.permute(0, 2, 3, 1).reshape(-1, 4) for x in reg_pred], 0)            
        agn_hm_pred = torch.cat([x.permute(0, 2, 3, 1).reshape(-1) \
            for x in agn_hm_pred], 0) if self.with_agn_hm else None
        return clss, reg_pred, agn_hm_pred

    def reduce_sum(self, tensor):
        world_size = self.get_world_size()
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

    def get_world_size(self) -> int:
        if not dist.is_available():
            return 1
        if not dist.is_initialized():
            return 1
        return dist.get_world_size()

    def get_bboxes(self, clss_per_level, reg_pred_per_level, agn_hm_pred_per_level, img_metas, cfg=None):
        
        grids = self.compute_grids(agn_hm_pred_per_level)

        # if self.more_pos:
        #     # add more pixels as positive if \
        #     #   1. they are within the center3x3 region of an object
        #     #   2. their regression losses are small (<self.more_pos_thresh)
        #     pos_inds, labels = self._add_more_pos(
        #         reg_pred, gt_bboxes, gt_labels, shapes_per_level)
            
        proposals = None
        image_sizes = []
        for i in range(len(img_metas)):
            image_sizes.append(img_metas[i]['ori_shape'][:2])
        if self.only_proposal:
            agn_hm_pred_per_level = [x.sigmoid() for x in agn_hm_pred_per_level]
            proposals = self.predict_instances(
                grids, agn_hm_pred_per_level, reg_pred_per_level, 
                image_sizes, [None for _ in agn_hm_pred_per_level])
        elif self.as_proposal:  # category specific bbox as agnostic proposals
            clss_per_level = [x.sigmoid() for x in clss_per_level]
            proposals = self.predict_instances(
                grids, clss_per_level, reg_pred_per_level,
                image_sizes, agn_hm_pred_per_level)
        # if self.only_proposal or self.as_proposal:
        #     for p in range(len(proposals)):
        #         proposals[p].proposal_boxes = proposals[p].get('pred_boxes')
        #         proposals[p].objectness_logits = proposals[p].get('scores')
        #         proposals[p].remove('pred_boxes')
        #         proposals[p].remove('scores')
        #         proposals[p].remove('pred_classes')

        # losses = self.losses(
        #     pos_inds, labels, reg_targets, flattened_hms,
        #     logits_pred, reg_pred, agn_hm_pred)

        # if self.debug:
        #     debug_train(
        #         [self.denormalizer(x) for x in images], 
        #         gt_instances, flattened_hms, reg_targets, 
        #         labels, pos_inds, shapes_per_level, grids, self.strides)
        return proposals

    def inference(self, images, clss_per_level, reg_pred_per_level, 
                    agn_hm_pred_per_level, grids):
        logits_pred = [x.sigmoid() if x is not None else None \
            for x in clss_per_level]
        agn_hm_pred_per_level = [x.sigmoid() if x is not None else None \
            for x in agn_hm_pred_per_level]

        if self.only_proposal:
            proposals, proposals_class = self.predict_instances(
                grids, agn_hm_pred_per_level, reg_pred_per_level, 
                images.image_sizes, [None for _ in agn_hm_pred_per_level])
        else:
            proposals, proposals_class = self.predict_instances(
                grids, logits_pred, reg_pred_per_level, 
                images.image_sizes, agn_hm_pred_per_level)
        # if self.as_proposal or self.only_proposal:
        #     for p in range(len(proposals)):
        #         proposals[p].proposal_boxes = proposals[p].get('pred_boxes')
        #         proposals[p].objectness_logits = proposals[p].get('scores')
        #         proposals[p].remove('pred_boxes')

        # if self.debug:
        #     debug_test(
        #         [self.denormalizer(x) for x in images], 
        #         logits_pred, reg_pred_per_level, 
        #         agn_hm_pred_per_level, preds=proposals,
        #         vis_thresh=self.vis_thresh, 
        #         debug_show_name=False)
        return proposals, proposals_class, {}


    def predict_instances(self, grids, logits_pred, reg_pred, image_sizes, 
                            agn_hm_pred, is_proposal=False):
        sampled_boxes = []
        for l in range(len(grids)):
            sampled_boxes.append(self.predict_single_level(
                grids[l], logits_pred[l], reg_pred[l] * self.strides[l],
                image_sizes, agn_hm_pred[l], l, is_proposal=is_proposal))
        boxlists = list(zip(*sampled_boxes))
        boxlists = [torch.cat(boxlist) for boxlist in boxlists]
        final_boxlists = []
        for l in range(len(boxlists)):
            final_boxlists.append(self.nms_and_topK(boxlists[l], nms=True))

        # boxes_scores_lists = boxlists[:, :5]
        # per_box_class = boxlists[:,5]
        return final_boxlists

    def predict_single_level(self, grids, heatmap, reg_pred, image_sizes,
                                agn_hm, level, is_proposal=False):
        N, C, H, W = heatmap.shape
        # put in the same format as grids
        if self.center_nms:
            heatmap_nms = nn.functional.max_pool2d(
                heatmap, (3, 3), stride=1, padding=1)
            heatmap = heatmap * (heatmap_nms == heatmap).float()
        heatmap = heatmap.permute(0, 2, 3, 1) # N x H x W x C
        heatmap = heatmap.reshape(N, -1, C) # N x HW x C
        box_regression = reg_pred.view(N, 4, H, W).permute(0, 2, 3, 1) # N x H x W x 4 
        box_regression = box_regression.reshape(N, -1, 4)

        candidate_inds = heatmap > self.score_thresh # 0.05
        pre_nms_top_n = candidate_inds.view(N, -1).sum(1) # N
        pre_nms_topk = self.pre_nms_topk_train if self.training else self.pre_nms_topk_test
        pre_nms_top_n = pre_nms_top_n.clamp(max=pre_nms_topk) # N

        if agn_hm is not None:
            agn_hm = agn_hm.view(N, 1, H, W).permute(0, 2, 3, 1)
            agn_hm = agn_hm.reshape(N, -1)
            heatmap = heatmap * agn_hm[:, :, None]

        results = []
        for i in range(N):
            per_box_cls = heatmap[i] # HW x C
            per_candidate_inds = candidate_inds[i] # n
            per_box_cls = per_box_cls[per_candidate_inds] # n

            per_candidate_nonzeros = per_candidate_inds.nonzero(as_tuple=False) # n
            per_box_loc = per_candidate_nonzeros[:, 0] # n
            per_class = per_candidate_nonzeros[:, 1] # n

            per_box_regression = box_regression[i] # HW x 4
            per_box_regression = per_box_regression[per_box_loc] # n x 4
            per_grids = grids[per_box_loc] # n x 2

            per_pre_nms_top_n = pre_nms_top_n[i] # 1

            if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
                per_box_cls, top_k_indices = \
                    per_box_cls.topk(per_pre_nms_top_n, sorted=False)
                per_class = per_class[top_k_indices]
                per_box_regression = per_box_regression[top_k_indices]
                per_grids = per_grids[top_k_indices]
            
            detections = distance2bbox(per_grids, per_box_regression, max_shape=None)

            detections[:, 2] = torch.max(detections[:, 2].clone(), detections[:, 0].clone() + 0.01)
            detections[:, 3] = torch.max(detections[:, 3].clone(), detections[:, 1].clone() + 0.01)
            # boxlist = Instances(image_sizes[i])
            scores = torch.sqrt(per_box_cls) \
                if self.with_agn_hm else per_box_cls # n
            scores = torch.unsqueeze(scores,1) # n x 1
            # per_class = torch.unsqueeze(per_class, 1)
            # boxlist.pred_classes = per_class
            boxlist = torch.cat([detections, scores], dim=1)
            results.append(boxlist)
        return results

    def nms_and_topK(self, boxlist, nms=True):

        cfg = self.test_cfg
        result = self.ml_nms(boxlist, cfg) if nms else boxlist
        # if self.debug:
        #     print('#proposals before nms', len(boxlists[i]))
        #     print('#proposals after nms', len(result))
        num_dets = len(result)
        post_nms_topk = self.post_nms_topk_train if self.training else \
            self.post_nms_topk_test
        if num_dets > post_nms_topk:
            cls_scores = result[:,4]
            image_thresh, _ = torch.kthvalue(
                cls_scores.cpu(),
                num_dets - post_nms_topk + 1
            )
            keep = cls_scores >= image_thresh.item()
            keep = torch.nonzero(keep, as_tuple=False).squeeze(1)
            result = result[keep]
        # if self.debug:
        #     print('#proposals after filter', len(result))

        return result

    def ml_nms(self, boxlist, cfg, max_proposals=-1,
            score_field="scores", label_field="labels"):
        """
        Performs non-maximum suppression on a boxlist, with scores specified
        in a boxlist field via score_field.
        Arguments:
            boxlist(BoxList)
            nms_thresh (float)
            max_proposals (int): if > 0, then only the top max_proposals are kept
                after non-maximum suppression
            score_field (str)
        """
        # if nms_thresh <= 0:
        #     return boxlist
        # if boxlist.has('pred_boxes'):
        #     boxes = boxlist.pred_boxes.tensor
        #     labels = boxlist.pred_classes
        # else:
        #     boxes = boxlist.proposal_boxes.tensor
        #     labels = boxlist.proposal_boxes.tensor.new_zeros(
        #         len(boxlist.proposal_boxes.tensor))
        boxes = boxlist[:,:4]
        labels = boxlist[:,-1]
        scores = boxlist[:,4]
        
        _, keep = batched_nms(boxes, scores.contiguous(), labels, cfg.nms)
        if max_proposals > 0:
            keep = keep[: max_proposals]
        boxlist = boxlist[keep]
        return boxlist

    def _add_more_pos(self, reg_pred, gt_instances, shapes_per_level):
        labels, level_masks, c33_inds, c33_masks, c33_regs = \
            self._get_c33_inds(gt_instances, shapes_per_level)
        N, L, K = labels.shape[0], len(self.strides), 9
        c33_inds[c33_masks == 0] = 0
        reg_pred_c33 = reg_pred[c33_inds].detach() # N x L x K
        invalid_reg = c33_masks == 0
        c33_regs_expand = c33_regs.view(N * L * K, 4).clamp(min=0)
        if N > 0:
            with torch.no_grad():
                c33_reg_loss = self.iou_loss(
                    reg_pred_c33.view(N * L * K, 4), 
                    c33_regs_expand, None,
                    reduction='none').view(N, L, K).detach() # N x L x K
        else:
            c33_reg_loss = reg_pred_c33.new_zeros((N, L, K)).detach()
        c33_reg_loss[invalid_reg] = INF # N x L x K
        c33_reg_loss.view(N * L, K)[level_masks.view(N * L), 4] = 0 # real center
        c33_reg_loss = c33_reg_loss.view(N, L * K)
        if N == 0:
            loss_thresh = c33_reg_loss.new_ones((N)).float()
        else:
            loss_thresh = torch.kthvalue(
                c33_reg_loss, self.more_pos_topk, dim=1)[0] # N
        loss_thresh[loss_thresh > self.more_pos_thresh] = self.more_pos_thresh # N
        new_pos = c33_reg_loss.view(N, L, K) < \
            loss_thresh.view(N, 1, 1).expand(N, L, K)
        pos_inds = c33_inds[new_pos].view(-1) # P
        labels = labels.view(N, 1, 1).expand(N, L, K)[new_pos].view(-1)
        return pos_inds, labels
        
    
    def _get_c33_inds(self, gt_instances, shapes_per_level):
        '''
        TODO (Xingyi): The current implementation is ugly. Refactor.
        Get the center (and the 3x3 region near center) locations of each objects
        Inputs:
            gt_instances: [n_i], sum n_i = N
            shapes_per_level: L x 2 [(h_l, w_l)]_L
        '''
        labels = []
        level_masks = []
        c33_inds = []
        c33_masks = []
        c33_regs = []
        L = len(self.strides)
        B = len(gt_instances)
        shapes_per_level = shapes_per_level.long()
        loc_per_level = (shapes_per_level[:, 0] * shapes_per_level[:, 1]).long() # L
        level_bases = []
        s = 0
        for l in range(L):
            level_bases.append(s)
            s = s + B * loc_per_level[l]
        level_bases = shapes_per_level.new_tensor(level_bases).long() # L
        strides_default = shapes_per_level.new_tensor(self.strides).float() # L
        K = 9
        dx = shapes_per_level.new_tensor([-1, 0, 1, -1, 0, 1, -1, 0, 1]).long()
        dy = shapes_per_level.new_tensor([-1, -1, -1, 0, 0, 0, 1, 1, 1]).long()
        for im_i in range(B):
            targets_per_im = gt_instances[im_i]
            bboxes = targets_per_im.gt_boxes.tensor # n x 4
            n = bboxes.shape[0]
            if n == 0:
                continue
            centers = ((bboxes[:, [0, 1]] + bboxes[:, [2, 3]]) / 2) # n x 2
            centers = centers.view(n, 1, 2).expand(n, L, 2)

            strides = strides_default.view(1, L, 1).expand(n, L, 2) # 
            centers_inds = (centers / strides).long() # n x L x 2
            center_grids = centers_inds * strides + strides // 2# n x L x 2
            l = center_grids[:, :, 0] - bboxes[:, 0].view(n, 1).expand(n, L)
            t = center_grids[:, :, 1] - bboxes[:, 1].view(n, 1).expand(n, L)
            r = bboxes[:, 2].view(n, 1).expand(n, L) - center_grids[:, :, 0]
            b = bboxes[:, 3].view(n, 1).expand(n, L) - center_grids[:, :, 1] # n x L
            reg = torch.stack([l, t, r, b], dim=2) # n x L x 4
            reg = reg / strides_default.view(1, L, 1).expand(n, L, 4).float()
            
            Ws = shapes_per_level[:, 1].view(1, L).expand(n, L)
            Hs = shapes_per_level[:, 0].view(1, L).expand(n, L)
            expand_Ws = Ws.view(n, L, 1).expand(n, L, K)
            expand_Hs = Hs.view(n, L, 1).expand(n, L, K)
            label = targets_per_im.gt_classes.view(n).clone()
            mask = reg.min(dim=2)[0] >= 0 # n x L
            mask = mask & self.assign_fpn_level(bboxes)
            labels.append(label) # n
            level_masks.append(mask) # n x L

            Dy = dy.view(1, 1, K).expand(n, L, K)
            Dx = dx.view(1, 1, K).expand(n, L, K)
            c33_ind = level_bases.view(1, L, 1).expand(n, L, K) + \
                       im_i * loc_per_level.view(1, L, 1).expand(n, L, K) + \
                       (centers_inds[:, :, 1:2].expand(n, L, K) + Dy) * expand_Ws + \
                       (centers_inds[:, :, 0:1].expand(n, L, K) + Dx) # n x L x K
            
            c33_mask = \
                ((centers_inds[:, :, 1:2].expand(n, L, K) + dy) < expand_Hs) & \
                ((centers_inds[:, :, 1:2].expand(n, L, K) + dy) >= 0) & \
                ((centers_inds[:, :, 0:1].expand(n, L, K) + dx) < expand_Ws) & \
                ((centers_inds[:, :, 0:1].expand(n, L, K) + dx) >= 0)
            # TODO (Xingyi): think about better way to implement this
            # Currently it hard codes the 3x3 region
            c33_reg = reg.view(n, L, 1, 4).expand(n, L, K, 4).clone()
            c33_reg[:, :, [0, 3, 6], 0] -= 1
            c33_reg[:, :, [0, 3, 6], 2] += 1
            c33_reg[:, :, [2, 5, 8], 0] += 1
            c33_reg[:, :, [2, 5, 8], 2] -= 1
            c33_reg[:, :, [0, 1, 2], 1] -= 1
            c33_reg[:, :, [0, 1, 2], 3] += 1
            c33_reg[:, :, [6, 7, 8], 1] += 1
            c33_reg[:, :, [6, 7, 8], 3] -= 1
            c33_mask = c33_mask & (c33_reg.min(dim=3)[0] >= 0) # n x L x K
            c33_inds.append(c33_ind)
            c33_masks.append(c33_mask)
            c33_regs.append(c33_reg)
        
        if len(level_masks) > 0:
            labels = torch.cat(labels, dim=0)
            level_masks = torch.cat(level_masks, dim=0)
            c33_inds = torch.cat(c33_inds, dim=0).long()
            c33_regs = torch.cat(c33_regs, dim=0)
            c33_masks = torch.cat(c33_masks, dim=0)
        else:
            labels = shapes_per_level.new_zeros((0)).long()
            level_masks = shapes_per_level.new_zeros((0, L)).bool()
            c33_inds = shapes_per_level.new_zeros((0, L, K)).long()
            c33_regs = shapes_per_level.new_zeros((0, L, K, 4)).float()
            c33_masks = shapes_per_level.new_zeros((0, L, K)).bool()
        return labels, level_masks, c33_inds, c33_masks, c33_regs # N x L, N x L x K
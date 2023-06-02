# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import bias_init_with_prob, normal_init
from mmcv.runner import force_fp32

from mmdet.core import multi_apply
from mmdet.core.anchor.point_generator import MlvlPointGenerator
from mmdet.core.bbox import bbox_overlaps
from mmdet.models import HEADS
from mmdet.models.dense_heads.atss_head import reduce_mean
from mmdet.models.dense_heads.fcos_head import FCOSHead
from mmdet.models.dense_heads.paa_head import levels_to_images

EPS = 1e-12


class CenterPrior(nn.Module):
    """Center Weighting module to adjust the category-specific prior
    distributions.

    Args:
        force_topk (bool): 是否执行tok_k操作. 默认值: False.
        topk (int): 当gt范围内不存在point时,强制选择离gt中心最近的k个point
            作为匹配的point. 默认值: 9.
        num_classes (int): 数据集类别数, 默认值: 80.
        strides (tuple[int]): 各输入特征的下采样倍数. 默认值: (8, 16, 32, 64, 128).
    """

    def __init__(self,
                 force_topk=False,
                 topk=9,
                 num_classes=80,
                 strides=(8, 16, 32, 64, 128)):
        super(CenterPrior, self).__init__()
        # 均值和标准差是可学习的,目的是需要通过这两个参数学习到和类别相关的特定先验分布
        self.mean = nn.Parameter(torch.zeros(num_classes, 2))
        self.sigma = nn.Parameter(torch.ones(num_classes, 2))
        self.strides = strides
        self.force_topk = force_topk
        self.topk = topk

    def forward(self, anchor_points_list, gt_bboxes, labels,
                inside_gt_bbox_mask):
        """计算point-gt的距离高斯矩阵,以及point-gt的匹配矩阵.

        Args:
            anchor_points_list (list[Tensor]): [[h * w, 2], ] * nl.
            gt_bboxes (Tensor): [num_gt, 4].
            labels (Tensor): [num_gt, ].
            inside_gt_bbox_mask (Tensor): [nl * h * w, num_gt].bool型数据.

        Returns:
            tuple(Tensor):

                - center_prior_weights(Tensor): [nl * h * w, num_gt].
                    表示point到gt中心的距离高斯权重矩阵,当point与gt不存在匹配
                    关系时,两者距离越近值越大,∈(0, 1].凡是没匹配到一律为0.
                - inside_gt_bbox_mask (Tensor): [nl * h * w, num_gt].
                    bool型数据,表示该位置的point与gt是否匹配,匹配规则如下:
                    1.当gt内部存在point时,其内部所有point值都为True.
                    2.当gt内部不存在point时,与其最近的top_k个point值都为True.
                    规则2需要self.force_topk为True,不过该参数默认为False.
        """
        inside_gt_bbox_mask = inside_gt_bbox_mask.clone()
        num_gts = len(labels)
        num_points = sum([len(item) for item in anchor_points_list])
        if num_gts == 0:
            return gt_bboxes.new_zeros(num_points,
                                       num_gts), inside_gt_bbox_mask
        # 注释代码是先cat到一起再执行逻辑操作,而原代码则先在不同层级间执行逻辑,再cat到一起.
        # 因为stride数据shape比较特殊无法直接与所有层级上的point进行广播，所以需要先生成
        # 与anchor_points_list的shape一致的数据,再逐层级更改stride,再cat到一起.
        # 而point则一开始就可以将所有层级cat到一起与gt生成point-gt矩阵,计算weight与mask.
        # anchor_points_list = torch.cat(anchor_points_list, dim=0)
        # anchor_points_list = anchor_points_list[:, None, :].expand(
        #         (anchor_points_list.size(0), num_gts, 2))
        # gt_center_x = ((gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2)
        # gt_center_y = ((gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2)
        # gt_center = torch.stack((gt_center_x, gt_center_y), dim=1)
        # gt_center = gt_center[None]
        # instance_center = self.mean[labels][None]
        # instance_sigma = self.sigma[labels][None]
        # stride = torch.ones_like(anchor_points_list)
        # ind = 0
        # for i, anchor in enumerate(anchor_points_list):
        #     stride[ind:ind+anchor.shape[0]] *= self.strides[i]
        #     ind += anchor.shape[0]
        # distance = (((anchor_points_list - gt_center) / stride -
        #              instance_center) ** 2)
        # center_prior_weights = torch.exp(-distance /
        #                          (2 * instance_sigma ** 2)).prod(dim=-1)
        center_prior_list = []
        for slvl_points, stride in zip(anchor_points_list, self.strides):
            # [h*w, 2] -> [h*w, 1, 2] -> [h*w, num_gt, 2]
            single_level_points = slvl_points[:, None, :].expand(
                (slvl_points.size(0), len(gt_bboxes), 2))
            gt_center_x = ((gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2)
            gt_center_y = ((gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2)
            # [num_gt, ] [num_gt, ] -> [num_gt, 2] -> [1, num_gt, 2]
            gt_center = torch.stack((gt_center_x, gt_center_y), dim=1)
            gt_center = gt_center[None]
            # [1, num_gt, 2]
            instance_center = self.mean[labels][None]
            instance_sigma = self.sigma[labels][None]
            # 广播机制, [num_points, num_gt, 2]. prod相乘, [num_points, num_gt]
            distance = (((single_level_points - gt_center) / float(stride) -
                         instance_center)**2)
            center_prior = torch.exp(-distance /
                                     (2 * instance_sigma**2)).prod(dim=-1)
            center_prior_list.append(center_prior)
        # [nl * num_points, num_gt], 表示point到gt中心的高斯分布矩阵,越接近值越大,最大为1.
        center_prior_weights = torch.cat(center_prior_list, dim=0)

        if self.force_topk:
            # inside_gt_bbox_mask: [nl * h * w, num_gt].
            # 获取那些范围内没有point的gt_index, 进行reshape是因为nonzero对[n, ] -> [n, 1]
            gt_inds_no_points_inside = torch.nonzero(
                inside_gt_bbox_mask.sum(0) == 0).reshape(-1)
            # 哪些情况可能会存在?gt的宽或高小于p3层级下采样倍数时.
            # 1.gt范围在grid内部的小物体.
            # 2.宽高比异常的物体,整个gt box内部没有完整包含一个grid.
            if gt_inds_no_points_inside.numel():
                # 在范围内没有point的gt中筛选出最近的top_k个point索引
                # [top_k, num_no_point_inside], 注意torch.top_k返回val, ind
                topk_center_index = \
                    center_prior_weights[:, gt_inds_no_points_inside].topk(
                                                             self.topk,
                                                             dim=0)[1]
                # empty_gt_ind会由[n,]广播至[top_k, n].而下面两行代码比scatter更易读.
                empty_gt_ind = torch.arange(gt_inds_no_points_inside.numel(), dtype=torch.long)
                inside_gt_bbox_mask[topk_center_index, gt_inds_no_points_inside[empty_gt_ind]] = True

                # [nl * h * w, num_no_point_inside].代表内部没有point的point-gt矩阵.
                # temp_mask = inside_gt_bbox_mask[:, gt_inds_no_points_inside]
                # scatter操作参考: https://zhuanlan.zhihu.com/p/339043454
                # 这步操作的目的是为了在那些范围内没有point的point-gt矩阵切片上,
                # 将前top_k个最近的point索引位置的值修改为True
                # inside_gt_bbox_mask[:, gt_inds_no_points_inside] = \
                #     torch.scatter(temp_mask,
                #                   dim=0,
                #                   index=topk_center_index,
                #                   src=torch.ones_like(
                #                     topk_center_index,
                #                     dtype=torch.bool))
        # 将gt外部的point权重值为0
        center_prior_weights[~inside_gt_bbox_mask] = 0
        return center_prior_weights, inside_gt_bbox_mask


@HEADS.register_module()
class AutoAssignHead(FCOSHead):
    """AutoAssignHead head used in AutoAssign.

    More details can be found in the `paper
    <https://arxiv.org/abs/2007.03496>`_ .

    Args:
        force_topk (bool): Used in center prior initialization to
            handle extremely small gt. Default is False.
        topk (int): The number of points used to calculate the
            center prior when no point falls in gt_bbox. Only work when
            force_topk if True. Defaults to 9.
        pos_loss_weight (float): The loss weight of positive loss
            and with default value 0.25.
        neg_loss_weight (float): The loss weight of negative loss
            and with default value 0.75.
        center_loss_weight (float): The loss weight of center prior
            loss and with default value 0.75.
    """

    def __init__(self,
                 *args,
                 force_topk=False,
                 topk=9,
                 pos_loss_weight=0.25,
                 neg_loss_weight=0.75,
                 center_loss_weight=0.75,
                 **kwargs):
        super().__init__(*args, conv_bias=True, **kwargs)
        self.center_prior = CenterPrior(
            force_topk=force_topk,
            topk=topk,
            num_classes=self.num_classes,
            strides=self.strides)
        self.pos_loss_weight = pos_loss_weight
        self.neg_loss_weight = neg_loss_weight
        self.center_loss_weight = center_loss_weight
        self.prior_generator = MlvlPointGenerator(self.strides, offset=0)

    def init_weights(self):
        """head部分的权重初始化.注意,cls conv和reg conv的bias进行了特殊初始化"""
        super(AutoAssignHead, self).init_weights()
        bias_cls = bias_init_with_prob(0.02)
        normal_init(self.conv_cls, std=0.01, bias=bias_cls)
        normal_init(self.conv_reg, std=0.01, bias=4.0)

    def forward_single(self, x, scale, stride):
        """单层级上的前向传播.

        Args:
            x (Tensor): [bs, c, h, w].
            scale (:obj: `mmcv.cnn.Scale`): 用于放缩reg值的可学习参数
            stride (int): 当前输入x对应的下采样倍数, 用于对reg进行标准化.

        Returns:
            tuple: 基于输入x生成的cls score, box reg, obj score.
        """
        cls_score, bbox_pred, cls_feat, reg_feat = super(
            FCOSHead, self).forward_single(x)
        centerness = self.conv_centerness(reg_feat)
        # 对不同层级输出的reg,进行不同的scale缩放,以避免在启用FP16时溢出
        bbox_pred = scale(bbox_pred).float()
        # 使用 PyTorch 1.10 运行时,梯度计算所需的 bbox_pred 会被
        # F.relu(bbox_pred)修改. 所以使用bbox_pred.clamp(min=0)来替换此操作
        bbox_pred = bbox_pred.clamp(min=0)
        bbox_pred *= stride
        return cls_score, bbox_pred, centerness

    def get_pos_loss_single(self, cls_score, objectness, reg_loss, gt_labels,
                            center_prior_weights):
        """计算所有层级特征图中的正样本loss. 以下两个概念是比较重要的.
            正样本区域置信度:用于衡量属于正样本的概率,值越大越可能是正样本,需要同时考虑三个分支的输出.
            正样本区域权重:用于对正样本区域 loss 进行加权,需要同时考虑中心先验分布和正样本区域置信度,
            且仅仅考虑 gt bbox 内部区域,因为 gt bbox 外部肯定不是正样本
            参考:https://zhuanlan.zhihu.com/p/378581552
        Args:
            cls_score (Tensor): [nl * h * w, nc].
            objectness (Tensor): [nl * h * w, 1].
            reg_loss (Tensor): [nl * h * w, num_gt].
            gt_labels (Tensor): [num_gt, ].
            center_prior_weights (Tensor): [nl * h * w, num_gt].
                point-gt的距离高斯权重矩阵,∈[0, 1],point与gt不匹配时为0.否则越接近越大.

        Returns:
            tuple[Tensor]:

                - pos_loss (Tensor): 单张图像上,所有gt内部的正样本loss加权总和.
        """
        # pos_reg_score,从正样本loss定义上来看.
        # 因为pos_loss = pos_cls_loss + pos_reg_loss
        #             = -log(pos_cls_score) + pos_reg_loss
        #             = -log(pos_cls_score * e^(-pos_reg_loss))
        p_loc = torch.exp(-reg_loss)
        # 先将cls_score与obj_score相乘,再获取它在gt类别上的值(正样本gt上预测的score)
        p_cls = (cls_score * objectness)[:, gt_labels]
        # 从直接上来看.正样本的概率值由以下几个值决定.
        # 即pos_score取决于obj_score * cls_score * reg_score(通过iou_loss反映).
        # 即目标置信度*分类置信度*定位置信度(iou loss越小,定位置信度越高,box与gt越接近)
        p_pos = p_cls * p_loc
        # 3 是一个超参数,平衡好的回归与差的回归之间的loss差距.
        confidence_weight = torch.exp(p_pos * 3)
        # 统计gt内部所有box权重的比重.比重总和为1,再与gt内部所有box的置信度相乘,
        # 最后再求和即求得gt内部所有box的加权置信度.只不过这个加权因子是box的权重.
        # 也就是说最后每个gt都计算出一个"加权"box置信度,来参与计算交叉熵loss.
        p_pos_weight = (confidence_weight * center_prior_weights) / (
            (confidence_weight * center_prior_weights).sum(
                0, keepdim=True)).clamp(min=EPS)
        reweighted_p_pos = (p_pos * p_pos_weight).sum(0)
        pos_loss = F.binary_cross_entropy(
            reweighted_p_pos,
            torch.ones_like(reweighted_p_pos),
            reduction='none')
        pos_loss = pos_loss.sum() * self.pos_loss_weight
        return pos_loss,

    def get_neg_loss_single(self, cls_score, objectness, gt_labels, ious,
                            inside_gt_bbox_mask):
        """计算所有层级特征图中的负样本loss.
            负样本区域置信度:用于衡量属于负样本的概率,值越大越可能是负样本,同时考虑了cls与obj分支.
            负样本区域权重:用于对负样本区域 loss 进行加权,仅考虑gt内部各box的iou大小,越大权重越小.
        Args:
            cls_score (Tensor): [nl * h * w, nc].
            objectness (Tensor): [nl * h * w, 1].
            gt_labels (Tensor): [num_gt, ].
            ious (Tensor): [nl * h * w, num_gt],box-gt的IOU矩阵.
            inside_gt_bbox_mask (Tensor): [nl * h * w, num_gt],
                表示point是否与gt匹配.

        Returns:
            tuple[Tensor]:

                - neg_loss (Tensor): 单张图像上,所有负样本loss加权总和.
        """
        num_gts = len(gt_labels)
        joint_conf = (cls_score * objectness)
        # 默认特征图上全是背景,如果存在gt则在gt范围内重新根据IOU大小赋予权重
        p_neg_weight = torch.ones_like(joint_conf)
        if num_gts > 0:
            # 维度的顺序会影响p_neg_weight的值,我们严格按照作者的实现.
            # [nl * h * w, num_gt] -> [num_gt, nl * h * w]
            inside_gt_bbox_mask = inside_gt_bbox_mask.permute(1, 0)
            ious = ious.permute(1, 0)

            # [[inside_gt_index, ] [inside_point_index, ]]
            foreground_idxs = torch.nonzero(inside_gt_bbox_mask, as_tuple=True)
            # 在gt内部的IOU权重值,[num_inside, ]
            temp_weight = (1 / (1 - ious[foreground_idxs]).clamp_(EPS))

            def normalize(x):
                return (x - x.min() + EPS) / (x.max() - x.min() + EPS)

            for instance_idx in range(num_gts):
                idxs = foreground_idxs[0] == instance_idx
                if idxs.any():
                    temp_weight[idxs] = normalize(temp_weight[idxs])
            # 对gt内部的负样本权重进行重新赋值,归一化的IOU值越大,越有可能是前景,neg权重越小
            # 归一化是因为temp_weight可能大于1,
            p_neg_weight[foreground_idxs[1],
                         gt_labels[foreground_idxs[0]]] = 1 - temp_weight

        logits = (joint_conf * p_neg_weight)
        # logits**2 控制gt内外的负样本loss差异,与正样本loss中的3类似?
        neg_loss = (
            logits**2 * F.binary_cross_entropy(
                logits, torch.zeros_like(logits), reduction='none'))
        neg_loss = neg_loss.sum() * self.neg_loss_weight
        return neg_loss,

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'objectnesses'))
    def loss(self,
             cls_scores,
             bbox_preds,
             objectnesses,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """计算head部分的loss.
            在AutoAssign之前的模型中,计算loss部分往往分为3个分支,cls,reg,obj(有些没有)
            cls:考虑正负样本,其中负样本区域的权重一般为1,正样本区域为可调节参数pos_weight.
            用来控制正负样本loss的比例.该值一般默认也为1.
            reg:仅考虑正样本,一般模型的gt中心所落在的grid权重为1,其他grid为0,但也有
            gt范围内所在的grid权重>0,其余grid为0,而gt范围内的grid权重则取决于
            各grid所代表的point到gt中心的距离,越近权重越大->1.距离越远权重越小->0.
            代表有FCOS, ATSS.
            obj:仅考虑正样本,权重是gt范围所在的grid区域皆为1,其余grid为0.
            而在AutoAssign中,仅区分正负样本loss.
            在计算正样本loss时将reg, cls, obj都揉到一起.在计算负样本loss时仅计算cls.
        Args:
            cls_scores (list[Tensor]): [[bs, nc, h, w], ] * nl.
            bbox_preds (list[Tensor]): [[bs, 4, h, w], ] * nl.
            objectnesses (list[Tensor]): [[bs, 1, h, w], ] * nl.
            gt_bboxes (list[Tensor]): [[num_gt, 4], ] * bs, xyxy格式.
            gt_labels (list[Tensor]): [[num_gt, ], ] * bs.
            img_metas (list[dict]): [dict(), ] * bs.
            gt_bboxes_ignore (None | list[Tensor]): [[num_gt_ignore, ], ] * bs.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        assert len(cls_scores) == len(bbox_preds) == len(objectnesses)
        all_num_gt = sum([len(item) for item in gt_bboxes])
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        # [[h * w, 2], ] * nl, 以grid左上角为中心.
        all_level_points = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=bbox_preds[0].dtype,
            device=bbox_preds[0].device)
        # [[nl * h * w, num_gt], ] * bs, [[nl * h * w, num_gt, 4], ] * bs
        inside_gt_bbox_mask_list, bbox_targets_list = self.get_targets(
            all_level_points, gt_bboxes)

        center_prior_weight_list = []
        temp_inside_gt_bbox_mask_list = []
        for gt_bboxe, gt_label, inside_gt_bbox_mask in zip(
                gt_bboxes, gt_labels, inside_gt_bbox_mask_list):
            # 二者都是[nl * h * w, num_gt].前者是point-gt的距离高斯矩阵
            # 后者是匹配矩阵,force_topk为True时会改变范围内没有point的point-gt值.
            center_prior_weight, inside_gt_bbox_mask = \
                self.center_prior(all_level_points, gt_bboxe, gt_label,
                                  inside_gt_bbox_mask)
            center_prior_weight_list.append(center_prior_weight)
            temp_inside_gt_bbox_mask_list.append(inside_gt_bbox_mask)
        inside_gt_bbox_mask_list = temp_inside_gt_bbox_mask_list
        mlvl_points = torch.cat(all_level_points, dim=0)
        # [[bs, c, h, w], ] * nl -> [[nl * h * w, c], ] * bs.
        bbox_preds = levels_to_images(bbox_preds)
        cls_scores = levels_to_images(cls_scores)
        objectnesses = levels_to_images(objectnesses)

        reg_loss_list = []
        ious_list = []
        num_points = len(mlvl_points)

        # 这几个变量我更愿称之为pred_reg, target_reg, point_in_gt,shape分别为
        # [nl * h * w, 4], [nl * h * w, num_gt, 4], [nl * h * w, num_gt]
        for bbox_pred, encoded_targets, inside_gt_bbox_mask in zip(
                bbox_preds, bbox_targets_list, inside_gt_bbox_mask_list):
            temp_num_gt = encoded_targets.size(1)
            # -> [nl*h*w, 1, 2] -> [nl*h*w, num_gt, 2] -> [nl*h*w*num_gt, 2]
            expand_mlvl_points = mlvl_points[:, None, :].expand(
                num_points, temp_num_gt, 2).reshape(-1, 2)
            # [nl * h * w, num_gt, 4] -> [nl * h * w * num_gt, 4]
            encoded_targets = encoded_targets.reshape(-1, 4)
            # -> [nl*h*w, 1, 4] -> [nl*h*w, num_gt, 4] -> [nl*h*w*num_gt, 4]
            expand_bbox_pred = bbox_pred[:, None, :].expand(
                num_points, temp_num_gt, 4).reshape(-1, 4)
            decoded_bbox_preds = self.bbox_coder.decode(
                expand_mlvl_points, expand_bbox_pred)
            # 这里为何不直接使用gt box作为target? TODO 待测试精度
            decoded_target_preds = self.bbox_coder.decode(
                expand_mlvl_points, encoded_targets)
            with torch.no_grad():
                ious = bbox_overlaps(
                    decoded_bbox_preds, decoded_target_preds, is_aligned=True)
                ious = ious.reshape(num_points, temp_num_gt)
                if temp_num_gt:
                    # 计算出任意pred_box与多个gt的最大IOU值,再复制num_gt倍.主要是为了得到
                    # box-gt的IOU矩阵,同时方便通过inside_gt_bbox_mask置为0.
                    ious = ious.max(
                        dim=-1, keepdim=True).values.repeat(1, temp_num_gt)
                else:
                    ious = ious.new_zeros(num_points, temp_num_gt)
                ious[~inside_gt_bbox_mask] = 0
                ious_list.append(ious)
            loss_bbox = self.loss_bbox(
                decoded_bbox_preds,
                decoded_target_preds,
                weight=None,
                reduction_override='none')
            # [nl * h * w * num_gt, ] -> [nl * h * w, num_gt]
            reg_loss_list.append(loss_bbox.reshape(num_points, temp_num_gt))

        cls_scores = [item.sigmoid() for item in cls_scores]
        objectnesses = [item.sigmoid() for item in objectnesses]
        pos_loss_list, = multi_apply(self.get_pos_loss_single, cls_scores,
                                     objectnesses, reg_loss_list, gt_labels,
                                     center_prior_weight_list)
        pos_avg_factor = reduce_mean(
            bbox_pred.new_tensor(all_num_gt)).clamp_(min=1)
        pos_loss = sum(pos_loss_list) / pos_avg_factor

        neg_loss_list, = multi_apply(self.get_neg_loss_single, cls_scores,
                                     objectnesses, gt_labels, ious_list,
                                     inside_gt_bbox_mask_list)
        neg_avg_factor = sum(item.data.sum()
                             for item in center_prior_weight_list)
        neg_avg_factor = reduce_mean(neg_avg_factor).clamp_(min=1)
        neg_loss = sum(neg_loss_list) / neg_avg_factor

        center_loss = []
        for i in range(len(img_metas)):

            if inside_gt_bbox_mask_list[i].any():
                # 为了能够锐化可学习类别先验距离高斯分布,使得各个类别gt的正样本中心权重变大,
                # 而周围值变小,注意并非gt中心,而是"类别正样本中心",
                # 不同的类别可能距离gt中心有不同的偏移.
                center_loss.append(
                    len(gt_bboxes[i]) /
                    center_prior_weight_list[i].sum().clamp_(min=EPS))
            # 当 gt_bbox 的宽度或高度小于p3层下采样倍数时,如果force_tok_k还为False
            # 那么是有可能所有特征图是不存在正样本区域的.
            else:
                center_loss.append(center_prior_weight_list[i].sum() * 0)

        center_loss = torch.stack(center_loss).mean() * self.center_loss_weight

        # avoid dead lock in DDP
        if all_num_gt == 0:
            pos_loss = bbox_preds[0].sum() * 0
            dummy_center_prior_loss = self.center_prior.mean.sum(
            ) * 0 + self.center_prior.sigma.sum() * 0
            center_loss = objectnesses[0].sum() * 0 + dummy_center_prior_loss

        loss = dict(
            loss_pos=pos_loss, loss_neg=neg_loss, loss_center=center_loss)

        return loss

    def get_targets(self, points, gt_bboxes_list):
        """计算batch张图像的target_reg,以及point是否在gt内部的Bool类型的mask.

        Args:
            points (list[Tensor]): [[h * w, 2], ] * nl.
            gt_bboxes_list (list[Tensor]): [[num_gt, 4], ] * bs.

        Returns:
            tuple(list[Tensor]):
                - inside_gt_bbox_mask_list (list[Tensor]): point是否
                    在gt内部的Bool类型的mask, [[nl * h * w, num_gt], ] * bs
                - concat_lvl_bbox_targets (list[Tensor]): point到gt四条边的距离.
                    [[nl * h * w, num_gt, 4], ] * bs
        """

        concat_points = torch.cat(points, dim=0)
        inside_gt_bbox_mask_list, bbox_targets_list = multi_apply(
            self._get_target_single, gt_bboxes_list, points=concat_points)
        return inside_gt_bbox_mask_list, bbox_targets_list

    def _get_target_single(self, gt_bboxes, points):
        """计算单张图像的target_reg,以及point是否在gt内部的Bool类型的mask.

        Args:
            gt_bboxes (Tensor): [num_gt, 4].
            points (Tensor): [nl * h * w, 2].

        Returns:
            tuple[Tensor]: Containing the following Tensors:

                - inside_gt_bbox_mask (Tensor): [nl * h * w, num_gt].
                    Bool类型值,表示point是否在gt内部.
                - bbox_targets (Tensor): [nl * h * w, num_gt, 4].
                    表示point到gt四条边的距离.
        """
        num_points = points.size(0)
        num_gts = gt_bboxes.size(0)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None]
        ys = ys[:, None]
        # [nl * h * w, 1] - [nl * h * w, num_gts]
        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        # 当四个方向的值皆大于0时才表示point在gt内部.否则在外部.
        bbox_targets = torch.stack((left, top, right, bottom), -1)
        if num_gts:
            inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0
        else:
            inside_gt_bbox_mask = bbox_targets.new_zeros((num_points, num_gts),
                                                         dtype=torch.bool)

        return inside_gt_bbox_mask, bbox_targets

    def _get_points_single(self,
                           featmap_size,
                           stride,
                           dtype,
                           device,
                           flatten=False):
        """Almost the same as the implementation in fcos, we remove half stride
        offset to align with the original implementation.

        This function will be deprecated soon.
        """
        warnings.warn(
            '`_get_points_single` in `AutoAssignHead` will be '
            'deprecated soon, we support a multi level point generator now'
            'you can get points of a single level feature map '
            'with `self.prior_generator.single_level_grid_priors` ')
        y, x = super(FCOSHead,
                     self)._get_points_single(featmap_size, stride, dtype,
                                              device)
        points = torch.stack((x.reshape(-1) * stride, y.reshape(-1) * stride),
                             dim=-1)
        return points

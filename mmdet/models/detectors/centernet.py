# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdet.core import bbox2result
from mmdet.models.builder import DETECTORS
from ...core.utils import flip_tensor
from .single_stage import SingleStageDetector


@DETECTORS.register_module()
class CenterNet(SingleStageDetector):
    """Implementation of CenterNet(Objects as Points)

    <https://arxiv.org/abs/1904.07850>.
    """

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super(CenterNet, self).__init__(backbone, neck, bbox_head, train_cfg,
                                        test_cfg, init_cfg)

    def merge_aug_results(self, aug_results, with_nms):
        """将TTA后的检测结果合并到一起.
            与CornerNet不同的是,CenterNet默认配置并不对TTA结果做NMS.

        Args:
            aug_results (list[list[tuple(Tensor,Tensor)]]): 检测结果.
                外部列表表示TTA列表,内部列表表示batch张图像.最里层元组是(box,label).
                其中box -> [bs*k, 5] [x, y, x, y, score], label -> [bs*k,]
            with_nms (bool): 如果为True, 在返回box前实行nms操作.

        Returns:
            tuple: (out_bboxes, out_labels)
        """
        recovered_bboxes, aug_labels = [], []
        # 需要说明的是,CenterNet默认测试配置中是不应用TTA的,即使其flip参数为True.
        # 也仅是针对原始图像与翻转图像进行前向传播再对二者的网络输出(cls/wh)取均值
        # xy取原始heatmap,将其作为单张图片的网络输出进行后续处理以得出预测box.
        # 也就是说aug_results是[[(box, label),],]结构
        for single_result in aug_results:
            recovered_bboxes.append(single_result[0][0])
            aug_labels.append(single_result[0][1])

        bboxes = torch.cat(recovered_bboxes, dim=0).contiguous()
        labels = torch.cat(aug_labels).contiguous()
        if with_nms:
            out_bboxes, out_labels = self.bbox_head._bboxes_nms(
                bboxes, labels, self.bbox_head.test_cfg)
        else:
            out_bboxes, out_labels = bboxes, labels

        return out_bboxes, out_labels

    def aug_test(self, imgs, img_metas, rescale=True):
        """CenterNet的测试增强. 该增强必须有翻转后的图像对,
        并且与CornerNet不同的是, 它将对cls与wh特征图取基础特征图与翻转特征图的平均值.
        再作为单张图片的特征图送入get_bboxes方法以获取检测结果

        Args:
            imgs (list[Tensor]): TTA图像列表.
            img_metas (list[list[dict]]): 外部列表表示TTA列表(多尺度、翻转等),
                内部列表表示batch张图像,其中每个字典都有图像信息,
            rescale (bool): 如果为True, 将预测的box缩放回原始图像尺寸下.默认: True.

        Note:
            ``imgs`` 必须包括翻转的图像.

        Returns:
            list[list[np.ndarray]]: [[[box,] * num_box], ] * bs.
                外层列表对应每张图片. 内部列表对应每个类.
        """
        img_inds = list(range(len(imgs)))
        assert img_metas[0][0]['flip'] + img_metas[1][0]['flip'], (
            'TTA图像列表中必须有一个翻转后的图像')
        aug_results = []
        # 这种for循环方式理论上是对未翻转+翻转的图像cat到一起为[2, 3, h, w]
        # 然后送入网络然后对cls与wh(xy直接取未翻转)的heatmap求和再除以2作为单张图片参加后续操作
        # 按照代码逻辑来看TTA长度是为2的倍数,即[原始,水平]或者[原始,水平,原始,竖直].但后者就冗余了
        # 一个原始图片,不过这里的代码显然是针对TTA长度为2固定了的,不太方便进行比如[原始,水平,竖直]
        # 的TTA方式
        for ind, flip_ind in zip(img_inds[0::2], img_inds[1::2]):
            flip_direction = img_metas[flip_ind][0]['flip_direction']
            img_pair = torch.cat([imgs[ind], imgs[flip_ind]])
            x = self.extract_feat(img_pair)
            # 因为CenterNet没有FPN结构,所以层级固定为1.所以forward函数返回multi_apply结果
            # 是这样的结构[[bs, nc/2, h, w],],下面的assert也是确保层级为1
            center_heatmap_preds, wh_preds, offset_preds = self.bbox_head(x)
            assert len(center_heatmap_preds) == len(wh_preds) == len(
                offset_preds) == 1

            # 对cls wh的heatmap取原始+翻转的均值,wh直接取原始特征图
            center_heatmap_preds[0] = (
                center_heatmap_preds[0][0:1] +
                flip_tensor(center_heatmap_preds[0][1:2], flip_direction)) / 2
            wh_preds[0] = (wh_preds[0][0:1] +
                           flip_tensor(wh_preds[0][1:2], flip_direction)) / 2

            bbox_list = self.bbox_head.get_bboxes(
                center_heatmap_preds,
                wh_preds, [offset_preds[0][0:1]],
                img_metas[ind],
                rescale=rescale,
                with_nms=False)
            aug_results.append(bbox_list)

        nms_cfg = self.bbox_head.test_cfg.get('nms_cfg', None)
        if nms_cfg is None:
            with_nms = False
        else:
            with_nms = True
        bbox_list = [self.merge_aug_results(aug_results, with_nms)]
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results

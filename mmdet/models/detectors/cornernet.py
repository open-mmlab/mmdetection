# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdet.core import bbox2result, bbox_mapping_back
from ..builder import DETECTORS
from .single_stage import SingleStageDetector


@DETECTORS.register_module()
class CornerNet(SingleStageDetector):
    """CornerNet的实现,<https://arxiv.org/abs/1808.01244>."""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super(CornerNet, self).__init__(backbone, neck, bbox_head, train_cfg,
                                        test_cfg, init_cfg)

    def merge_aug_results(self, aug_results, img_metas):
        """将TTA后的检测结果合并到一起.再对这些结果做NMS.

        Args:
            aug_results (list[list[Tensor]]): 检测结果.
                外部列表表示TTA列表(多尺度、翻转等),内部列表表示batch张图像.
            img_metas (list[list[dict]]): 数据结构同上,其中每个字典都有图像信息.

        Returns:
            tuple: (bboxes, labels)
        """
        recovered_bboxes, aug_labels = [], []
        # 这里表示对不同TTA状态的图片进行前向传播,因为CornerNet的len(TTA)==2,
        # 所有for循环会进行两次(第一次原始,第二次翻转),但是由于TTA与bs不能都大于1,
        # 所以下面的img_info实际长度为1,这也是img_info固定取[0]的原因
        for bboxes_labels, img_info in zip(aug_results, img_metas):
            img_shape = img_info[0]['img_shape']  # 在进行Padding之前的图像尺寸
            scale_factor = img_info[0]['scale_factor']
            flip = img_info[0]['flip']
            bboxes, labels = bboxes_labels
            bboxes, scores = bboxes[:, :4], bboxes[:, -1:]
            bboxes = bbox_mapping_back(bboxes, img_shape, scale_factor, flip)
            recovered_bboxes.append(torch.cat([bboxes, scores], dim=-1))
            aug_labels.append(labels)

        bboxes = torch.cat(recovered_bboxes, dim=0)
        labels = torch.cat(aug_labels)

        if bboxes.shape[0] > 0:
            out_bboxes, out_labels = self.bbox_head._bboxes_nms(
                bboxes, labels, self.bbox_head.test_cfg)
        else:
            out_bboxes, out_labels = bboxes, labels

        return out_bboxes, out_labels

    def aug_test(self, imgs, img_metas, rescale=False):
        """CornerNet的测试增强.

        Args:
            imgs (list[Tensor]): 增强后的TTA图像列表.
            img_metas (list[list[dict]]): 外部列表表示TTA列表(多尺度、翻转等),
                内部列表表示batch张图像,其中每个字典都有图像信息.
            rescale (bool): 如果为True, 将预测的box缩放回原始图像尺寸下.默认: False.

        注意:
            ``imgs`` 必须包括翻转的图像.

        Returns:
            list[list[np.ndarray]]: [[[box,] * num_box], ] * bs.
                外层列表对应每张图片. 内部列表对应每个类.
        """
        img_inds = list(range(len(imgs)))
        assert img_metas[0][0]['flip'] + img_metas[1][0]['flip'], (
            'TTA图像列表中必须有翻转后的图像')
        aug_results = []
        for ind, flip_ind in zip(img_inds[0::2], img_inds[1::2]):
            # 原始图像和翻转后的图像cat到一起,[2,3,h,w]
            img_pair = torch.cat([imgs[ind], imgs[flip_ind]])
            # CornerNet的主干网络HourglassNet-104默认会输出两个层级的特征图
            # [[2, 256, h/4, w/4], * num_stack], 这里num_stack=2
            # 但是与训练时的情况不同,训练时是在num_stack个特征图上计算loss,每个特征图上的
            # 要拟合的目标即target都是相同的(生成一个target然后直接复制num_stack份).
            # 但在测试阶段仅使用倒数第一个特征图进行预测.
            x = self.extract_feat(img_pair)
            outs = self.bbox_head(x)
            # 需要注意为什么CornerNet不需要rescale,(不是不需要,而是使用了其他方案代替了)
            # 因为 `RandomCenterCropPad` 是在 CPU 上使用 numpy 完成的,并且在导出到
            # ONNX时它不是动态可跟踪的,因此 'border' 不会作为 'img_meta'中的键出现.
            # 作为一个临时解决方案,在完成导出到ONNX之后,我们将这部分移动到模型后处理中解决.
            bbox_list = self.bbox_head.get_bboxes(
                *outs, [img_metas[ind], img_metas[flip_ind]], False, False)
            # 因为TTA只有原图和翻转后状态,所以bbox_list长度也为2,即只取两个索引
            aug_results.append(bbox_list[0])
            aug_results.append(bbox_list[1])

        bboxes, labels = self.merge_aug_results(aug_results, img_metas)
        # bbox_results为[box,]*num_class
        # 其中box.shape=[n, 5], n∈[0,max_per_img]表示该类别检测出多少个box
        bbox_results = bbox2result(bboxes, labels, self.bbox_head.num_classes)

        return [bbox_results]

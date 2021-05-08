import torch

from mmdet.core import bbox2result
from mmdet.models.builder import DETECTORS
from .cornernet import CornerNet


@DETECTORS.register_module()
class CenterNet(CornerNet):
    """Implementation of CenterNet(Objects as Points)

    <https://arxiv.org/abs/1904.07850>.
    """

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(CenterNet, self).__init__(backbone, neck, bbox_head, train_cfg,
                                        test_cfg, pretrained, init_cfg)

        def aug_test(self, imgs, img_metas, rescale=False):
            """Augment testing of CornerNet.

            Args:
                imgs (list[Tensor]): Augmented images.
                img_metas (list[list[dict]]): Meta information of each
                image, e.g.,
                    image size, scaling factor, etc.
                rescale (bool): If True, return boxes in original image space.
                    Default: False.

            Note:
                ``imgs`` must including flipped image pairs.

            Returns:
                list[list[np.ndarray]]: BBox results of each image and classes.
                    The outer list corresponds to each image. The inner list
                    corresponds to each class.
            """
            img_inds = list(range(len(imgs)))

            assert img_metas[0][0]['flip'] + img_metas[1][0]['flip'], (
                'aug test must have flipped image pair')
            aug_results = []
            for ind, flip_ind in zip(img_inds[0::2], img_inds[1::2]):
                img_pair = torch.cat([imgs[ind], imgs[flip_ind]])
                x = self.extract_feat(img_pair)
                outs = self.bbox_head(x)
                bbox_list = self.bbox_head.get_bboxes(
                    *outs, [img_metas[ind], img_metas[flip_ind]],
                    rescale=rescale,
                    with_nms=False)
                aug_results.append(bbox_list)
            bbox_list = [self.merge_aug_results(aug_results)]
            bbox_results = [
                bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
                for det_bboxes, det_labels in bbox_list
            ]
            return bbox_results

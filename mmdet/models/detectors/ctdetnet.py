import torch

from mmdet.core import bbox2result
from mmdet.models.builder import DETECTORS
from .single_stage import SingleStageDetector


@DETECTORS.register_module()
class CTDetNet(SingleStageDetector):
    """Implementation of RetinaNetMultiHead."""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(CTDetNet, self).__init__(backbone, neck, bbox_head, train_cfg,
                                       test_cfg, pretrained)

    def extract_feat(self, img):
        # import cv2
        # image = img[0].permute(1, 2, 0).cpu().numpy()
        # print(image.shape)
        # cv2.imshow('img', image)
        # cv2.waitKey()
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def merge_aug_results(self, aug_results):
        """Merge augmented detection bboxes and score.

        Args:
            aug_results (list[list[Tensor]]): Det_bboxes and det_labels of each
                image.
            img_metas (list[list[dict]]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            tuple: (bboxes, labels)
        """
        recovered_bboxes, aug_labels = [], []
        for single_result in aug_results:
            recovered_bboxes.append(single_result[0][0])
            aug_labels.append(single_result[0][1])

        bboxes = torch.cat(recovered_bboxes, dim=0).contiguous()
        labels = torch.cat(aug_labels).contiguous()

        if bboxes.shape[0] > 0:
            out_bboxes, out_labels = self.bbox_head._bboxes_nms(
                bboxes, labels, self.bbox_head.test_cfg)
        else:
            out_bboxes, out_labels = bboxes, labels

        return out_bboxes, out_labels

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        bbox_list = self.bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)
        # skip post-processing when exporting to ONNX
        if torch.onnx.is_in_onnx_export():
            return bbox_list
        # import cv2
        # image = cv2.imread(img_metas[0]['filename'])
        # bboxes = bbox_list[0][0]
        # for box in bboxes[:40]:
        #     x1, y1, x2, y2 = box[:4].astype(int)
        #     cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
        # cv2.imshow('ssssss', image)
        # cv2.waitKey()
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results

    def aug_test(self, imgs, img_metas, rescale=False, *args, **kwargs):
        """Augment testing of CornerNet.

        Args:
            imgs (list[Tensor]): Augmented images.
            img_metas (list[list[dict]]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Note:
            ``imgs`` must including flipped image pairs.

        Returns:
            bbox_results (tuple[np.ndarry]): Detection result of each class.
        """
        aug_results = []
        for img, img_meta in zip(imgs, img_metas):
            x = self.extract_feat(img)
            outs = self.bbox_head(x)
            bbox_list = self.bbox_head.get_bboxes(*outs, img_meta, rescale)
            aug_results.append(bbox_list)
        bbox_list = [self.merge_aug_results(aug_results)]
        # import pdb
        # pdb.set_trace()
        # bbox_results = bbox2result(bboxes, labels,
        # self.bbox_head.num_classes)
        # import pdb
        # pdb.set_trace()
        # import cv2
        # image = cv2.imread(img_metas[0][0]['filename'])
        # bboxes = bboxes[0][0]
        # for bboxes in bbox_results:
        #     for box in bboxes:
        #         x1, y1, x2, y2 = box[:4].astype(int)
        #         cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
        # cv2.imshow('ssssss', image)
        # cv2.waitKey()
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results

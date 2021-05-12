from ...core.post_processor.bbox_nms import NMS
from ..builder import DETECTORS
from .single_stage import SingleStageDetector


@DETECTORS.register_module()
class DETR(SingleStageDetector):
    r"""Implementation of `DETR: End-to-End Object Detection with
    Transformers <https://arxiv.org/pdf/2005.12872>`_"""

    def __init__(self,
                 backbone,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 aug_bbox_post_processes=[
                     dict(type='MergeResults'),
                     dict(
                         type='NaiveNMS',
                         iou_threshold=0.5,
                         class_agnostic=False,
                         max_num=100)
                 ]):
        super(DETR, self).__init__(
            backbone,
            None,
            bbox_head,
            train_cfg,
            test_cfg,
            pretrained,
            init_cfg,
            aug_bbox_post_processes=aug_bbox_post_processes)

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        list[list[np.ndarray]]: BBox results of each image and classes.
            The outer list corresponds to each image. The inner list
            corresponds to each class.
        """
        batch_size = len(img_metas)
        assert batch_size == 1, 'Currently only batch_size 1 for inference ' \
            f'mode is supported. Found batch_size {batch_size}.'
        x = self.extract_feat(img)
        outs = self.bbox_head(x, img_metas)
        results_list = self.bbox_head.get_bboxes(*outs, img_metas)

        return [results.export('bbox') for results in results_list]

    def aug_test_bboxes(self, feats, img_metas):
        # Almost same with the implementation in SingleStageDetector
        # except the we need to pass the img_meta to bbox_head

        # remove the nms op from head at first iteration
        if not hasattr(self, 'remove_head_nms'):
            head_post_processes = \
                self.bbox_head.bbox_post_processes.process_list
            for index, operation in enumerate(head_post_processes):
                if isinstance(operation, NMS):
                    head_post_processes.pop(index)
                    break
            # Some heads do not hav nms, such as detr
            self.remove_head_nms = True

        aug_resutls_list = []
        for x, img_meta in zip(feats, img_metas):
            # only one image in the batch
            outs = self.bbox_head(x, img_meta)
            results = self.bbox_head.get_bboxes(*outs, img_meta)[0]
            aug_resutls_list.append(results)

        aug_resutls_list = self.aug_bbox_post_processes(aug_resutls_list)
        return [results.export('bbox') for results in aug_resutls_list]

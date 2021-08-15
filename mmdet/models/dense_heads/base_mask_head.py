from abc import ABCMeta, abstractmethod

from mmcv.runner import BaseModule


class BaseMaskHead(BaseModule, metaclass=ABCMeta):
    """Base class for heads used in One-Stage Instance Segmentation."""

    def __init__(self, init_cfg):
        super(BaseMaskHead, self).__init__(init_cfg)

    @abstractmethod
    def loss(self, **kwargs):
        pass

    @abstractmethod
    def get_results(self, **kwargs):
        """Get precessed :obj:`DetectionResults` of multiple images."""
        pass

    def forward_train(self,
                      x,
                      gt_labels,
                      gt_masks,
                      img_metas,
                      gt_bboxes=None,
                      gt_bboxes_ignore=None,
                      positive_infos=None,
                      **kwargs):
        if positive_infos is None:
            outs = self(x)
        else:
            outs = self(x, positive_infos)

        assert isinstance(outs, tuple), 'Forward results should be a tuple, ' \
                                        'even if only one item is returned'
        loss = self.loss(
            *outs,
            gt_labels=gt_labels,
            gt_masks=gt_masks,
            img_metas=img_metas,
            gt_bboxes=gt_bboxes,
            gt_bboxes_ignore=gt_bboxes_ignore,
            positive_infos=positive_infos,
            **kwargs)
        return loss

    def simple_test(self,
                    feats,
                    img_metas,
                    rescale=False,
                    det_results=None,
                    **kwargs):
        """Test function without test-time augmentation.

        Args:
            feats (tuple[torch.Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.
            det_results (list[obj:`InstanceResults`]): Detection
                Results of each image after the post process.

        Returns:
            list[obj:`DetectionResults`]: Instance segmentation
                results of each image after the post process.
        """
        if det_results is None:
            outs = self(feats)
        else:
            outs = self(feats, det_results=det_results)
        mask_inputs = outs + (img_metas, )
        results_list = self.get_results(
            *mask_inputs, rescale=rescale, det_results=det_results, **kwargs)
        return results_list

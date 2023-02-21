# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleTrackList
from mmdet.utils import OptConfigType, OptMultiConfig
from .base import BaseMOTModel


@MODELS.register_module()
class QDTrack(BaseMOTModel):
    """Quasi-Dense Similarity Learning for Multiple Object Tracking.

    This multi object tracker is the implementation of `QDTrack
    <https://arxiv.org/abs/2006.06664>`_.

    Args:
        detector (dict): Configuration of detector. Defaults to None.
        track_head (dict): Configuration of track head. Defaults to None.
        tracker (dict): Configuration of tracker. Defaults to None.
        freeze_detector (bool): If True, freeze the detector weights.
            Defaults to False.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
           config of :class:`TrackDataPreprocessor`.  it usually includes,
            ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
        init_cfg (dict or list[dict]): Configuration of initialization.
            Defaults to None.
    """

    def __init__(self,
                 detector: Optional[dict] = None,
                 track_head: Optional[dict] = None,
                 tracker: Optional[dict] = None,
                 freeze_detector: bool = False,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(data_preprocessor, init_cfg)
        if detector is not None:
            self.detector = MODELS.build(detector)

        if track_head is not None:
            self.track_head = MODELS.build(track_head)

        if tracker is not None:
            self.tracker = MODELS.build(tracker)

        self.freeze_detector = freeze_detector
        if self.freeze_detector:
            self.freeze_module('detector')

    def predict(self,
                inputs: Tensor,
                data_samples: SampleTrackList,
                rescale: bool = True,
                **kwargs) -> SampleTrackList:
        """Predict results from a video and data samples with post- processing.

        Args:
            inputs (Tensor): of shape (N, T, C, H, W) encoding
                input images. The N denotes batch size.
                The T denotes the number of key frames
                and reference frames.
            data_samples (list[:obj:`TrackDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance`.
            rescale (bool, Optional): If False, then returned bboxes and masks
                will fit the scale of img, otherwise, returned bboxes and masks
                will fit the scale of original image shape. Defaults to True.

        Returns:
            SampleTrackList: List[TrackDataSample]
            Tracking results of the input videos.
            Each DetDataSample usually contains ``pred_track_instances``.
        """
        img = inputs
        assert img.dim() == 5, 'The img must be 5D Tensor (N, T, C, H, W).'
        assert img.size(0) == 1, \
            'QDTrack inference only support 1 batch size per gpu for now.'
        img = img[:, 0]

        assert len(data_samples) == 1, \
            'QDTrack only support 1 batch size per gpu for now.'

        track_data_sample = data_samples[0]
        video_len = len(track_data_sample)

        for frame_id in range(video_len):
            img_data_sample = track_data_sample[frame_id]
            x = self.detector.extract_feat(img)
            rpn_results_list = self.detector.rpn_head.predict(
                x, [img_data_sample])
            det_results = self.detector.roi_head.predict(
                x, rpn_results_list, [img_data_sample], rescale=rescale)
            # det_results List[InstanceData]
            assert len(det_results) == 1, 'Batch inference is not supported.'
            img_data_sample.pred_det_instances = \
                det_results[0].pred_instances.clone()
            frame_pred_track_instances = self.tracker.track(
                model=self,
                img=img,
                feats=x,
                data_sample=img_data_sample,
                **kwargs)
            img_data_sample.pred_instances = frame_pred_track_instances

        return [track_data_sample]

    # TODO: QDTrack loss

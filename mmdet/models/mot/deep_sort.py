# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import TrackSampleList
from mmdet.utils import OptConfigType
from .base import BaseMOTModel


@MODELS.register_module()
class DeepSORT(BaseMOTModel):
    """Simple online and realtime tracking with a deep association metric.

    Details can be found at `DeepSORT<https://arxiv.org/abs/1703.07402>`_.

    Args:
        detector (dict): Configuration of detector. Defaults to None.
        reid (dict): Configuration of reid. Defaults to None
        tracker (dict): Configuration of tracker. Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
           config of :class:`TrackDataPreprocessor`.  it usually includes,
            ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
        init_cfg (dict or list[dict]): Configuration of initialization.
            Defaults to None.
    """

    def __init__(self,
                 detector: Optional[dict] = None,
                 reid: Optional[dict] = None,
                 tracker: Optional[dict] = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptConfigType = None):
        super().__init__(data_preprocessor, init_cfg)

        if detector is not None:
            self.detector = MODELS.build(detector)

        if reid is not None:
            self.reid = MODELS.build(reid)

        if tracker is not None:
            self.tracker = MODELS.build(tracker)

        self.preprocess_cfg = data_preprocessor

    def loss(self, inputs: Tensor, data_samples: TrackSampleList,
             **kwargs) -> dict:
        """Calculate losses from a batch of inputs and data samples."""
        raise NotImplementedError(
            'Please train `detector` and `reid` models firstly, then \
                inference with SORT/DeepSORT.')

    def predict(self,
                inputs: Tensor,
                data_samples: TrackSampleList,
                rescale: bool = True,
                **kwargs) -> TrackSampleList:
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
            TrackSampleList: List[TrackDataSample]
            Tracking results of the input videos.
            Each DetDataSample usually contains ``pred_track_instances``.
        """
        assert inputs.dim() == 5, 'The img must be 5D Tensor (N, T, C, H, W).'
        assert inputs.size(0) == 1, \
            'SORT/DeepSORT inference only support ' \
            '1 batch size per gpu for now.'

        assert len(data_samples) == 1, \
            'SORT/DeepSORT inference only support ' \
            '1 batch size per gpu for now.'

        track_data_sample = data_samples[0]
        video_len = len(track_data_sample)
        if track_data_sample[0].frame_id == 0:
            self.tracker.reset()

        for frame_id in range(video_len):
            img_data_sample = track_data_sample[frame_id]
            single_img = inputs[:, frame_id].contiguous()
            # det_results List[DetDataSample]
            det_results = self.detector.predict(single_img, [img_data_sample])
            assert len(det_results) == 1, 'Batch inference is not supported.'

            pred_track_instances = self.tracker.track(
                model=self,
                img=single_img,
                feats=None,
                data_sample=det_results[0],
                data_preprocessor=self.preprocess_cfg,
                rescale=rescale,
                **kwargs)
            img_data_sample.pred_track_instances = pred_track_instances

        return [track_data_sample]

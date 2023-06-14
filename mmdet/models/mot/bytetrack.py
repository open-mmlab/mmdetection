# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional

from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList, TrackSampleList
from mmdet.utils import OptConfigType, OptMultiConfig
from .base import BaseMOTModel


@MODELS.register_module()
class ByteTrack(BaseMOTModel):
    """ByteTrack: Multi-Object Tracking by Associating Every Detection Box.

    This multi object tracker is the implementation of `ByteTrack
    <https://arxiv.org/abs/2110.06864>`_.

    Args:
        detector (dict): Configuration of detector. Defaults to None.
        tracker (dict): Configuration of tracker. Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
           config of :class:`TrackDataPreprocessor`.  it usually includes,
            ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
        init_cfg (dict or list[dict]): Configuration of initialization.
            Defaults to None.
    """

    def __init__(self,
                 detector: Optional[dict] = None,
                 tracker: Optional[dict] = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(data_preprocessor, init_cfg)

        if detector is not None:
            self.detector = MODELS.build(detector)

        if tracker is not None:
            self.tracker = MODELS.build(tracker)

    def loss(self, inputs: Tensor, data_samples: SampleList, **kwargs) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Tensor): of shape (N, C, H, W) encoding
                input images. Typically these should be mean centered and std
                scaled. The N denotes batch size
            data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance`.

        Returns:
            dict: A dictionary of loss components.
        """
        return self.detector.loss(inputs, data_samples, **kwargs)

    def predict(self, inputs: Dict[str, Tensor], data_samples: TrackSampleList,
                **kwargs) -> TrackSampleList:
        """Predict results from a video and data samples with post-processing.

        Args:
            inputs (Tensor): of shape (N, T, C, H, W) encoding
                input images. The N denotes batch size.
                The T denotes the number of frames in a video.
            data_samples (list[:obj:`TrackDataSample`]): The batch
                data samples. It usually includes information such
                as `video_data_samples`.
        Returns:
            TrackSampleList: Tracking results of the inputs.
        """
        assert inputs.dim() == 5, 'The img must be 5D Tensor (N, T, C, H, W).'
        assert inputs.size(0) == 1, \
            'Bytetrack inference only support ' \
            '1 batch size per gpu for now.'

        assert len(data_samples) == 1, \
            'Bytetrack inference only support 1 batch size per gpu for now.'

        track_data_sample = data_samples[0]
        video_len = len(track_data_sample)

        for frame_id in range(video_len):
            img_data_sample = track_data_sample[frame_id]
            single_img = inputs[:, frame_id].contiguous()
            # det_results List[DetDataSample]
            det_results = self.detector.predict(single_img, [img_data_sample])
            assert len(det_results) == 1, 'Batch inference is not supported.'

            pred_track_instances = self.tracker.track(
                data_sample=det_results[0], **kwargs)
            img_data_sample.pred_track_instances = pred_track_instances

        return [track_data_sample]

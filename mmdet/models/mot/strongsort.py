# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import numpy as np
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures import TrackSampleList
from mmdet.utils import OptConfigType
from .deep_sort import DeepSORT


@MODELS.register_module()
class StrongSORT(DeepSORT):
    """StrongSORT: Make DeepSORT Great Again.

    Details can be found at `StrongSORT<https://arxiv.org/abs/2202.13514>`_.

    Args:
        detector (dict): Configuration of detector. Defaults to None.
        reid (dict): Configuration of reid. Defaults to None
        tracker (dict): Configuration of tracker. Defaults to None.
        kalman (dict): Configuration of Kalman filter. Defaults to None.
        cmc (dict): Configuration of camera model compensation.
            Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
           config of :class:`TrackDataPreprocessor`.  it usually includes,
            ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
        init_cfg (dict or list[dict]): Configuration of initialization.
            Defaults to None.
    """

    def __init__(self,
                 detector: Optional[dict] = None,
                 reid: Optional[dict] = None,
                 cmc: Optional[dict] = None,
                 tracker: Optional[dict] = None,
                 postprocess_model: Optional[dict] = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptConfigType = None):
        super().__init__(detector, reid, tracker, data_preprocessor, init_cfg)

        if cmc is not None:
            self.cmc = TASK_UTILS.build(cmc)

        if postprocess_model is not None:
            self.postprocess_model = TASK_UTILS.build(postprocess_model)

    @property
    def with_cmc(self):
        """bool: whether the framework has a camera model compensation
                model.
        """
        return hasattr(self, 'cmc') and self.cmc is not None

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

        video_track_instances = []
        for frame_id in range(video_len):
            img_data_sample = track_data_sample[frame_id]
            single_img = inputs[:, frame_id].contiguous()
            # det_results List[DetDataSample]
            det_results = self.detector.predict(single_img, [img_data_sample])
            assert len(det_results) == 1, 'Batch inference is not supported.'

            pred_track_instances = self.tracker.track(
                model=self,
                img=single_img,
                data_sample=det_results[0],
                data_preprocessor=self.preprocess_cfg,
                rescale=rescale,
                **kwargs)
            for i in range(len(pred_track_instances.instances_id)):
                video_track_instances.append(
                    np.array([
                        frame_id + 1,
                        pred_track_instances.instances_id[i].cpu(),
                        pred_track_instances.bboxes[i][0].cpu(),
                        pred_track_instances.bboxes[i][1].cpu(),
                        (pred_track_instances.bboxes[i][2] -
                         pred_track_instances.bboxes[i][0]).cpu(),
                        (pred_track_instances.bboxes[i][3] -
                         pred_track_instances.bboxes[i][1]).cpu(),
                        pred_track_instances.scores[i].cpu()
                    ]))
        video_track_instances = np.array(video_track_instances).reshape(-1, 7)
        video_track_instances = self.postprocess_model.forward(
            video_track_instances)
        for frame_id in range(video_len):
            track_data_sample[frame_id].pred_track_instances = \
                    InstanceData(bboxes=video_track_instances[
                        video_track_instances[:, 0] == frame_id + 1, :])

        return [track_data_sample]

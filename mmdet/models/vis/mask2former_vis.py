# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Union

from torch import Tensor

from mmdet.models.mot import BaseMOTModel
from mmdet.registry import MODELS
from mmdet.structures import TrackDataSample, TrackSampleList
from mmdet.utils import OptConfigType, OptMultiConfig


@MODELS.register_module()
class Mask2FormerVideo(BaseMOTModel):
    r"""Implementation of `Masked-attention Mask
    Transformer for Universal Image Segmentation
    <https://arxiv.org/pdf/2112.01527>`_.

    Args:
        backbone (dict): Configuration of backbone. Defaults to None.
        track_head (dict): Configuration of track head. Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
           config of :class:`TrackDataPreprocessor`.  it usually includes,
            ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
            Defaults to None.
        init_cfg (dict or list[dict]): Configuration of initialization.
            Defaults to None.
    """

    def __init__(self,
                 backbone: Optional[dict] = None,
                 track_head: Optional[dict] = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):
        super(BaseMOTModel, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        if backbone is not None:
            self.backbone = MODELS.build(backbone)

        if track_head is not None:
            self.track_head = MODELS.build(track_head)

        self.num_classes = self.track_head.num_classes

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """Overload in order to load mmdet pretrained ckpt."""
        for key in list(state_dict):
            if key.startswith('panoptic_head'):
                state_dict[key.replace('panoptic',
                                       'track')] = state_dict.pop(key)

        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)

    def loss(self, inputs: Tensor, data_samples: TrackSampleList,
             **kwargs) -> Union[dict, tuple]:
        """
        Args:
            inputs (Tensor): Input images of shape (N, T, C, H, W).
                These should usually be mean centered and std scaled.
            data_samples (list[:obj:`TrackDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        assert inputs.dim() == 5, 'The img must be 5D Tensor (N, T, C, H, W).'
        # shape (N * T, C, H, W)
        img = inputs.flatten(0, 1)

        x = self.backbone(img)
        losses = self.track_head.loss(x, data_samples)

        return losses

    def predict(self,
                inputs: Tensor,
                data_samples: TrackSampleList,
                rescale: bool = True) -> TrackSampleList:
        """Predict results from a batch of inputs and data samples with
        postprocessing.

        Args:
            inputs (Tensor): of shape (N, T, C, H, W) encoding
                input images. The N denotes batch size.
                The T denotes the number of frames in a video.
            data_samples (list[:obj:`TrackDataSample`]): The batch
                data samples. It usually includes information such
                as `video_data_samples`.
            rescale (bool, Optional): If False, then returned bboxes and masks
                will fit the scale of img, otherwise, returned bboxes and masks
                will fit the scale of original image shape. Defaults to True.

        Returns:
            TrackSampleList: Tracking results of the inputs.
        """
        assert inputs.dim() == 5, 'The img must be 5D Tensor (N, T, C, H, W).'

        assert len(data_samples) == 1, \
            'Mask2former only support 1 batch size per gpu for now.'

        # [T, C, H, W]
        img = inputs[0]
        track_data_sample = data_samples[0]
        feats = self.backbone(img)
        pred_track_ins_list = self.track_head.predict(feats, track_data_sample,
                                                      rescale)

        det_data_samples_list = []
        for idx, pred_track_ins in enumerate(pred_track_ins_list):
            img_data_sample = track_data_sample[idx]
            img_data_sample.pred_track_instances = pred_track_ins
            det_data_samples_list.append(img_data_sample)

        results = TrackDataSample()
        results.video_data_samples = det_data_samples_list
        return [results]

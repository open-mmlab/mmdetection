# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
from torch import Tensor

from mmdet.models.mot import BaseMOTModel
from mmdet.registry import MODELS
from mmdet.structures import TrackSampleList
from mmdet.utils import OptConfigType, OptMultiConfig


@MODELS.register_module()
class MaskTrackRCNN(BaseMOTModel):
    """Video Instance Segmentation.

    This video instance segmentor is the implementation of`MaskTrack R-CNN
    <https://arxiv.org/abs/1905.04804>`_.

    Args:
        detector (dict): Configuration of detector. Defaults to None.
        track_head (dict): Configuration of track head. Defaults to None.
        tracker (dict): Configuration of tracker. Defaults to None.
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
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(data_preprocessor, init_cfg)

        if detector is not None:
            self.detector = MODELS.build(detector)
        assert hasattr(self.detector, 'roi_head'), \
            'MaskTrack R-CNN only supports two stage detectors.'

        if track_head is not None:
            self.track_head = MODELS.build(track_head)
        if tracker is not None:
            self.tracker = MODELS.build(tracker)

    def loss(self, inputs: Tensor, data_samples: TrackSampleList,
             **kwargs) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Dict[str, Tensor]): of shape (N, T, C, H, W) encoding
                input images. Typically these should be mean centered and std
                scaled. The N denotes batch size. The T denotes the number of
                frames.
            data_samples (list[:obj:`TrackDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance`.

        Returns:
            dict: A dictionary of loss components.
        """

        assert inputs.dim() == 5, 'The img must be 5D Tensor (N, T, C, H, W).'
        assert inputs.size(1) == 2, \
            'MaskTrackRCNN can only have 1 key frame and 1 reference frame.'

        # split the data_samples into two aspects: key frames and reference
        # frames
        ref_data_samples, key_data_samples = [], []
        key_frame_inds, ref_frame_inds = [], []

        # set cat_id of gt_labels to 0 in RPN
        for track_data_sample in data_samples:
            key_data_sample = track_data_sample.get_key_frames()[0]
            key_data_samples.append(key_data_sample)
            ref_data_sample = track_data_sample.get_ref_frames()[0]
            ref_data_samples.append(ref_data_sample)
            key_frame_inds.append(track_data_sample.key_frames_inds[0])
            ref_frame_inds.append(track_data_sample.ref_frames_inds[0])

        key_frame_inds = torch.tensor(key_frame_inds, dtype=torch.int64)
        ref_frame_inds = torch.tensor(ref_frame_inds, dtype=torch.int64)
        batch_inds = torch.arange(len(inputs))
        key_imgs = inputs[batch_inds, key_frame_inds].contiguous()
        ref_imgs = inputs[batch_inds, ref_frame_inds].contiguous()

        x = self.detector.extract_feat(key_imgs)
        ref_x = self.detector.extract_feat(ref_imgs)

        losses = dict()

        # RPN forward and loss
        if self.detector.with_rpn:
            proposal_cfg = self.detector.train_cfg.get(
                'rpn_proposal', self.detector.test_cfg.rpn)

            rpn_losses, rpn_results_list = self.detector.rpn_head. \
                loss_and_predict(x,
                                 key_data_samples,
                                 proposal_cfg=proposal_cfg,
                                 **kwargs)

            # avoid get same name with roi_head loss
            keys = rpn_losses.keys()
            for key in keys:
                if 'loss' in key and 'rpn' not in key:
                    rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
            losses.update(rpn_losses)
        else:
            # TODO: Not support currently, should have a check at Fast R-CNN
            assert key_data_samples[0].get('proposals', None) is not None
            # use pre-defined proposals in InstanceData for the second stage
            # to extract ROI features.
            rpn_results_list = [
                key_data_sample.proposals
                for key_data_sample in key_data_samples
            ]

        losses_detect = self.detector.roi_head.loss(x, rpn_results_list,
                                                    key_data_samples, **kwargs)
        losses.update(losses_detect)

        losses_track = self.track_head.loss(x, ref_x, rpn_results_list,
                                            data_samples, **kwargs)
        losses.update(losses_track)

        return losses

    def predict(self,
                inputs: Tensor,
                data_samples: TrackSampleList,
                rescale: bool = True,
                **kwargs) -> TrackSampleList:
        """Test without augmentation.

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
            'MaskTrackRCNN only support 1 batch size per gpu for now.'

        track_data_sample = data_samples[0]
        video_len = len(track_data_sample)
        if track_data_sample[0].frame_id == 0:
            self.tracker.reset()

        for frame_id in range(video_len):
            img_data_sample = track_data_sample[frame_id]
            single_img = inputs[:, frame_id].contiguous()
            x = self.detector.extract_feat(single_img)

            rpn_results_list = self.detector.rpn_head.predict(
                x, [img_data_sample])
            # det_results List[InstanceData]
            det_results = self.detector.roi_head.predict(
                x, rpn_results_list, [img_data_sample], rescale=rescale)
            assert len(det_results) == 1, 'Batch inference is not supported.'
            assert 'masks' in det_results[0], 'There are no mask results.'

            img_data_sample.pred_instances = det_results[0]
            frame_pred_track_instances = self.tracker.track(
                model=self, feats=x, data_sample=img_data_sample, **kwargs)
            img_data_sample.pred_track_instances = frame_pred_track_instances

        return [track_data_sample]

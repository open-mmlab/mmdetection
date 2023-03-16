# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

from mmengine.structures import BaseDataElement, InstanceData, PixelData


class DetDataSample(BaseDataElement):
    """A data structure interface of MMDetection. They are used as interfaces
    between different components.

    The attributes in ``DetDataSample`` are divided into several parts:

        - ``proposals``(InstanceData): Region proposals used in two-stage
            detectors.
        - ``gt_instances``(InstanceData): Ground truth of instance annotations.
        - ``pred_instances``(InstanceData): Instances of detection predictions.
        - ``pred_track_instances``(InstanceData): Instances of tracking
            predictions.
        - ``ignored_instances``(InstanceData): Instances to be ignored during
            training/testing.
        - ``gt_panoptic_seg``(PixelData): Ground truth of panoptic
            segmentation.
        - ``pred_panoptic_seg``(PixelData): Prediction of panoptic
           segmentation.
        - ``gt_sem_seg``(PixelData): Ground truth of semantic segmentation.
        - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.

    Examples:
         >>> import torch
         >>> import numpy as np
         >>> from mmengine.structures import InstanceData
         >>> from mmdet.structures import DetDataSample

         >>> data_sample = DetDataSample()
         >>> img_meta = dict(img_shape=(800, 1196),
         ...                 pad_shape=(800, 1216))
         >>> gt_instances = InstanceData(metainfo=img_meta)
         >>> gt_instances.bboxes = torch.rand((5, 4))
         >>> gt_instances.labels = torch.rand((5,))
         >>> data_sample.gt_instances = gt_instances
         >>> assert 'img_shape' in data_sample.gt_instances.metainfo_keys()
         >>> len(data_sample.gt_instances)
         5
         >>> print(data_sample)
        <DetDataSample(

            META INFORMATION

            DATA FIELDS
            gt_instances: <InstanceData(

                    META INFORMATION
                    pad_shape: (800, 1216)
                    img_shape: (800, 1196)

                    DATA FIELDS
                    labels: tensor([0.8533, 0.1550, 0.5433, 0.7294, 0.5098])
                    bboxes:
                    tensor([[9.7725e-01, 5.8417e-01, 1.7269e-01, 6.5694e-01],
                            [1.7894e-01, 5.1780e-01, 7.0590e-01, 4.8589e-01],
                            [7.0392e-01, 6.6770e-01, 1.7520e-01, 1.4267e-01],
                            [2.2411e-01, 5.1962e-01, 9.6953e-01, 6.6994e-01],
                            [4.1338e-01, 2.1165e-01, 2.7239e-04, 6.8477e-01]])
                ) at 0x7f21fb1b9190>
        ) at 0x7f21fb1b9880>
         >>> pred_instances = InstanceData(metainfo=img_meta)
         >>> pred_instances.bboxes = torch.rand((5, 4))
         >>> pred_instances.scores = torch.rand((5,))
         >>> data_sample = DetDataSample(pred_instances=pred_instances)
         >>> assert 'pred_instances' in data_sample

         >>> pred_track_instances = InstanceData(metainfo=img_meta)
         >>> pred_track_instances.bboxes = torch.rand((5, 4))
         >>> pred_track_instances.scores = torch.rand((5,))
         >>> data_sample = DetDataSample(
         ...    pred_track_instances=pred_track_instances)
         >>> assert 'pred_track_instances' in data_sample

         >>> data_sample = DetDataSample()
         >>> gt_instances_data = dict(
         ...                        bboxes=torch.rand(2, 4),
         ...                        labels=torch.rand(2),
         ...                        masks=np.random.rand(2, 2, 2))
         >>> gt_instances = InstanceData(**gt_instances_data)
         >>> data_sample.gt_instances = gt_instances
         >>> assert 'gt_instances' in data_sample
         >>> assert 'masks' in data_sample.gt_instances

         >>> data_sample = DetDataSample()
         >>> gt_panoptic_seg_data = dict(panoptic_seg=torch.rand(2, 4))
         >>> gt_panoptic_seg = PixelData(**gt_panoptic_seg_data)
         >>> data_sample.gt_panoptic_seg = gt_panoptic_seg
         >>> print(data_sample)
        <DetDataSample(

            META INFORMATION

            DATA FIELDS
            _gt_panoptic_seg: <BaseDataElement(

                    META INFORMATION

                    DATA FIELDS
                    panoptic_seg: tensor([[0.7586, 0.1262, 0.2892, 0.9341],
                                [0.3200, 0.7448, 0.1052, 0.5371]])
                ) at 0x7f66c2bb7730>
            gt_panoptic_seg: <BaseDataElement(

                    META INFORMATION

                    DATA FIELDS
                    panoptic_seg: tensor([[0.7586, 0.1262, 0.2892, 0.9341],
                                [0.3200, 0.7448, 0.1052, 0.5371]])
                ) at 0x7f66c2bb7730>
        ) at 0x7f66c2bb7280>
        >>> data_sample = DetDataSample()
        >>> gt_segm_seg_data = dict(segm_seg=torch.rand(2, 2, 2))
        >>> gt_segm_seg = PixelData(**gt_segm_seg_data)
        >>> data_sample.gt_segm_seg = gt_segm_seg
        >>> assert 'gt_segm_seg' in data_sample
        >>> assert 'segm_seg' in data_sample.gt_segm_seg
    """

    @property
    def proposals(self) -> InstanceData:
        return self._proposals

    @proposals.setter
    def proposals(self, value: InstanceData):
        self.set_field(value, '_proposals', dtype=InstanceData)

    @proposals.deleter
    def proposals(self):
        del self._proposals

    @property
    def gt_instances(self) -> InstanceData:
        return self._gt_instances

    @gt_instances.setter
    def gt_instances(self, value: InstanceData):
        self.set_field(value, '_gt_instances', dtype=InstanceData)

    @gt_instances.deleter
    def gt_instances(self):
        del self._gt_instances

    @property
    def pred_instances(self) -> InstanceData:
        return self._pred_instances

    @pred_instances.setter
    def pred_instances(self, value: InstanceData):
        self.set_field(value, '_pred_instances', dtype=InstanceData)

    @pred_instances.deleter
    def pred_instances(self):
        del self._pred_instances

    # directly add ``pred_track_instances`` in ``DetDataSample``
    # so that the ``TrackDataSample`` does not bother to access the
    # instance-level information.
    @property
    def pred_track_instances(self) -> InstanceData:
        return self._pred_track_instances

    @pred_track_instances.setter
    def pred_track_instances(self, value: InstanceData):
        self.set_field(value, '_pred_track_instances', dtype=InstanceData)

    @pred_track_instances.deleter
    def pred_track_instances(self):
        del self._pred_track_instances

    @property
    def ignored_instances(self) -> InstanceData:
        return self._ignored_instances

    @ignored_instances.setter
    def ignored_instances(self, value: InstanceData):
        self.set_field(value, '_ignored_instances', dtype=InstanceData)

    @ignored_instances.deleter
    def ignored_instances(self):
        del self._ignored_instances

    @property
    def gt_panoptic_seg(self) -> PixelData:
        return self._gt_panoptic_seg

    @gt_panoptic_seg.setter
    def gt_panoptic_seg(self, value: PixelData):
        self.set_field(value, '_gt_panoptic_seg', dtype=PixelData)

    @gt_panoptic_seg.deleter
    def gt_panoptic_seg(self):
        del self._gt_panoptic_seg

    @property
    def pred_panoptic_seg(self) -> PixelData:
        return self._pred_panoptic_seg

    @pred_panoptic_seg.setter
    def pred_panoptic_seg(self, value: PixelData):
        self.set_field(value, '_pred_panoptic_seg', dtype=PixelData)

    @pred_panoptic_seg.deleter
    def pred_panoptic_seg(self):
        del self._pred_panoptic_seg

    @property
    def gt_sem_seg(self) -> PixelData:
        return self._gt_sem_seg

    @gt_sem_seg.setter
    def gt_sem_seg(self, value: PixelData):
        self.set_field(value, '_gt_sem_seg', dtype=PixelData)

    @gt_sem_seg.deleter
    def gt_sem_seg(self):
        del self._gt_sem_seg

    @property
    def pred_sem_seg(self) -> PixelData:
        return self._pred_sem_seg

    @pred_sem_seg.setter
    def pred_sem_seg(self, value: PixelData):
        self.set_field(value, '_pred_sem_seg', dtype=PixelData)

    @pred_sem_seg.deleter
    def pred_sem_seg(self):
        del self._pred_sem_seg


SampleList = List[DetDataSample]
OptSampleList = Optional[SampleList]

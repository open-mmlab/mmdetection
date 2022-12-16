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
        - ``pred_instances``(InstanceData): Instances of model predictions.
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
        >>> img_meta = dict(img_shape=(800, 1196, 3),
        ...                 pad_shape=(800, 1216, 3))
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
            _gt_instances: <InstanceData(

                    META INFORMATION
                    pad_shape: (800, 1216, 3)
                    img_shape: (800, 1196, 3)

                    DATA FIELDS
                    labels: tensor([0.8533, 0.1550, 0.5433, 0.7294, 0.5098])
                    bboxes: tensor([[0.9773, 0.5842, 0.1727, 0.6569],
                                [0.1789, 0.5178, 0.7059, 0.4859],
                                [0.7039, 0.6677, 0.1752, 0.1427],
                                [0.2241, 0.5196, 0.9695, 0.6699],
                                [0.4134, 0.2117, 0.2724, 0.6848]])
                ) at 0x7f21fb1b9190>
            gt_instances: <InstanceData(

                    META INFORMATION
                    pad_shape: (800, 1216, 3)
                    img_shape: (800, 1196, 3)

                    DATA FIELDS
                    labels: tensor([0.8533, 0.1550, 0.5433, 0.7294, 0.5098])
                    bboxes: tensor([[0.9773, 0.5842, 0.1727, 0.6569],
                                [0.1789, 0.5178, 0.7059, 0.4859],
                                [0.7039, 0.6677, 0.1752, 0.1427],
                                [0.2241, 0.5196, 0.9695, 0.6699],
                                [0.4134, 0.2117, 0.2724, 0.6848]])
                ) at 0x7f21fb1b9190>
        ) at 0x7f21fb1b9880>
        >>> pred_instances = InstanceData(metainfo=img_meta)
        >>> pred_instances.bboxes = torch.rand((5, 4))
        >>> pred_instances.scores = torch.rand((5,))
        >>> data_sample = DetDataSample(pred_instances=pred_instances)
        >>> assert 'pred_instances' in data_sample

        >>> data_sample = DetDataSample()
        >>> gt_instances_data = dict(
        ...                        bboxes=torch.rand(2, 4),
        ...                        labels=torch.rand(2),
        ...                        masks=np.random.rand(2, 2, 2))
        >>> gt_instances = InstanceData(**gt_instances_data)
        >>> data_sample.gt_instances = gt_instances
        >>> assert 'gt_instances' in data_sample
        >>> assert 'masks' in data_sample.gt_instances

        >>> from mmengine.structures import PixelData
        >>> data_sample = DetDataSample()
        >>> gt_panoptic_seg_data = dict(panoptic_seg=torch.rand(2, 4))
        >>> gt_panoptic_seg = PixelData(**gt_panoptic_seg_data)
        >>> data_sample.gt_panoptic_seg = gt_panoptic_seg
        >>> print(data_sample)
        <DetDataSample(

            META INFORMATION

            DATA FIELDS
            gt_panoptic_seg: <PixelData(

                    META INFORMATION

                    DATA FIELDS
                    panoptic_seg: tensor([[[0.7586, 0.1262, 0.2892, 0.9341],
                                 [0.3200, 0.7448, 0.1052, 0.5371]]])
                ) at 0x7f66c2bb7730>
            _gt_panoptic_seg: <PixelData(

                    META INFORMATION

                    DATA FIELDS
                    panoptic_seg: tensor([[[0.7586, 0.1262, 0.2892, 0.9341],
                                 [0.3200, 0.7448, 0.1052, 0.5371]]])
                ) at 0x7f66c2bb7730>
        ) at 0x7f66c2bb7280>
        >>> data_sample = DetDataSample()
        >>> gt_sem_seg_data = dict(sem_seg=torch.rand(2, 2, 2))
        >>> gt_sem_seg = PixelData(**gt_sem_seg_data)
        >>> data_sample.gt_sem_seg = gt_sem_seg
        >>> assert 'gt_sem_seg' in data_sample
        >>> assert 'sem_seg' in data_sample.gt_sem_seg
    """

    @property
    def proposals(self) -> InstanceData:
        return self._proposals

    @proposals.setter
    def proposals(self, value: InstanceData) -> None:
        self.set_field(value, '_proposals', dtype=InstanceData)

    @proposals.deleter
    def proposals(self) -> None:
        del self._proposals

    @property
    def gt_instances(self) -> InstanceData:
        return self._gt_instances

    @gt_instances.setter
    def gt_instances(self, value: InstanceData) -> None:
        self.set_field(value, '_gt_instances', dtype=InstanceData)

    @gt_instances.deleter
    def gt_instances(self) -> None:
        del self._gt_instances

    @property
    def pred_instances(self) -> InstanceData:
        return self._pred_instances

    @pred_instances.setter
    def pred_instances(self, value: InstanceData) -> None:
        self.set_field(value, '_pred_instances', dtype=InstanceData)

    @pred_instances.deleter
    def pred_instances(self) -> None:
        del self._pred_instances

    @property
    def ignored_instances(self) -> InstanceData:
        return self._ignored_instances

    @ignored_instances.setter
    def ignored_instances(self, value: InstanceData) -> None:
        self.set_field(value, '_ignored_instances', dtype=InstanceData)

    @ignored_instances.deleter
    def ignored_instances(self) -> None:
        del self._ignored_instances

    @property
    def gt_panoptic_seg(self) -> PixelData:
        return self._gt_panoptic_seg

    @gt_panoptic_seg.setter
    def gt_panoptic_seg(self, value: PixelData) -> None:
        self.set_field(value, '_gt_panoptic_seg', dtype=PixelData)

    @gt_panoptic_seg.deleter
    def gt_panoptic_seg(self) -> None:
        del self._gt_panoptic_seg

    @property
    def pred_panoptic_seg(self) -> PixelData:
        return self._pred_panoptic_seg

    @pred_panoptic_seg.setter
    def pred_panoptic_seg(self, value: PixelData) -> None:
        self.set_field(value, '_pred_panoptic_seg', dtype=PixelData)

    @pred_panoptic_seg.deleter
    def pred_panoptic_seg(self) -> None:
        del self._pred_panoptic_seg

    @property
    def gt_sem_seg(self) -> PixelData:
        return self._gt_sem_seg

    @gt_sem_seg.setter
    def gt_sem_seg(self, value: PixelData) -> None:
        self.set_field(value, '_gt_sem_seg', dtype=PixelData)

    @gt_sem_seg.deleter
    def gt_sem_seg(self) -> None:
        del self._gt_sem_seg

    @property
    def pred_sem_seg(self) -> PixelData:
        return self._pred_sem_seg

    @pred_sem_seg.setter
    def pred_sem_seg(self, value: PixelData) -> None:
        self.set_field(value, '_pred_sem_seg', dtype=PixelData)

    @pred_sem_seg.deleter
    def pred_sem_seg(self) -> None:
        del self._pred_sem_seg


SampleList = List[DetDataSample]
OptSampleList = Optional[SampleList]

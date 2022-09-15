from unittest import TestCase

import numpy as np
import pytest
import torch
from mmengine.structures import InstanceData, PixelData

from mmdet.structures import DetDataSample


def _equal(a, b):
    if isinstance(a, (torch.Tensor, np.ndarray)):
        return (a == b).all()
    else:
        return a == b


class TestDetDataSample(TestCase):

    def test_init(self):
        meta_info = dict(
            img_size=[256, 256],
            scale_factor=np.array([1.5, 1.5]),
            img_shape=torch.rand(4))

        det_data_sample = DetDataSample(metainfo=meta_info)
        assert 'img_size' in det_data_sample
        assert det_data_sample.img_size == [256, 256]
        assert det_data_sample.get('img_size') == [256, 256]

    def test_setter(self):
        det_data_sample = DetDataSample()
        # test gt_instances
        gt_instances_data = dict(
            bboxes=torch.rand(4, 4),
            labels=torch.rand(4),
            masks=np.random.rand(4, 2, 2))
        gt_instances = InstanceData(**gt_instances_data)
        det_data_sample.gt_instances = gt_instances
        assert 'gt_instances' in det_data_sample
        assert _equal(det_data_sample.gt_instances.bboxes,
                      gt_instances_data['bboxes'])
        assert _equal(det_data_sample.gt_instances.labels,
                      gt_instances_data['labels'])
        assert _equal(det_data_sample.gt_instances.masks,
                      gt_instances_data['masks'])

        # test pred_instances
        pred_instances_data = dict(
            bboxes=torch.rand(2, 4),
            labels=torch.rand(2),
            masks=np.random.rand(2, 2, 2))
        pred_instances = InstanceData(**pred_instances_data)
        det_data_sample.pred_instances = pred_instances
        assert 'pred_instances' in det_data_sample
        assert _equal(det_data_sample.pred_instances.bboxes,
                      pred_instances_data['bboxes'])
        assert _equal(det_data_sample.pred_instances.labels,
                      pred_instances_data['labels'])
        assert _equal(det_data_sample.pred_instances.masks,
                      pred_instances_data['masks'])

        # test proposals
        proposals_data = dict(bboxes=torch.rand(4, 4), labels=torch.rand(4))
        proposals = InstanceData(**proposals_data)
        det_data_sample.proposals = proposals
        assert 'proposals' in det_data_sample
        assert _equal(det_data_sample.proposals.bboxes,
                      proposals_data['bboxes'])
        assert _equal(det_data_sample.proposals.labels,
                      proposals_data['labels'])

        # test ignored_instances
        ignored_instances_data = dict(
            bboxes=torch.rand(4, 4), labels=torch.rand(4))
        ignored_instances = InstanceData(**ignored_instances_data)
        det_data_sample.ignored_instances = ignored_instances
        assert 'ignored_instances' in det_data_sample
        assert _equal(det_data_sample.ignored_instances.bboxes,
                      ignored_instances_data['bboxes'])
        assert _equal(det_data_sample.ignored_instances.labels,
                      ignored_instances_data['labels'])

        # test gt_panoptic_seg
        gt_panoptic_seg_data = dict(panoptic_seg=torch.rand(5, 4))
        gt_panoptic_seg = PixelData(**gt_panoptic_seg_data)
        det_data_sample.gt_panoptic_seg = gt_panoptic_seg
        assert 'gt_panoptic_seg' in det_data_sample
        assert _equal(det_data_sample.gt_panoptic_seg.panoptic_seg,
                      gt_panoptic_seg_data['panoptic_seg'])

        # test pred_panoptic_seg
        pred_panoptic_seg_data = dict(panoptic_seg=torch.rand(5, 4))
        pred_panoptic_seg = PixelData(**pred_panoptic_seg_data)
        det_data_sample.pred_panoptic_seg = pred_panoptic_seg
        assert 'pred_panoptic_seg' in det_data_sample
        assert _equal(det_data_sample.pred_panoptic_seg.panoptic_seg,
                      pred_panoptic_seg_data['panoptic_seg'])

        # test gt_sem_seg
        gt_segm_seg_data = dict(segm_seg=torch.rand(5, 4, 2))
        gt_segm_seg = PixelData(**gt_segm_seg_data)
        det_data_sample.gt_segm_seg = gt_segm_seg
        assert 'gt_segm_seg' in det_data_sample
        assert _equal(det_data_sample.gt_segm_seg.segm_seg,
                      gt_segm_seg_data['segm_seg'])

        # test pred_segm_seg
        pred_segm_seg_data = dict(segm_seg=torch.rand(5, 4, 2))
        pred_segm_seg = PixelData(**pred_segm_seg_data)
        det_data_sample.pred_segm_seg = pred_segm_seg
        assert 'pred_segm_seg' in det_data_sample
        assert _equal(det_data_sample.pred_segm_seg.segm_seg,
                      pred_segm_seg_data['segm_seg'])

        # test type error
        with pytest.raises(AssertionError):
            det_data_sample.pred_instances = torch.rand(2, 4)

        with pytest.raises(AssertionError):
            det_data_sample.pred_panoptic_seg = torch.rand(2, 4)

        with pytest.raises(AssertionError):
            det_data_sample.pred_sem_seg = torch.rand(2, 4)

    def test_deleter(self):
        gt_instances_data = dict(
            bboxes=torch.rand(4, 4),
            labels=torch.rand(4),
            masks=np.random.rand(4, 2, 2))

        det_data_sample = DetDataSample()
        gt_instances = InstanceData(data=gt_instances_data)
        det_data_sample.gt_instances = gt_instances
        assert 'gt_instances' in det_data_sample
        del det_data_sample.gt_instances
        assert 'gt_instances' not in det_data_sample

        pred_panoptic_seg_data = torch.rand(5, 4)
        pred_panoptic_seg = PixelData(data=pred_panoptic_seg_data)
        det_data_sample.pred_panoptic_seg = pred_panoptic_seg
        assert 'pred_panoptic_seg' in det_data_sample
        del det_data_sample.pred_panoptic_seg
        assert 'pred_panoptic_seg' not in det_data_sample

        pred_segm_seg_data = dict(segm_seg=torch.rand(5, 4, 2))
        pred_segm_seg = PixelData(**pred_segm_seg_data)
        det_data_sample.pred_segm_seg = pred_segm_seg
        assert 'pred_segm_seg' in det_data_sample
        del det_data_sample.pred_segm_seg
        assert 'pred_segm_seg' not in det_data_sample

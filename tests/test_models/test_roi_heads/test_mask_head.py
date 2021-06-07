import mmcv
import torch

from mmdet.models.roi_heads.mask_heads import FCNMaskHead, MaskIoUHead
from .utils import _dummy_bbox_sampling


def test_mask_head_loss():
    """Test mask head loss when mask target is empty."""
    self = FCNMaskHead(
        num_convs=1,
        roi_feat_size=6,
        in_channels=8,
        conv_out_channels=8,
        num_classes=8)

    # Dummy proposals
    proposal_list = [
        torch.Tensor([[23.6667, 23.8757, 228.6326, 153.8874]]),
    ]

    gt_bboxes = [
        torch.Tensor([[23.6667, 23.8757, 238.6326, 151.8874]]),
    ]
    gt_labels = [torch.LongTensor([2])]
    sampling_results = _dummy_bbox_sampling(proposal_list, gt_bboxes,
                                            gt_labels)

    # create dummy mask
    import numpy as np
    from mmdet.core import BitmapMasks
    dummy_mask = np.random.randint(0, 2, (1, 160, 240), dtype=np.uint8)
    gt_masks = [BitmapMasks(dummy_mask, 160, 240)]

    # create dummy train_cfg
    train_cfg = mmcv.Config(dict(mask_size=12, mask_thr_binary=0.5))

    # Create dummy features "extracted" for each sampled bbox
    num_sampled = sum(len(res.bboxes) for res in sampling_results)
    dummy_feats = torch.rand(num_sampled, 8, 6, 6)

    mask_pred = self.forward(dummy_feats)
    mask_targets = self.get_targets(sampling_results, gt_masks, train_cfg)
    pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
    loss_mask = self.loss(mask_pred, mask_targets, pos_labels)

    onegt_mask_loss = sum(loss_mask['loss_mask'])
    assert onegt_mask_loss.item() > 0, 'mask loss should be non-zero'

    # test mask_iou_head
    mask_iou_head = MaskIoUHead(
        num_convs=1,
        num_fcs=1,
        roi_feat_size=6,
        in_channels=8,
        conv_out_channels=8,
        fc_out_channels=8,
        num_classes=8)

    pos_mask_pred = mask_pred[range(mask_pred.size(0)), pos_labels]
    mask_iou_pred = mask_iou_head(dummy_feats, pos_mask_pred)
    pos_mask_iou_pred = mask_iou_pred[range(mask_iou_pred.size(0)), pos_labels]

    mask_iou_targets = mask_iou_head.get_targets(sampling_results, gt_masks,
                                                 pos_mask_pred, mask_targets,
                                                 train_cfg)
    loss_mask_iou = mask_iou_head.loss(pos_mask_iou_pred, mask_iou_targets)
    onegt_mask_iou_loss = loss_mask_iou['loss_mask_iou'].sum()
    assert onegt_mask_iou_loss.item() >= 0


def test_can_skip_post_processing_in_onnx():
    self = FCNMaskHead()
    num_classes = 80
    num_bboxes = 10
    mask_shape = [28, 28]
    original_shape = [800, 1200]
    mask_pred = torch.rand(
        [num_bboxes, num_classes] + mask_shape, dtype=torch.float32)
    det_bboxes = torch.rand([num_bboxes, 4], dtype=torch.float32)
    det_labels = torch.randint(0, num_classes, [num_bboxes], dtype=torch.int64)
    from mmcv.utils.config import ConfigDict
    rcnn_test_cfg = ConfigDict({
        'mask_thr_binary': 0.5,
        'rescale_mask_to_input_shape': True
    })
    ori_shape = torch.tensor(original_shape, dtype=torch.int64)

    rcnn_test_cfg.rescale_mask_to_input_shape = True
    masks = self.onnx_export(mask_pred, det_bboxes, det_labels, rcnn_test_cfg,
                             ori_shape)
    assert masks.shape[1:] == torch.Size(original_shape)

    rcnn_test_cfg.rescale_mask_to_input_shape = False
    masks = self.onnx_export(mask_pred, det_bboxes, det_labels, rcnn_test_cfg,
                             ori_shape)
    assert masks.shape[1:] == torch.Size(mask_shape)

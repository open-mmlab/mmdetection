import mmcv
import numpy as np
import torch
import torch.nn as nn
from utils import verify_model

from mmdet.models.dense_heads import RetinaHead, YOLOV3Head

# Control the forward of the Test class
test_step_names = {
    'forward_single': 0,
    'forward': 1,
    '_get_bboxes_single': 2,
    'get_bboxes': 3
}

data_path = 'tests/test_onnx/data/'


class AnchorHeadTest(nn.Module):
    """AnchorHead Test Class.

    Args:
        test_head (AnchorHead): The head to be testd.
        head_cfg (dict): Config of head.
        test_cfg (mmcv.Config): Testing config of head.
        img_metas (list): List of image information.
        test_step (int): Specify the part of AnchorHead you want to test, which
            the name corresponded the test_step is in the test_step_names.
        with_nms (bool): If true, do nms before return boxes. Default: True.
    """

    def __init__(self,
                 test_head,
                 head_cfg,
                 test_cfg,
                 img_metas,
                 test_step,
                 with_nms=True,
                 num_classes=4,
                 in_channels=1):
        super(AnchorHeadTest, self).__init__()
        self.head = test_head(
            num_classes=num_classes,
            in_channels=in_channels,
            test_cfg=test_cfg,
            **head_cfg)
        self.head_cfg = head_cfg
        self.img_metas = img_metas
        self.test_step = test_step
        self.with_nms = with_nms

    def forward(self, feat):
        """Forward feature according test_step.

        Args:
            feats (list[Tensor]): A list of tensors from torch.rand
                to simulate input, each is a 4D-tensor.

        Returns:
            list[Tensor]: A list of outputs from unit test, each is
                a 4D-tensor.
        """
        if (self.test_step == 0):  # forward_single test
            cls_score, bbox_pred = self.head.forward_single(feat[0])
            return [cls_score] + [bbox_pred]
        cls_scores, bbox_preds = self.head.forward(feat)
        if (self.test_step == 1):  # forward test
            return cls_scores + bbox_preds
        if (self.test_step == 2):  # get_bboxes test
            num_levels = len(cls_scores)
            featmap_sizes = [
                cls_scores[i][0].shape[-2:] for i in range(num_levels)
            ]
            cls_score_list = [
                cls_scores[i][0].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][0].detach() for i in range(num_levels)
            ]
            mlvl_anchors = self.head.anchor_generator.grid_anchors(
                featmap_sizes, device='cpu')
            img_shape = self.img_metas[0]['img_shape']
            scale_factor = self.img_metas[0]['scale_factor']
            if self.with_nms:
                det_bboxes, det_labels = self.head._get_bboxes_single(
                    cls_score_list, bbox_pred_list, mlvl_anchors, img_shape,
                    scale_factor, None)
                return [det_bboxes] + [det_labels]
            else:
                mlvl_bboxes, mlvl_scores = self._get_bboxes_single(
                    cls_score_list, bbox_pred_list, mlvl_anchors, img_shape,
                    scale_factor, None, self.with_nms)
                return [mlvl_bboxes] + [mlvl_scores]
        result_list = self.head.get_bboxes(cls_scores, bbox_preds,
                                           self.img_metas)
        if (self.test_step == 3):  # get_bboxes test
            return list(result_list[0])


def retinanet_config(test_step_name):
    """RetinaNet Head Test Config.

    Args:
        test_step_name (str): The unit test to be used.

    Returns:
        AnchorHeadTest: A AnchorHeadTest which initialized with the specified
            test_step_name
        list[Tensor]: A list of tensors from torch.rand to simulate input,
            each is a 4D-tensor.
    """
    s = 128
    img_metas = [{
        'img_shape': (s, s, 3),
        'scale_factor': 1,
        'pad_shape': (s, s, 3)
    }]

    head_cfg = dict(
        stacked_convs=6,
        feat_channels=2,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]))

    test_cfg = mmcv.Config(
        dict(
            nms_pre=1000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100))

    model = AnchorHeadTest(
        RetinaHead,
        head_cfg,
        test_cfg,
        img_metas,
        test_step=test_step_names[test_step_name],
        num_classes=4,
        in_channels=1)
    model.cpu().eval()

    feat = []
    retina_head_data = 'retina_head_'
    for i in range(len(model.head.anchor_generator.strides)):
        data_name = data_path + retina_head_data + str(i) + '.npy'
        feat.append(torch.tensor(np.load(data_name)))

    if (test_step_name != 'forward_single'):
        return model, feat
    else:
        return model, [feat[0]]


class YoloV3HeadTest(nn.Module):
    """YoloV3Head Test Class.

    Args:
        test_head (YoloV3Head): The YoloV3 head which need to be testd.
        test_cfg (mmcv.Config): Testing config of head.
        img_metas (list): List of image information.
        test_step (int): Specify the part of AnchorHead you want to test,
            which the name corresponded the test_step is in the test_step_names
        with_nms (bool): If true, do nms before return boxes. Default: True
    """

    def __init__(self,
                 test_head,
                 head_cfg,
                 test_cfg,
                 img_metas,
                 test_step,
                 with_nms=True,
                 num_classes=4,
                 in_channels=[512, 256, 128],
                 out_channels=[1024, 512, 256]):
        super(YoloV3HeadTest, self).__init__()
        self.head = test_head(
            num_classes=num_classes,
            in_channels=in_channels,
            out_channels=out_channels,
            test_cfg=test_cfg,
            **head_cfg)
        self.img_metas = img_metas
        self.test_step = test_step
        self.with_nms = with_nms

    def forward(self, feat):
        """Forward feature according self.test_step.

        Args:
            feats (list[Tensor]): A list of tensors from torch.rand
                to simulate input, each is a 4D-tensor.

        Returns:
            list[Tensor]: A list of all of outputs from unit test, each is
                a 4D-tensor.
        """
        pred_maps = self.head.forward(feat)
        if (self.test_step == 1):  # forward test
            return list(pred_maps[0])
        if (self.test_step == 2):  # get_bboxes test
            num_levels = len(pred_maps[0])
            pred_maps_list = [
                pred_maps[0][i][0].detach() for i in range(num_levels)
            ]
            scale_factor = self.img_metas[0]['scale_factor']
            if self.with_nms:
                det_bboxes, det_labels = self.head._get_bboxes_single(
                    pred_maps_list, scale_factor, None)
                return list(det_bboxes) + list(det_labels)
            else:
                outs = self._get_bboxes_single(pred_maps_list, scale_factor,
                                               None, self.with_nms)
                return outs[0] + outs[1] + outs[2]
        result_list = self.head.get_bboxes(pred_maps[0], self.img_metas)
        if (self.test_step == 3):  # get_bboxes test
            return list(result_list[0])


def yolo_config(test_step_name):
    """YoloV3 Head Test Config.

    Args:
        test_step_name (str): The unit test to be used.

    Returns:
        YoloV3HeadTest: A YoloV3HeadTest which initialized with the specified
            test_step_name
        list[Tensor]: A list of tensors from torch.rand to simulate input,
            each is a 4D-tensor.
    """
    s = 256
    img_metas = [{
        'img_shape': (s, s, 3),
        'scale_factor': 1,
        'pad_shape': (s, s, 3)
    }]

    head_cfg = dict(
        anchor_generator=dict(
            type='YOLOAnchorGenerator',
            base_sizes=[[(116, 90), (156, 198), (373, 326)],
                        [(30, 61), (62, 45), (59, 119)],
                        [(10, 13), (16, 30), (33, 23)]],
            strides=[32, 16, 8]),
        bbox_coder=dict(type='YOLOBBoxCoder'))

    test_cfg = mmcv.Config(
        dict(
            nms_pre=1000,
            min_bbox_size=0,
            score_thr=0.05,
            conf_thr=0.005,
            nms=dict(type='nms', iou_threshold=0.45),
            max_per_img=100))

    model = YoloV3HeadTest(
        YOLOV3Head,
        head_cfg,
        test_cfg,
        img_metas,
        in_channels=[1, 1, 1],
        out_channels=[32, 16, 8],
        num_classes=4,
        test_step=test_step_names[test_step_name])
    model.cpu().eval()

    feat = []
    yolov3_head_data = 'yolov3_head_'
    for i in range(len(model.head.anchor_generator.strides)):
        data_name = data_path + yolov3_head_data + str(i) + '.npy'
        feat.append(torch.tensor(np.load(data_name)))

    return model, feat


def test_retinanet_head_forward_single():
    outs = retinanet_config('forward_single')
    verify_model(*outs)


def test_retinanet_head_forward():
    outs = retinanet_config('forward')
    verify_model(*outs)


def test_retinanet_head_get_bboxes_single():
    outs = retinanet_config('_get_bboxes_single')
    verify_model(*outs)


def test_retinanet_head_get_bboxes():
    outs = retinanet_config('get_bboxes')
    verify_model(*outs)


def test_yolov3_head_forward():
    outs = yolo_config('forward')
    verify_model(*outs)


def test_yolov3_head_get_bboxes_single():
    outs = yolo_config('_get_bboxes_single')
    verify_model(*outs)


def test_yolov3_head_get_bboxes():
    outs = yolo_config('get_bboxes')
    verify_model(*outs)

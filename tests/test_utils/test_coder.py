import torch

from mmdet.core.bbox.coder import YOLOBBoxCoder


def test_yolo_bbox_coder():
    coder = YOLOBBoxCoder()
    bboxes = torch.Tensor([[-42., -29., 74., 61.], [-10., -29., 106., 61.],
                           [22., -29., 138., 61.], [54., -29., 170., 61.]])
    pred_bboxes = torch.Tensor([[0.4709, 0.6152, 0.1690, -0.4056],
                                [0.5399, 0.6653, 0.1162, -0.4162],
                                [0.4654, 0.6618, 0.1548, -0.4301],
                                [0.4786, 0.6197, 0.1896, -0.4479]])
    grid_size = 32
    expected_decode_bboxes = torch.Tensor(
        [[-53.6102, -10.3096, 83.7478, 49.6824],
         [-15.8700, -8.3901, 114.4236, 50.9693],
         [11.1822, -8.0924, 146.6034, 50.4476],
         [41.2068, -8.9232, 181.4236, 48.5840]])
    assert expected_decode_bboxes.allclose(
        coder.decode(bboxes, pred_bboxes, grid_size))

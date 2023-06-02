# Copyright (c) OpenMMLab. All rights reserved.
"""
CommandLine:
    pytest tests/test_utils/test_anchor.py
    xdoctest tests/test_utils/test_anchor.py zero

"""
import pytest
import torch


def test_standard_points_generator():
    from mmdet.core.anchor import build_prior_generator

    # teat init
    anchor_generator_cfg = dict(
        type='MlvlPointGenerator', strides=[4, 8], offset=0)
    anchor_generator = build_prior_generator(anchor_generator_cfg)
    assert anchor_generator is not None
    assert anchor_generator.num_base_priors == [1, 1]
    # test_stride
    from mmdet.core.anchor import MlvlPointGenerator

    # Square strides
    mlvl_points = MlvlPointGenerator(strides=[4, 10], offset=0)
    mlvl_points_half_stride_generator = MlvlPointGenerator(
        strides=[4, 10], offset=0.5)
    assert mlvl_points.num_levels == 2

    # assert self.num_levels == len(featmap_sizes)
    with pytest.raises(AssertionError):
        mlvl_points.grid_priors(featmap_sizes=[(2, 2)], device='cpu')
    priors = mlvl_points.grid_priors(
        featmap_sizes=[(2, 2), (4, 8)], device='cpu')
    priors_with_stride = mlvl_points.grid_priors(
        featmap_sizes=[(2, 2), (4, 8)], with_stride=True, device='cpu')
    assert len(priors) == 2

    # assert last dimension is (coord_x, coord_y, stride_w, stride_h).
    assert priors_with_stride[0].size(1) == 4
    assert priors_with_stride[0][0][2] == 4
    assert priors_with_stride[0][0][3] == 4
    assert priors_with_stride[1][0][2] == 10
    assert priors_with_stride[1][0][3] == 10

    stride_4_feat_2_2 = priors[0]
    assert (stride_4_feat_2_2[1] - stride_4_feat_2_2[0]).sum() == 4
    assert stride_4_feat_2_2.size(0) == 4
    assert stride_4_feat_2_2.size(1) == 2

    stride_10_feat_4_8 = priors[1]
    assert (stride_10_feat_4_8[1] - stride_10_feat_4_8[0]).sum() == 10
    assert stride_10_feat_4_8.size(0) == 4 * 8
    assert stride_10_feat_4_8.size(1) == 2

    # assert the offset of 0.5 * stride
    priors_half_offset = mlvl_points_half_stride_generator.grid_priors(
        featmap_sizes=[(2, 2), (4, 8)], device='cpu')

    assert (priors_half_offset[0][0] - priors[0][0]).sum() == 4 * 0.5 * 2
    assert (priors_half_offset[1][0] - priors[1][0]).sum() == 10 * 0.5 * 2
    if torch.cuda.is_available():
        anchor_generator_cfg = dict(
            type='MlvlPointGenerator', strides=[4, 8], offset=0)
        anchor_generator = build_prior_generator(anchor_generator_cfg)
        assert anchor_generator is not None
        # Square strides
        mlvl_points = MlvlPointGenerator(strides=[4, 10], offset=0)
        mlvl_points_half_stride_generator = MlvlPointGenerator(
            strides=[4, 10], offset=0.5)
        assert mlvl_points.num_levels == 2

        # assert self.num_levels == len(featmap_sizes)
        with pytest.raises(AssertionError):
            mlvl_points.grid_priors(featmap_sizes=[(2, 2)], device='cuda')
        priors = mlvl_points.grid_priors(
            featmap_sizes=[(2, 2), (4, 8)], device='cuda')
        priors_with_stride = mlvl_points.grid_priors(
            featmap_sizes=[(2, 2), (4, 8)], with_stride=True, device='cuda')
        assert len(priors) == 2

        # assert last dimension is (coord_x, coord_y, stride_w, stride_h).
        assert priors_with_stride[0].size(1) == 4
        assert priors_with_stride[0][0][2] == 4
        assert priors_with_stride[0][0][3] == 4
        assert priors_with_stride[1][0][2] == 10
        assert priors_with_stride[1][0][3] == 10

        stride_4_feat_2_2 = priors[0]
        assert (stride_4_feat_2_2[1] - stride_4_feat_2_2[0]).sum() == 4
        assert stride_4_feat_2_2.size(0) == 4
        assert stride_4_feat_2_2.size(1) == 2

        stride_10_feat_4_8 = priors[1]
        assert (stride_10_feat_4_8[1] - stride_10_feat_4_8[0]).sum() == 10
        assert stride_10_feat_4_8.size(0) == 4 * 8
        assert stride_10_feat_4_8.size(1) == 2

        # assert the offset of 0.5 * stride
        priors_half_offset = mlvl_points_half_stride_generator.grid_priors(
            featmap_sizes=[(2, 2), (4, 8)], device='cuda')

        assert (priors_half_offset[0][0] - priors[0][0]).sum() == 4 * 0.5 * 2
        assert (priors_half_offset[1][0] - priors[1][0]).sum() == 10 * 0.5 * 2


def test_sparse_prior():
    from mmdet.core.anchor import MlvlPointGenerator
    mlvl_points = MlvlPointGenerator(strides=[4, 10], offset=0)
    prior_indexs = torch.Tensor([0, 2, 4, 5, 6, 9]).long()

    featmap_sizes = [(3, 5), (6, 4)]
    grid_anchors = mlvl_points.grid_priors(
        featmap_sizes=featmap_sizes, with_stride=False, device='cpu')
    sparse_prior = mlvl_points.sparse_priors(
        prior_idxs=prior_indexs,
        featmap_size=featmap_sizes[0],
        level_idx=0,
        device='cpu')

    assert not sparse_prior.is_cuda
    assert (sparse_prior == grid_anchors[0][prior_indexs]).all()
    sparse_prior = mlvl_points.sparse_priors(
        prior_idxs=prior_indexs,
        featmap_size=featmap_sizes[1],
        level_idx=1,
        device='cpu')
    assert (sparse_prior == grid_anchors[1][prior_indexs]).all()

    from mmdet.core.anchor import AnchorGenerator
    mlvl_anchors = AnchorGenerator(
        strides=[16, 32], ratios=[1.], scales=[1.], base_sizes=[4, 8])
    prior_indexs = torch.Tensor([0, 2, 4, 5, 6, 9]).long()

    featmap_sizes = [(3, 5), (6, 4)]
    grid_anchors = mlvl_anchors.grid_priors(
        featmap_sizes=featmap_sizes, device='cpu')
    sparse_prior = mlvl_anchors.sparse_priors(
        prior_idxs=prior_indexs,
        featmap_size=featmap_sizes[0],
        level_idx=0,
        device='cpu')
    assert (sparse_prior == grid_anchors[0][prior_indexs]).all()
    sparse_prior = mlvl_anchors.sparse_priors(
        prior_idxs=prior_indexs,
        featmap_size=featmap_sizes[1],
        level_idx=1,
        device='cpu')
    assert (sparse_prior == grid_anchors[1][prior_indexs]).all()

    # for ssd
    from mmdet.core.anchor.anchor_generator import SSDAnchorGenerator
    featmap_sizes = [(38, 38), (19, 19), (10, 10)]
    anchor_generator = SSDAnchorGenerator(
        scale_major=False,
        input_size=300,
        basesize_ratio_range=(0.15, 0.9),
        strides=[8, 16, 32],
        ratios=[[2], [2, 3], [2, 3]])
    ssd_anchors = anchor_generator.grid_anchors(featmap_sizes, device='cpu')
    for i in range(len(featmap_sizes)):
        sparse_ssd_anchors = anchor_generator.sparse_priors(
            prior_idxs=prior_indexs,
            level_idx=i,
            featmap_size=featmap_sizes[i],
            device='cpu')
        assert (sparse_ssd_anchors == ssd_anchors[i][prior_indexs]).all()

    # for yolo
    from mmdet.core.anchor.anchor_generator import YOLOAnchorGenerator
    featmap_sizes = [(38, 38), (19, 19), (10, 10)]
    anchor_generator = YOLOAnchorGenerator(
        strides=[32, 16, 8],
        base_sizes=[
            [(116, 90), (156, 198), (373, 326)],
            [(30, 61), (62, 45), (59, 119)],
            [(10, 13), (16, 30), (33, 23)],
        ])
    yolo_anchors = anchor_generator.grid_anchors(featmap_sizes, device='cpu')
    for i in range(len(featmap_sizes)):
        sparse_yolo_anchors = anchor_generator.sparse_priors(
            prior_idxs=prior_indexs,
            level_idx=i,
            featmap_size=featmap_sizes[i],
            device='cpu')
        assert (sparse_yolo_anchors == yolo_anchors[i][prior_indexs]).all()

    if torch.cuda.is_available():
        mlvl_points = MlvlPointGenerator(strides=[4, 10], offset=0)
        prior_indexs = torch.Tensor([0, 3, 4, 5, 6, 7, 1, 2, 4, 5, 6,
                                     9]).long().cuda()

        featmap_sizes = [(6, 8), (6, 4)]
        grid_anchors = mlvl_points.grid_priors(
            featmap_sizes=featmap_sizes, with_stride=False, device='cuda')
        sparse_prior = mlvl_points.sparse_priors(
            prior_idxs=prior_indexs,
            featmap_size=featmap_sizes[0],
            level_idx=0,
            device='cuda')
        assert (sparse_prior == grid_anchors[0][prior_indexs]).all()
        sparse_prior = mlvl_points.sparse_priors(
            prior_idxs=prior_indexs,
            featmap_size=featmap_sizes[1],
            level_idx=1,
            device='cuda')
        assert (sparse_prior == grid_anchors[1][prior_indexs]).all()
        assert sparse_prior.is_cuda
        mlvl_anchors = AnchorGenerator(
            strides=[16, 32],
            ratios=[1., 2.5],
            scales=[1., 5.],
            base_sizes=[4, 8])
        prior_indexs = torch.Tensor([4, 5, 6, 7, 0, 2, 50, 4, 5, 6,
                                     9]).long().cuda()

        featmap_sizes = [(13, 5), (16, 4)]
        grid_anchors = mlvl_anchors.grid_priors(
            featmap_sizes=featmap_sizes, device='cuda')
        sparse_prior = mlvl_anchors.sparse_priors(
            prior_idxs=prior_indexs,
            featmap_size=featmap_sizes[0],
            level_idx=0,
            device='cuda')
        assert (sparse_prior == grid_anchors[0][prior_indexs]).all()
        sparse_prior = mlvl_anchors.sparse_priors(
            prior_idxs=prior_indexs,
            featmap_size=featmap_sizes[1],
            level_idx=1,
            device='cuda')
        assert (sparse_prior == grid_anchors[1][prior_indexs]).all()

        # for ssd
        from mmdet.core.anchor.anchor_generator import SSDAnchorGenerator
        featmap_sizes = [(38, 38), (19, 19), (10, 10)]
        anchor_generator = SSDAnchorGenerator(
            scale_major=False,
            input_size=300,
            basesize_ratio_range=(0.15, 0.9),
            strides=[8, 16, 32],
            ratios=[[2], [2, 3], [2, 3]])
        ssd_anchors = anchor_generator.grid_anchors(
            featmap_sizes, device='cuda')
        for i in range(len(featmap_sizes)):
            sparse_ssd_anchors = anchor_generator.sparse_priors(
                prior_idxs=prior_indexs,
                level_idx=i,
                featmap_size=featmap_sizes[i],
                device='cuda')
            assert (sparse_ssd_anchors == ssd_anchors[i][prior_indexs]).all()

        # for yolo
        from mmdet.core.anchor.anchor_generator import YOLOAnchorGenerator
        featmap_sizes = [(38, 38), (19, 19), (10, 10)]
        anchor_generator = YOLOAnchorGenerator(
            strides=[32, 16, 8],
            base_sizes=[
                [(116, 90), (156, 198), (373, 326)],
                [(30, 61), (62, 45), (59, 119)],
                [(10, 13), (16, 30), (33, 23)],
            ])
        yolo_anchors = anchor_generator.grid_anchors(
            featmap_sizes, device='cuda')
        for i in range(len(featmap_sizes)):
            sparse_yolo_anchors = anchor_generator.sparse_priors(
                prior_idxs=prior_indexs,
                level_idx=i,
                featmap_size=featmap_sizes[i],
                device='cuda')
            assert (sparse_yolo_anchors == yolo_anchors[i][prior_indexs]).all()


def test_standard_anchor_generator():
    from mmdet.core.anchor import build_anchor_generator
    anchor_generator_cfg = dict(
        type='AnchorGenerator',
        scales=[8],
        ratios=[0.5, 1.0, 2.0],
        strides=[4, 8])

    anchor_generator = build_anchor_generator(anchor_generator_cfg)
    assert anchor_generator.num_base_priors == \
           anchor_generator.num_base_anchors
    assert anchor_generator.num_base_priors == [3, 3]
    assert anchor_generator is not None


def test_strides():
    from mmdet.core import AnchorGenerator

    # Square strides
    self = AnchorGenerator([10], [1.], [1.], [10])
    anchors = self.grid_anchors([(2, 2)], device='cpu')

    expected_anchors = torch.tensor([[-5., -5., 5., 5.], [5., -5., 15., 5.],
                                     [-5., 5., 5., 15.], [5., 5., 15., 15.]])

    assert torch.equal(anchors[0], expected_anchors)

    # Different strides in x and y direction
    self = AnchorGenerator([(10, 20)], [1.], [1.], [10])
    anchors = self.grid_anchors([(2, 2)], device='cpu')

    expected_anchors = torch.tensor([[-5., -5., 5., 5.], [5., -5., 15., 5.],
                                     [-5., 15., 5., 25.], [5., 15., 15., 25.]])

    assert torch.equal(anchors[0], expected_anchors)


def test_ssd_anchor_generator():
    from mmdet.core.anchor import build_anchor_generator
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    # min_sizes max_sizes must set at the same time
    with pytest.raises(AssertionError):
        anchor_generator_cfg = dict(
            type='SSDAnchorGenerator',
            scale_major=False,
            min_sizes=[48, 100, 150, 202, 253, 300],
            max_sizes=None,
            strides=[8, 16, 32, 64, 100, 300],
            ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]])
        build_anchor_generator(anchor_generator_cfg)

    # length of min_sizes max_sizes must be the same
    with pytest.raises(AssertionError):
        anchor_generator_cfg = dict(
            type='SSDAnchorGenerator',
            scale_major=False,
            min_sizes=[48, 100, 150, 202, 253, 300],
            max_sizes=[100, 150, 202, 253],
            strides=[8, 16, 32, 64, 100, 300],
            ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]])
        build_anchor_generator(anchor_generator_cfg)

    # test setting anchor size manually
    anchor_generator_cfg = dict(
        type='SSDAnchorGenerator',
        scale_major=False,
        min_sizes=[48, 100, 150, 202, 253, 304],
        max_sizes=[100, 150, 202, 253, 304, 320],
        strides=[16, 32, 64, 107, 160, 320],
        ratios=[[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]])

    featmap_sizes = [(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)]
    anchor_generator = build_anchor_generator(anchor_generator_cfg)

    expected_base_anchors = [
        torch.Tensor([[-16.0000, -16.0000, 32.0000, 32.0000],
                      [-26.6410, -26.6410, 42.6410, 42.6410],
                      [-25.9411, -8.9706, 41.9411, 24.9706],
                      [-8.9706, -25.9411, 24.9706, 41.9411],
                      [-33.5692, -5.8564, 49.5692, 21.8564],
                      [-5.8564, -33.5692, 21.8564, 49.5692]]),
        torch.Tensor([[-34.0000, -34.0000, 66.0000, 66.0000],
                      [-45.2372, -45.2372, 77.2372, 77.2372],
                      [-54.7107, -19.3553, 86.7107, 51.3553],
                      [-19.3553, -54.7107, 51.3553, 86.7107],
                      [-70.6025, -12.8675, 102.6025, 44.8675],
                      [-12.8675, -70.6025, 44.8675, 102.6025]]),
        torch.Tensor([[-43.0000, -43.0000, 107.0000, 107.0000],
                      [-55.0345, -55.0345, 119.0345, 119.0345],
                      [-74.0660, -21.0330, 138.0660, 85.0330],
                      [-21.0330, -74.0660, 85.0330, 138.0660],
                      [-97.9038, -11.3013, 161.9038, 75.3013],
                      [-11.3013, -97.9038, 75.3013, 161.9038]]),
        torch.Tensor([[-47.5000, -47.5000, 154.5000, 154.5000],
                      [-59.5332, -59.5332, 166.5332, 166.5332],
                      [-89.3356, -17.9178, 196.3356, 124.9178],
                      [-17.9178, -89.3356, 124.9178, 196.3356],
                      [-121.4371, -4.8124, 228.4371, 111.8124],
                      [-4.8124, -121.4371, 111.8124, 228.4371]]),
        torch.Tensor([[-46.5000, -46.5000, 206.5000, 206.5000],
                      [-58.6651, -58.6651, 218.6651, 218.6651],
                      [-98.8980, -9.4490, 258.8980, 169.4490],
                      [-9.4490, -98.8980, 169.4490, 258.8980],
                      [-139.1044, 6.9652, 299.1044, 153.0348],
                      [6.9652, -139.1044, 153.0348, 299.1044]]),
        torch.Tensor([[8.0000, 8.0000, 312.0000, 312.0000],
                      [4.0513, 4.0513, 315.9487, 315.9487],
                      [-54.9605, 52.5198, 374.9604, 267.4802],
                      [52.5198, -54.9605, 267.4802, 374.9604],
                      [-103.2717, 72.2428, 423.2717, 247.7572],
                      [72.2428, -103.2717, 247.7572, 423.2717]])
    ]

    base_anchors = anchor_generator.base_anchors
    for i, base_anchor in enumerate(base_anchors):
        assert base_anchor.allclose(expected_base_anchors[i])

    # check valid flags
    expected_valid_pixels = [2400, 600, 150, 54, 24, 6]
    multi_level_valid_flags = anchor_generator.valid_flags(
        featmap_sizes, (320, 320), device)
    for i, single_level_valid_flag in enumerate(multi_level_valid_flags):
        assert single_level_valid_flag.sum() == expected_valid_pixels[i]

    # check number of base anchors for each level
    assert anchor_generator.num_base_anchors == [6, 6, 6, 6, 6, 6]

    # check anchor generation
    anchors = anchor_generator.grid_anchors(featmap_sizes, device)
    assert len(anchors) == 6

    # test vgg ssd anchor setting
    anchor_generator_cfg = dict(
        type='SSDAnchorGenerator',
        scale_major=False,
        input_size=300,
        basesize_ratio_range=(0.15, 0.9),
        strides=[8, 16, 32, 64, 100, 300],
        ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]])

    featmap_sizes = [(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)]
    anchor_generator = build_anchor_generator(anchor_generator_cfg)

    # check base anchors
    expected_base_anchors = [
        torch.Tensor([[-6.5000, -6.5000, 14.5000, 14.5000],
                      [-11.3704, -11.3704, 19.3704, 19.3704],
                      [-10.8492, -3.4246, 18.8492, 11.4246],
                      [-3.4246, -10.8492, 11.4246, 18.8492]]),
        torch.Tensor([[-14.5000, -14.5000, 30.5000, 30.5000],
                      [-25.3729, -25.3729, 41.3729, 41.3729],
                      [-23.8198, -7.9099, 39.8198, 23.9099],
                      [-7.9099, -23.8198, 23.9099, 39.8198],
                      [-30.9711, -4.9904, 46.9711, 20.9904],
                      [-4.9904, -30.9711, 20.9904, 46.9711]]),
        torch.Tensor([[-33.5000, -33.5000, 65.5000, 65.5000],
                      [-45.5366, -45.5366, 77.5366, 77.5366],
                      [-54.0036, -19.0018, 86.0036, 51.0018],
                      [-19.0018, -54.0036, 51.0018, 86.0036],
                      [-69.7365, -12.5788, 101.7365, 44.5788],
                      [-12.5788, -69.7365, 44.5788, 101.7365]]),
        torch.Tensor([[-44.5000, -44.5000, 108.5000, 108.5000],
                      [-56.9817, -56.9817, 120.9817, 120.9817],
                      [-76.1873, -22.0937, 140.1873, 86.0937],
                      [-22.0937, -76.1873, 86.0937, 140.1873],
                      [-100.5019, -12.1673, 164.5019, 76.1673],
                      [-12.1673, -100.5019, 76.1673, 164.5019]]),
        torch.Tensor([[-53.5000, -53.5000, 153.5000, 153.5000],
                      [-66.2185, -66.2185, 166.2185, 166.2185],
                      [-96.3711, -23.1855, 196.3711, 123.1855],
                      [-23.1855, -96.3711, 123.1855, 196.3711]]),
        torch.Tensor([[19.5000, 19.5000, 280.5000, 280.5000],
                      [6.6342, 6.6342, 293.3658, 293.3658],
                      [-34.5549, 57.7226, 334.5549, 242.2774],
                      [57.7226, -34.5549, 242.2774, 334.5549]]),
    ]
    base_anchors = anchor_generator.base_anchors
    for i, base_anchor in enumerate(base_anchors):
        assert base_anchor.allclose(expected_base_anchors[i])

    # check valid flags
    expected_valid_pixels = [5776, 2166, 600, 150, 36, 4]
    multi_level_valid_flags = anchor_generator.valid_flags(
        featmap_sizes, (300, 300), device)
    for i, single_level_valid_flag in enumerate(multi_level_valid_flags):
        assert single_level_valid_flag.sum() == expected_valid_pixels[i]

    # check number of base anchors for each level
    assert anchor_generator.num_base_anchors == [4, 6, 6, 6, 4, 4]

    # check anchor generation
    anchors = anchor_generator.grid_anchors(featmap_sizes, device)
    assert len(anchors) == 6


def test_anchor_generator_with_tuples():
    from mmdet.core.anchor import build_anchor_generator
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    anchor_generator_cfg = dict(
        type='SSDAnchorGenerator',
        scale_major=False,
        input_size=300,
        basesize_ratio_range=(0.15, 0.9),
        strides=[8, 16, 32, 64, 100, 300],
        ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]])

    featmap_sizes = [(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)]
    anchor_generator = build_anchor_generator(anchor_generator_cfg)
    anchors = anchor_generator.grid_anchors(featmap_sizes, device)

    anchor_generator_cfg_tuples = dict(
        type='SSDAnchorGenerator',
        scale_major=False,
        input_size=300,
        basesize_ratio_range=(0.15, 0.9),
        strides=[(8, 8), (16, 16), (32, 32), (64, 64), (100, 100), (300, 300)],
        ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]])

    anchor_generator_tuples = build_anchor_generator(
        anchor_generator_cfg_tuples)
    anchors_tuples = anchor_generator_tuples.grid_anchors(
        featmap_sizes, device)
    for anchor, anchor_tuples in zip(anchors, anchors_tuples):
        assert torch.equal(anchor, anchor_tuples)


def test_yolo_anchor_generator():
    from mmdet.core.anchor import build_anchor_generator
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    anchor_generator_cfg = dict(
        type='YOLOAnchorGenerator',
        strides=[32, 16, 8],
        base_sizes=[
            [(116, 90), (156, 198), (373, 326)],
            [(30, 61), (62, 45), (59, 119)],
            [(10, 13), (16, 30), (33, 23)],
        ])

    featmap_sizes = [(14, 18), (28, 36), (56, 72)]
    anchor_generator = build_anchor_generator(anchor_generator_cfg)

    # check base anchors
    expected_base_anchors = [
        torch.Tensor([[-42.0000, -29.0000, 74.0000, 61.0000],
                      [-62.0000, -83.0000, 94.0000, 115.0000],
                      [-170.5000, -147.0000, 202.5000, 179.0000]]),
        torch.Tensor([[-7.0000, -22.5000, 23.0000, 38.5000],
                      [-23.0000, -14.5000, 39.0000, 30.5000],
                      [-21.5000, -51.5000, 37.5000, 67.5000]]),
        torch.Tensor([[-1.0000, -2.5000, 9.0000, 10.5000],
                      [-4.0000, -11.0000, 12.0000, 19.0000],
                      [-12.5000, -7.5000, 20.5000, 15.5000]])
    ]
    base_anchors = anchor_generator.base_anchors
    for i, base_anchor in enumerate(base_anchors):
        assert base_anchor.allclose(expected_base_anchors[i])

    # check number of base anchors for each level
    assert anchor_generator.num_base_anchors == [3, 3, 3]

    # check anchor generation
    anchors = anchor_generator.grid_anchors(featmap_sizes, device)
    assert len(anchors) == 3


def test_retina_anchor():
    from mmdet.models import build_head
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    # head configs modified from
    # configs/nas_fpn/retinanet_r50_fpn_crop640_50e.py
    bbox_head = dict(
        type='RetinaSepBNHead',
        num_classes=4,
        num_ins=5,
        in_channels=4,
        stacked_convs=1,
        feat_channels=4,
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

    retina_head = build_head(bbox_head)
    assert retina_head.anchor_generator is not None

    # use the featmap sizes in NASFPN setting to test retina head
    featmap_sizes = [(80, 80), (40, 40), (20, 20), (10, 10), (5, 5)]
    # check base anchors
    expected_base_anchors = [
        torch.Tensor([[-22.6274, -11.3137, 22.6274, 11.3137],
                      [-28.5088, -14.2544, 28.5088, 14.2544],
                      [-35.9188, -17.9594, 35.9188, 17.9594],
                      [-16.0000, -16.0000, 16.0000, 16.0000],
                      [-20.1587, -20.1587, 20.1587, 20.1587],
                      [-25.3984, -25.3984, 25.3984, 25.3984],
                      [-11.3137, -22.6274, 11.3137, 22.6274],
                      [-14.2544, -28.5088, 14.2544, 28.5088],
                      [-17.9594, -35.9188, 17.9594, 35.9188]]),
        torch.Tensor([[-45.2548, -22.6274, 45.2548, 22.6274],
                      [-57.0175, -28.5088, 57.0175, 28.5088],
                      [-71.8376, -35.9188, 71.8376, 35.9188],
                      [-32.0000, -32.0000, 32.0000, 32.0000],
                      [-40.3175, -40.3175, 40.3175, 40.3175],
                      [-50.7968, -50.7968, 50.7968, 50.7968],
                      [-22.6274, -45.2548, 22.6274, 45.2548],
                      [-28.5088, -57.0175, 28.5088, 57.0175],
                      [-35.9188, -71.8376, 35.9188, 71.8376]]),
        torch.Tensor([[-90.5097, -45.2548, 90.5097, 45.2548],
                      [-114.0350, -57.0175, 114.0350, 57.0175],
                      [-143.6751, -71.8376, 143.6751, 71.8376],
                      [-64.0000, -64.0000, 64.0000, 64.0000],
                      [-80.6349, -80.6349, 80.6349, 80.6349],
                      [-101.5937, -101.5937, 101.5937, 101.5937],
                      [-45.2548, -90.5097, 45.2548, 90.5097],
                      [-57.0175, -114.0350, 57.0175, 114.0350],
                      [-71.8376, -143.6751, 71.8376, 143.6751]]),
        torch.Tensor([[-181.0193, -90.5097, 181.0193, 90.5097],
                      [-228.0701, -114.0350, 228.0701, 114.0350],
                      [-287.3503, -143.6751, 287.3503, 143.6751],
                      [-128.0000, -128.0000, 128.0000, 128.0000],
                      [-161.2699, -161.2699, 161.2699, 161.2699],
                      [-203.1873, -203.1873, 203.1873, 203.1873],
                      [-90.5097, -181.0193, 90.5097, 181.0193],
                      [-114.0350, -228.0701, 114.0350, 228.0701],
                      [-143.6751, -287.3503, 143.6751, 287.3503]]),
        torch.Tensor([[-362.0387, -181.0193, 362.0387, 181.0193],
                      [-456.1401, -228.0701, 456.1401, 228.0701],
                      [-574.7006, -287.3503, 574.7006, 287.3503],
                      [-256.0000, -256.0000, 256.0000, 256.0000],
                      [-322.5398, -322.5398, 322.5398, 322.5398],
                      [-406.3747, -406.3747, 406.3747, 406.3747],
                      [-181.0193, -362.0387, 181.0193, 362.0387],
                      [-228.0701, -456.1401, 228.0701, 456.1401],
                      [-287.3503, -574.7006, 287.3503, 574.7006]])
    ]
    base_anchors = retina_head.anchor_generator.base_anchors
    for i, base_anchor in enumerate(base_anchors):
        assert base_anchor.allclose(expected_base_anchors[i])

    # check valid flags
    expected_valid_pixels = [57600, 14400, 3600, 900, 225]
    multi_level_valid_flags = retina_head.anchor_generator.valid_flags(
        featmap_sizes, (640, 640), device)
    for i, single_level_valid_flag in enumerate(multi_level_valid_flags):
        assert single_level_valid_flag.sum() == expected_valid_pixels[i]

    # check number of base anchors for each level
    assert retina_head.anchor_generator.num_base_anchors == [9, 9, 9, 9, 9]

    # check anchor generation
    anchors = retina_head.anchor_generator.grid_anchors(featmap_sizes, device)
    assert len(anchors) == 5


def test_guided_anchor():
    from mmdet.models import build_head
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    # head configs modified from
    # configs/guided_anchoring/ga_retinanet_r50_fpn_1x_coco.py
    bbox_head = dict(
        type='GARetinaHead',
        num_classes=8,
        in_channels=4,
        stacked_convs=1,
        feat_channels=4,
        approx_anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        square_anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            scales=[4],
            strides=[8, 16, 32, 64, 128]))

    ga_retina_head = build_head(bbox_head)
    assert ga_retina_head.approx_anchor_generator is not None

    # use the featmap sizes in NASFPN setting to test ga_retina_head
    featmap_sizes = [(100, 152), (50, 76), (25, 38), (13, 19), (7, 10)]
    # check base anchors
    expected_approxs = [
        torch.Tensor([[-22.6274, -11.3137, 22.6274, 11.3137],
                      [-28.5088, -14.2544, 28.5088, 14.2544],
                      [-35.9188, -17.9594, 35.9188, 17.9594],
                      [-16.0000, -16.0000, 16.0000, 16.0000],
                      [-20.1587, -20.1587, 20.1587, 20.1587],
                      [-25.3984, -25.3984, 25.3984, 25.3984],
                      [-11.3137, -22.6274, 11.3137, 22.6274],
                      [-14.2544, -28.5088, 14.2544, 28.5088],
                      [-17.9594, -35.9188, 17.9594, 35.9188]]),
        torch.Tensor([[-45.2548, -22.6274, 45.2548, 22.6274],
                      [-57.0175, -28.5088, 57.0175, 28.5088],
                      [-71.8376, -35.9188, 71.8376, 35.9188],
                      [-32.0000, -32.0000, 32.0000, 32.0000],
                      [-40.3175, -40.3175, 40.3175, 40.3175],
                      [-50.7968, -50.7968, 50.7968, 50.7968],
                      [-22.6274, -45.2548, 22.6274, 45.2548],
                      [-28.5088, -57.0175, 28.5088, 57.0175],
                      [-35.9188, -71.8376, 35.9188, 71.8376]]),
        torch.Tensor([[-90.5097, -45.2548, 90.5097, 45.2548],
                      [-114.0350, -57.0175, 114.0350, 57.0175],
                      [-143.6751, -71.8376, 143.6751, 71.8376],
                      [-64.0000, -64.0000, 64.0000, 64.0000],
                      [-80.6349, -80.6349, 80.6349, 80.6349],
                      [-101.5937, -101.5937, 101.5937, 101.5937],
                      [-45.2548, -90.5097, 45.2548, 90.5097],
                      [-57.0175, -114.0350, 57.0175, 114.0350],
                      [-71.8376, -143.6751, 71.8376, 143.6751]]),
        torch.Tensor([[-181.0193, -90.5097, 181.0193, 90.5097],
                      [-228.0701, -114.0350, 228.0701, 114.0350],
                      [-287.3503, -143.6751, 287.3503, 143.6751],
                      [-128.0000, -128.0000, 128.0000, 128.0000],
                      [-161.2699, -161.2699, 161.2699, 161.2699],
                      [-203.1873, -203.1873, 203.1873, 203.1873],
                      [-90.5097, -181.0193, 90.5097, 181.0193],
                      [-114.0350, -228.0701, 114.0350, 228.0701],
                      [-143.6751, -287.3503, 143.6751, 287.3503]]),
        torch.Tensor([[-362.0387, -181.0193, 362.0387, 181.0193],
                      [-456.1401, -228.0701, 456.1401, 228.0701],
                      [-574.7006, -287.3503, 574.7006, 287.3503],
                      [-256.0000, -256.0000, 256.0000, 256.0000],
                      [-322.5398, -322.5398, 322.5398, 322.5398],
                      [-406.3747, -406.3747, 406.3747, 406.3747],
                      [-181.0193, -362.0387, 181.0193, 362.0387],
                      [-228.0701, -456.1401, 228.0701, 456.1401],
                      [-287.3503, -574.7006, 287.3503, 574.7006]])
    ]
    approxs = ga_retina_head.approx_anchor_generator.base_anchors
    for i, base_anchor in enumerate(approxs):
        assert base_anchor.allclose(expected_approxs[i])

    # check valid flags
    expected_valid_pixels = [136800, 34200, 8550, 2223, 630]
    multi_level_valid_flags = ga_retina_head.approx_anchor_generator \
        .valid_flags(featmap_sizes, (800, 1216), device)
    for i, single_level_valid_flag in enumerate(multi_level_valid_flags):
        assert single_level_valid_flag.sum() == expected_valid_pixels[i]

    # check number of base anchors for each level
    assert ga_retina_head.approx_anchor_generator.num_base_anchors == [
        9, 9, 9, 9, 9
    ]

    # check approx generation
    squares = ga_retina_head.square_anchor_generator.grid_anchors(
        featmap_sizes, device)
    assert len(squares) == 5

    expected_squares = [
        torch.Tensor([[-16., -16., 16., 16.]]),
        torch.Tensor([[-32., -32., 32., 32]]),
        torch.Tensor([[-64., -64., 64., 64.]]),
        torch.Tensor([[-128., -128., 128., 128.]]),
        torch.Tensor([[-256., -256., 256., 256.]])
    ]
    squares = ga_retina_head.square_anchor_generator.base_anchors
    for i, base_anchor in enumerate(squares):
        assert base_anchor.allclose(expected_squares[i])

    # square_anchor_generator does not check valid flags
    # check number of base anchors for each level
    assert (ga_retina_head.square_anchor_generator.num_base_anchors == [
        1, 1, 1, 1, 1
    ])

    # check square generation
    anchors = ga_retina_head.square_anchor_generator.grid_anchors(
        featmap_sizes, device)
    assert len(anchors) == 5

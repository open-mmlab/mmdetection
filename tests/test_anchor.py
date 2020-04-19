"""
CommandLine:
    pytest tests/test_anchor.py
    xdoctest tests/test_anchor.py zero

"""
import torch


def test_standard_anchor_generator():
    from mmdet.core.anchor import build_anchor_generator
    anchor_generator_cfg = dict(
        type='AnchorGenerator',
        scales=[8],
        ratios=[0.5, 1.0, 2.0],
        strides=[4, 8])

    anchor_generator = build_anchor_generator(anchor_generator_cfg)
    assert anchor_generator is not None
    print(anchor_generator)


def test_ssd_anchor_generator():
    from mmdet.core.anchor import build_anchor_generator
    anchor_generator_cfg = dict(
        type='SSDAnchorGenerator',
        num_levels=6,
        scale_major=False,
        input_size=300,
        basesize_ratio_range=(0.15, 0.9),
        strides=(8, 16, 32, 64, 100, 300),
        ratios=([2], [2, 3], [2, 3], [2, 3], [2], [2]))

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
    expect_valid_flags = [5776, 2166, 600, 150, 36, 4]
    multi_level_valid_flags = anchor_generator.valid_flags(
        featmap_sizes, (300, 300), 'cpu')
    for i, single_level_valid_flag in enumerate(multi_level_valid_flags):
        assert single_level_valid_flag.sum() == expect_valid_flags[i]

    # check number of base anchors for each level
    assert anchor_generator.num_base_anchors == [4, 6, 6, 6, 4, 4]
    print(anchor_generator)

    # check anchor generation
    anchors = anchor_generator.grid_anchors(featmap_sizes, 'cpu')
    assert len(anchors) == 6

_base_ = [
    '_base_/xdecoder-tiny_ref-seg.py', 'mmdet::_base_/datasets/refcocog.py'
]

test_dataloader = dict(dataset=dict(split='val'))

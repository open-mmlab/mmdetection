_base_ = [
    '_base_/xdecoder-tiny_ref-semseg.py', 'mmdet::_base_/datasets/refcocog.py'
]

test_dataloader = dict(dataset=dict(split='val'))
test_evaluator = dict(eval_first_text=True)

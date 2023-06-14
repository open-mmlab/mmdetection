_base_ = [
    './deepsort_faster-rcnn_r50_fpn_8xb2-4e_mot17halftrain'
    '_test-mot17halfval.py'
]

# dataloader
val_dataloader = dict(
    dataset=dict(ann_file='annotations/train_cocoformat.json'))
test_dataloader = dict(
    dataset=dict(
        ann_file='annotations/test_cocoformat.json',
        data_prefix=dict(img_path='test')))

# evaluator
test_evaluator = dict(format_only=True, outfile_prefix='./mot_17_test_res')

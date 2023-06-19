_base_ = [
    './bytetrack/bytetrack_yolox_x_8xb4-amp-80e_crowdhuman-'
    'mot17halftrain_test-mot17halfval.py'
]

test_dataloader = dict(
    dataset=dict(
        data_root='data/MOT17/',
        ann_file='annotations/test_cocoformat.json',
        data_prefix=dict(img_path='test')))
test_evaluator = dict(
    type='MOTChallengeMetrics',
    postprocess_tracklet_cfg=[
        dict(type='InterpolateTracklets', min_num_frames=5, max_num_frames=20)
    ],
    format_only=True,
    outfile_prefix='./mot_17_test_res')

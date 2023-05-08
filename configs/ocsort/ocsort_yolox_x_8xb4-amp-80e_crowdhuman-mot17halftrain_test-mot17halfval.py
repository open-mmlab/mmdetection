_base_ = [
    '../bytetrack/bytetrack_yolox_x_8xb4-amp-80e_crowdhuman-mot17halftrain_test-mot17halfval.py',  # noqa: E501
]

model = dict(
    type='OCSORT',
    tracker=dict(
        _delete_=True,
        type='OCSORTTracker',
        motion=dict(type='KalmanFilter'),
        obj_score_thr=0.3,
        init_track_thr=0.7,
        weight_iou_with_det_scores=True,
        match_iou_thr=0.3,
        num_tentatives=3,
        vel_consist_weight=0.2,
        vel_delta_t=3,
        num_frames_retain=30))

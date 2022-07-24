_base_ = ['semi_base_faster_rcnn_r50_fpn_coco_90k.py']

model_wrapper = dict(
    type='SoftTeacher',
    semi_train_cfg=dict(
        pseudo_label_initial_score_thr=0.5,
        rpn_pseudo_thr=0.9,
        reg_pseudo_thr=0.02,
        jitter_times=10,
        jitter_scale=0.06))

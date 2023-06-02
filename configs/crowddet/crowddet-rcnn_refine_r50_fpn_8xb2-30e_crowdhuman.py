_base_ = './crowddet-rcnn_r50_fpn_8xb2-30e_crowdhuman.py'

model = dict(roi_head=dict(bbox_head=dict(with_refine=True)))

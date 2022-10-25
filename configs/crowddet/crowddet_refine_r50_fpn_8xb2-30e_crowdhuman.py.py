_base_ = ['./crowddet_r50_fpn_8xb2-30e_crowdhuman.py']

model = dict(roi_head=dict(bbox_head=dict(refine_flag=True)))

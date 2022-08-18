_base_ = 'deformable-detr-refine_r50_16x2-50e_coco.py'
model = dict(bbox_head=dict(as_two_stage=True))

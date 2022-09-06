_base_ = 'deformable-detr_refine_r50_16xb2-50e_coco.py'
model = dict(bbox_head=dict(as_two_stage=True))

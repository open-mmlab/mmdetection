_base_ = 'deformable-detr_refine_r50_16xb2-50e_coco.py'
model = dict(as_two_stage=True, bbox_head=dict(num_pred=7, as_two_stage=True))

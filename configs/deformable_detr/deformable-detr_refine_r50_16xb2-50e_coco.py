_base_ = 'deformable-detr_r50_16xb2-50e_coco.py'
model = dict(bbox_head=dict(with_box_refine=True))

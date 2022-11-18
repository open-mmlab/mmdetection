_base_ = 'deformable-detr_r50_16xb2-50e_coco-91cls.py'
model = dict(with_box_refine=True)

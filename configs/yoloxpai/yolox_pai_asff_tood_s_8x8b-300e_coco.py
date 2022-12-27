_base_ = './yolox_pai_asff_s_8x8b-300e_coco.py'

model = dict(bbox_head=dict(type='YOLOXTOODHead'))

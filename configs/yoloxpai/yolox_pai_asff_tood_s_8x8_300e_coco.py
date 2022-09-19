_base_ = './yolox_pai_asff_s_8x8_300e_coco.py'

model = dict(bbox_head=dict(type='YOLOXTOODHead'))

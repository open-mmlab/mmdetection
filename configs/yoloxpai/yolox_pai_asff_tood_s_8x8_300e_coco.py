_base_ = './yolox_pai_s_8x8_300e_coco.py'

model = dict(bbox_head=dict(type='YOLOXTOODHead'))
find_unused_parameters = True

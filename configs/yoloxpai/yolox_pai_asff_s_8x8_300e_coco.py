_base_ = './yolox_pai_s_8x8_300e_coco.py'

model = dict(neck=dict(type='YOLOXASFFPAFPN'))

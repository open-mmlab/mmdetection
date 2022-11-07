_base_ = './yolox_pai_s_8x8b-300e_coco.py'

model = dict(neck=dict(type='YOLOXASFFPAFPN'))

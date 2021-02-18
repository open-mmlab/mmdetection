_base_ = 'tridentnet_r50_caffe_mstrain_1x_coco.py'

lr_config = dict(step=[28, 34])
runner = dict(max_epochs=36)

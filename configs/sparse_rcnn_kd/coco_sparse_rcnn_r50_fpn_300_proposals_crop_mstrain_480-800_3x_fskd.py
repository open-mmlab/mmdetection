_base_ = '../sparse_rcnn/coco_sparse_rcnn_r50_fpn_mstrain_480-800_3x.py'

model = dict(type='SparseRCNN_TS',
             roi_head=dict(
                type='ContSparseRoIHead',
             ))


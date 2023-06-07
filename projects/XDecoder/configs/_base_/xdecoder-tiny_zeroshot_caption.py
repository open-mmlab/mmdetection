_base_ = 'xdecoder-tiny_zeroshot_open-vocab-semseg.py'

model = dict(head=dict(task='caption'))

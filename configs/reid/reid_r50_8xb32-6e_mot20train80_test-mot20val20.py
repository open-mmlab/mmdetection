_base_ = ['./reid_r50_8xb32-6e_mot17train80_test-mot17val20.py']
model = dict(head=dict(num_classes=1701))
# data
data_root = 'data/MOT20/'
train_dataloader = dict(dataset=dict(data_root=data_root))
val_dataloader = dict(dataset=dict(data_root=data_root))
test_dataloader = val_dataloader

# train, val, test setting
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=6, val_interval=7)

_base_ = ['./reid_r50_8xb32-6e_mot17train80_test-mot17val20.py']
model = dict(head=dict(num_classes=371))
# data
data_root = 'data/MOT16/'
train_dataloader = dict(dataset=dict(data_root=data_root))
val_dataloader = dict(dataset=dict(data_root=data_root))
test_dataloader = val_dataloader

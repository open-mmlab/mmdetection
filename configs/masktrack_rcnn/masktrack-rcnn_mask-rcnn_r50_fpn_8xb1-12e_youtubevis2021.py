_base_ = ['./masktrack-rcnn_mask-rcnn_r50_fpn_8xb1-12e_youtubevis2019.py']

data_root = 'data/youtube_vis_2021/'
dataset_version = data_root[-5:-1]

# dataloader
train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        dataset_version=dataset_version,
        ann_file='annotations/youtube_vis_2021_train.json'))
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        dataset_version=dataset_version,
        ann_file='annotations/youtube_vis_2021_valid.json'))
test_dataloader = val_dataloader

from mmdet.datasets import OpenImagesDataset

if __name__ == '__main__':
    ann_file = '/home/PJLAB/wangyudong/DATA/DATA/OpenImage/'
    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True)
    train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='Resize', img_scale=(1000, 600), keep_ratio=True),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='Pad', size_divisor=32),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
    ]
    dataset = OpenImagesDataset(
        ann_file=ann_file + 'validation-annotations-bbox.csv',
        pipeline=train_pipeline,
        img_prefix=ann_file,
        label_csv_path=ann_file + 'class-descriptions-boxable.csv')

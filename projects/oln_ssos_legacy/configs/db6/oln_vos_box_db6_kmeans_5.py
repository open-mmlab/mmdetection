_base_ = ['../oln_box/oln_box_model.py',
          '../_base_/datasets/db6_split_detection.py',
          '/media/capture/hdd8tb/MMDet3.x/mmdetection/configs/_base_/schedules/schedule_1x.py',
          '//media/capture/hdd8tb/MMDet3.x/mmdetection/configs/_base_/default_runtime.py'
          ]

custom_imports = dict(
    imports=[
        'projects.oln_ssos_legacy.vos.models.roi_heads.oln_mask_vos_roi_head',
        'projects.oln_ssos_legacy.vos.models.detectors.epoch_faster_rcnn',
        'projects.oln_ssos_legacy.vos.models.roi_heads.bbox_heads.oln_vos_bbox_head',
        'projects.oln_ssos_legacy.vos.datasets.vos_coco',
        'projects.oln_ssos_legacy.epochs.pseudo_label_runner',
        'projects.oln_ssos_legacy.vos.datasets.pipelines.loading',
        'projects.oln_ssos_legacy.vos.datasets.pipelines.formatting'
    ],
    allow_failed_imports=False)

model = dict(
    type='EpochFasterRCNN',
    calculate_pseudo_labels_from_epoch=0,
    roi_head=dict(
        type='OLNKMeansVOSRoIHead',
        start_epoch=2,
        logistic_regression_hidden_dim=512,
        negative_sampling_size=10000,
        bottomk_epsilon_dist=1,
        ood_loss_weight=0.1,
        pseudo_label_loss_weight=1.,
        k=5,
        repeat_ood_sampling=4,
        use_all_proposals_ood=False,
        bbox_head=dict(
            type='VOSShared2FCBBoxScoreHead',
        reg_class_agnostic=True))
    )

checkpoint_config = dict(interval=1)
dataset_type = "VOSDB6SplitDataset"

optimizer = dict(type='SGD', lr=0.004, momentum=0.9, weight_decay=0.0001)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=0.001,
    step=[5])
total_epochs = 7

load_from = './work_dirs/oln_box/epoch_8.pth'
custom_hooks = [dict(type='SetEpochInfoHook',
                priority='NORMAL')]


train_cfg = dict(
    type='PseudoLabelEpochBasedTrainLoop',
    max_epochs=12,
    val_interval=1
)

_base_ = [
    '../../../configs/_base_/datasets/coco_detection.py',
    '../../../configs/_base_/schedules/schedule_1x.py', '../../../configs/_base_/default_runtime.py'
]
model = dict(
    type='COXNet',
    backbone=dict(
        type='ResNetDual',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=4),
    neck_t=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=4),
    bbox_head=dict(
        type='GFLHead',
        num_classes=3,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[4, 8, 16, 32]),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25),
        reg_max=16,
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0)),
    reduction=16,
    num_layers=4,
    fs_type='dasr',
    use_wf=[True, True, True, True],
    wf_initialize=[True, True, True, True],
    wf_use_sa=[False, False, False, False],
    wf_use_ca=[False, False, False, False],
    # training and testing settings
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.3),
        max_per_img=100))

# dataset settings
dataset_type = 'DronePerson'
data_root = '/data3/pengpeiran/datasets/RGBTDronePerson/'
img_norm_cfg = dict(
    mean_list=([115.37, 121.82, 122.63], [93.10, 93.10, 93.10]),
    std_list=([85.13, 89.01, 88.27], [50.24, 50.24, 50.24]), to_rgb=True)
train_pipeline = [
    dict(type='LoadImagePairFromFile', spectrals=('visible', 'thermal')),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(640, 512), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='MultiNormalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImagePairFromFile', spectrals=('visible', 'thermal')),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='MultiNormalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train_thermal.json',
        img_prefix=data_root + 'train/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'val_thermal.json',
        img_prefix=data_root + 'val/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'val_thermal.json',
        img_prefix=data_root + 'val/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(_delete_=True, max_norm=35, norm_type=2))

work_dir = 'work_dir/rgbtdroneperson/ablation_new/baseline_star+dasr_wf1111_init'

# dataset settings
dataset_type = 'VTUAVdet'
data_root = '/mnt/nfs_200T/optics/data/dataset_ppr/Public/VTUAV-yolo/'
img_norm_cfg = dict(
    mean_list=([83.20, 92.24, 97.70], [134.84, 134.84, 134.84]),
    std_list=([57.77, 57.41, 57.69], [81.58, 81.58, 81.58]), to_rgb=True)
train_pipeline = [
    dict(type='LoadYOLOImagePairFromFile', spectrals=('visible', 'infrared')),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(640, 512), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='MultiNormalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadYOLOImagePairFromFile', spectrals=('visible', 'infrared')),
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
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train_ir.json',
        img_prefix=data_root + 'train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'val_ir.json',
        img_prefix=data_root + 'val',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'val_ir.json',
        img_prefix=data_root + 'val',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
_base_ = './gfl_rgbt_6ch_avg_r50_fpn_1x.py'
model = dict(
    backbone=dict(
        type='ResNet',
        depth=101,
        in_channels=6,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained',
                      checkpoint='checkpoints/resnet101_rgbt_6ch_avg.pth')))

work_dir = 'work_dir/rgbtdroneperson/gfl/gfl_rgbt_6ch_avg_r101_fpn_1x'
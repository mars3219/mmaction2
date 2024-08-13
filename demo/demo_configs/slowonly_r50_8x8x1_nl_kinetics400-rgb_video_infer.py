# Copyright (c) OpenMMLab. All rights reserved.
_base_ = ['../../configs/_base_/models/slowonly_r50.py']

# model settings
model = dict(
    backbone=dict(
        pretrained='https://download.pytorch.org/models/resnet50-11ad3fa6.pth',
        non_local=((0, 0, 0), (1, 0, 1, 0), (1, 0, 1, 0, 1, 0), (0, 0, 0)),
        non_local_cfg=dict(
            sub_sample=True,
            use_scale=True,
            norm_cfg=dict(type='BN3d', requires_grad=True),
            mode='embedded_gaussian')))

# dataset settings
dataset_type = 'VideoDataset'
test_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=8,
        num_clips=10,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='ThreeCrop', crop_size=256),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        ann_file=None,
        data_prefix=None,
        pipeline=test_pipeline))

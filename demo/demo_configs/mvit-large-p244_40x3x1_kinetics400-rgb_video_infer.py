# Copyright (c) OpenMMLab. All rights reserved.
_base_ = ['../../configs/_base_/models/mvit_small.py']

model = dict(
    backbone=dict(
        arch='large',
        temporal_size=40,
        spatial_size=312,
        drop_path_rate=0.75,
    ),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[114.75, 114.75, 114.75],
        std=[57.375, 57.375, 57.375],
        format_shape='NCTHW'),
    cls_head=dict(in_channels=1152),
    test_cfg=dict(max_testing_views=5))

# dataset settings
dataset_type = 'VideoDataset'
test_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=40,
        frame_interval=3,
        num_clips=5,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 312)),
    dict(type='ThreeCrop', crop_size=312),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        ann_file=None,
        data_prefix=None,
        pipeline=test_pipeline))

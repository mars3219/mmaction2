# Copyright (c) OpenMMLab. All rights reserved.
_base_ = ['../../configs/_base_/models/ircsn_r152.py']

# model settings
model = dict(
    backbone=dict(bottleneck_mode='ip', pretrained=None),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[110.2008, 100.63983, 95.99475],
        std=[58.14765, 56.46975, 55.332195],
        format_shape='NCTHW'))

# dataset settings
dataset_type = 'VideoDataset'
test_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=2,
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
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        ann_file=None,
        data_prefix=None,
        pipeline=test_pipeline))

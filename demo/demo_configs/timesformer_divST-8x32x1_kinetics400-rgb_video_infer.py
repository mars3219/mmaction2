# Copyright (c) OpenMMLab. All rights reserved.
_base_ = ['../../configs/recognition/timesformer/timesformer_spaceOnly_8xb8-8x32x1-15e_kinetics400-rgb.py']

model = dict(backbone=dict(attention_type='divided_space_time'))

# dataset settings
dataset_type = 'VideoDataset'
test_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=32,
        num_clips=1,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='ThreeCrop', crop_size=224),
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

# Copyright (c) OpenMMLab. All rights reserved.
_base_ = ['../../configs/skeleton/posec3d/slowonly_r50_8xb32-u48-240e_k400-keypoint.py']

dataset_type = 'PoseDataset'
left_kp = [1, 3, 5, 7, 9, 11, 13, 15]
right_kp = [2, 4, 6, 8, 10, 12, 14, 16]

# dataset settings
test_pipeline = [
    dict(type='DecompressPose', squeeze=True),
    dict(
        type='UniformSampleFrames', clip_len=48, num_clips=10, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(64, 64), keep_ratio=False),
    dict(
        type='GeneratePoseTarget',
        with_kp=True,
        with_limb=False,
        double=True,
        left_kp=left_kp,
        right_kp=right_kp),
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),
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

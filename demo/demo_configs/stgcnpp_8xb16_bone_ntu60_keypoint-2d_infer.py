# Copyright (c) OpenMMLab. All rights reserved.
_base_ = ['../../configs/skeleton/stgcnpp/stgcnpp_8xb16-bone-u100-80e_ntu60-xsub-keypoint-2d.py']

# dataset settings
dataset_type = 'PoseDataset'
test_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['b']),
    dict(
        type='UniformSampleFrames', clip_len=100, num_clips=10,
        test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
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

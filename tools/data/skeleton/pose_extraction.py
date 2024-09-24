# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm

import os
import cv2
import mmcv
import logging
import argparse
import os.path as osp
from collections import defaultdict
from tempfile import TemporaryDirectory

import mmengine
import numpy as np

from mmaction.apis import pose_inference
from mmengine.structures import InstanceData

from typing import List, Optional, Tuple, Union

# 기본 설정 및 로깅 설정
class Args:
    def __init__(self):
        self.det_config = '/workspace/demo/demo_configs/faster-rcnn_r50_fpn_2x_coco_infer.py'
        self.det_checkpoint = '/workspace/tools/data/skeleton/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth'
        self.det_score_thr = 0.5
        self.pose_config = '/workspace/demo/demo_configs/td-hm_hrnet-w32_8xb64-210e_coco-256x192_infer.py'
        self.pose_checkpoint = '/workspace/tools/data/skeleton/hrnet_w32_coco_256x192-c78dce93_20200708.pth'

args = Args()

def setup_logging(log_file):
    logging.basicConfig(filename=log_file,
                        level=logging.ERROR,
                        format='%(asctime)s - %(levelname)s - %(message)s')


def intersection(b0, b1):
    l, r = max(b0[0], b1[0]), min(b0[2], b1[2])
    u, d = max(b0[1], b1[1]), min(b0[3], b1[3])
    return max(0, r - l) * max(0, d - u)


def iou(b0, b1):
    i = intersection(b0, b1)
    u = area(b0) + area(b1) - i
    return i / u


def area(b):
    return (b[2] - b[0]) * (b[3] - b[1])


def removedup(bbox):

    def inside(box0, box1, threshold=0.8):
        return intersection(box0, box1) / area(box0) > threshold

    num_bboxes = bbox.shape[0]
    if num_bboxes == 1 or num_bboxes == 0:
        return bbox
    valid = []
    for i in range(num_bboxes):
        flag = True
        for j in range(num_bboxes):
            if i != j and inside(bbox[i],
                                 bbox[j]) and bbox[i][4] <= bbox[j][4]:
                flag = False
                break
        if flag:
            valid.append(i)
    return bbox[valid]


def is_easy_example(det_results, num_person):
    threshold = 0.95

    def thre_bbox(bboxes, threshold=threshold):
        shape = [sum(bbox[:, -1] > threshold) for bbox in bboxes]
        ret = np.all(np.array(shape) == shape[0])
        return shape[0] if ret else -1

    if thre_bbox(det_results) == num_person:
        det_results = [x[x[..., -1] > 0.95] for x in det_results]
        return True, np.stack(det_results)
    return False, thre_bbox(det_results)


def bbox2tracklet(bbox):
    iou_thre = 0.6
    tracklet_id = -1
    tracklet_st_frame = {}
    tracklets = defaultdict(list)
    for t, box in enumerate(bbox):
        for idx in range(box.shape[0]):
            matched = False
            for tlet_id in range(tracklet_id, -1, -1):
                cond1 = iou(tracklets[tlet_id][-1][-1], box[idx]) >= iou_thre
                cond2 = (
                    t - tracklet_st_frame[tlet_id] - len(tracklets[tlet_id]) <
                    10)
                cond3 = tracklets[tlet_id][-1][0] != t
                if cond1 and cond2 and cond3:
                    matched = True
                    tracklets[tlet_id].append((t, box[idx]))
                    break
            if not matched:
                tracklet_id += 1
                tracklet_st_frame[tracklet_id] = t
                tracklets[tracklet_id].append((t, box[idx]))
    return tracklets


def drop_tracklet(tracklet):
    tracklet = {k: v for k, v in tracklet.items() if len(v) > 5}

    def meanarea(track):
        boxes = np.stack([x[1] for x in track]).astype(np.float32)
        areas = (boxes[..., 2] - boxes[..., 0]) * (
            boxes[..., 3] - boxes[..., 1])
        return np.mean(areas)

    tracklet = {k: v for k, v in tracklet.items() if meanarea(v) > 5000}
    return tracklet


def distance_tracklet(tracklet):
    dists = {}
    for k, v in tracklet.items():
        bboxes = np.stack([x[1] for x in v])
        c_x = (bboxes[..., 2] + bboxes[..., 0]) / 2.
        c_y = (bboxes[..., 3] + bboxes[..., 1]) / 2.
        c_x -= 480
        c_y -= 270
        c = np.concatenate([c_x[..., None], c_y[..., None]], axis=1)
        dist = np.linalg.norm(c, axis=1)
        dists[k] = np.mean(dist)
    return dists


def tracklet2bbox(track, num_frame):
    # assign_prev
    bbox = np.zeros((num_frame, 5))
    trackd = {}
    for k, v in track:
        bbox[k] = v
        trackd[k] = v
    for i in range(num_frame):
        if bbox[i][-1] <= 0.5:
            mind = np.Inf
            for k in trackd:
                if np.abs(k - i) < mind:
                    mind = np.abs(k - i)
            bbox[i] = bbox[k]
    return bbox


def tracklets2bbox(tracklet, num_frame):
    dists = distance_tracklet(tracklet)
    sorted_inds = sorted(dists, key=lambda x: dists[x])
    dist_thre = np.Inf
    for i in sorted_inds:
        if len(tracklet[i]) >= num_frame / 2:
            dist_thre = 2 * dists[i]
            break

    dist_thre = max(50, dist_thre)

    bbox = np.zeros((num_frame, 5))
    bboxd = {}
    for idx in sorted_inds:
        if dists[idx] < dist_thre:
            for k, v in tracklet[idx]:
                if bbox[k][-1] < 0.01:
                    bbox[k] = v
                    bboxd[k] = v
    bad = 0
    for idx in range(num_frame):
        if bbox[idx][-1] < 0.01:
            bad += 1
            mind = np.Inf
            mink = None
            for k in bboxd:
                if np.abs(k - idx) < mind:
                    mind = np.abs(k - idx)
                    mink = k
            bbox[idx] = bboxd[mink]
    return bad, bbox[:, None, :]


def bboxes2bbox(bbox, num_frame):
    ret = np.zeros((num_frame, 2, 5))
    for t, item in enumerate(bbox):
        if item.shape[0] <= 2:
            ret[t, :item.shape[0]] = item
        else:
            inds = sorted(
                list(range(item.shape[0])), key=lambda x: -item[x, -1])
            ret[t] = item[inds[:2]]
    for t in range(num_frame):
        if ret[t, 0, -1] <= 0.01:
            ret[t] = ret[t - 1]
        elif ret[t, 1, -1] <= 0.01:
            if t:
                if ret[t - 1, 0, -1] > 0.01 and ret[t - 1, 1, -1] > 0.01:
                    if iou(ret[t, 0], ret[t - 1, 0]) > iou(
                            ret[t, 0], ret[t - 1, 1]):
                        ret[t, 1] = ret[t - 1, 1]
                    else:
                        ret[t, 1] = ret[t - 1, 0]
    return ret


def ntu_det_postproc(vid, det_results):
    det_results = [removedup(x) for x in det_results]
    label = int(vid.split('/')[-1].split('A')[1][:3])
    mpaction = list(range(50, 61)) + list(range(106, 121))
    n_person = 2 if label in mpaction else 1
    is_easy, bboxes = is_easy_example(det_results, n_person)
    if is_easy:
        print('\nEasy Example')
        return bboxes

    tracklets = bbox2tracklet(det_results)
    tracklets = drop_tracklet(tracklets)

    print(f'\nHard {n_person}-person Example, found {len(tracklets)} tracklet')
    if n_person == 1:
        if len(tracklets) == 1:
            tracklet = list(tracklets.values())[0]
            det_results = tracklet2bbox(tracklet, len(det_results))
            return np.stack(det_results)
        else:
            bad, det_results = tracklets2bbox(tracklets, len(det_results))
            return det_results
    # n_person is 2
    if len(tracklets) <= 2:
        tracklets = list(tracklets.values())
        bboxes = []
        for tracklet in tracklets:
            bboxes.append(tracklet2bbox(tracklet, len(det_results))[:, None])
        bbox = np.concatenate(bboxes, axis=1)
        return bbox
    else:
        return bboxes2bbox(det_results, len(det_results))


def pose_inference_with_align(model, args, frame, det_result):
    # filter frame without det bbox
    
    if det_result.shape[0] > 0:
        pose = pose_inference(model, frame, det_result)
    else:
        pose = None 
    return pose


def frame_extract(video_path: str,
                  short_side: Optional[int] = None,):

    vid = cv2.VideoCapture(video_path)
    resize = False

    flag, frame = vid.read()
    new_h, new_w = None, None
    if short_side is not None:
        if new_h is None:
            h, w, _ = frame.shape
            new_w, new_h = mmcv.rescale_size((w, h), (short_side, np.Inf))
            shape = (new_h, new_w)
            resize = True
    else:
        shape = frame.shape
        
    while flag:
        if resize:
            frame = mmcv.imresize(frame, (new_w, new_h))

        yield frame
        flag, frame = vid.read()

def detection_inference(model, frame,
                        det_score_thr: float = 0.9,
                        det_cat_id: int = 0,
                        with_score: bool = False) -> tuple:
    try:
        from mmdet.apis import inference_detector
        from mmdet.structures import DetDataSample
    except (ImportError, ModuleNotFoundError):
        raise ImportError('Failed to import `inference_detector` and '
                          '`init_detector` from `mmdet.apis`. These apis are '
                          'required in this inference api! ')

    # print('Performing Human Detection for each frame')
    # for frame_path in track_iter_progress(frame_paths):
    # for frame_path in frame_paths:
    det_data_sample: DetDataSample = inference_detector(model, frame)
    pred_instance = det_data_sample.pred_instances.cpu().numpy()
    bboxes = pred_instance.bboxes
    scores = pred_instance.scores
    # We only keep human detection bboxs with score larger
    # than `det_score_thr` and category id equal to `det_cat_id`.
    valid_idx = np.logical_and(pred_instance.labels == det_cat_id,
                                pred_instance.scores > det_score_thr)
    bboxes = bboxes[valid_idx]
    scores = scores[valid_idx]

    if with_score:
        bboxes = np.concatenate((bboxes, scores[:, None]), axis=-1)
    return bboxes

def pose_inference(model, frame,
                   det_result) -> tuple:

    try:
        from mmpose.apis import inference_topdown
        from mmpose.structures import PoseDataSample, merge_data_samples
    except (ImportError, ModuleNotFoundError):
        raise ImportError('Failed to import `inference_topdown` and '
                          '`init_model` from `mmpose.apis`. These apis '
                          'are required in this inference api! ')

    # print('Performing Human Pose Estimation for each frame')
    # for f, d in track_iter_progress(list(zip(frame_paths, det_results))):

    pose_data_samples: List[PoseDataSample] \
        = inference_topdown(model, frame, det_result[..., :4], bbox_format='xyxy')
    pose_data_sample = merge_data_samples(pose_data_samples)
    pose_data_sample.dataset_meta = model.dataset_meta
    # make fake pred_instances
    if not hasattr(pose_data_sample, 'pred_instances'):
        num_keypoints = model.dataset_meta['num_keypoints']
        pred_instances_data = dict(
            keypoints=np.empty(shape=(0, num_keypoints, 2)),
            keypoints_scores=np.empty(shape=(0, 17), dtype=np.float32),
            bboxes=np.empty(shape=(0, 4), dtype=np.float32),
            bbox_scores=np.empty(shape=(0), dtype=np.float32))
        pose_data_sample.pred_instances = InstanceData(
            **pred_instances_data)

    pose = pose_data_sample.pred_instances.to_dict()

    return pose

def keypoint_scores(pose_results):
    # align the num_person among frames
    num_frames = len(pose_results)
    num_persons = max([pose['keypoints'].shape[0] for pose in pose_results])
    num_points = pose_results[0]['keypoints'].shape[1]
    num_frames = len(pose_results)
    keypoints = np.zeros((num_persons, num_frames, num_points, 2),
                         dtype=np.float32)
    scores = np.zeros((num_persons, num_frames, num_points), dtype=np.float32)

    for f_idx, frm_pose in enumerate(pose_results):
        frm_num_persons = frm_pose['keypoints'].shape[0]
        for p_idx in range(frm_num_persons):
            keypoints[p_idx, f_idx] = frm_pose['keypoints'][p_idx]
            scores[p_idx, f_idx] = frm_pose['keypoint_scores'][p_idx]

    return keypoints, scores


def init_detect_model(det_config: Union[str, Path, mmengine.Config, nn.Module],
            det_checkpoint: str, device: Union[str, torch.device] = 'cuda:0'):
    try:
        from mmdet.apis import init_detector
    except (ImportError, ModuleNotFoundError):
        raise ImportError('Failed to import `inference_detector` and '
                          '`init_detector` from `mmdet.apis`. These apis are '
                          'required in this inference api! ')
    if isinstance(det_config, nn.Module):
        model = det_config
    else:
        model = init_detector(
            config=det_config, checkpoint=det_checkpoint, device=device)
    return model

def init_pose_model(pose_config: Union[str, Path, mmengine.Config, nn.Module],
                   pose_checkpoint: str,
                   device: Union[str, torch.device] = 'cuda:0') -> tuple:

    try:
        from mmpose.apis import init_model
    except (ImportError, ModuleNotFoundError):
        raise ImportError('Failed to import `inference_topdown` and '
                          '`init_model` from `mmpose.apis`. These apis '
                          'are required in this inference api! ')
    if isinstance(pose_config, nn.Module):
        model = pose_config
    else:
        model = init_model(pose_config, pose_checkpoint, device)
    return model


def pose_extraction(dmodel, pmodel, vid, label, skip_postproc=False):
    frame_gen = frame_extract(vid, 720)
    # frame_gen = tqdm(frame_gen, desc="Processing Frames", unit="frame")
    first = True
    pose_results = []
    for frame in frame_gen:
        if first:
            img_shape = frame[:2]
            first = False
        result = detection_inference(
            dmodel,
            frame,
            args.det_score_thr,
            with_score=True)

        # if not skip_postproc:
        #     det_results = ntu_det_postproc(vid, det_results)

        pose = pose_inference_with_align(pmodel, args, frame, result)
        
        if pose is not None:
            pose_results.append(pose)

    keypoints, scores = keypoint_scores(pose_results)
    

    anno = dict()
    anno['keypoint'] = keypoints
    anno['keypoint_score'] = scores
    anno['frame_dir'] = osp.splitext(osp.basename(vid))[0]
    anno['img_shape'] = img_shape
    anno['original_shape'] = img_shape 
    anno['total_frames'] = keypoints.shape[1]
    anno['label'] = label

    return anno



def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate Pose Annotation for a single NTURGB-D video')
    parser.add_argument('--video', default="/data/test/violence/fight/Fighting002_x264.mp4", type=str, help='source video')
    parser.add_argument('--txt-file', default='/data/aihub/violence/output/custom_train1.txt', type=str, help='path to txt file containing video paths')
    parser.add_argument('--output', default="/data/aihub/violence/train_pkl", type=str, help='output pickle name')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--skip-postproc', action='store_false')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    global_args = parse_args()
    args.device = global_args.device
    args.video = global_args.video
    args.output = global_args.output
    args.skip_postproc = global_args.skip_postproc

    dmodel = init_detect_model( args.det_config,
                        args.det_checkpoint,
                        device=args.device)
    
    pmodel = init_pose_model( args.pose_config,
                        args.pose_checkpoint,
                        device=args.device)
    
    # 로그 파일 설정
    log_file = osp.join(osp.dirname(global_args.output), 'error_log.txt')
    setup_logging(log_file)

    # 텍스트 파일에서 비디오 경로 읽기
    with open(global_args.txt_file, 'r') as f:
        lines = f.readlines()

    tasks = []

    # 각 비디오 경로에 대해 포즈 추출 및 pkl 파일 저장
    for line in lines:
        line_parts = line.split()
        video_path = line_parts[0]  # 비디오 경로 추출
        label = int(line_parts[1])  # 레이블
        # print(f'Processing video: {video_path} with label: {label}')

        # 비디오 파일이 존재하는지 확인
        if not osp.exists(video_path):
            print(f"Video file {video_path} does not exist. Skipping.")
            continue

        try:
            anno = pose_extraction(dmodel, pmodel, video_path, label, args.skip_postproc)
            pkl_name = osp.splitext(osp.basename(video_path))[0] + ".pkl"
            pkl_path = osp.join(args.output, pkl_name)
            mmengine.dump(anno, pkl_path)

            torch.cuda.empty_cache()

        except Exception as e:
            # 오류 로그 파일에 기록
            logging.error(f'Error processing {video_path}: {e}')
            print(f'Error processing {video_path}: {e}. Check the log file for details.')

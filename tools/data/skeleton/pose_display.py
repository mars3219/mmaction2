import cv2
import pickle
import os
import mmcv
import numpy as np

# 피클 파일에서 자세 추정 데이터를 불러오는 함수
def load_pose_data(pkl_file):
    with open(pkl_file, 'rb') as f:
        pose_data = pickle.load(f)
    return pose_data

# 키포인트를 비디오에 그리는 함수
def draw_keypoints(frame, keypoints):
    for i, keypoint in enumerate(keypoints):
        # 각 키포인트에 원을 그리기
        cv2.circle(frame, (int(keypoint[0]), int(keypoint[1])), 5, (0, 255, 0), -1)
        cv2.putText(frame, str(i), (int(keypoint[0]), int(keypoint[1])), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return frame

# 비디오에 키포인트를 표시하는 함수
def create_video_with_keypoints(video_file, pose_data_file, output_file, short_side):
    resize = False

    # 비디오 불러오기
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_file}")
        return
    
    # 비디오 정보 가져오기
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    video_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 해상도 리사이즈
    new_h, new_w = None, None
    if short_side is not None:
        resize = True
        if new_h is None:
            new_w, new_h = mmcv.rescale_size((frame_width, frame_height), (short_side, np.Inf))

    # 출력 비디오 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (new_w, new_h))

    # 자세 추정 데이터 로드
    pose_data = load_pose_data(pose_data_file)

    # 현재 프레임의 키포인트 가져오기
    keypoints = pose_data['keypoint']  # (max_num_people, total_frames, num_keypoints, 2)
    scores = pose_data['keypoint_score']  # (num_people, num_keypoints)
    total_frames = keypoints.shape[1]

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_idx >= total_frames:
            break
        
        if resize:
            frame = mmcv.imresize(frame, (new_w, new_h))

        # 프레임에 키포인트 그리기
        person_keypoints = keypoints[:, frame_idx, :, :]
        frame = draw_keypoints(frame, person_keypoints[0])
        
        # 비디오에 프레임 쓰기
        out.write(frame)
        frame_idx += 1

    # 비디오 처리 완료 후 자원 해제
    cap.release()
    out.release()
    print(f"Output video saved as {output_file}")

# 파일 경로 설정
video_file = '/data/aihub/violence/output/event/assault/Assault019_x264.mp4'  # 원본 비디오 파일 경로
pose_data_file = '/data/aihub/violence/train_pkl_m/Assault019_x264.pkl'  # 피클 파일 경로
output_file = '/workspace/tools/data/skeleton/output_video_with_keypoints.mp4'  # 출력 비디오 파일 경로
short_side = 720

# 비디오 생성 함수 실행
create_video_with_keypoints(video_file, pose_data_file, output_file, short_side)

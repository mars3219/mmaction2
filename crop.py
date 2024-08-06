import cv2

# 비디오 파일 경로
video_path = 'tmp_violence_cut.mp4'

# 비디오 캡처 객체 생성
cap = cv2.VideoCapture(video_path)

# 비디오의 프레임 크기와 FPS 가져오기
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# 크롭할 영역의 좌표 (x, y, w, h)
x, y, w, h = 100, 50, 200, 200  # 예시 좌표와 크기

# 출력 비디오 파일 설정
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_cropped_video.mp4', fourcc, fps, (w, h))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 프레임에서 크롭할 영역 추출
    cropped_frame = frame[y:y+h, x:x+w]

    # 출력 비디오 파일에 크롭된 프레임 쓰기
    out.write(cropped_frame)

    # # 크롭된 프레임을 화면에 표시 (옵션)
    # cv2.imshow('Cropped Frame', cropped_frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# 모든 리소스 해제
cap.release()
out.release()
cv2.destroyAllWindows()

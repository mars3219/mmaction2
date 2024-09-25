import os

# 파일 경로 설정
txt_file_path = '/data/aihub/violence/output/custom_train1.txt'
pkl_folder_path = '/data/aihub/violence/train_pkl'
output_txt_file_path = '/data/aihub/violence/output/custom_train1_missing_in_pkl.txt'

# 텍스트 파일에서 파일명 추출 (MP4 파일명에서 확장자를 제거한 부분만 저장, 빈 줄 제외)
with open(txt_file_path, 'r') as f:
    txt_lines = [line.strip() for line in f if line.strip()]  # 빈 줄 제외
    txt_files = set(line.split()[0].split('/')[-1].replace('.mp4', '') for line in txt_lines)

# PKL 폴더에서 파일명 추출 (확장자를 제외한 파일명만 저장)
pkl_files = set(os.path.splitext(file)[0] for file in os.listdir(pkl_folder_path) if file.endswith('.pkl'))

# PKL 폴더에 없는 txt 파일 찾기
missing_in_pkl = txt_files - pkl_files

# 원본 txt 파일에서 PKL 폴더에 없는 파일만 남겨 새로운 txt 파일에 저장
with open(output_txt_file_path, 'w') as output_file:
    for line in txt_lines:
        filename = line.split()[0].split('/')[-1].replace('.mp4', '')
        if filename in missing_in_pkl:
            output_file.write(line + '\n')

# 결과 출력
print(f"새로운 텍스트 파일 '{output_txt_file_path}'이(가) 생성되었습니다.")
print(f"PKL 폴더에 없는 파일 수: {len(missing_in_pkl)}")
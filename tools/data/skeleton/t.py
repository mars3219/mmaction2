import os

# 파일 경로 정의
custom_train1_path = '/data/aihub/violence/output/custom_train1.txt'
train_pkl_folder = '/data/aihub/violence/train_pkl'
output_file_path = '/data/aihub/violence/output/matching_files.txt'  # 일치하는 파일의 출력 경로

# custom_train1.txt 파일에서 파일 이름 읽기
with open(custom_train1_path, 'r') as f:
    txt_lines = [line.strip() for line in f if line.strip()]  # 빈 줄 제외
    # 원본 파일의 전체 경로와 파일 이름 저장
    custom_files = {line.split()[0]: line for line in txt_lines}

# train_pkl 폴더에서 .pkl 파일 가져오기
pkl_files = {os.path.splitext(f)[0]: f for f in os.listdir(train_pkl_folder) if f.endswith('.pkl')}

# 일치하는 파일 찾기
matching_files = {path: line for path, line in custom_files.items() if os.path.splitext(os.path.basename(path))[0] in pkl_files}

# 일치하는 항목을 원본 형식으로 새 파일에 작성
with open(output_file_path, 'w') as output_file:
    for original_path in custom_files.keys():
        if original_path in matching_files:
            output_file.write(matching_files[original_path] + '\n')  # 원본 형식 유지

print(f"일치하는 파일이 {output_file_path}에 작성되었습니다.")

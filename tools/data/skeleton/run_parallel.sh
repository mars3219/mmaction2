#!/bin/bash

# Usage: ./run_parallel.sh <txt-file-path> <output-path> <num-processes>
# Example: ./run_parallel.sh /data/aihub/violence/output /data/aihub/violence/train_pkl 4

TXT_FILE=$1  # txt-file 경로
OUTPUT_PATH=$2  # output 경로
NUM_PROCESSES=$3  # 실행할 프로세스 개수

if [[ -z "$TXT_FILE" || -z "$OUTPUT_PATH" || -z "$NUM_PROCESSES" ]]; then
  echo "Usage: $0 <txt-file-path> <output-path> <num-processes>"
  exit 1
fi

# PID 파일
PID_FILE="${OUTPUT_PATH}/process_pids.txt"

# 파이썬 명령을 실행할 함수 정의
run_python_process() {
  local idx=$1  # 인덱스 (1, 2, 3, 4 ...)
  local txt_file_suffix="custom_train${idx}.txt"
  local output_suffix="${OUTPUT_PATH}/output${idx}.pkl"
  
  # Python 명령 실행
  nohup python3 pose_extraction.py --txt-file "${TXT_FILE}/${txt_file_suffix}" --output "${OUTPUT_PATH}" --device cuda:0 > "${OUTPUT_PATH}/process${idx}.log" 2>&1 &
  echo $! >> $PID_FILE
}

# 복수개의 프로세스를 백그라운드에서 실행
for i in $(seq 1 $NUM_PROCESSES); do
  run_python_process $i
done

echo "All processes started. Check log files in ${OUTPUT_PATH}."


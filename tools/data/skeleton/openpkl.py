import pickle

# pickle 파일 열기
with open('/workspace/tools/data/skeleton/ntu60_2d.pkl', 'rb') as f:
    data = pickle.load(f)

# data 확인
print(data)

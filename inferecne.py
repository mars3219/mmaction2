from operator import itemgetter
from mmaction.apis import init_recognizer, inference_recognizer

#c3d
config_file = '/workspace/configs/recognition/c3d/custom_c3d_sports1m-pretrained_8xb30-16x1x1-45e_ucf101-rgb.py'
checkpoint_file = '/workspace/work_dir/c3d/best_acc_top1_epoch_15.pth'
video_file = '/workspace/tmp_violence_cut.mp4'
# video_file = '/data/aihub/violence/output/event/fight/cut_1-1_cam01_fight04_place02_night_spring.mp4'
label_file = '/workspace/label_map_custom.txt'

model = init_recognizer(config_file, checkpoint_file, device='cuda:0')
pred_result = inference_recognizer(model, video_file)

pred_scores = pred_result.pred_score.tolist()
score_tuples = tuple(zip(range(len(pred_scores)), pred_scores))
score_sorted = sorted(score_tuples, key=itemgetter(1), reverse=True)
top2_label = score_sorted[:2]

labels = open(label_file).readlines()
labels = [x.strip() for x in labels]
results = [(labels[k[0]], k[1]) for k in top2_label]

print('The top-2 labels with corresponding scores are:')
for result in results:
    print(f'{result[0]}: ', result[1])
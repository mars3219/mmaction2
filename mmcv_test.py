from mmcv.ops import batched_nms
import os
import torch

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TORCH_USE_CUDA_DSA"] = '1'

def check_mmcv():

    device = torch.device('cuda:0')

    bboxes = torch.randn(2, 4, device=device)
    scores = torch.randn(2, device=device)
    labels = torch.zeros(2, dtype=torch.long, device=device)
    det_bboxes, keep_idxs = batched_nms(bboxes.to(torch.float32), scores.to(torch.float32), labels, {
        'type': 'nms',
        'iou_threshold': 0.6
    })

    print('OK.')


if __name__ == '__main__':
    check_mmcv()

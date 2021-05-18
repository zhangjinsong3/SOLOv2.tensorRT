from mmdet.apis import init_detector, inference_detector, show_result_pyplot, show_result_ins
import mmcv
import torch


# Decoupled solo
# config_file = '../configs/solo/decoupled_solo_r50_fpn_8gpu_3x.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
# checkpoint_file = '../weights/DECOUPLED_SOLO_R50_3x.pth'

#  Decoupled solo lite
# config_file = '../configs/solo/decoupled_solo_light_r50_fpn_8gpu_3x.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
# checkpoint_file = '../weights/DECOUPLED_SOLO_LIGHT_R50_3x.pth'

# SOLOv2 lite
config_file = '../configs/solov2/solov2_light_448_r34_fpn_8gpu_3x.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
checkpoint_file = '../weights/SOLOv2_LIGHT_448_R34_3x.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image
img = 'images/demo.jpg'
result = inference_detector(model, img)

show_result_ins(img, result, model.CLASSES, score_thr=0.25, out_file="images/demo_out_torch.jpg")

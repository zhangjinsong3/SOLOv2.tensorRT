# -*- coding: utf-8 -*-
"""
Created on 20-9-28

@author: zjs (01376022)

Full inference script to run on onnxruntime and tensorRT!

# TODO (zjs): Make it more flexible, currently run *DecoupledSOLO* successfully on coco model with static input shape
# TODO (zjs): Remove torch op in preprocess and postprocess
# TODO (zjs): ADD TensorRT support

```
demo input shape 1 x 3x 427 x  640
resized shape    1 x 3x 800 x 1199
paded shape      1 x 3x 800 x 1216

seg_preds_x (also marked as mask_preds_x):
        1, 40, 200, 304
        1, 36, 200, 304
        1, 24, 200, 304
        1, 16, 200, 304
        1, 12, 200, 304

seg_preds_y (also marked as mask_preds_y):
        1, 40, 200, 304
        1, 36, 200, 304
        1, 24, 200, 304
        1, 16, 200, 304
        1, 12, 200, 304

cate_preds (with eval=True):
       1, 40, 40, 80
       1, 36, 36, 80
       1, 24, 24, 80
       1, 16, 16, 80
       1, 12, 12, 80

<optional, if use vanilla SOLO, should be:>
seg_preds (also marked as mask_preds):
        1, 40*40, 200, 304
        1, 36*36, 200, 304
        1, 24*24, 200, 304
        1, 16*16, 200, 304
        1, 12*12, 200, 304

```

"""
from mmdet.datasets.pipelines import Compose
from mmdet.apis.inference import LoadImage
from mmdet.apis import inference_detector, show_result_ins
from mmdet.core import get_classes
import mmcv
import torch
import onnxruntime as rt


def run_onnxruntime():
    config_file = '../configs/solo/decoupled_solo_r50_fpn_8gpu_3x.py'
    # checkpoint_file = '../weights/DECOUPLED_SOLO_R50_3x.pth'
    onnx_file = 'weights/DSOLO_R50.onnx'
    input_names = ['input']
    # output_names = ['C0', 'C1', 'C2', 'C3', 'C4']
    output_names = ['seg_preds_x_0', 'seg_preds_x_1', 'seg_preds_x_2', 'seg_preds_x_3', 'seg_preds_x_4',
                    'seg_preds_y_0', 'seg_preds_y_1', 'seg_preds_y_2', 'seg_preds_y_3', 'seg_preds_y_4',
                    'cate_pred_0', 'cate_pred_1', 'cate_pred_2', 'cate_pred_3', 'cate_pred_4']
    if isinstance(config_file, str):
        cfg = mmcv.Config.fromfile(config_file)
    elif not isinstance(config_file, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        'but got {}'.format(type(config_file)))

    # 1. Preprocess
    # input demo img size 427x640 --> resized 800x1199 --> pad 800x1216
    img = 'images/demo.jpg'
    # build the data pipeline
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    # prepare data
    data = dict(img=img)
    data = test_pipeline(data)

    # 2. Run inference on onnxruntime
    print("Load onnx model from {}.".format(onnx_file))
    sess = rt.InferenceSession(onnx_file)
    onnx_output = sess.run(output_names, {input_names[0]: data['img'][0].unsqueeze(0).cpu().numpy()})

    # 3. Get seg
    # 调用pytorch定义的获取分割图和matrix nms 以及后处理
    from mmdet.models.anchor_heads.decoupled_solo_head import DecoupledSOLOHead
    dsolo_head = DecoupledSOLOHead(num_classes=81,
                                   in_channels=256,
                                   num_grids=[40, 36, 24, 16, 12],
                                   strides=[8, 8, 16, 32, 32],
                                   loss_ins=cfg.model.bbox_head.loss_ins,
                                   loss_cate=cfg.model.bbox_head.loss_cate)
    seg_preds_x = [torch.from_numpy(x) for x in onnx_output[:5]]
    seg_preds_y = [torch.from_numpy(x) for x in onnx_output[5:10]]
    cate_preds =  [torch.from_numpy(x) for x in onnx_output[10:]]
    result = dsolo_head.get_seg(seg_preds_x, seg_preds_y, cate_preds, [data['img_meta'][0].data], cfg.test_cfg, rescale=True)
    show_result_ins(img, result, get_classes('coco'), score_thr=0.25, out_file="images/demo_out_onnxrt_decoupled_solo.jpg")


if __name__ == '__main__':
    run_onnxruntime()

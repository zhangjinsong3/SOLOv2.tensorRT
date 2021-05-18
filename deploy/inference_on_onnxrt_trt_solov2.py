# -*- coding: utf-8 -*-
"""
Created on 21-5-11

@author: zjs (01376022)

Full inference script to run on onnxruntime and tensorRT!

# TODO (zjs): Make it more flexible, currently run *SOLOv2 light R34* successfully on coco model with static input shape
# TODO (zjs): Remove torch op in preprocess and postprocess

```
demo input shape 1 x 3x 427 x  640
resized shape    1 x 3x 448 x 671
paded shape      1 x 3x 448 x 672

cate_preds:
        1, 40, 40, 80
        1, 36, 36, 80
        1, 24, 24, 80
        1, 16, 16, 80
        1, 12, 12, 80

kernel_preds:
        1, 128, 40, 40
        1, 128, 36, 36
        1, 128, 24, 24
        1, 128, 16, 16
        1, 128, 12, 12

seg_pred:
       1, 128, 112, 168


After reshape,permute,concate
Per batch
cate_pred:
        [3872, 80]
kernel_pred:
        [3872, 128]
seg_pred:
        [1, 128, 112, 168]

```

"""
from mmdet.datasets.pipelines import Compose
from mmdet.apis.inference import LoadImage
from mmdet.apis import show_result_ins
from mmdet.core import get_classes
import mmcv
import torch
import onnxruntime as rt
import cv2


def run_on_onnxruntime():
    config_file = '../configs/solov2/solov2_light_448_r34_fpn_8gpu_3x.py'
    onnx_file = 'weights/SOLOv2_light_R34.onnx'
    input_names = ['input']
    # output_names = ['C0', 'C1', 'C2', 'C3', 'C4']
    # output_names = ['cate_pred_0', 'cate_pred_1', 'cate_pred_2', 'cate_pred_3', 'cate_pred_4',
    #                 'kernel_pred_0', 'kernel_pred_1', 'kernel_pred_2', 'kernel_pred_3', 'kernel_pred_4',
    #                 'seg_pred']  # Origin
    output_names = ['cate_pred', 'kernel_pred', 'seg_pred']  # add permute & concate
    if isinstance(config_file, str):
        cfg = mmcv.Config.fromfile(config_file)
    elif not isinstance(config_file, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        'but got {}'.format(type(config_file)))

    # 1. Preprocess
    # input demo img size 427x640 --> resized 448x671 --> pad 448x672
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
    tic = cv2.getTickCount()
    onnx_output = sess.run(output_names, {input_names[0]: data['img'][0].unsqueeze(0).cpu().numpy()})
    print('-----> onnxruntime inference time: {}ms'.format((cv2.getTickCount() - tic) * 1000/ cv2.getTickFrequency()))

    # 3. Get seg
    # 调用pytorch定义的获取分割图和matrix nms 以及后处理
    from mmdet.models.anchor_heads.solov2_head import SOLOv2Head
    solov2_head = SOLOv2Head(num_classes=81,
                             in_channels=256,
                             num_grids=[40, 36, 24, 16, 12],
                             strides=[8, 8, 16, 32, 32],
                             ins_out_channels = 128,
                             loss_ins=cfg.model.bbox_head.loss_ins,
                             loss_cate=cfg.model.bbox_head.loss_cate)
    cate_preds = [torch.from_numpy(x) for x in onnx_output[:1]]
    kernel_preds = [torch.from_numpy(x) for x in onnx_output[1:2]]
    seg_pred = torch.from_numpy(onnx_output[2])
    result = solov2_head.get_seg(cate_preds, kernel_preds, seg_pred, [data['img_meta'][0].data], cfg.test_cfg, rescale=True)
    show_result_ins(img, result, get_classes('coco'), score_thr=0.25, out_file="images/demo_out_onnxrt_solov2.jpg")
    print('Script done!')


def run_on_tensorrt():
    from deploy import common

    config_file = '../configs/solov2/solov2_light_448_r34_fpn_8gpu_3x.py'
    onnx_file = 'weights/SOLOv2_light_R34.onnx'
    input_names = ['input']
    # output_names = ['C0', 'C1', 'C2', 'C3', 'C4']
    # output_names = ['cate_pred_0', 'cate_pred_1', 'cate_pred_2', 'cate_pred_3', 'cate_pred_4',
    #                 'kernel_pred_0', 'kernel_pred_1', 'kernel_pred_2', 'kernel_pred_3', 'kernel_pred_4',
    #                 'seg_pred']  # Origin
    output_names = ['cate_pred', 'kernel_pred', 'seg_pred']  # add permute & concate
    if isinstance(config_file, str):
        cfg = mmcv.Config.fromfile(config_file)
    elif not isinstance(config_file, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        'but got {}'.format(type(config_file)))

    # 1. Preprocess
    # input demo img size 427x640 --> resized 448x671 --> pad 448x672
    img = 'images/demo.jpg'
    # build the data pipeline
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    # prepare data
    data = dict(img=img)
    data = test_pipeline(data)

    # 2. Run inference on trt
    print("Load onnx model from {}.".format(onnx_file))
    image_shape = data['img_meta'][0].data['pad_shape']
    input_shapes = ((1, image_shape[2], image_shape[0], image_shape[1]),)  # explict shape
    # input_shapes = ((1, 3, 448, 448), (1, 3, 608, 608), (1, 3, 768, 768))  # dynamic shape
    # shape_matrix = [
    # [1, 40, 40, 80],
    # [1, 36, 36, 80],
    # [1, 24, 24, 80],
    # [1, 16, 16, 80],
    # [1, 12, 12, 80],
    # [1, 128, 40, 40],
    # [1, 128, 36, 36],
    # [1, 128, 24, 24],
    # [1, 128, 16, 16],
    # [1, 128, 12, 12],
    # [1, 128, image_shape[0]//4, image_shape[1]//4]
    # ]
    shape_matrix = [
        [3872, 80],
        [3872, 128],
        [1, 128, image_shape[0] // 4, image_shape[1] // 4]
    ]
    with common.get_engine(onnx_file, onnx_file.replace(".onnx", ".engine"),
                           input_shapes=input_shapes, force_rebuild=False) \
            as engine, engine.create_execution_context() as context:
        # Notice: Here we only allocate device memory for speed up
        # DYNAMIC shape
        # context.active_optimization_profile = 0
        # [context.set_binding_shape(x, tuple(y)) for x, y in enumerate(shape_matrix)]
        # inputs, outputs, bindings, stream = common.allocate_buffersV2(engine, context)
        # EXPLICIT shape
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)

        # Speed test: cpu(0.976s) vs gpu(0.719s)
        # ==> Set host input to the data.
        # The common.do_inference function will copy the input to the GPU before executing.
        inputs[0].host = data['img'][0].unsqueeze(0).cpu().numpy()  # for torch.Tensor
        # ==> Or set device input to the data.
        # in this mode, common.do_inference function should not copy inputs.host to inputs.device anymore.
        # c_type_pointer = ctypes.c_void_p(int(inputs[0].device))
        # x.cpu().numpy().copy_to_external(c_type_pointer)
        tic = cv2.getTickCount()
        trt_outputs = common.do_inferenceV2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream,
                                            batch_size=1, h_=image_shape[0], w_=image_shape[1])
        print('-----> tensorRT inference time: {}ms'.format((cv2.getTickCount() - tic) * 1000 / cv2.getTickFrequency()))

    # 3. Get seg
    # 调用pytorch定义的获取分割图和matrix nms 以及后处理
    from mmdet.models.anchor_heads.solov2_head import SOLOv2Head
    solov2_head = SOLOv2Head(num_classes=81,
                             in_channels=256,
                             num_grids=[40, 36, 24, 16, 12],
                             strides=[8, 8, 16, 32, 32],
                             ins_out_channels = 128,
                             loss_ins=cfg.model.bbox_head.loss_ins,
                             loss_cate=cfg.model.bbox_head.loss_cate)
    # TODO: tensorrt output order is different from pytorch?
    # Origin
    # ids = [8, 9, 7, 6, 5, 3, 4, 2, 1, 0, 10]
    # ids = [9, 8, 7, 5, 6, 4, 3, 2, 0, 1, 10]  # TODO: tensorrt output order is different from pytorch?
    # Add permute & concate
    ids = [1, 0, 2]

    cate_preds = [torch.from_numpy(trt_outputs[x]).reshape(y) for x, y in zip(ids[:1], shape_matrix[:1])]
    kernel_preds = [torch.from_numpy(trt_outputs[x]).reshape(y) for x, y in zip(ids[1:2], shape_matrix[1:2])]
    seg_pred = torch.from_numpy(trt_outputs[2]).reshape(shape_matrix[2])
    result = solov2_head.get_seg(cate_preds, kernel_preds, seg_pred, [data['img_meta'][0].data], cfg.test_cfg, rescale=True)
    show_result_ins(img, result, get_classes('coco'), score_thr=0.25, out_file="images/demo_out_trt_solov2.jpg")
    print('Script done!')


if __name__ == '__main__':
    run_on_onnxruntime()
    run_on_tensorrt()


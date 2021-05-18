# -*- coding: utf-8 -*-
"""
Created on 20-9-25

@author: zjs (01376022)

export onnx model from torch checkpoint

@Notice: GN normalize method seems not well implemented on onnx and tensorrt!  SEE: https://github.com/aim-uofa/AdelaiDet/issues/31#issuecomment-625217956 ; https://github.com/aim-uofa/AdelaiDet/pull/25#issue-401785580

@TODO: use onnx-simplifier may help run successfully! SEE: https://github.com/aim-uofa/AdelaiDet/issues/83#issuecomment-635718543

@Usage:
    python onnx_exporter.py ../configs/solo/decoupled_solo_r50_fpn_8gpu_3x.py  weights/DSOLO_R50.onnx --checkpoint ../weights/DECOUPLED_SOLO_R50_3x.pth --shape 800 1216

"""
import argparse
import cv2
import numpy as np
import torch
import mmcv
from mmdet.apis import init_detector

input_names = ['input']
# output_names = ['C0', 'C1', 'C2', 'C3', 'C4']
output_names = ['seg_preds_x_0', 'seg_preds_x_1', 'seg_preds_x_2', 'seg_preds_x_3', 'seg_preds_x_4',
                'seg_preds_y_0', 'seg_preds_y_1', 'seg_preds_y_2', 'seg_preds_y_3', 'seg_preds_y_4',
                'cate_pred_0', 'cate_pred_1', 'cate_pred_2', 'cate_pred_3', 'cate_pred_4']


def parse_args():
    parser = argparse.ArgumentParser(description='Export a torch model to onnx model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('out', help='output ONNX file')
    parser.add_argument('--checkpoint', help='checkpoint file of the model')
    parser.add_argument('--shape', type=int, nargs='+', default=[224], help='input image size')
    args = parser.parse_args()
    return args


def to_list(inputs):
    outputs = []
    for item in inputs:
        if isinstance(item, tuple) or isinstance(item, list):
            for tp in item:
                if isinstance(tp, tuple) or isinstance(tp, list):
                    for lt in tp:
                        if isinstance(lt, tuple) or isinstance(lt, list):
                            print("result is still packed strucure")
                        elif isinstance(lt, torch.Tensor):
                            print("collect tensor:", lt.shape)
                            outputs.append(lt)
                elif isinstance(tp, torch.Tensor):
                    print("collect tensor:", tp.shape)
                    outputs.append(tp)
        elif isinstance(item, torch.Tensor):
            print("collect tensor:", item.shape)
            outputs.append(item)
    print("output item count: %d" % len(outputs))
    return outputs


def convert2onnx(args):
    ''' Convert torch model to onnx model '''
    if len(args.shape) == 1:
        img_shape = (1, 3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        img_shape = (1, 3) + tuple(args.shape)
    elif len(args.shape) == 4:
        img_shape = tuple(args.shape)
    else:
        raise ValueError('invalid input shape')
    dummy_input = torch.randn(*img_shape, device='cuda:0')

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device='cuda:0')

    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
        # model.forward = model.extract_feat
    else:
        raise NotImplementedError(
            'ONNX exporting is not currently supported with {}'.
            format(model.__class__.__name__))

    # torch.onnx.export(model, dummy_input, args.out, input_names=['input'], output_names=['outputs'], verbose=True)
    torch.onnx.export(model, dummy_input, args.out, input_names=input_names, output_names=output_names, verbose=True, opset_version=11)
    # traced_script_module = torch.jit.trace(model, dummy_input)
    # traced_script_module.save(args.out.replace('onnx', 'pt'))


def check(args):
    ''' Checkthe Converted onnx model on onnxruntime and tensorrt '''
    import onnx
    import onnxruntime as rt

    if len(args.shape) == 1:
        img_shape = (1, 3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        img_shape = (1, 3) + tuple(args.shape)
    elif len(args.shape) == 4:
        img_shape = tuple(args.shape)
    else:
        raise ValueError('invalid input shape')
    dummy_input = torch.randn(*img_shape, device='cuda:0')

    # ======================= Run pytorch model =========================================
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device='cuda:0')
    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
        # model.forward = model.extract_feat
    else:
        raise NotImplementedError(
            'ONNX exporting is not currently supported with {}'.
            format(model.__class__.__name__))

    with torch.no_grad():
        torch_output = model(dummy_input)
        torch_output = to_list(torch_output)

    # ======================= Run onnx on onnxruntime =========================================
    print("Load onnx model from {}.".format(args.out))
    sess = rt.InferenceSession(args.out)

    # check input and output
    for in_blob in sess.get_inputs():
        if in_blob.name not in input_names:
            print("Input blob name not match that in the mode")
        else:
            print("Input {}, shape {} and type {}".format(in_blob.name, in_blob.shape, in_blob.type))
    for out_blob in sess.get_outputs():
        if out_blob.name not in output_names:
            print("Output blob name not match that in the mode")
        else:
            print("Output {}, shape {} and type {}".format(out_blob.name, out_blob.shape, out_blob.type))

    onnx_output = sess.run(output_names, {input_names[0]: dummy_input.cpu().numpy()})

    print("onnxruntime")
    for i, out in enumerate(onnx_output):
        try:
            np.testing.assert_allclose(torch_output[i].cpu().detach().numpy(), out, rtol=1e-03, atol=2e-04)
        except AssertionError as e:
            print("ouput {} mismatch {}".format(output_names[i], e))
            continue
        print("ouput {} match\n".format(output_names[i]))

    # ======================= Run onnx on tensorrt backend =========================================
    # TODO: Always failed but onnx2trt executable file run successfully!
    # print("Load onnx model from {}.".format(args.out))
    # import onnx_tensorrt.backend as tensorrt_backend
    # import tensorrt as trt
    # TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
    #
    # onnx_model = onnx.load(args.out)
    # onnx.checker.check_model(onnx_model)
    #
    # model_trt_backend = tensorrt_backend.prepare(onnx_model)
    # print("Run onnx model on tensorrt backend ")
    # tensorrt_output = model_trt_backend.run(dummy_input.data.numpy())
    #
    # print("tensorrt")
    # for i, name in enumerate(output_names):
    #     try:
    #         out = tensorrt_output[name]
    #         np.testing.assert_allclose(torch_output[i].cpu().detach().numpy(), out, rtol=1e-03, atol=2e-04)
    #     except AssertionError as e:
    #         print("ouput {} mismatch {}".format(output_names[i], e))
    #         continue
    #     print("ouput {} match\n".format(output_names[i]))

    print("script done")


if __name__ == '__main__':
    args = parse_args()
    convert2onnx(args)
    check(args)

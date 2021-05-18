# -*- coding: utf-8 -*-
"""
Created on 20-9-25

@author: zjs (01376022)

export onnx model from torch checkpoint

@Notice: use onnx-simplifier may help run successfully! SEE: https://github.com/aim-uofa/AdelaiDet/issues/83#issuecomment-635718543

@Notice: 在不同的位置import tensorrt 可能会出现不同的结果,尽量在开始的时候先import tenosrrt

@Notice: GN normalize method seems not well implemented on onnx and tensorrt!  SEE: https://github.com/aim-uofa/AdelaiDet/issues/31#issuecomment-625217956 ; https://github.com/aim-uofa/AdelaiDet/pull/25#issue-401785580
(but I run well on onnxruntime with version onnx==1.8.0, onnxruntime==1.6.0, pytorch==1.4.0, opset==11)

@Notice: 加载反序列化engine前,确保运行 trt.init_libnvinfer_plugins(TRT_LOGGER, '')

@TODO: 1. tensorrt 出来的output顺序与pytorch定义的不一致
@TODO: 2. 支持动态输入, backbone+neck 动态输入ok, head出现以下error(应该是forward_single中的某个操作导致,怀疑是linspace)
        ```
        [TensorRT] INTERNAL ERROR: Assertion failed: validateInputsCutensor(src, dst)
        ../rtSafe/cuda/cutensorReformat.cpp:227
        Aborting...
        [TensorRT] VERBOSE: Builder timing cache: created 115 entries, 415 hit(s)
        [TensorRT] ERROR: ../rtSafe/cuda/cutensorReformat.cpp (227) - Assertion Error in executeCutensor: 0 (validateInputsCutensor(src, dst))

        ```


@Usage:
    python onnx_exporter.py ../configs/solov2/solov2_light_448_r34_fpn_8gpu_3x.py  weights/SOLOv2_light_R34.onnx --checkpoint ../weights/SOLOv2_LIGHT_448_R34_3x.pth --shape 448 672

"""
import torch
import argparse
import numpy as np

from deploy import common

from mmdet.apis import init_detector

input_names = ['input']
# output_names = ['output']
# output_names = ['C0', 'C1', 'C2', 'C3']  # for backbone
# output_names = ['C0', 'C1', 'C2', 'C3', 'C4']  # for backbone + neck
# output_names = ['cate_pred_0', 'cate_pred_1', 'cate_pred_2', 'cate_pred_3', 'cate_pred_4',
#                 'kernel_pred_0', 'kernel_pred_1', 'kernel_pred_2', 'kernel_pred_3', 'kernel_pred_4',
#                 'seg_pred']  # Origin
output_names = ['cate_pred', 'kernel_pred', 'seg_pred']  # add permute & concate


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


def convert2onnx(args, dummy_input):
    ''' Convert torch model to onnx model '''
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device='cuda:0')

    if hasattr(model, 'forward_dummy'):
        # model.forward = model.extract_feat
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'ONNX exporting is not currently supported with {}'.
            format(model.__class__.__name__))
    # torch.onnx.export(model, dummy_input, args.out, input_names=['input'], output_names=['outputs'], verbose=True, opset_version=11)
    torch.onnx.export(model, dummy_input, args.out, input_names=input_names, output_names=output_names, verbose=True, opset_version=11)
    # traced_script_module = torch.jit.trace(model, dummy_input)
    # traced_script_module.save(args.out.replace('onnx', 'pt'))


def check(args, dummy_input, check_onnx=True, check_trt=True):
    ''' Check the Converted onnx model on onnxruntime and tensorrt '''
    # ======================= Run pytorch model =========================================
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device='cuda:0')

    if hasattr(model, 'forward_dummy'):
        # model.forward = model.extract_feat
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'ONNX exporting is not currently supported with {}'.
            format(model.__class__.__name__))

    with torch.no_grad():
        torch_output = model(dummy_input)
        torch_output = to_list(torch_output)

    # ======================= Run onnx on onnxruntime =========================================
    if check_onnx:
        import onnxruntime as rt
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

    # ======================= Run onnx on tensorrt =========================================
    if check_trt:
        input_shapes = ((1, 3, args.shape[0], args.shape[1]),)  # explict shape
        # input_shapes = ((1, 3, 448, 448), (1, 3, 608, 608), (1, 3, 768, 768))  # dynamic shape
        # shape_matrix = [
        #     [1, 3, args.shape[0], args.shape[1]],
        #     [1, 40, 40, 80],
        #     [1, 36, 36, 80],
        #     [1, 24, 24, 80],
        #     [1, 16, 16, 80],
        #     [1, 12, 12, 80],
        #     [1, 128, 40, 40],
        #     [1, 128, 36, 36],
        #     [1, 128, 24, 24],
        #     [1, 128, 16, 16],
        #     [1, 128, 12, 12],
        #     [1, 128, args.shape[0] // 4, args.shape[1] // 4]
        # ]
        # shape_matrix = [
        #     [1, 3, args.shape[0], args.shape[1]],
        #     [3872, 80],
        #     [3872, 128],
        #     [1, 128, args.shape[0] // 4, args.shape[1] // 4]
        # ]
        with common.get_engine(args.out, args.out.replace(".onnx", ".engine"), input_shapes=input_shapes, force_rebuild=False) \
                as engine, engine.create_execution_context() as context:
            # Notice: Here we only allocate device memory for speed up

            # DYNAMIC shape
            # context.active_optimization_profile = 0
            # [context.set_binding_shape(x, tuple(y)) for x, y in enumerate(shape_matrix)]
            # inputs, outputs, bindings, stream = common.allocate_buffersV2(engine, context)

            # EXPLICIT shape
            inputs, outputs, bindings, stream = common.allocate_buffers(engine)

            # The common.do_inference function will copy the input to the GPU before executing.
            inputs[0].host = dummy_input.cpu().numpy()  # for torch.Tensor
            # ==> Or set device input to the data.
            # in this mode, common.do_inference function should not copy inputs.host to inputs.device anymore.
            # c_type_pointer = ctypes.c_void_p(int(inputs[0].device))
            # x.cpu().numpy().copy_to_external(c_type_pointer)
            trt_outputs = common.do_inferenceV2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream,
                                                batch_size=1, h_=args.shape[0], w_=args.shape[1])
        print("tensorrt")
        # TODO: tensorrt output order is different from pytorch? Origin
        # Origin
        # ids = [8, 9, 7, 6, 5, 3, 4, 2, 1, 0, 10]
        # Add permute & concate
        ids = [1, 0, 2]
        for i, (trt_output, id) in enumerate(zip(trt_outputs, ids)):
            try:
                np.testing.assert_allclose(torch_output[id].cpu().detach().numpy().reshape(-1), trt_output, rtol=1e-03, atol=2e-04)
            except AssertionError as e:
                print("ouput {} mismatch {}".format(output_names[id], e))
                continue
            print("ouput {} match\n".format(output_names[id]))

    print("script done")


if __name__ == '__main__':
    args = parse_args()
    if len(args.shape) == 1:
        img_shape = (1, 3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        img_shape = (1, 3) + tuple(args.shape)
    elif len(args.shape) == 4:
        img_shape = tuple(args.shape)
    else:
        raise ValueError('invalid input shape')
    dummy_input = torch.randn(*img_shape, device='cuda:0')

    convert2onnx(args, dummy_input)
    check(args, dummy_input, check_onnx=False, check_trt=True)

# -*- coding: utf-8 -*-
"""
Created on 21-5-11

@author: zjs (01376022)

export an torch model to onnx model and tensorRT engine

此工程可以用来快速确认 环境和模型 是否可以转换到tensorRT上
我们常常遇到 torch版本 opset版本 tensorRT版本不兼容的问题,可以线运行此小工程,确定环境与版本无误,再继续后续探索

eg.
OK:    pytorch 1.1.0   -- opset-9  -- tensorRT 6.0.1.5
OK:    pytorch 1.1.0   -- opset-10 -- tensorRT 6.0.1.5
OK:    pytorch 1.3.0   -- opset-9  -- tensorRT 6.0.1.5 --onnx-tensorrt 6.0
OK:    pytorch 1.3.0   -- opset-10 -- tensorRT 6.0.1.5 --onnx-tensorrt 6.0
OK:    pytorch 1.3.0   -- opset-11 -- tensorRT 6.0.1.5 --onnx-tensorrt 6.0
WRONG: pytorch 1.4.0   -- opset-11 -- tensorRT 6.0.1.5
OK:    pytorch 1.4.0   -- opset-11 -- tensorRT 6.0.1.5 --onnx-tensorrt 6.0
OK:    pytorch 1.4.0   -- opset-11 -- tensorRT 7.2.1.6 --onnx-tensorrt 7.2.1


@Reference: https://blog.csdn.net/yangjf91/article/details/102666456
@Reference: https://www.jianshu.com/p/dcd450aa2e41
@Reference: https://www.yuque.com/bajiu-dmdxl/ogyuv8/pxnfgo
@Reference: https://blog.csdn.net/xxradon/article/details/103959758

"""
import os
# import onnx  # TODO: Notice that onnx can`t be imported with trt on some version
import time
import torch
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = str('0')

import tensorrt as trt
from deploy import common
from deploy.common import TRT_LOGGER

torch.cuda.synchronize()  # TODO: Notice that the cuda.synchronize() should be done after import trt
input_names = ['inputs']
output_names = ['outputs']

# # TODO: opset 9 还不支持 upsample_bilinear2d, 但是opset-11已经支持了.
# import torch.onnx.symbolic
# @torch.onnx.symbolic.parse_args('v', 'is', 'i')
# def upsample_bilinear2d(g, input, output_size, align_corners):
#     height_scale = float(output_size[-2]) / input.type().sizes()[-2]  # 8
#     width_scale = float(output_size[-1]) / input.type().sizes()[-1]  # 8
#     return g.op("Upsample", input,
#                 scales_f=(1, 1, height_scale, width_scale),
#                 mode_s="linear")
# torch.onnx.symbolic.upsample_bilinear2d = upsample_bilinear2d


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


def convert2onnx(model, dummy_input, save_path):
    ''' Convert torch model to onnx model '''
    torch.onnx.export(model, dummy_input, save_path,
                      input_names=input_names, output_names=output_names, verbose=True, opset_version=11)


def get_engine(onnx_file_path, engine_file_path=""):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(common.EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            builder.max_workspace_size = 1 << 30  # 1 GiB
            builder.max_batch_size = 1
            # builder.fp16_mode = False
            # builder.strict_type_constraints = True

            # Parse model file
            # Try to load a previously generated graph in ONNX format:
            if not os.path.exists(onnx_file_path):
                print('ONNX file {} not found, please generate it first.'.format(onnx_file_path))
                exit(0)
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                if not parser.parse(model.read()):
                    print('ERROR: Failed to parse the ONNX file.')
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None

            # Reference: https://blog.csdn.net/weixin_43953045/article/details/103937295
            # last_layer = network.get_layer(network.num_layers - 1)
            # if not last_layer.get_output(0):
            #     network.mark_output(last_layer.get_output(0))
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            engine = builder.build_cuda_engine(network)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())
            return engine

    return build_engine()


def check(model, device, input_tensor, name, check_onnx=True, check_trt=True):
    ''' Check the Converted onnx model on onnxruntime and tensorrt '''
    x = input_tensor
    x.to(device)
    # ======================= Run pytorch model =========================================
    tic = time.time()
    with torch.no_grad():
        torch_output = model(x)
        # print(torch_output.shape)
        torch_output = to_list([torch_output])
    print(time.time() - tic)
    print(name)
    # print(model)
    # torch.save(model.state_dict(), 'weights/{}.pth'.format(name))

    # ======================= Run onnx on onnxruntime =========================================
    # export onnx model
    convert2onnx(model, x, 'weights/{}.onnx'.format(name))
    if check_onnx:
        print("Load onnx model from {}.".format('weights/{}.onnx'.format(name)))
        import onnxruntime as rt
        sess = rt.InferenceSession('weights/{}.onnx'.format(name))

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

        onnx_output = sess.run(output_names, {input_names[0]: x.cpu().numpy()})

        print("========================== Check output between onnxruntime and torch! ================================")
        for i, out in enumerate(onnx_output):
            try:
                np.testing.assert_allclose(torch_output[i].cpu().detach().numpy(), out, rtol=1e-03, atol=2e-04)
            except AssertionError as e:
                print("ouput {} mismatch {}".format(output_names[i], e))
                continue
            print("ouput {} match\n".format(output_names[i]))

    # ======================= Run onnx on tensorrt backend =========================================
    if check_trt:
        # use API V2 here
        print("Load onnx model from {}.".format('weights/{}.onnx'.format(name)))

        with get_engine('weights/{}.onnx'.format(name), 'weights/{}.engine'.format(name)) as engine, engine.create_execution_context() as context:
            # Notice: Here we only allocate device memory for speed up
            inputs, outputs, bindings, stream = common.allocate_buffers(engine)

            # Speed test: cpu(0.976s) vs gpu(0.719s)
            # ==> Set host input to the data.
            # The common.do_inference function will copy the input to the GPU before executing.
            inputs[0].host = x.cpu().numpy()  # for torch.Tensor
            # ==> Or set device input to the data.
            # in this mode, common.do_inference function should not copy inputs.host to inputs.device anymore.
            # c_type_pointer = ctypes.c_void_p(int(inputs[0].device))
            # x.cpu().numpy().copy_to_external(c_type_pointer)
            trt_outputs = common.do_inferenceV2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream, batch_size=1, h_=224, w_=224)

        print("====================== Check output between tensorRT and torch =====================================")
        for i, trt_output in enumerate(trt_outputs):
            try:
                np.testing.assert_allclose(torch_output[i].cpu().detach().numpy().reshape(-1), trt_output, rtol=1e-03, atol=2e-04)
            except AssertionError as e:
                print("ouput {} mismatch {}".format(output_names[i], e))
                continue
            print("ouput {} match\n".format(output_names[i]))

        print("script done")


if __name__ == '__main__':
    import torch.nn as nn
    import torchvision

    class minimal_model(nn.Module):
        def __init__(self, out):
            super(minimal_model, self).__init__()
            self.conv1 = nn.Conv2d(3, out, kernel_size=7, stride=2, padding=3,
                                   bias=False)
            self.bn1 = nn.BatchNorm2d(out)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            return x

    device = torch.device('cuda:0')
    batch_size = 1
    x = torch.randn(batch_size, 3, 224, 224).to(device)
    # model = minimal_model(32)
    model = torchvision.models.resnet18(pretrained=False)
    model.to(device)
    model.eval()
    check(model, device, x, 'resnet18', check_onnx=True, check_trt=True)

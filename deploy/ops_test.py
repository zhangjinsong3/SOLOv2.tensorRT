# -*- coding: utf-8 -*-
"""
Created on 21-5-12

@author: zjs (01376022)



"""
import torch
import torch.nn as nn


class TestUpsample(nn.Module):
    def __int__(self):
        super(TestUpsample, self).__init__()

    def forward(self, x):
        x = nn.Upsample(scale_factor=2, mode="nearest")(x)
        return x


class TestPermute(nn.Module):
    def __init__(self):
        super(TestPermute, self).__init__()

    def forward(self, x):
        x = x[0].permute(1, 2, 0).view(-1, 128)
        return x


class TestOps(nn.Module):
    def __init__(self):
        super(TestOps, self).__init__()

    def forward(self, x):
        x = x.view(-1, 128)
        return x


image = torch.ones([1, 128, 40, 40])
torch_model = TestOps()
torch_out = torch.onnx._export(torch_model, image, 'weights/test.onnx', verbose=True, opset_version=11)
from deploy import common

with common.get_engine('weights/test.onnx', 'weights/test.engine', input_shapes=((1, 128, 40, 40),), force_rebuild=True) \
        as engine, engine.create_execution_context() as context:
    # DYNAMIC shape
    # context.active_optimization_profile = 0
    # [context.set_binding_shape(x, tuple(y)) for x, y in enumerate(shape_matrix)]
    # inputs, outputs, bindings, stream = common.allocate_buffersV2(engine, context)

    # EXPLICIT shape
    inputs, outputs, bindings, stream = common.allocate_buffers(engine)

    # The common.do_inference function will copy the input to the GPU before executing.
    inputs[0].host = image.cpu().numpy()  # for torch.Tensor
    trt_outputs = common.do_inferenceV2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream,
                                        batch_size=1, h_=40, w_=40, c=128)
print('Done!')

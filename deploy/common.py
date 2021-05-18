#
# Copyright 1993-2019 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO LICENSEE:
#
# This source code and/or documentation ("Licensed Deliverables") are
# subject to NVIDIA intellectual property rights under U.S. and
# international Copyright laws.
#
# These Licensed Deliverables contained herein is PROPRIETARY and
# CONFIDENTIAL to NVIDIA and is being provided under the terms and
# conditions of a form of NVIDIA software license agreement by and
# between NVIDIA and Licensee ("License Agreement") or electronically
# accepted by Licensee.  Notwithstanding any terms or conditions to
# the contrary in the License Agreement, reproduction or disclosure
# of the Licensed Deliverables to any third party without the express
# written consent of NVIDIA is prohibited.
#
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
# SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
# PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
# NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
# DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
# NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
# SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
# DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
# WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
# ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
# OF THESE LICENSED DELIVERABLES.
#
# U.S. Government End Users.  These Licensed Deliverables are a
# "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
# 1995), consisting of "commercial computer software" and "commercial
# computer software documentation" as such terms are used in 48
# C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
# only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
# 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
# U.S. Government End Users acquire the Licensed Deliverables with
# only those rights set forth herein.
#
# Any use of the Licensed Deliverables in individual and commercial
# software must include, in the user documentation and internal
# comments to the code, the above Disclaimer and U.S. Government End
# Users Notice.
#

from itertools import chain
import argparse
import os

import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

import tensorrt as trt
TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
# TRT_LOGGER = trt.Logger()
trt.init_libnvinfer_plugins(TRT_LOGGER, '')

try:
    # Sometimes python2 does not understand FileNotFoundError
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError

EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

def GiB(val):
    return val * 1 << 30


# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


# Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream

# # Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
# def allocate_buffers(engine):
#     inputs = []
#     outputs = []
#     bindings = []
#     stream = cuda.Stream()
#     for binding in engine:
#         size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
#         dtype = engine.get_binding_dtype(binding)
#         # Allocate only device buffers
#         # host_mem = cuda.pagelocked_empty(size, dtype)
#         device_mem = cuda.mem_alloc(size * dtype.itemsize)
#         # Append the device buffer to device bindings.
#         bindings.append(int(device_mem))
#         # Append to the appropriate list.
#         if engine.binding_is_input(binding):
#             inputs.append(HostDeviceMem(None, device_mem))
#         else:
#             # host_mem = cuda.pagelocked_empty(size, trt.nptype(dtype))
#             outputs.append(HostDeviceMem(None, device_mem))
#     return inputs, outputs, bindings, stream


# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # 1. Transfer input data to the GPU if need.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # 2. Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # 3. Transfer predictions back from the GPU if need.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # 4. Synchronize the stream
    stream.synchronize()
    # 5. Return only the host outputs or only the device outputs
    return [out.host for out in outputs]
    # return [out.device for out in outputs]


# ============================= NEW API FOR FULL-DIMS AND DYNAMIC SHAPE ====================================
# Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
# Modify for dynamic input shape, which maybe network differentable
def allocate_buffersV2(engine, context):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    down_stride = 1
    print('engine.get_binding_format_desc', engine.get_binding_format_desc(0))
    for count, binding in enumerate(engine):
        # print('binding:', binding)
        size = trt.volume(context.get_binding_shape(count)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # print('dtype:', dtype)
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
        print('count: ', count)
        print('binding: ', binding)
        print('binding shape: ', engine.get_binding_shape(binding))
        print('binding shape: ', context.get_binding_shape(count))
        print('max_batch_size:', engine.max_batch_size)
        print('size:', size)
        print('input:', inputs)
        print('output:', outputs)
        print('------------------')
    return inputs, outputs, bindings, stream


def do_inferenceV2(context, bindings, inputs, outputs, stream, batch_size, h_, w_, c=3):
    # Transfer input data to the GPU.
    context.set_binding_shape(0, (batch_size, c, h_, w_))  # if comment: [TensorRT] ERROR: Parameter check failed at: engine.cpp::resolveSlots::1024, condition: allInputDimensionsSpecified(routine), but get correct result!

    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]

# ============================= NEW API FOR FULL-DIMS AND DYNAMIC SHAPE ====================================


def get_engine(onnx_file_path, engine_file_path="", input_shapes=((1, 3, 640, 640)), force_rebuild=False):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""

    assert len(input_shapes) in [1, 3], 'length of input_shapes should be 1 or 3, 3 for dynamic input size, got {}'.format(len(input_shapes))

    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(
                network, TRT_LOGGER) as parser, builder.create_builder_config() as config:

            builder.strict_type_constraints = True
            # builder.max_workspace_size = 1 << 30  # deprecated, use config to set max_workspace_size
            # builder.fp16_mode = True   # deprecated, use config to set FP16 mode
            # builder.max_batch_size = 1  # deprecated, use EXPLICIT_BATCH

            config.set_flag(trt.BuilderFlag.FP16)
            config.max_workspace_size=GiB(1)

            # Parse model file
            # Try to load a previously generated graph in ONNX format:
            if not os.path.exists(onnx_file_path):
                print('ONNX file {} not found, please generate it first.'.format(
                    onnx_file_path))
                exit(0)
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                if not parser.parse(model.read()):
                    print('ERROR: Failed to parse the ONNX file.')
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    exit(0)

            # Reference: https://blog.csdn.net/weixin_43953045/article/details/103937295
            last_layer = network.get_layer(network.num_layers - 1)
            if not last_layer.get_output(0):
                network.mark_output(last_layer.get_output(0))

            print("input shape {}".format(network.get_input(0).shape))
            network.get_input(0).shape = [1, 3, -1, -1] if len(input_shapes) != 1 else input_shapes[0]
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            # ########################  SET DYNAMIC INPUT SHAPE #################################
            if len(input_shapes) == 3:
                profile = builder.create_optimization_profile()
                profile.set_shape(network.get_input(0).name, *input_shapes)
                config.add_optimization_profile(profile)
                engine = builder.build_engine(network, config)
            else:
                engine = builder.build_cuda_engine(network)
            # ########################################################
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())
            return engine

    # return build_engine()
    if os.path.exists(engine_file_path) and not force_rebuild:
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()


## Deploy SOLO to onnx and tensorRT

#### Requirements

- pytorch == 1.4.0
- onnx == 1.6.0
- onnxruntime == 1.6.0
- tensorRT == 6.0.1.5
- onnx-tensorrt == 6.0  
- cuda == 10.1
- protobuf == 3.13.0
- pycuda == 2019.1.2

#### Requirements 2021

- pytorch == 1.4.0
- onnx == 1.8.0
- onnxruntime == 1.6.0
- tensorRT == 7.2.1.6
- onnx-tensorrt == 7.2.1  
- cuda == 11.1
- protobuf == 3.13.0
- pycuda == 2019.1.2

#### Modification
0. 增加ONNX_EXPORT和ONNX_BATCH_SIZE 参数控制导出,需要导出onnx时设置`ONNX_EXPORT=True`

1. 当前版本的 onnx 支持 F.interpolate()函数,但是tensorRT 6.0 不支持 bilinear 模式(TensorRT 7.2.1支持),
    
2. 当前版本pytorch 1.4 不支持onnx导出 torch.linspace(), 后面的版本应该修复了 
    see: `https://github.com/pytorch/pytorch/pull/39403`
    修改：
    - 参考上面的pull, 修改 `torch/onnx/symbolic_opset11.py`, 增加如下代码段:
    ```python
    def linspace(g, start, end, steps, dtype, *options):
        parsed_dtype = sym_help._get_const(dtype, 'i', 'dtype')
        steps_ = sym_help._get_const(steps, 'i', 'steps')
        if steps_ == 1:
            return g.op("Cast", start, to_i=sym_help.scalar_type_to_onnx[parsed_dtype])
        diff = g.op("Cast", g.op("Sub", end, start), to_i=sym_help.cast_pytorch_to_onnx['Double'])
        delta = g.op("Div", diff, g.op("Constant", value_t=torch.tensor(steps_ - 1, dtype=torch.double)))
        end = g.op("Add", g.op("Cast", end, to_i=sym_help.cast_pytorch_to_onnx['Double']), delta)
        start = g.op("Cast", start, to_i=sym_help.cast_pytorch_to_onnx['Double'])
        return g.op("Cast", g.op("Range", start, end, delta), to_i=sym_help.scalar_type_to_onnx[parsed_dtype])
    ```

3. head 中将size() cast到 int,修改如下:
    ```python
       # Modify for onnx export, frozen the input size = 800x800, batch size = 1
        size = {0: 100, 1: 100, 2: 50, 3: 25, 4: 25}
        feat_h, feat_w = ins_feat.shape[-2], ins_feat.shape[-1]
        feat_h, feat_w = int(feat_h.cpu().numpy() if isinstance(feat_h, torch.Tensor) else feat_h),\
                         int(feat_w.cpu().numpy() if isinstance(feat_w, torch.Tensor) else feat_w)
        x_range = torch.linspace(-1, 1, feat_w, device=ins_feat.device)
        y_range = torch.linspace(-1, 1, feat_h, device=ins_feat.device)
        y, x = torch.meshgrid(y_range, x_range)
        y = y.expand([1, 1, -1, -1])
        x = x.expand([1, 1, -1, -1])

        # Origin from SOLO
        # x_range = torch.linspace(-1, 1, ins_feat.shape[-1], device=ins_feat.device)
        # y_range = torch.linspace(-1, 1, ins_feat.shape[-2], device=ins_feat.device)
        # y, x = torch.meshgrid(y_range, x_range)
        # y = y.expand([ins_feat.shape[0], 1, -1, -1])
        # x = x.expand([ins_feat.shape[0], 1, -1, -1])
    ```

    *PS: 至此,应该可以成功导出onnx模型,并在onnxruntime上运行,得到一致的结果,
    但是由于目前版本的onnx不支持动态尺寸,onnx输入为固定尺寸;由于修改点3,输入batchsize也是固定的.
    按照修改点1修改之后,应该也能在tensorRT上运行(还未尝试).*

4. 在 导出ONNX 时,将solov_head.py中get_seg() 如下操作放入onnx model中,减少自写后处理的工作量
    ```python
        cate_pred_list = [
            cate_preds[i][img_id].view(-1, self.cate_out_channels).detach() for i in range(num_levels)
        ]
        seg_pred_list = seg_pred[img_id, ...].unsqueeze(0)
        kernel_pred_list = [
            kernel_preds[i][img_id].permute(1, 2, 0).view(-1, self.kernel_out_channels).detach()
                            for i in range(num_levels)
        ]
        cate_pred_list = torch.cat(cate_pred_list, dim=0)
        kernel_pred_list = torch.cat(kernel_pred_list, dim=0)
    ```
 
 5. 动态卷积的支持: 使用动态卷积可以在allocate_buffers时分配最大的size对应的显存计算.
        
        ```
        @TODO: 支持动态输入, backbone+neck 动态输入ok, head出现以下error(应该是forward_single中的某个操作导致,怀疑是linspace)
        ```
        
        ```
        [TensorRT] INTERNAL ERROR: Assertion failed: validateInputsCutensor(src, dst)
        ../rtSafe/cuda/cutensorReformat.cpp:227
        Aborting...
        [TensorRT] VERBOSE: Builder timing cache: created 115 entries, 415 hit(s)
        [TensorRT] ERROR: ../rtSafe/cuda/cutensorReformat.cpp (227) - Assertion Error in executeCutensor: 0 (validateInputsCutensor(src, dst))

        ```
 
 6. pytorch 输出顺序与 onnx的一致,但是tensorRT的输出顺序不一致,尚未找到问题所在(但是tensorRT输出顺序是稳定的)
 
 
 #### Speed
 使用模型 `weights/SOLOv2_LIGHT_448_R34_3x.pth` 推理 `demo.jpg`,速度对比如下
 
 | 模型             | 推理框架            | size     | time | gpu memory |
 | ----            | --------           |------    |------| ----------- |
 | SOLOv2 LIGHT R34|  pytorch(FPN FP16) |  448x672 | 18ms |1107M        |
 | SOLOv2 LIGHT R34| onnxruntime(cpu)   |  448x672 |220ms | 320M        |
 | SOLOv2 LIGHT R34|  tensorRT(FP32)    |  448x672 | 18ms | 1031M       |
 
     ``` tensorRT
    -----> tensorRT inference time: 25.519246ms
    -----> tensorRT inference time: 18.540499ms
    -----> tensorRT inference time: 17.316972ms
    -----> tensorRT inference time: 17.140708ms
    -----> tensorRT inference time: 17.055805ms
    -----> tensorRT inference time: 16.243325ms
    -----> tensorRT inference time: 16.143518ms
    -----> tensorRT inference time: 16.121296ms
    -----> tensorRT inference time: 16.093398ms
    -----> tensorRT inference time: 16.300199ms
    
    ```
    
    ``` pytorch
    -----> pytorch inference time: 107.31163ms
    -----> pytorch inference time: 18.357391ms
    -----> pytorch inference time: 18.342804ms
    -----> pytorch inference time: 18.3574ms
    -----> pytorch inference time: 18.330068ms
    -----> pytorch inference time: 17.675493ms
    -----> pytorch inference time: 17.446083ms
    -----> pytorch inference time: 17.452661ms
    -----> pytorch inference time: 17.522118ms
    -----> pytorch inference time: 17.361028ms
    -----> pytorch inference time: 17.374755ms

    ```
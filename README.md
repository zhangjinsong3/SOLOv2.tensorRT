# DBNet.tensorRT: 

**NOTE**: code based on [WXinlong/SOLO](https://github.com/WXinlong/SOLO)

add support for TensorRT inference

- [x] onnxruntime
- [x] tensorRT
- [ ] full_dims and dynamic shape
- [ ] postprocess without torch

```
git clone https://github.com/zhangjinsong3/SOLOv2.tensorRT
```

## Requirements

* pytorch 1.4+
* torchvision 0.5+
* gcc 4.9+
* tensorRT 7.2.1
* onnx-tensorrt 7.2
* onnx 1.8.0
* onnxruntime 1.6.0


### Train
- set `ONNX_EXPORT=False` in  `mmdet.deploy_params.py`
- Training according the step in <a href="README_SOLO.md" alt="链接">SOLO🔗</a>


### Deploy
- set `ONNX_EXPORT=True` in  `mmdet.deploy_params.py`

- `cd deploy`

- run inference on onnxruntime & tensorRT
```bash
# run on pytorch
python inference_demo.py

# Convert pth to onnx model, then run on onnxrt & trt
# SOLOv2 light R34
python onnx_exporter_solov2.py ../configs/solov2/solov2_light_448_r34_fpn_8gpu_3x.py  weights/SOLOv2_light_R34.onnx --checkpoint ../weights/SOLOv2_LIGHT_448_R34_3x.pth --shape 448 672
python inference_on_onnxrt_trt_solov2.py
```



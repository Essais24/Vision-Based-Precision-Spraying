# GPU weed-mask template

Build this directory on the Jetson with the installed DeepStream SDK:

```bash
make -C weed_mask_template
```

The library receives RGBA NVMM batches directly after `nvinfer`. For every frame it reads `NvDsFrameMeta`/`NvDsObjectMeta`, runs `2G-R-B > exg-threshold` in CUDA, and excludes every detector box (the deployed YOLO has one crop/chilli class). It marks surviving weed pixels magenta; `deepstream_pipeline.py` converts that exact marker back into the binary `uint8` mask passed to `process_frame`. No PyTorch model or CUDA tensor operation is used for ExG.

`exg-threshold` is an nvdsvideotemplate creation property, so change it by recreating `DeepStreamCamera` (rather than expecting a live slider to alter a running custom library).

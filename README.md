
# Weed Sprayer - Jetson DeepStream deployment

This project uses NVIDIA DeepStream for the YOLO crop-detector stage --
that's the *only* inference backend now, there's no PyTorch/ultralytics
fallback. The green-vegetation (ExG) segmentation and crop-box exclusion run in a
DeepStream CUDA custom library. ExG weed boxes are added to DeepStream
metadata before native NvSORT, so crop and weed IDs are assigned in the same
pipeline. It is designed for a Jetson Orin Nano
running DeepStream 9.1.

## Problem and fix: track chilli and weeds together

**Problem:** the old flow created chilli boxes in `nvinfer`, then generated
weed boxes with Python/OpenCV after `appsink`. Since `nvtracker` only sees
metadata inside the GStreamer pipeline, it could keep chilli IDs but never
assign persistent IDs to those later weed boxes.

**Fix:** a CUDA ExG kernel in the `weed_bbox_meta` custom library runs after
YOLO and before `nvtracker`. It suppresses the class-0 chilli boxes, extracts
weed regions, and adds them as class-1 `NvDsObjectMeta`. NvSORT receives both
classes in one frame and keeps their tracks separate by class. Python reads
the tracked boxes and IDs from returned metadata instead of extracting weed
boxes after `appsink`.

The CUDA kernel creates the ExG mask; OpenCV connected components in the C++
template library extracts boxes from a host copy of that mask. Large connected
regions use distance-transform/watershed splitting. YOLO and NvSORT stay in
DeepStream.

## Processing architecture (Baseline / Previous Pipeline)

> **Note**: This section documents the original baseline architecture. The updated, optimized pipeline architecture is documented below in [Updated Pipeline Architecture](#updated-pipeline-architecture-and-optimizations).

```text
CSI camera / video / RTSP
        |
        v
DeepStream Service Maker
  decode -> nvstreammux -> TensorRT YOLO crop detection
                                  |
                                  v
                  CUDA ExG + crop-box exclusion (nvdsvideotemplate)
                                  |
                                  v
                 GPU RGB conversion -> Python frame receiver
                                  |
                                  v
                    CPU connected components -> weed boxes
   (large blobs still split via CPU/OpenCV watershed)
        |
        v
pixel-to-world conversion -> sprayer controller
```

`bestM.pt` is a **YOLOv8 crop detector**. The ExG/green-inference algorithm is
not part of the YOLO model and is intentionally not exported to ONNX -- it
runs as a CUDA/C++ custom library inside the DeepStream pipeline instead.

## Target platform

| Component | Required deployment target |
| --- | --- |
| Compute module | NVIDIA Jetson Orin Nano |
| OS | Ubuntu 24.04 (L4T 39.2 / JetPack 7.2 GA) |
| DeepStream | NVIDIA DeepStream 9.1 for Jetson |
| Python | Python 3.12 |
| Pipeline API | `pyservicemaker` bundled with the DeepStream SDK |
| Inference runtime | TensorRT, built into the installed JetPack/DeepStream stack |
| Detector format | YOLOv8 ONNX with DeepStream-Yolo custom parser |
| Precision | FP16 (`network-mode=2`) |
| ExG/crop exclusion | CUDA/C++ `nvdsvideotemplate` custom library (no PyTorch runtime) |

Install a matching JetPack and DeepStream release on the Jetson. DeepStream
9.1 supports Jetson Orin with JetPack 7.2/Ubuntu 24.04. `pyservicemaker` comes
with the DeepStream SDK; do not attempt to substitute an unrelated PyPI package.

PyTorch and kornia are not used or imported by this project at run time. The
custom CUDA library computes the ExG mask and excludes all YOLO crop/chilli
boxes before frames leave DeepStream. `detection_worldcoord.py` uses OpenCV
connected-components for weed boxes and requires that CUDA-produced mask. The
only Python CUDA dependency is `cuda-bindings`; it performs one final device-to-
host copy of the RGB frame for the GUI preview and is not used for inference or
image processing.

## Camera configuration

The intended camera is a **Waveshare dual-IMX CSI camera module** connected to
the Jetson Orin Nano. The application uses one sensor at a time; select the
active CSI input with `sensor-id` (`0` or `1`).

Deployment capture target:

| Setting | Value |
| --- | --- |
| Interface | Jetson CSI / MIPI camera interface |
| Source element | `nvarguscamerasrc` |
| Resolution | 1920 \D7 1080 |
| Frame rate | 30 FPS |
| Native capture format | NV12 in NVMM memory |
| First camera | `sensor-id=0` |
| Second camera | `sensor-id=1` |

Before running the application, validate each connected camera independently:

```bash
gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! \
  'video/x-raw(memory:NVMM),width=1920,height=1080,format=NV12,framerate=30/1' ! \
  nvvidconv ! xvimagesink sync=false
```

Repeat with `sensor-id=1` for the other camera. The code deliberately sets
`flip-method=0`, so DeepStream preserves the source orientation. Do not add a
rotation/flip after inference: that would make the displayed frame, crop boxes,
and world-coordinate calibration disagree.

## Project files required on the Jetson

Place these files together in the project directory (for example,
`/home/archi/Desktop/BTP-I`):

```text
deepstream_pipeline.py
weed_sprayer_gui.py
detection_worldcoord.py
sprayer_control.py
config_infer_primary_yolo.txt

bestM.onnx
bestM.onnx.data                 # required when ONNX uses external weights
labels.txt
libnvdsinfer_custom_impl_Yolo.so
weed_mask_template/             # CUDA custom library source + Makefile
libweed_mask_template.so         # build on this Jetson; required at run time
model_b1_gpu0_fp16.engine       # created automatically after the first build
```

`bestM.onnx.data` must remain beside `bestM.onnx`. Newer ONNX exports can keep
weights in this external data file; without it TensorRT cannot parse the model.

## Exporting the YOLOv8 model

Use the matching DeepStream-Yolo exporter for YOLOv8, not the YOLO26 exporter:

```bash
git clone https://github.com/marcoslucianops/DeepStream-Yolo.git
git clone https://github.com/ultralytics/ultralytics.git

cd ~/Desktop/ultralytics
python3 -m pip install -e . onnx onnxslim onnxscript
cp ~/DeepStream-Yolo/utils/export_yoloV8.py .
cp /home/archi/Desktop/BTP-I/bestM.pt .
python3 export_yoloV8.py -w bestM.pt --simplify --opset 18
```

Copy every generated ONNX artifact into the project directory:

```bash
cp bestM.onnx bestM.onnx.data labels.txt /home/archi/Desktop/BTP-I/
```

If the exporter does not create `bestM.onnx.data`, copy only the files that
exist. Validate an exported model before deployment:

```bash
python3 -c "import onnx; m=onnx.load('bestM.onnx'); onnx.checker.check_model(m); print('Valid ONNX; opset:', m.opset_import[0].version)"
```

## DeepStream-Yolo parser library

Build the parser on the same Jetson that will run the application. This keeps
the CUDA/TensorRT binary compatible with the installed JetPack release.

```bash
cd ~/DeepStream-Yolo
make -C nvdsinfer_custom_impl_Yolo clean
make -C nvdsinfer_custom_impl_Yolo
cp nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so \
  /home/archi/Desktop/BTP-I/
```

The labels file must contain the YOLO detector's class names, one per line. In
this project it represents the crop/plant classes used for crop exclusion, not
the final ExG-derived weed boxes.


## CUDA ExG and native weed-box metadata libraries

Build the project-local `nvdsvideotemplate` library on the same Jetson/DeepStream
installation that runs the pipeline:

```bash
cd /home/archi/Desktop/BTP-I
make -C weed_mask_template
```

Build the weed metadata library on the Jetson:

```bash
make -C weed_bbox_meta DS_ROOT=/opt/nvidia/deepstream/deepstream
```

The build needs the matching DeepStream 9.1 headers/libraries, CUDA `nvcc`,
GStreamer development files, and OpenCV 4 development files. The compiled
`libweed_bbox_meta.so` is written inside `weed_bbox_meta/` and loaded by the
pipeline from there.

The pipeline uses:

```text
nvstreammux ! nvinfer ! nvvideoconvert(RGBA) ! weed_bbox_meta(CUDA ExG + boxes) \
            ! nvtracker(NvSORT) ! nvvideoconvert(RGB) ! appsink
```

The custom CUDA kernel reads YOLO `NvDsObjectMeta`, excludes crop/chilli
pixels, and creates an ExG mask. C++ connected components converts its host
copy into class-1 `NvDsObjectMeta` before NvSORT; chilli detections are class
0. NvSORT therefore returns persistent IDs for both classes, and Python
consumes weed boxes from metadata instead of extracting them after appsink.
The ExG threshold is a construction-time property; recreate the camera after
changing it.

## Inference configuration and engine cache

`config_infer_primary_yolo.txt` points to the ONNX model, labels, and parser.
The engine filename must stay aligned with the DeepStream-Yolo custom engine
creator:

```ini
onnx-file=bestM.onnx
model-engine-file=model_b1_gpu0_fp16.engine
labelfile-path=labels.txt
custom-lib-path=libnvdsinfer_custom_impl_Yolo.so
network-mode=2
```

The first run can take several minutes while TensorRT creates the FP16 engine.
On success the log must include:

```text
serialize cuda engine to file: .../model_b1_gpu0_fp16.engine successfully
```

Later starts should load this same file quickly. If DeepStream tries to load a
different engine name (such as `bestM.onnx_b1_gpu0_fp16.engine`) while saving
`model_b1_gpu0_fp16.engine`, correct the configuration above; otherwise it will
rebuild every time. Do not copy this engine to a different Jetson/GPU.

## DeepStream frame handoff requirements

The final conversion stage in `deepstream_pipeline.py` is intentionally set to
RGB and GPU compute:

```python
self.pipeline.add("nvvideoconvert", "convert", {
    "gpu-id": gpu_id,
    "compute-hw": 1,
    "nvbuf-memory-type": 2,
    "flip-method": 0,
})
self.pipeline.add("capsfilter", "output-caps", {
    "caps": "video/x-raw(memory:NVMM),format=RGB",
})
```

This is required because Service Maker's `BufferRetriever.extract()` supplies
RGB tensors for the Python receiver. `compute-hw=1` forces CUDA conversion;
Jetson VIC cannot perform the required NV12-to-RGB/BGR transform and will fail
with `RGB/BGR Format transformation is not supported by VIC`.

## Run

Activate the project environment, change to the project directory, and launch:

```bash
source btp_env/bin/activate
cd /home/archi/Desktop/BTP-I
python3 weed_sprayer_gui.py
```

DeepStream is the only backend now, so there's no toggle to enable -- the
Start button is simply disabled with an explanation if `pyservicemaker`
isn't importable. Use a numeric source value for a CSI camera (for example
`0`) or provide a local video path / RTSP URI.

## Live weed-parameter tuning

Use the browser calibration view to choose the ExG and bbox values against
the live camera image and the YOLO chilli detections. It works headlessly, so
you can start it over SSH and open the page from another device:

```bash
python3 weed_tuning.py --source 0 --host 0.0.0.0 --port 5001
```

Browse to `http://<jetson-ip>:5001` on the same network. The page shows the
camera view with chilli YOLO boxes in green, weed boxes in red, and the ExG
mask alongside it. Browser sliders adjust:

For access through an SSH connection, forward the port from your computer:
`ssh -L 5001:localhost:5001 <user>@<jetson-ip>`, then open
`http://localhost:5001` locally.

| Control | Effect |
| --- | --- |
| ExG threshold | Pixel passes the vegetation mask when `2G - R - B` is above this value. |
| Min weed pixel area | Rejects small connected regions. |
| Cluster split area | Runs watershed splitting on a connected region above this area. |
| Peak separation px | Controls how close watershed seeds may be before they merge. |
| Max bbox area | Rejects an extracted box larger than this area; zero disables the limit. |

Click **Save profile** to write the selected values to
`weed_tuning_profile.json`. The DeepStream camera loads that profile at the
next start and passes the ExG, area, cluster, and separation values into the
native weed metadata stage before NvSORT. Restart `web_gui.py` or
`weed_sprayer_gui.py` after tuning. The tuner computes weed boxes with the
OpenCV fallback while displaying the live DeepStream camera and YOLO chilli
boxes; its preview FPS is not a measurement of native pipeline performance.
The main `web_gui.py` no longer exposes weed tuning sliders; use this page to
calibrate the profile.

## Performance and world-coordinate cost

The GUI's **Pipeline FPS** is the EMA of unique frames that complete the active
path: DeepStream decode/mux/YOLO/CUDA ExG, native weed metadata, NvSORT,
pixel-to-world conversion, sprayer hand-off, and GUI queueing. The periodic log
also reports DeepStream appsink FPS and post-processing time separately.

Pixel-to-world conversion is only a few 3×3 matrix/vector operations per final
weed box. The measured cost on this Jetson is **0.032 ms per weed** (about
**1.6 ms for 50 weeds**). The native weed metadata stage uses connected
components and watershed cluster splitting.

## Troubleshooting

| Symptom | Cause | Resolution |
| --- | --- | --- |
| `DeepStream YOLO assets are missing` | Deployment artifacts were not copied | Put ONNX, optional ONNX `.data`, labels, and custom `.so` beside the config. |
| `Failed to open ... bestM.onnx.data` | ONNX external weights file is absent | Copy `bestM.onnx.data` beside `bestM.onnx`. |
| Engine rebuilds every start | Configured engine name differs from saved engine name | Use `model-engine-file=model_b1_gpu0_fp16.engine`. |
| `Only RGB format is supported for being extracted as a tensor` | Appsink receives RGBA | Use RGB output caps and `cv2.COLOR_RGB2BGR`. |
| `RGB/BGR Format transformation is not supported by VIC` | VIC is attempting RGB conversion | Set `compute-hw=1` and `nvbuf-memory-type=2` on `nvvideoconvert`. |
| No preview during first launch | TensorRT is still compiling | Wait for the engine serialization success message. |
| Camera is unavailable | CSI device, cable, or sensor selection problem | Test with the GStreamer command above and try `sensor-id=0` / `1`. |
| ExG mask missing | Custom library was not built for this Jetson/DeepStream SDK | Run `make -C weed_mask_template` and restart the application. |
| `DeepStream weed-bbox library is missing` | Native weed metadata stage was not built | Run `make -C weed_bbox_meta` on the Jetson and restart the application. |

## Updated Pipeline Architecture and Optimizations

> **Important**: The previous pipeline implementation remains fully preserved and functional (accessible via `use_updated=False` or `_build_pipeline_legacy`). The updated pipeline is the new active default (`use_updated=True`), specifically addressing the identified DeepStream bottlenecks, sensor rotation, and color formatting issues.

```text
CSI camera / video / RTSP
        |
        v
DeepStream Service Maker
  decode -> nvstreammux -> TensorRT YOLO crop boxes (class 0)
                                  |
                  nvof -> CUDA ExG mask with crop exclusion
                                  |
                     C++ weed_bbox_meta template
                  mask boxes become weed objects (class 1)
                                  |
                      native nvtracker / NvSORT
                      chilli IDs + weed IDs
                                  |
                RGB conversion -> appsink -> sprayer
```

`weed_bbox_meta` runs a CUDA ExG kernel, excludes YOLO crop boxes, and adds
weed objects before NvSORT. Its connected-components step receives the mask
from CUDA and adds class-1 `NvDsObjectMeta`. The live application consumes
tracked weed boxes and IDs instead of extracting boxes with Python/OpenCV
after `appsink`.

### Summary of Changes and Explicit Reasons

#### 1. GPU Batch World-Coordinate Homography (Replacing CPU-bound Per-Box Math)
* **Problem**: In the previous pipeline, `detection_worldcoord.py` iterated through every weed detection in a Python `for` loop, calling `pixel_to_world(u, v, K, Rt)` individually. Each iteration allocated NumPy vectors, computed 3D ray projections, and performed ray-plane intersections in pure Python under the GIL. When dense weed patches produced dozens or hundreds of detections, Python function call overhead and host allocations stalled the pipeline, destroying throughput even though TensorRT was running at 30 FPS.
* **Changes Applied**:
  * In `detection_worldcoord.py`, the ground-plane projection matrix ($Z=0$) is precomputed once at startup as a single $3 \times 3$ homography matrix:
    $$M = K \begin{bmatrix} r_1 & r_2 & t \end{bmatrix}, \quad H_{p2w} = M^{-1}$$
  * Created `batch_world_coords(centroids_px)` which transforms all $N$ bbox centroids simultaneously using a single matrix multiplication:
    $$\begin{bmatrix} X \\ Y \\ 1 \end{bmatrix} \sim H_{p2w} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}$$
  * Implemented using CuPy for direct GPU execution without PyTorch (satisfying the project's zero-PyTorch constraint), with an automatic vectorized NumPy fallback if CuPy is not yet compiled.
  * In `process_frame()`, all detections in a frame are batched into an $N \times 2$ matrix, reducing $N$ CPU function calls to a single kernel/vectorized operation.

#### 2. Camera Distance Traveled via Hardware Optical Flow (`Gst-nvof`)
* **Problem**: The rover odometry in `sprayer_control.py` was previously a fixed mock (constant 2 km/h). Tracking true rover displacement using traditional CPU feature tracking (e.g., OpenCV Lucas-Kanade or ORB) would cause severe CPU bottlenecks and device-to-host memory thrashing at 1080p.
* **Changes Applied**:
  * DeepStream's native hardware Optical Flow plugin (`nvof`) was added directly into the pipeline (`_build_pipeline_updated` in `deepstream_pipeline.py`).
  * `nvof` runs on NVMM NV12 surfaces entirely on dedicated Jetson hardware, producing dense motion vector fields with zero CPU copies.
  * `weed_mask_template.cu` intercepts `NVDS_OPTICAL_FLOW_META` attached to the frame metadata, averages the flow vectors across the frame grid, and exports `get_latest_optical_flow(float *fx, float *fy)` via C ABI.
  * `DeepStreamCamera.get_optical_flow()` queries these motion vectors via ctypes and feeds them into `SprayerController.update_optical_flow(flow_x, flow_y)`.
  * `RoverState.update_displacement()` integrates the scaled motion vectors into real-world cumulative distance traveled (meters) and rover position $(x, y)$.

#### 3. Rotated Display Correction (`flip-method=1`)
* **Problem**: Due to physical CSI camera sensor orientation on the rover mount, the preview and coordinate space were rotated 90° counterclockwise.
* **Changes Applied**:
  * In `deepstream_pipeline.py`, the post-template `nvvideoconvert` element now accepts a configurable `flip-method` (default `flip_method=1`, which applies a 90° clockwise rotation to restore upright orientation).
  * Added `flip_var` control slider and parameter persistence in `weed_sprayer_gui.py` (`0=none`, `1=90° CW`, `2=180°`, `3=90° CCW`).

#### 4. Resolution of Pink / Magenta ExG Output and Channel Artifacts
* **Problem**: The ExG output had an unintended pink/magenta cast across the frame. In DeepStream pipelines, this is caused by:
  1. Partial channel writes leaving uninitialized or stale R and B channels when modifying RGBA NVMM surfaces.
  2. Missing capsfilters allowing format ambiguity between RGBA and BGRx across custom in-place videotemplate elements.
* **Changes Applied**:
  * In `weed_mask_template.cu`, `exg_mask_rgba` was updated to explicitly write all four channels ($R, G, B, A=255$). In mask mode (mode 1), $R=G=B=\text{val}$ and $A=255$ are written explicitly for every pixel, eliminating uninitialized or stale channel values.
  * Added configurable `exg-mode` (0 = legacy carrier, 1 = explicit binary mask with matching RGB, 2 = balanced green overlay).
  * In `deepstream_pipeline.py`, strict `capsfilter` elements (`video/x-raw(memory:NVMM),format=RGBA`) were placed both immediately before and immediately after `nvdsvideotemplate` to guarantee strict RGBA negotiation through the CUDA transform.

#### 5. Preservation of Legacy Pipeline
* **Design Decision**: The original baseline pipeline has been retained completely intact in `deepstream_pipeline.py` as `_build_pipeline_legacy`. Existing workflows can select `DeepStreamCamera(use_updated=False)` at any time to run the original baseline configuration without regressions.

---

## References

- [NVIDIA DeepStream 9.1 installation and Jetson compatibility](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Installation.html)
- [NVIDIA DeepStream Service Maker for Python](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_service_maker_python.html)
- [Gst-nvdsvideotemplate custom processing](https://docs.nvidia.com/metropolis/deepstream/9.1/text/DS_plugin_gst-nvdsvideotemplate.html)
- [DeepStream metadata APIs](https://docs.nvidia.com/metropolis/deepstream/9.1/sdk-api/group__metadata__structures.html)
- [Gst-nvtracker and NvSORT](https://docs.nvidia.com/metropolis/deepstream/text/DS_plugin_gst-nvtracker.html)
- [DeepStream-Yolo custom parser project](https://github.com/marcoslucianops/DeepStream-Yolo)
- [Waveshare Jetson Orin Nano dual-camera preview example](https://www.waveshare.com/wiki/Jetson_Orin_Nano)

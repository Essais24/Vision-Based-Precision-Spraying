# Weed Sprayer — Jetson DeepStream Deployment

Weed detection and precision spraying for an NVIDIA Jetson Orin Nano. The
runtime uses NVIDIA DeepStream 9.1, TensorRT YOLOv8 detection, a CSI camera,
and an ESP32-controlled sprayer over UART. A browser UI (`web_gui.py`) and a
desktop UI (`weed_sprayer_gui.py`) use the same backend.

> **Safety:** this software can operate pumps, valves, and servos. Test with
> the ESP32 disconnected or with the fluid system disabled first. Do not make
> the web UI publicly reachable.

## What to upload to GitHub

Commit the source, configuration, build recipes, labels, Python dependency
list, and documentation:

```text
*.py
weed_bbox_meta/weed_bbox_meta.cu
weed_bbox_meta/Makefile
weed_mask_template/weed_mask_template.cu
weed_mask_template/Makefile
config_infer_primary_yolo.txt
config_infer_weed_yolov8.txt
config_tracker_NvSORT.yml
labels.txt
labels_weed.txt
requirements.txt
weed_tuning_profile.json
weed-sprayer-web.service
README.md
README_actuation.md
```

Also commit `camera_calibration.npz` and/or `camera_calibration/` **only when
the replacement machine uses the same camera, lens, resolution, orientation,
and physical mounting**. Otherwise, do not reuse them: capture new images and
run `calibrate_camera.py` on the deployed rig.

### Models: use Git LFS or a release asset

The ONNX model weights are too large for normal GitHub Git objects:

```text
bestM.onnx + bestM.onnx.data          # crop/chilli detector
best_weed.onnx + best_weed.onnx.data  # weed detector, used by the web UI
```

Keep each `.onnx` file beside its matching `.onnx.data` file. Upload them
with Git LFS (recommended) or attach them to a private GitHub Release/cloud
artifact, then copy them into the project directory after cloning. `.pt`
files (`bestM.pt`, `best_weed.pt`) are training/export sources; keep them in
LFS or a release only if someone needs to re-export the ONNX models.

Example LFS setup:

```bash
git lfs install
git lfs track '*.onnx' '*.onnx.data' '*.pt'
git add .gitattributes
```

## Do not upload these generated or machine-specific files

The included `.gitignore` excludes these items:

| Item | Why |
| --- | --- |
| `btp_env/` | A 5+ GB local Python virtual environment; recreate it. |
| `*.engine` | TensorRT engine caches are tied to the Jetson GPU, TensorRT, CUDA, and model. |
| `*.so` | Native DeepStream/CUDA parser libraries must match the target JetPack/DeepStream installation. |
| `calib_images/`, `calibration_images/` | Local calibration captures; keep them separately if desired. |
| `runs/`, logs, detection CSV/JSONL output | Generated runtime data. |
| `report_output/`, `*.whl` | Generated reports and downloaded package artifacts. |

`libnvdsinfer_custom_impl_Yolo.so`, `libweed_mask_template.so`,
`weed_bbox_meta/libweed_bbox_meta.so`, `model_b1_gpu0_fp16.engine`, and
`model_weed_b1_gpu0_fp16.engine` are therefore deliberately not versioned.

## Fresh-machine requirements

The production target is a **Jetson Orin Nano** running:

| Component | Required version / role |
| --- | --- |
| OS / platform | JetPack 7.2 on Ubuntu 24.04 (L4T 39.2) |
| DeepStream | DeepStream 9.1 for Jetson, including `pyservicemaker` |
| CUDA / TensorRT | Installed by the matching JetPack + DeepStream stack |
| Python | Python 3.12 |
| Hardware | CSI camera and optional ESP32 UART controller |

Install matching NVIDIA JetPack and DeepStream before setting up this repo.
After it is installed this must succeed:

```bash
python3 -c "import pyservicemaker; print('DeepStream Service Maker OK')"
```

Install build dependencies for the two local CUDA plugins:

```bash
sudo apt update
sudo apt install -y build-essential pkg-config gstreamer1.0-tools \
  libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libopencv-dev
```

The DeepStream-Yolo parser is an external dependency. Clone it and build its
custom parser on the *same Jetson*:

```bash
git clone https://github.com/marcoslucianops/DeepStream-Yolo.git ~/DeepStream-Yolo
make -C ~/DeepStream-Yolo/nvdsinfer_custom_impl_Yolo
cp ~/DeepStream-Yolo/nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so .
```

## Setup on a different machine

From a clone of this repository:

```bash
python3 -m venv btp_env
source btp_env/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` includes the GUI/pipeline basics, Flask for the web
interface, and pyserial for the ESP32 link. `tkinter` is normally supplied by
the OS; install `python3-tk` if the desktop UI cannot import it.

Retrieve the model artifacts (from Git LFS or the chosen release location) and
place them in the project root. Then build both project-local CUDA libraries:

```bash
make -C weed_mask_template
make -C weed_bbox_meta DS_ROOT=/opt/nvidia/deepstream/deepstream
```

Finally start an application. The first run generates the TensorRT `.engine`
files and may take a few minutes; later runs reuse the cache.

```bash
# Browser interface: open http://<jetson-ip>:8000
python web_gui.py

# Or the local desktop UI
python weed_sprayer_gui.py
```

The web UI enables the weed YOLO model, so it requires both model pairs.
The desktop UI defaults to the crop YOLO plus native ExG path and requires the
primary model plus the CUDA libraries.

## Machine-specific configuration checklist

Before spraying on a new device:

1. Check the CSI camera using `gst-launch-1.0 nvarguscamerasrc ...`; select
   the correct sensor in the UI (normally `0` or `1`).
2. Calibrate the installed camera/mount and verify the camera height and axis
   flags in `detection_worldcoord.py`.
3. Confirm model/label pairs match their configs: `bestM.onnx` with
   `labels.txt`, and `best_weed.onnx` with `labels_weed.txt`.
4. Verify the ESP32 UART port and wiring in `spray_actuation.py`. Add the
   service user to `dialout` if serial access is denied.
5. Test detection, servo aim, valve commands, and pump commands without
   chemical or pressure first.

## Optional web service

Edit `weed-sprayer-web.service` before installing it: update `User=`,
`WorkingDirectory=`, and the virtual-environment Python path to match the new
machine. Then install it:

```bash
sudo cp weed-sprayer-web.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now weed-sprayer-web.service
sudo systemctl status weed-sprayer-web.service
```

Keep port 8000 on a trusted local network or behind a VPN/authenticated reverse
proxy. The application itself has no login and can command physical hardware.

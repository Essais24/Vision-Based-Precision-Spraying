"""DeepStream 9.1 / Jetson Orin capture and YOLO inference backend.

This module uses DeepStream Service Maker (``pyservicemaker``), the supported
Python API for DeepStream 9.1. It provides both:
1. Legacy Pipeline: The original baseline pipeline (preserved untouched).
2. Updated Pipeline: The optimized pipeline resolving:
   - On-chip hardware Optical Flow (Gst-nvof) for zero-CPU distance tracking.
   - Optional display rotation via configurable `flip-method`.
   - Strict matching RGBA NVMM caps around nvdsvideotemplate to prevent pink ExG tint.

Target platform
---------------
DeepStream 9.1 officially supports Jetson Orin Nano, NX and AGX.  Its Jetson
package is built for JetPack 7.2 / Ubuntu 24.04 / Python 3.12.
"""

import os
import ctypes
import configparser
import threading
import time

import cv2
import numpy as np

DEEPSTREAM_AVAILABLE = True
DEEPSTREAM_UNAVAILABLE_REASON = ""
try:
    from pyservicemaker import BufferRetriever, Pipeline, Receiver
except Exception as exc:
    DEEPSTREAM_AVAILABLE = False
    DEEPSTREAM_UNAVAILABLE_REASON = (
        f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
    )


class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int32), ("device_id", ctypes.c_int32)]


class _DLDataType(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8),
                ("lanes", ctypes.c_uint16)]


class _DLTensor(ctypes.Structure):
    _fields_ = [("data", ctypes.c_void_p), ("device", _DLDevice),
                ("ndim", ctypes.c_int32), ("dtype", _DLDataType),
                ("shape", ctypes.POINTER(ctypes.c_int64)),
                ("strides", ctypes.POINTER(ctypes.c_int64)),
                ("byte_offset", ctypes.c_uint64)]


class _DLManagedTensor(ctypes.Structure):
    pass


_DLManagedTensor._fields_ = [
    ("dl_tensor", _DLTensor), ("manager_ctx", ctypes.c_void_p),
    ("deleter", ctypes.CFUNCTYPE(None, ctypes.POINTER(_DLManagedTensor))),
]


def _service_maker_tensor_to_numpy(tensor):
    """Copy Service Maker's CUDA DLPack RGB tensor without PyTorch/CuPy."""
    try:
        from cuda.bindings import driver as cuda_driver
    except ImportError as exc:
        raise RuntimeError("CUDA Python bindings are required for GUI frame copy") from exc

    capsule = tensor.__dlpack__(None)
    get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
    get_pointer.argtypes = (ctypes.py_object, ctypes.c_char_p)
    get_pointer.restype = ctypes.c_void_p
    managed = ctypes.cast(get_pointer(capsule, b"dltensor"),
                           ctypes.POINTER(_DLManagedTensor))
    dl_tensor = managed.contents.dl_tensor
    if dl_tensor.device.device_type != 2 or dl_tensor.dtype.code != 1 or dl_tensor.dtype.bits != 8:
        raise RuntimeError("Expected a CUDA uint8 Service Maker RGB tensor")
    shape = tuple(dl_tensor.shape[i] for i in range(dl_tensor.ndim))
    host = np.empty(shape, dtype=np.uint8)
    result = cuda_driver.cuMemcpyDtoH(
        host.ctypes.data,
        cuda_driver.CUdeviceptr(dl_tensor.data + dl_tensor.byte_offset),
        host.nbytes,
    )
    if result[0] != cuda_driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA frame copy failed: {result[0]}")
    # The PyCapsule owns the managed tensor and invokes its deleter when it
    # leaves scope. Calling it here would double-free Service Maker storage.
    return host


def _is_camera_source(source):
    """``int``/``None`` means a CSI sensor id; all other values are URIs."""
    return source is None or isinstance(source, int)


def _to_uri(path_or_uri):
    """Return an existing URI unchanged or convert a local path to file://."""
    if "://" in path_or_uri:
        return path_or_uri
    return "file://" + os.path.abspath(path_or_uri).replace("\\", "/")



def _probe_video_layout(source, fallback_width, fallback_height):
    """Return encoded dimensions and MP4 display rotation separately.

    OpenCV enables ``CAP_PROP_ORIENTATION_AUTO`` by default. That makes its
    reported dimensions describe the display image while ``nvurisrcbin`` still
    produces the encoded image. Probe without auto-orientation, then rotate
    explicitly later in the pipeline.
    """
    probe = cv2.VideoCapture(source)
    if not probe.isOpened():
        return fallback_width, fallback_height, 0
    try:
        orientation_auto = getattr(cv2, "CAP_PROP_ORIENTATION_AUTO", None)
        orientation_meta = getattr(cv2, "CAP_PROP_ORIENTATION_META", None)
        if orientation_auto is not None:
            probe.set(orientation_auto, 0)
        width = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH)) or fallback_width
        height = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT)) or fallback_height
        rotation = int(round(probe.get(orientation_meta))) % 360 if orientation_meta is not None else 0
        return width, height, rotation if rotation in (0, 90, 180, 270) else 0
    finally:
        probe.release()


def _rotation_to_flip_method(rotation):
    """Map clockwise MP4 display rotation to nvvideoconvert flip-method."""
    return {0: 0, 90: 3, 180: 2, 270: 1}[rotation]


if DEEPSTREAM_AVAILABLE:
    class _FrameRetriever(BufferRetriever):
        """Copy one appsink frame and its inference metadata for the consumer."""

        def __init__(self, camera):
            super().__init__()
            self._camera = camera

        def consume(self, buffer):
            try:
                plants = []
                plant_ids = []
                weeds = []
                weed_ids = []
                for frame_meta in buffer.batch_meta.frame_items:
                    for object_meta in frame_meta.object_items:
                        rect = object_meta.rect_params
                        box = [
                            int(rect.left),
                            int(rect.top),
                            int(rect.left + rect.width),
                            int(rect.top + rect.height),
                        ]
                        component_id = int(getattr(object_meta, "unique_component_id", 0))
                        # The normal production chain contains two full-frame
                        # GIEs.  Both models use class 0, so class_id alone is
                        # not sufficient to identify the model that created an
                        # object.  gie-unique-id is retained in object metadata.
                        if component_id == self._camera.weed_gie_id:
                            weeds.append(box)
                            track_id = int(object_meta.object_id)
                            weed_ids.append(-1 if track_id == (1 << 64) - 1 else track_id)
                        elif component_id == self._camera.primary_gie_id:
                            plants.append(box)
                            track_id = int(object_meta.object_id)
                            plant_ids.append(-1 if track_id == (1 << 64) - 1 else track_id)
                        # Compatibility with legacy single-GIE + CUDA ExG
                        # pipelines, where ExG boxes use class 1.
                        elif not self._camera.enable_weed_yolo:
                            if int(object_meta.class_id) == 1:
                                weeds.append(box)
                                weed_ids.append(-1)
                            else:
                                plants.append(box)
                                plant_ids.append(-1)

                # The GPU-to-CPU copy for preview uses CUDA Python driver
                # bindings, not PyTorch or another tensor framework.
                tensor = buffer.extract(0).clone()
                frame_rgba = _service_maker_tensor_to_numpy(tensor)
                # Service Maker's Buffer.extract() currently supports RGB
                # tensors only.  Keep the caps below in sync with this
                # conversion; RGBA produces an invalid DLPack capsule.
                frame_bgr = cv2.cvtColor(frame_rgba, cv2.COLOR_RGB2BGR)

                # Optional ExG overlay is separate from the native weed boxes.
                if not self._camera.enable_weed_mask:
                    weed_mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
                # In mode 1 (clean binary mask), weed pixels are R=G=B=255.
                # In mode 0 (legacy), weed pixels are marked pure magenta (255, 0, 255).
                elif self._camera.exg_mode == 1:
                    weed_mask = cv2.inRange(frame_bgr, np.array((250, 250, 250), np.uint8),
                                            np.array((255, 255, 255), np.uint8))
                else:
                    weed_mask = cv2.inRange(frame_bgr, np.array((255, 0, 255), np.uint8),
                                            np.array((255, 0, 255), np.uint8))

                self._camera._store_latest(
                    frame_bgr, self._camera._transform_boxes_for_output(plants), weed_mask,
                    plant_ids=plant_ids,
                    weed_boxes=self._camera._transform_boxes_for_output(weeds),
                    weed_ids=weed_ids,
                )
                return 1
            except Exception as exc:
                print(f"[DeepStream][ERROR] Could not consume frame: {exc}")
                return 0


class DeepStreamCamera:
    """Run a DeepStream 9.1 pipeline and expose its latest frame, boxes, and flow."""

    def __init__(self, source=0, width=1920, height=1080, framerate=30,
                 wbmode=0, config_path="config_infer_primary_yolo.txt",
                 gpu_id=0, loop=True, exg_threshold=20, exg_mode=0,
                 weed_template_lib="libweed_mask_template.so",
                 use_updated=True, flip_method=0, enable_optical_flow=True,
                 enable_weed_mask=True,
                 tracker_lib="/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so",
                 tracker_config="config_tracker_NvSORT.yml",
                 weed_bbox_lib="weed_bbox_meta/libweed_bbox_meta.so",
                 weed_tuning_profile="weed_tuning_profile.json",
                 weed_min_area=3000, weed_max_box_area=100000,
                 weed_cluster_area=30000, weed_separation_px=9,
                 enable_native_weed_bboxes=True,
                 weed_config_path="config_infer_weed_yolov8.txt",
                 enable_weed_yolo=False):
        if not DEEPSTREAM_AVAILABLE:
            raise RuntimeError(
                "DeepStream 9.1 Service Maker is unavailable. Install the "
                "matching DeepStream SDK on the Jetson, then verify with: "
                "python3 -c \"import pyservicemaker\""
            )

        self.is_camera = _is_camera_source(source)
        self.loop = loop and not self.is_camera
        self.config_path = os.path.abspath(config_path)
        self.weed_config_path = os.path.abspath(weed_config_path)
        self.enable_weed_yolo = bool(enable_weed_yolo)
        self.weed_template_lib = os.path.abspath(weed_template_lib)
        tuning_path = os.path.abspath(weed_tuning_profile)
        if os.path.isfile(tuning_path):
            try:
                import json
                with open(tuning_path, "r", encoding="utf-8") as profile_file:
                    tuned = json.load(profile_file)
                exg_threshold = tuned.get("exg_threshold", exg_threshold)
                weed_min_area = tuned.get("min_area", weed_min_area)
                weed_max_box_area = tuned.get("max_box_area", weed_max_box_area)
                weed_cluster_area = tuned.get("cluster_area_threshold", weed_cluster_area)
                weed_separation_px = tuned.get("separation_px", weed_separation_px)
            except (OSError, ValueError, TypeError, AttributeError) as exc:
                print(f"[DeepStream][WARN] Ignoring invalid weed tuning profile: {exc}")
        self.exg_threshold = int(exg_threshold)
        self.weed_min_area = max(1, int(weed_min_area))
        self.weed_max_box_area = max(0, int(weed_max_box_area))
        self.weed_cluster_area = max(1, int(weed_cluster_area))
        self.weed_separation_px = max(1, int(weed_separation_px))
        self.exg_mode = int(exg_mode)
        self.use_updated = bool(use_updated)
        self.flip_method = int(flip_method)
        self.display_rotation = 0
        self.enable_optical_flow = bool(enable_optical_flow)
        self.enable_weed_mask = bool(enable_weed_mask)
        self.tracker_lib = os.path.abspath(tracker_lib)
        self.tracker_config = os.path.abspath(tracker_config)
        self.weed_bbox_lib = os.path.abspath(weed_bbox_lib)
        self.enable_native_weed_bboxes = bool(enable_native_weed_bboxes)

        self.primary_gie_id = self._validate_model_assets(self.config_path, "chilli")
        self.weed_gie_id = None
        if self.enable_weed_yolo:
            self.weed_gie_id = self._validate_model_assets(self.weed_config_path, "weed")
            if self.weed_gie_id == self.primary_gie_id:
                raise ValueError("Chilli and weed GIE configs must use different gie-unique-id values")
        if self.enable_weed_mask and not os.path.isfile(self.weed_template_lib):
            raise FileNotFoundError(
                f"DeepStream weed-mask library is missing: {self.weed_template_lib}. "
                "Build it on the target Jetson with `make -C weed_mask_template`."
            )
        if self.enable_native_weed_bboxes and not os.path.isfile(self.weed_bbox_lib):
            raise FileNotFoundError(
                f"DeepStream weed-bbox library is missing: {self.weed_bbox_lib}. "
                "Build it with `make -C weed_bbox_meta` on the target Jetson."
            )
        self._custom_lib = None
        if self.enable_weed_mask:
            try:
                self._custom_lib = ctypes.CDLL(self.weed_template_lib)
            except Exception as exc:
                print(f"[DeepStream][WARN] Could not load ctypes custom lib: {exc}")

        self.width = width
        self.height = height
        self._lock = threading.Lock()
        self._latest_frame = None
        self._latest_plants = []
        self._latest_plant_ids = []
        self._latest_weed_boxes = []
        self._latest_weed_ids = []
        self._latest_weed_mask = None
        self._latest_flow = (0.0, 0.0)
        self._frame_sequence = 0
        self._fps = 0.0
        self._last_frame_time = None
        self._fps_alpha = 0.15
        self._running = False

        if not self.is_camera:
            # Probe the decoder's dimensions, not OpenCV's auto-rotated view.
            self.width, self.height, self.display_rotation = _probe_video_layout(
                source, width, height
            )

        # An explicit flip is for a physically rotated camera. Otherwise honor
        # a video container's display matrix exactly once.
        self.output_flip_method = self.flip_method
        if not self.is_camera and self.flip_method == 0:
            self.output_flip_method = _rotation_to_flip_method(self.display_rotation)
        elif not self.is_camera and self.display_rotation:
            print("[DeepStream][WARN] Ignoring video display rotation because "
                  f"an explicit flip_method={self.flip_method} was requested.")
        self.output_rotates_dims = self.output_flip_method in (1, 3, 5, 7)
        self.output_width = self.height if self.output_rotates_dims else self.width
        self.output_height = self.width if self.output_rotates_dims else self.height
        if not self.is_camera:
            print("[DeepStream][INFO] Video layout: "
                  f"encoded={self.width}x{self.height}, "
                  f"display-rotation={self.display_rotation}deg, "
                  f"output={self.output_width}x{self.output_height}")

        self.pipeline = Pipeline("weed-sprayer")
        self._build_pipeline(source, framerate, wbmode, self.config_path, gpu_id)
        self._retriever = _FrameRetriever(self)  # retain the Python callback
        self.pipeline.attach(
            "sink", Receiver("frame-receiver", self._retriever), tips="new-sample"
        )

    @staticmethod
    def _validate_model_assets(config_path, model_name):
        """Fail clearly before GStreamer starts if the YOLO deployment is incomplete."""
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"DeepStream {model_name} inference config not found: {config_path}")

        config = configparser.ConfigParser(interpolation=None)
        config.read(config_path)
        properties = config["property"] if config.has_section("property") else {}
        config_dir = os.path.dirname(config_path)
        required = ("onnx-file", "labelfile-path", "custom-lib-path")
        missing = []
        for key in required:
            value = properties.get(key)
            if not value:
                missing.append(f"{key} is not configured")
                continue
            asset = value if os.path.isabs(value) else os.path.join(config_dir, value)
            if not os.path.isfile(asset):
                missing.append(asset)
        if missing:
            raise FileNotFoundError(
                f"DeepStream {model_name} YOLO assets are missing. Missing: "
                + ", ".join(missing)
            )
        try:
            return int(properties.get("gie-unique-id", "0"))
        except ValueError as exc:
            raise ValueError(f"Invalid gie-unique-id in {config_path}") from exc

    def _build_pipeline(self, source, framerate, wbmode, config_path, gpu_id):
        """Dispatch to updated pipeline or legacy pipeline based on use_updated flag."""
        if self.use_updated:
            self._build_pipeline_updated(source, framerate, wbmode, config_path, gpu_id)
        else:
            self._build_pipeline_legacy(source, framerate, wbmode, config_path, gpu_id)

    def _build_pipeline_legacy(self, source, framerate, wbmode, config_path, gpu_id):
        """Original baseline pipeline preserved for backward compatibility."""
        mux_props = {
            "batch-size": 1,
            "width": self.width,
            "height": self.height,
            "batched-push-timeout": 40000,
            "gpu-id": gpu_id,
            "live-source": self.is_camera or str(source).startswith("rtsp://"),
        }
        if self.loop:
            mux_props["nvbuf-memory-type"] = 4

        if self.is_camera:
            sensor_id = source or 0
            self.pipeline.add("nvarguscamerasrc", "source", {
                "sensor-id": sensor_id, "wbmode": wbmode,
            })
            self.pipeline.add("capsfilter", "source-caps", {
                "caps": (
                    "video/x-raw(memory:NVMM),format=NV12,"
                    f"width={self.width},height={self.height},framerate={framerate}/1"
                ),
            })
            self.pipeline.link("source", "source-caps")
            self.pipeline.add("nvstreammux", "mux", mux_props)
            self.pipeline.link(("source-caps", "mux"), ("", "sink_%u"))
        else:
            self.pipeline.add("nvurisrcbin", "source", {
                "uri": _to_uri(source), "file-loop": self.loop,
            })
            self.pipeline.add("nvstreammux", "mux", mux_props)
            self.pipeline.link(("source", "mux"), ("", "sink_%u"))

        self.pipeline.add("nvinfer", "infer", {
            "config-file-path": config_path, "gpu-id": gpu_id,
        })
        if self.enable_weed_yolo:
            self.pipeline.add("nvinfer", "weed-infer", {
                "config-file-path": self.weed_config_path, "gpu-id": gpu_id,
            })
        self.pipeline.add("nvtracker", "tracker", {
            "ll-lib-file": self.tracker_lib,
            "ll-config-file": self.tracker_config,
            "tracker-width": 640, "tracker-height": 384,
            "gpu-id": gpu_id, "display-tracking-id": 1,
            "operate-on-class-ids": "0;1",
        })
        self.pipeline.add("nvvideoconvert", "to-rgba", {
            "gpu-id": gpu_id, "compute-hw": 1, "nvbuf-memory-type": 2,
        })
        self.pipeline.add("capsfilter", "rgba-caps", {
            "caps": "video/x-raw(memory:NVMM),format=RGBA",
        })
        if self.enable_native_weed_bboxes:
            self.pipeline.add("nvdsvideotemplate", "weed-bbox-meta", {
                "gpu-id": gpu_id, "customlib-name": self.weed_bbox_lib,
                "customlib-props": (f"exg-threshold:{self.exg_threshold};min-area:{self.weed_min_area};"
                                    f"max-box-area:{self.weed_max_box_area};"
                                    f"cluster-area:{self.weed_cluster_area};"
                                    f"separation-px:{self.weed_separation_px}"),
            })
        if self.enable_weed_mask:
            self.pipeline.add("nvdsvideotemplate", "weed-mask", {
                "gpu-id": gpu_id, "customlib-name": self.weed_template_lib,
                "customlib-props": f"exg-threshold:{self.exg_threshold};exg-mode:{self.exg_mode}",
            })
        self.pipeline.add("nvvideoconvert", "convert", {
            "gpu-id": gpu_id,
            "compute-hw": 1,
            "nvbuf-memory-type": 2,
            "flip-method": self.output_flip_method,
        })
        self.pipeline.add("capsfilter", "output-caps", {
            "caps": (
                "video/x-raw(memory:NVMM),format=RGB,"
                f"width={self.output_width},height={self.output_height}"
            ),
        })
        self.pipeline.add("appsink", "sink", {
            "emit-signals": True, "max-buffers": 1, "drop": True, "sync": False,
        })
        elements = ["mux", "infer"]
        if self.enable_weed_yolo:
            elements.append("weed-infer")
        elements.extend(["to-rgba", "rgba-caps"])
        if self.enable_native_weed_bboxes:
            elements.append("weed-bbox-meta")
        elements.append("tracker")
        if self.enable_weed_mask:
            elements.append("weed-mask")
        elements.extend(["convert", "output-caps", "sink"])
        self.pipeline.link(*elements)

    def _build_pipeline_updated(self, source, framerate, wbmode, config_path, gpu_id):
        """Updated DeepStream pipeline with Optical Flow, rotation fix, and strict RGBA caps."""
        mux_props = {
            "batch-size": 1,
            "width": self.width,
            "height": self.height,
            "batched-push-timeout": 40000,
            "gpu-id": gpu_id,
            "live-source": self.is_camera or str(source).startswith("rtsp://"),
        }
        if self.loop:
            mux_props["nvbuf-memory-type"] = 4

        if self.is_camera:
            sensor_id = source or 0
            self.pipeline.add("nvarguscamerasrc", "source", {
                "sensor-id": sensor_id, "wbmode": wbmode,
            })
            self.pipeline.add("capsfilter", "source-caps", {
                "caps": (
                    "video/x-raw(memory:NVMM),format=NV12,"
                    f"width={self.width},height={self.height},framerate={framerate}/1"
                ),
            })
            self.pipeline.link("source", "source-caps")
            self.pipeline.add("nvstreammux", "mux", mux_props)
            self.pipeline.link(("source-caps", "mux"), ("", "sink_%u"))
        else:
            self.pipeline.add("nvurisrcbin", "source", {
                "uri": _to_uri(source), "file-loop": self.loop,
            })
            self.pipeline.add("nvstreammux", "mux", mux_props)
            self.pipeline.link(("source", "mux"), ("", "sink_%u"))

        # YOLO inference via TensorRT custom engine
        self.pipeline.add("nvinfer", "infer", {
            "config-file-path": config_path, "gpu-id": gpu_id,
        })
        if self.enable_weed_yolo:
            self.pipeline.add("nvinfer", "weed-infer", {
                "config-file-path": self.weed_config_path, "gpu-id": gpu_id,
            })

        elements = ["mux", "infer"]
        if self.enable_weed_yolo:
            elements.append("weed-infer")

        # Hardware Optical Flow Engine (Gst-nvof): runs on-chip in NVMM NV12
        if self.enable_optical_flow:
            self.pipeline.add("nvof", "nvof", {
                "gpu-id": gpu_id,
                "preset-level": 0,  # NV_OF_PERF_LEVEL_FAST
            })
            elements.append("nvof")

        # ExG custom library operates in RGBA. Convert NV12 to RGBA using CUDA GPU compute.
        self.pipeline.add("nvvideoconvert", "to-rgba", {
            "gpu-id": gpu_id, "compute-hw": 1, "nvbuf-memory-type": 2,
        })
        # Strict input caps: enforce RGBA on input of nvdsvideotemplate
        self.pipeline.add("capsfilter", "rgba-caps-in", {
            "caps": "video/x-raw(memory:NVMM),format=RGBA",
        })
        if self.enable_native_weed_bboxes:
            self.pipeline.add("nvdsvideotemplate", "weed-bbox-meta", {
                "gpu-id": gpu_id, "customlib-name": self.weed_bbox_lib,
                "customlib-props": (f"exg-threshold:{self.exg_threshold};min-area:{self.weed_min_area};"
                                    f"max-box-area:{self.weed_max_box_area};"
                                    f"cluster-area:{self.weed_cluster_area};"
                                    f"separation-px:{self.weed_separation_px}"),
            })
        if self.enable_weed_mask:
            # CUDA ExG + object exclusion + optical flow reader.
            props = f"exg-threshold:{self.exg_threshold};exg-mode:{self.exg_mode}"
            self.pipeline.add("nvdsvideotemplate", "weed-mask", {
                "gpu-id": gpu_id, "customlib-name": self.weed_template_lib,
                "customlib-props": props,
            })
            self.pipeline.add("capsfilter", "rgba-caps-out", {
                "caps": "video/x-raw(memory:NVMM),format=RGBA",
            })

        # flip-method values 1/3 (90\B0/270\B0) and 5/7 (diagonal) swap width
        # and height; 0/2/4/6 (none/180/h-flip/v-flip) keep them as-is.
        # Stating the post-rotation size explicitly avoids ambiguous caps
        # negotiation here, which otherwise triggers nvvideoconvert's
        # "Cannot keep DAR" warning and stretches the frame to fit whatever
        # size it guessed instead of the actual rotated dimensions.
        out_width = self.output_width
        out_height = self.output_height

        # Final conversion applies either explicit camera orientation or the
        # video container's display rotation. Output caps use the matching size.
        self.pipeline.add("nvvideoconvert", "convert", {
            "gpu-id": gpu_id,
            "compute-hw": 1,
            "nvbuf-memory-type": 2,
            "flip-method": self.output_flip_method,
        })
        self.pipeline.add("capsfilter", "output-caps", {
            "caps": (
                "video/x-raw(memory:NVMM),format=RGB,"
                f"width={out_width},height={out_height}"
            ),
        })
        self.pipeline.add("appsink", "sink", {
            "emit-signals": True, "max-buffers": 1, "drop": True, "sync": False,
        })

        elements.extend(["to-rgba", "rgba-caps-in"])
        if self.enable_native_weed_bboxes:
            elements.append("weed-bbox-meta")
        self.pipeline.add("nvtracker", "tracker", {
            "ll-lib-file": self.tracker_lib,
            "ll-config-file": self.tracker_config,
            "tracker-width": 640, "tracker-height": 384,
            "gpu-id": gpu_id, "display-tracking-id": 1,
            "operate-on-class-ids": "0;1",
        })
        elements.append("tracker")
        if self.enable_weed_mask:
            elements.extend(["weed-mask", "rgba-caps-out"])
        elements.extend(["convert", "output-caps", "sink"])
        self.pipeline.link(*elements)

    def get_optical_flow(self):
        """Return (flow_x, flow_y, frame_count) in pixels/frame from Gst-nvof."""
        if self._custom_lib is not None:
            try:
                fx = ctypes.c_float()
                fy = ctypes.c_float()
                self._custom_lib.get_latest_optical_flow(ctypes.byref(fx), ctypes.byref(fy))
                cnt = self._custom_lib.get_optical_flow_frame_count()
                return float(fx.value), float(fy.value), int(cnt)
            except Exception:
                pass
        return 0.0, 0.0, 0

    def _transform_boxes_for_output(self, plants):
        """Keep inference boxes aligned with the explicitly rotated preview."""
        method = self.output_flip_method
        if method == 0:
            return plants
        width, height = self.width, self.height
        transformed = []
        for x1, y1, x2, y2 in plants:
            if method == 3:       # 90 degrees clockwise
                box = (height - y2, x1, height - y1, x2)
            elif method == 1:     # 90 degrees counter-clockwise
                box = (y1, width - x2, y2, width - x1)
            elif method == 2:     # 180 degrees
                box = (width - x2, height - y2, width - x1, height - y1)
            else:
                # Mirror/diagonal transforms are legacy GUI camera options.
                return plants
            x1o, y1o, x2o, y2o = box
            transformed.append([
                max(0, min(self.output_width, int(x1o))),
                max(0, min(self.output_height, int(y1o))),
                max(0, min(self.output_width, int(x2o))),
                max(0, min(self.output_height, int(y2o))),
            ])
        return transformed

    def _store_latest(self, frame_bgr, plants, weed_mask, plant_ids=None,
                      weed_boxes=None, weed_ids=None):
        # Every consumer (GUI, MJPEG preview, and world-coordinate path) uses
        # this contract: NumPy shape is (height, width), matching output caps.
        actual_height, actual_width = frame_bgr.shape[:2]
        if (actual_width, actual_height) != (self.output_width, self.output_height):
            raise RuntimeError(
                "DeepStream frame dimensions do not match negotiated output caps: "
                f"received {actual_width}x{actual_height}, expected "
                f"{self.output_width}x{self.output_height}"
            )
        now = time.monotonic()
        flow = self.get_optical_flow()[:2]
        with self._lock:
            self._latest_frame = frame_bgr
            self._latest_plants = [] if plants is None else list(plants)
            self._latest_plant_ids = [] if plant_ids is None else list(plant_ids)
            self._latest_weed_boxes = [] if weed_boxes is None else list(weed_boxes)
            self._latest_weed_ids = [] if weed_ids is None else list(weed_ids)
            self._latest_weed_mask = weed_mask
            self._latest_flow = flow
            self._frame_sequence += 1
            if self._last_frame_time is not None:
                dt = now - self._last_frame_time
                if dt > 0:
                    inst_fps = 1.0 / dt
                    self._fps = (self._fps + self._fps_alpha * (inst_fps - self._fps)
                                 if self._fps else inst_fps)
            self._last_frame_time = now

    def get_latest_plant_ids(self):
        """Return NvSORT IDs in the same order as the latest chilli boxes."""
        with self._lock:
            return list(self._latest_plant_ids)

    def start(self):
        if not self._running:
            self.pipeline.start()
            self._running = True

    def stop(self):
        if self._running:
            self.pipeline.stop()
            self._running = False

    def get_latest(self, timeout=0.1):
        """Return ``(BGR frame, crop boxes, GPU weed mask, fps, sequence)`` or empty.

        Maintains exact 5-element return tuple for full backward compatibility.
        Access optical flow via ``get_optical_flow()`` or ``get_latest_with_flow()``.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self._latest_frame is not None:
                    return (self._latest_frame, self._latest_plants, self._latest_weed_mask,
                            self._fps, self._frame_sequence)
            time.sleep(0.005)
        return None, [], None, 0.0, -1

    def get_latest_with_ids(self, timeout=0.1):
        """Return frame data plus atomically paired native tracker IDs."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self._latest_frame is not None:
                    return (self._latest_frame, self._latest_plants,
                            self._latest_plant_ids, self._latest_weed_mask,
                            self._fps, self._frame_sequence)
            time.sleep(0.005)
        return None, [], [], None, 0.0, -1

    def get_latest_tracking(self, timeout=0.1):
        """Return both classes' boxes/IDs as one atomic frame snapshot."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self._latest_frame is not None:
                    return (
                        self._latest_frame, self._latest_plants,
                        self._latest_plant_ids, self._latest_weed_boxes,
                        self._latest_weed_ids, self._latest_weed_mask,
                        self._fps, self._frame_sequence,
                    )
            time.sleep(0.005)
        return None, [], [], [], [], None, 0.0, -1

    def get_latest_with_flow(self, timeout=0.1):
        """Return ``(BGR frame, crop boxes, GPU weed mask, fps, sequence, flow_vector)``."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self._latest_frame is not None:
                    return (self._latest_frame, self._latest_plants, self._latest_weed_mask,
                            self._fps, self._frame_sequence, self._latest_flow)
            time.sleep(0.005)
        return None, [], None, 0.0, -1, (0.0, 0.0)

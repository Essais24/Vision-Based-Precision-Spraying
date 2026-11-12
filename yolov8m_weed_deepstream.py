#!/usr/bin/env python3
"""Standalone GPU-only YOLOv8 weed viewer using DeepStream and TensorRT.

This does not change or connect to the spraying pipeline.  DeepStream is used
for CSI camera capture and TensorRT inference. ``best_weed.onnx`` is compiled
by DeepStream into a local Orin-specific FP16 engine on its first run.

Run:
    btp_env/bin/python yolov8m_weed_deepstream.py --source 0
Then open: http://<jetson-ip>:8082
"""
import argparse
import os
import threading
import time

import cv2
from flask import Flask, Response, jsonify

from deepstream_pipeline import (
    DEEPSTREAM_AVAILABLE,
    DEEPSTREAM_UNAVAILABLE_REASON,
    DeepStreamCamera,
)


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_source(value):
    return int(value) if value.isdigit() else value


class DeepStreamYoloV8Viewer:
    def __init__(self, source, config):
        self.source = source
        self.config = config
        self.lock = threading.Lock()
        self.running = False
        self.status = "Starting"
        self.error = None
        self.jpeg = None
        self.fps = 0.0
        self.weed_count = 0
        self.worker = None
        self.camera = None

    def start(self):
        self.running = True
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def stop(self):
        self.running = False
        if self.worker:
            self.worker.join(timeout=3.0)

    def state(self):
        with self.lock:
            return {
                "running": self.running, "status": self.status, "error": self.error,
                "fps": round(self.fps, 1), "weed_count": self.weed_count,
                "source": str(self.source), "model": os.path.basename(self.config),
            }

    def _run(self):
        try:
            if not DEEPSTREAM_AVAILABLE:
                raise RuntimeError(f"DeepStream unavailable: {DEEPSTREAM_UNAVAILABLE_REASON}")
            if not os.path.isfile(self.config):
                raise FileNotFoundError(f"DeepStream weed config not found: {self.config}")
            # Same CSI-camera creation path used by interactive_spray_accuracy_test.py.
            self.camera = DeepStreamCamera(
                source=self.source,
                config_path=self.config,
                enable_optical_flow=False,
                enable_weed_mask=False,
                use_updated=True,
                enable_native_weed_bboxes=False,
            )
            self.camera.start()
            print(f"[INFO] DeepStream camera started: {self.source!r}", flush=True)
            with self.lock:
                self.status = "Running"

            previous = time.perf_counter()
            last_sequence = -1
            while self.running:
                # DeepStreamCamera is shared with the chilli/weed pipeline,
                # where only class_id == 1 is placed in ``weed_boxes``.  This
                # viewer's model has one class (labels_weed.txt: "weed"),
                # whose class ID is 0; those boxes therefore arrive in
                # ``plant_boxes``.  A single-class weed viewer must render
                # both collections, otherwise every valid model detection is
                # silently discarded and the UI always reports zero weeds.
                frame, plant_boxes, _plant_ids, weed_boxes, _weed_ids, _mask, _camera_fps, sequence = \
                    self.camera.get_latest_tracking(timeout=0.5)
                if frame is None or sequence == last_sequence:
                    continue
                last_sequence = sequence

                annotated = frame.copy()
                detections = [*plant_boxes, *weed_boxes]
                weed_count = len(detections)
                for x1, y1, x2, y2 in detections:
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(annotated, "weed", (x1, max(22, y1 - 7)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)

                now = time.perf_counter()
                inference_fps = 1.0 / max(now - previous, 1e-6)
                previous = now
                cv2.putText(
                    annotated, f"TensorRT weeds: {weed_count}  FPS: {inference_fps:.1f}",
                    (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA,
                )
                ok, encoded = cv2.imencode(".jpg", annotated)
                if ok:
                    with self.lock:
                        self.jpeg = encoded.tobytes()
                        self.weed_count = weed_count
                        self.fps = inference_fps
        except Exception as exc:
            with self.lock:
                self.error = f"{type(exc).__name__}: {exc}"
                self.status = "Error"
            print(f"[ERROR] {self.error}", flush=True)
        finally:
            if self.camera is not None:
                self.camera.stop()
            with self.lock:
                self.running = False
                if self.status == "Running":
                    self.status = "Stopped"


def create_app(viewer):
    app = Flask(__name__)

    @app.get("/")
    def index():
        return """<!doctype html><title>DeepStream TensorRT Weed Viewer</title>
<style>body{background:#172018;color:#edf5e9;font:16px system-ui;margin:2rem}img{max-width:100%;border:2px solid #6aa84f}</style>
<h1>DeepStream TensorRT Weed Detector</h1><p id=s>Connecting…</p><img src=/stream.mjpg>
<script>setInterval(async()=>{let x=await (await fetch('/api/state')).json();s.textContent=`${x.status} | weeds: ${x.weed_count} | FPS: ${x.fps}${x.error?' | '+x.error:''}`},500)</script>"""

    @app.get("/api/state")
    def state():
        return jsonify(viewer.state())

    @app.get("/stream.mjpg")
    def stream():
        def frames():
            while viewer.running:
                with viewer.lock:
                    jpg = viewer.jpeg
                if jpg:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                time.sleep(0.03)
        return Response(frames(), mimetype="multipart/x-mixed-replace; boundary=frame")
    return app


def main():
    parser = argparse.ArgumentParser(description="Run YOLOv8 weeds with DeepStream TensorRT GPU inference.")
    parser.add_argument("--source", default="0", help="CSI sensor id or file/stream path (default: 0)")
    parser.add_argument("--config", default=os.path.join(PROJECT_DIR, "config_infer_weed_yolov8.txt"))
    parser.add_argument("--port", type=int, default=8082)
    args = parser.parse_args()

    viewer = DeepStreamYoloV8Viewer(parse_source(args.source), os.path.abspath(args.config))
    viewer.start()
    print(f"[INFO] Open http://<device-ip>:{args.port}")
    try:
        create_app(viewer).run(host="0.0.0.0", port=args.port, threaded=True, use_reloader=False)
    finally:
        viewer.stop()


if __name__ == "__main__":
    main()

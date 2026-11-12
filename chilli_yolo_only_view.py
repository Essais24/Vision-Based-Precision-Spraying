"""
yolo_only_preview.py

Minimal, standalone test: runs the DeepStream pipeline (via
DeepStreamCamera in deepstream_pipeline.py) purely to look at YOLO's raw
plant-detection boxes. Ignores the ExG weed mask, world-coordinate math,
and spraying logic entirely -- none of that code runs or is imported here.

Does not modify deepstream_pipeline.py, detection_worldcoord.py, or
weed_sprayer_gui.py in any way; only imports DeepStreamCamera from the
first one.

Built for headless SSH use, since cv2.imshow needs a GUI-enabled OpenCV
build (GTK/Qt/Cocoa) that most Jetson venvs don't have -- this never
calls cv2.imshow at all. Instead it:
  1. Prints detection counts to the console every frame.
  2. Saves periodic annotated JPEG snapshots to disk.
  3. Serves a live MJPEG stream over plain HTTP that you can view in any
     browser, including on your laptop via SSH port forwarding.

Usage:
    python3 yolo_only_view.py [video_or_camera_source] [config_path]

Then, from your laptop:
    ssh -L 8080:localhost:8080 archi@192.168.137.181
and open http://localhost:8080 in a browser.
"""
import os
import sys
import time
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2

import detection_worldcoord as detector
from deepstream_pipeline import DeepStreamCamera

def _parse_source(value):
    """Treat an all-digit argument such as `0` as a CSI sensor id."""
    return int(value) if value.isdigit() else value


VIDEO_SOURCE = _parse_source(sys.argv[1]) if len(sys.argv) > 1 else 0
CONFIG_PATH = sys.argv[2] if len(sys.argv) > 2 else "config_infer_primary_yolo.txt"

SNAPSHOT_DIR = "yolo_preview_snapshots"
SNAPSHOT_EVERY_SEC = 2.0     # how often to also write a JPEG to disk
STREAM_PORT = 8080           # browse to http://<jetson>:8080 (or via ssh -L)

_latest_jpeg = None
_jpeg_lock = threading.Lock()


class _MJPEGHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            while True:
                with _jpeg_lock:
                    jpg = _latest_jpeg
                if jpg is not None:
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                    self.wfile.write(jpg)
                    self.wfile.write(b"\r\n")
                time.sleep(0.05)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, format, *args):
        pass  # silence the default per-request console logging


def _start_mjpeg_server(port):
    server = HTTPServer(("0.0.0.0", port), _MJPEGHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def draw_detection_boxes(frame_bgr, plants):
    """Overlay red ExG weed boxes and green YOLO boxes on untouched pixels."""
    annotated, weed_detections, _mask = detector.process_frame(
        frame_bgr, plants, weed_mask=None
    )
    cv2.putText(annotated,
                f"YOLO plants: {len(plants)}  ExG weeds: {len(weed_detections)}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
    return annotated, len(weed_detections)


def main():
    global _latest_jpeg
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    _start_mjpeg_server(STREAM_PORT)

    cam = DeepStreamCamera(
        source=VIDEO_SOURCE,
        config_path=CONFIG_PATH,
        enable_optical_flow=False,  # not needed just to check YOLO output
        enable_weed_mask=False,     # preserve original camera pixels
        use_updated=True,
    )
    cam.start()

    print(f"[INFO] YOLO-only preview running on: {VIDEO_SOURCE}")
    print(f"[INFO] Live view: http://<jetson-ip>:{STREAM_PORT}  "
          f"(or ssh -L {STREAM_PORT}:localhost:{STREAM_PORT} then http://localhost:{STREAM_PORT})")
    print(f"[INFO] Snapshots also saved every {SNAPSHOT_EVERY_SEC}s to ./{SNAPSHOT_DIR}/")
    print("Press Ctrl+C to stop.\n")

    last_snapshot = 0.0
    frame_count = 0
    try:
        while True:
            frame_bgr, plants, _weed_mask, fps, seq = cam.get_latest(timeout=0.5)
            if frame_bgr is None:
                continue

            frame_count += 1
            if frame_count == 1:
                frame_height, frame_width = frame_bgr.shape[:2]
                expected_width = cam.output_width
                expected_height = cam.output_height
                print(f"[INFO] Received frame: {frame_width}x{frame_height} (width x height); "
                      f"pipeline output: {expected_width}x{expected_height}")
                if (frame_width, frame_height) != (expected_width, expected_height):
                    raise RuntimeError(
                        "DeepStream output dimensions do not match negotiated caps: "
                        f"received {frame_width}x{frame_height}, expected "
                        f"{expected_width}x{expected_height}"
                    )
            annotated, weed_count = draw_detection_boxes(frame_bgr, plants)

            print(f"\r[seq {seq}] fps={fps:.1f} yolo={len(plants)} exg={weed_count}",
                  end="", flush=True)

            ok, buf = cv2.imencode(".jpg", annotated)
            if ok:
                with _jpeg_lock:
                    _latest_jpeg = buf.tobytes()

            now = time.time()
            if now - last_snapshot >= SNAPSHOT_EVERY_SEC:
                path = os.path.join(SNAPSHOT_DIR, f"frame_{frame_count:06d}.jpg")
                cv2.imwrite(path, annotated)
                last_snapshot = now
    except KeyboardInterrupt:
        pass
    finally:
        print("\n[INFO] Stopping pipeline...")
        cam.stop()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
capture_calibration_images.py
========================================================================
Headless camera calibration capture tool for Jetson.
Streams live camera feed over HTTP (MJPEG) so you can view it in your browser
over SSH / LAN, without requiring an X11 / GUI display.

Features:
  - Live browser preview: http://<jetson-ip>:5000/
  - Dual capture triggers:
      1. Press [ENTER] in your SSH terminal
      2. Click the "Capture Photo" button on the web page (or press Space/Enter on page)
  - Full-resolution (1920x1080) uncompressed frames saved to calibration_images/
  - Zero X11/DISPLAY dependencies.
========================================================================
"""

import os
import sys
import time
import socket
import logging
import threading
from datetime import datetime

# Allow importing system PyGObject / GStreamer bindings even within a venv
for p in ["/usr/lib/python3/dist-packages", "/usr/local/lib/python3/dist-packages"]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.append(p)

try:
    import gi
    gi.require_version("Gst", "1.0")
    from gi.repository import Gst
except ImportError as err:
    print(f"[ERROR] Failed to import GStreamer (gi.repository.Gst): {err}")
    print("[HINT] Ensure python3-gi and python3-gst-1.0 are installed on system.")
    sys.exit(1)

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template_string

# Mute noisy werkzeug HTTP request logs
logging.getLogger("werkzeug").setLevel(logging.ERROR)

# Configuration
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration_images")
SENSOR_ID = 0
CAPTURE_WIDTH = 1920
CAPTURE_HEIGHT = 1080
FRAMERATE = 30
WBMODE = 2
WEB_PORT = 5000

app = Flask(__name__)


class CameraState:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = True
        self.latest_raw_frame = None
        self.latest_jpeg = None
        self.photo_count = len([
            f for f in os.listdir(SAVE_DIR)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]) if os.path.exists(SAVE_DIR) else 0
        self.last_saved_file = None


state = CameraState()


def get_gstreamer_pipeline_str(sensor_id=SENSOR_ID, width=CAPTURE_WIDTH, height=CAPTURE_HEIGHT,
                              framerate=FRAMERATE, wbmode=WBMODE):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} wbmode={wbmode} ! "
        f"video/x-raw(memory:NVMM), width=(int){width}, height=(int){height}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! "
        f"appsink name=sink max-buffers=1 drop=True emit-signals=True"
    )


def extract_frame_from_sample(sample, width=CAPTURE_WIDTH, height=CAPTURE_HEIGHT):
    """Extract full 1920x1080 numpy BGR frame from Gst.Sample."""
    if sample is None:
        return None
    buf = sample.get_buffer()
    success, map_info = buf.map(Gst.MapFlags.READ)
    if not success:
        return None
    try:
        frame = np.ndarray((height, width, 3), buffer=map_info.data, dtype=np.uint8).copy()
        return frame
    finally:
        buf.unmap(map_info)


def camera_capture_loop(sink):
    """Background worker continuously pulling frames from GStreamer appsink."""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 75]
    while state.running:
        sample = sink.emit("pull-sample")
        if sample is None:
            time.sleep(0.01)
            continue

        raw_frame = extract_frame_from_sample(sample)
        if raw_frame is None:
            continue

        # Create web preview (optionally scaled to 960x540 for fast, smooth streaming)
        preview = cv2.resize(raw_frame, (960, 540), interpolation=cv2.INTER_AREA)
        ret, jpeg_buf = cv2.imencode(".jpg", preview, encode_param)

        if ret:
            with state.lock:
                state.latest_raw_frame = raw_frame
                state.latest_jpeg = jpeg_buf.tobytes()

        time.sleep(0.01)


def save_current_photo():
    """Thread-safe trigger to capture and save the current full-res photo."""
    with state.lock:
        if state.latest_raw_frame is None:
            return False, "Camera is not streaming yet."
        frame_to_save = state.latest_raw_frame.copy()
        state.photo_count += 1
        count = state.photo_count

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"calib_{count:03d}_{timestamp}.jpg"
    filepath = os.path.join(SAVE_DIR, filename)

    # Save full-resolution uncompressed image
    cv2.imwrite(filepath, frame_to_save)

    with state.lock:
        state.last_saved_file = filename

    msg = f"[CAPTURE #{count:03d}] Saved: {filepath} ({frame_to_save.shape[1]}x{frame_to_save.shape[0]})"
    print(f"\n{msg}\n[{count} photos] Press ENTER to capture ('q' to exit): ", end="", flush=True)
    return True, filename


# =========================================================================
# Web Server Routes
# =========================================================================
HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Jetson Camera Calibration View</title>
  <style>
    :root {
      --bg: #121316;
      --card: #1e2024;
      --accent: #238636;
      --accent-hover: #2ea043;
      --text: #e6edf3;
      --muted: #8b949e;
    }
    body {
      margin: 0;
      padding: 20px;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      display: flex;
      flex-direction: column;
      align-items: center;
    }
    .container {
      max-width: 980px;
      width: 100%;
      background: var(--card);
      border-radius: 12px;
      padding: 20px;
      box-shadow: 0 8px 24px rgba(0,0,0,0.5);
      border: 1px solid #30363d;
    }
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid #30363d;
      padding-bottom: 15px;
      margin-bottom: 15px;
    }
    h1 { margin: 0; font-size: 1.4rem; font-weight: 600; }
    .badge {
      background: #21262d;
      border: 1px solid #30363d;
      padding: 6px 14px;
      border-radius: 20px;
      font-size: 0.95rem;
      font-weight: 500;
    }
    .badge span { color: #58a6ff; font-weight: 700; }
    .video-wrap {
      position: relative;
      width: 100%;
      border-radius: 8px;
      overflow: hidden;
      background: #000;
      aspect-ratio: 16 / 9;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .video-wrap img {
      width: 100%;
      height: 100%;
      object-fit: contain;
    }
    .controls {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-top: 20px;
      gap: 15px;
    }
    button.btn-capture {
      flex: 1;
      background: var(--accent);
      color: #fff;
      border: none;
      padding: 14px 24px;
      font-size: 1.1rem;
      font-weight: 600;
      border-radius: 8px;
      cursor: pointer;
      transition: background 0.15s ease, transform 0.05s ease;
    }
    button.btn-capture:hover { background: var(--accent-hover); }
    button.btn-capture:active { transform: scale(0.98); }
    .hint {
      color: var(--muted);
      font-size: 0.85rem;
    }
    #toast {
      margin-top: 10px;
      padding: 10px 14px;
      border-radius: 6px;
      background: #1f3624;
      border: 1px solid #238636;
      color: #7ee787;
      font-size: 0.9rem;
      display: none;
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>🎯 Camera Calibration Live Feed</h1>
      <div class="badge">Photos Captured: <span id="photo-count">{{ count }}</span></div>
    </header>

    <div class="video-wrap">
      <img src="/video_feed" alt="Live Camera Stream">
    </div>

    <div class="controls">
      <button class="btn-capture" id="captureBtn" onclick="triggerCapture()">
        📸 Capture Calibration Photo
      </button>
    </div>
    <div class="hint" style="margin-top: 8px;">
      ⌨️ Shortcut: Press <b>Space</b> or <b>Enter</b> on this page, or press <b>Enter</b> in your SSH terminal.
    </div>

    <div id="toast"></div>
  </div>

  <script>
    async function triggerCapture() {
      const btn = document.getElementById('captureBtn');
      btn.disabled = true;
      btn.innerText = "Capturing...";
      try {
        const res = await fetch('/capture', { method: 'POST' });
        const data = await res.json();
        if (data.success) {
          document.getElementById('photo-count').innerText = data.count;
          showToast(`Saved: ${data.filename} (Total: ${data.count})`);
        } else {
          showToast(`Error: ${data.error}`, true);
        }
      } catch (err) {
        showToast(`Request failed: ${err}`, true);
      } finally {
        btn.disabled = false;
        btn.innerText = "📸 Capture Calibration Photo";
      }
    }

    function showToast(msg, isError = false) {
      const toast = document.getElementById('toast');
      toast.innerText = msg;
      toast.style.display = 'block';
      toast.style.borderColor = isError ? '#f85149' : '#238636';
      toast.style.color = isError ? '#ff7b72' : '#7ee787';
      toast.style.background = isError ? '#3e1b1b' : '#1f3624';
      clearTimeout(window._toastTimeout);
      window._toastTimeout = setTimeout(() => { toast.style.display = 'none'; }, 3000);
    }

    // Keyboard shortcuts in browser
    window.addEventListener('keydown', (e) => {
      if (e.code === 'Space' || e.code === 'Enter') {
        e.preventDefault();
        triggerCapture();
      }
    });
  </script>
</body>
</html>
"""


@app.route("/")
def index():
    with state.lock:
        count = state.photo_count
    return render_template_string(HTML_PAGE, count=count)


@app.route("/video_feed")
def video_feed():
    def generate():
        while state.running:
            with state.lock:
                frame_bytes = state.latest_jpeg
            if frame_bytes is not None:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
            time.sleep(0.03)

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/capture", methods=["POST"])
def web_capture():
    success, result = save_current_photo()
    if success:
        with state.lock:
            count = state.photo_count
        return jsonify({"success": True, "filename": result, "count": count})
    return jsonify({"success": False, "error": result}), 400


# =========================================================================
# Main Entry Point
# =========================================================================
def get_local_ips():
    ips = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ips.append(s.getsockname()[0])
        s.close()
    except Exception:
        pass
    return ips


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    Gst.init(None)

    pipe_str = get_gstreamer_pipeline_str()
    print("=" * 65)
    print("Initializing Headless GStreamer Pipeline:")
    print(pipe_str)
    print("=" * 65)

    try:
        pipeline = Gst.parse_launch(pipe_str)
    except Exception as exc:
        print(f"[ERROR] Failed to parse GStreamer pipeline: {exc}")
        sys.exit(1)

    sink = pipeline.get_by_name("sink")
    bus = pipeline.get_bus()

    ret = pipeline.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        print("[ERROR] Failed to start pipeline playing state.")
        pipeline.set_state(Gst.State.NULL)
        sys.exit(1)

    # Start background capture thread
    capture_thread = threading.Thread(target=camera_capture_loop, args=(sink,), daemon=True)
    capture_thread.start()

    # Start Flask web server in a background daemon thread
    flask_thread = threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=WEB_PORT, debug=False, use_reloader=False, threaded=True),
        daemon=True
    )
    flask_thread.start()

    # Wait for camera stream to deliver first frame
    print("[INFO] Connecting to camera and starting web server...")
    for _ in range(50):
        with state.lock:
            if state.latest_jpeg is not None:
                break
        time.sleep(0.1)

    local_ips = get_local_ips()
    primary_ip = local_ips[0] if local_ips else "127.0.0.1"

    print("=" * 65)
    print(f"🎉 Live Camera Feed is now accessible over HTTP:")
    print(f"   👉 http://{primary_ip}:{WEB_PORT}/")
    print(f"   👉 http://localhost:{WEB_PORT}/ (if SSH port-forwarded: -L {WEB_PORT}:localhost:{WEB_PORT})")
    print("=" * 65)
    print(f"[INFO] Photos folder: {SAVE_DIR}")
    print(f"[INFO] Current photo count: {state.photo_count}")
    print("-" * 65)
    print(">>> Press [ENTER] in this terminal (or use web button) to CAPTURE <<<")
    print(">>> Type 'q' and press [ENTER] to QUIT                          <<<")
    print("-" * 65)

    try:
        while True:
            # Check for GStreamer bus errors non-blockingly
            msg = bus.pop_filtered(Gst.MessageType.ERROR | Gst.MessageType.EOS)
            if msg:
                if msg.type == Gst.MessageType.ERROR:
                    err, debug = msg.parse_error()
                    print(f"\n[GSTREAMER ERROR] {err.message}")
                    if debug:
                        print(f"Debug: {debug}")
                    break
                elif msg.type == Gst.MessageType.EOS:
                    print("\n[GSTREAMER] End of Stream.")
                    break

            try:
                user_input = input(f"[{state.photo_count} photos] Press ENTER to capture ('q' to exit): ").strip()
            except EOFError:
                break

            if user_input.lower() in ("q", "quit", "exit"):
                print("\n[INFO] Exit command received.")
                break

            # Capture photo from terminal Enter press
            save_current_photo()

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user (Ctrl+C).")
    finally:
        state.running = False
        pipeline.set_state(Gst.State.NULL)
        print(f"[INFO] Camera pipeline and web server stopped.")
        print(f"[INFO] Total photos captured in {SAVE_DIR}: {state.photo_count}")


if __name__ == "__main__":
    main()

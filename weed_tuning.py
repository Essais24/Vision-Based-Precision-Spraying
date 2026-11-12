"""Browser based live ExG/weed-box calibration for a Jetson camera.

Run over SSH with ``python3 weed_tuning.py --source 0`` and open
``http://<jetson-ip>:5001`` from a browser. Tuning uses the CPU ExG and
watershed path with YOLO chilli boxes suppressed from the weed mask. Save
stores a profile for the next DeepStream pipeline start.
"""

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import cv2

import detection_worldcoord as detector
from deepstream_pipeline import DeepStreamCamera


DEFAULTS = {
    "exg_threshold": detector.EXG_THRESHOLD,
    "min_area": detector.MIN_CONTOUR_AREA,
    "cluster_area_threshold": detector.CLUSTER_AREA_THRESHOLD,
    "separation_px": detector.SEPARATION_KERNEL_PX,
    "max_box_area": detector.MAX_WEED_BOX_AREA,
}
RANGES = {
    "exg_threshold": (0, 100), "min_area": (0, 50000),
    "cluster_area_threshold": (0, 200000), "separation_px": (1, 41),
    "max_box_area": (0, 500000),
}
LABELS = {
    "exg_threshold": "ExG threshold", "min_area": "Minimum weed pixel area",
    "cluster_area_threshold": "Cluster split area", "separation_px": "Peak separation (px)",
    "max_box_area": "Maximum bbox area",
}


class Shared:
    lock = threading.Lock()
    settings = DEFAULTS.copy()
    frame_jpeg = None
    mask_jpeg = None
    status = "Starting camera"
    fps = 0.0
    stop = threading.Event()


def _parse_source(value):
    return int(value) if value.isdecimal() else value


def _read_profile(path):
    settings = DEFAULTS.copy()
    try:
        with open(path, "r", encoding="utf-8") as profile:
            saved = json.load(profile)
        for key in settings:
            if key in saved:
                lo, hi = RANGES[key]
                settings[key] = max(lo, min(hi, int(saved[key])))
    except (OSError, ValueError, TypeError):
        pass
    return settings


PAGE = """<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content='width=device-width,initial-scale=1'><title>Weed tuning</title>
<style>
body{font:15px system-ui;margin:0;background:#171a1d;color:#eee}header{padding:12px 18px;background:#252a2e;display:flex;gap:18px;align-items:center;flex-wrap:wrap}.layout{display:grid;grid-template-columns:minmax(0,2fr) minmax(290px,1fr);gap:14px;padding:14px}.panel{background:#252a2e;border-radius:8px;padding:12px}.views{display:grid;gap:10px}.views img{width:100%;height:auto;background:#000;border-radius:5px}.control{margin:14px 0}.row{display:flex;justify-content:space-between;gap:8px}.control input{width:100%}button{padding:8px 14px;cursor:pointer}small{color:#bbc4cb}@media(max-width:800px){.layout{grid-template-columns:1fr}}
</style></head><body>
<header><b>Live weed tuning</b><span id=status>Connecting…</span><span id=fps></span><button id=save>Save profile</button></header>
<div class=layout><section class=views><div class=panel><b>Camera & detections</b><br><small>Chilli / YOLO / DeepStream track ID: green · ExG weed candidates: red</small><img src=/stream.mjpg></div><div class=panel><b>ExG weed mask</b><img src=/mask.mjpg></div></section>
<section class=panel><b>Live controls</b><div id=controls></div><small>Changes apply to the next processed frame. Save writes the profile used by the main pipeline.</small></section></div>
<script>
const names={exg_threshold:'ExG threshold',min_area:'Minimum weed pixel area',cluster_area_threshold:'Cluster split area',separation_px:'Peak separation (px)',max_box_area:'Maximum bbox area'};
async function update(key,value){await fetch('/api/settings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({[key]:Number(value)})})}
async function init(){const d=await (await fetch('/api/settings')).json(),root=document.getElementById('controls');for(const [key,item] of Object.entries(d.settings)){let [lo,hi]=d.ranges[key],div=document.createElement('div');div.className='control';div.innerHTML=`<div class=row><label>${names[key]}</label><span id="v-${key}"></span></div><input type=range min=${lo} max=${hi} value=${item} id="i-${key}">`;root.appendChild(div);let input=div.querySelector('input'),value=div.querySelector('span');value.textContent=item;input.oninput=()=>{value.textContent=input.value;update(key,input.value)}}}
document.getElementById('save').onclick=async()=>{let r=await fetch('/api/save',{method:'POST'});let d=await r.json();alert(d.message)};
async function poll(){try{let s=await(await fetch('/api/status')).json();document.getElementById('status').textContent=s.status;document.getElementById('fps').textContent=s.fps.toFixed(1)+' camera FPS'}catch(e){}setTimeout(poll,1000)}init();poll();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print("[HTTP] " + fmt % args, flush=True)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            payload, kind = PAGE.encode(), "text/html; charset=utf-8"
        elif path in ("/stream.mjpg", "/mask.mjpg"):
            field = "frame_jpeg" if path == "/stream.mjpg" else "mask_jpeg"
            self.send_response(200)
            self.send_header("Cache-Control", "no-cache, no-store")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            while not Shared.stop.is_set():
                with Shared.lock:
                    jpg = getattr(Shared, field)
                if jpg:
                    try:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        break
                time.sleep(0.04)
            return
        elif path == "/api/settings":
            with Shared.lock:
                payload = json.dumps({"settings": Shared.settings, "ranges": RANGES}).encode()
            kind = "application/json"
        elif path == "/api/status":
            with Shared.lock:
                payload = json.dumps({"status": Shared.status, "fps": Shared.fps}).encode()
            kind = "application/json"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            size = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(size) or b"{}")
        except (ValueError, json.JSONDecodeError):
            self.send_error(400, "Invalid JSON")
            return
        if path == "/api/settings":
            with Shared.lock:
                for key, value in body.items():
                    if key in RANGES:
                        lo, hi = RANGES[key]
                        Shared.settings[key] = max(lo, min(hi, int(value)))
                payload = json.dumps({"settings": Shared.settings}).encode()
        elif path == "/api/save":
            settings = Shared.settings.copy()
            save_path = Path(self.server.profile_path)
            save_path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
            payload = json.dumps({"message": f"Saved {save_path.resolve()}"}).encode()
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="0", help="CSI sensor number, video path, or stream URI")
    parser.add_argument("--save", default="weed_tuning_profile.json", help="JSON profile path")
    parser.add_argument("--host", default="0.0.0.0", help="Web server interface (default: all interfaces)")
    parser.add_argument("--port", type=int, default=5001)
    args = parser.parse_args()

    with Shared.lock:
        Shared.settings = _read_profile(args.save)
        Shared.status = "Starting camera"
    camera = DeepStreamCamera(
        source=_parse_source(args.source), enable_weed_mask=False,
        enable_native_weed_bboxes=False, enable_optical_flow=False,
    )
    detector.reset_detection_ids()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    server.profile_path = args.save
    web_thread = threading.Thread(target=server.serve_forever, daemon=True)
    web_thread.start()
    print(f"Tuning page: http://{args.host}:{args.port} (use the Jetson IP from your browser)", flush=True)
    last_sequence, last_fps_time, fps_ema = -1, None, 0.0
    try:
        camera.start()
        with Shared.lock:
            Shared.status = "Running"
        while not Shared.stop.is_set():
            (frame, plants, plant_ids, _weed_boxes, _weed_ids,
             _weed_mask, camera_fps, sequence) = camera.get_latest_tracking(timeout=0.5)
            if frame is None or sequence == last_sequence:
                print(f"no new frame: frame None={frame is None}, seq={sequence}", flush=True)
                continue
            last_sequence = sequence
            with Shared.lock:
                settings = Shared.settings.copy()
            annotated, _detections, mask = detector.process_frame(
                frame, plants, exg_threshold=settings["exg_threshold"],
                min_area=settings["min_area"],
                cluster_area_threshold=settings["cluster_area_threshold"],
                separation_px=settings["separation_px"],
                max_box_area=settings["max_box_area"],
                plant_track_ids=plant_ids, pipeline_compatible=True,
            )
            cv2.putText(annotated, f"Camera {camera_fps:.1f} FPS", (10, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            ok1, frame_buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 82])
            ok2, mask_buf = cv2.imencode(".jpg", mask, [cv2.IMWRITE_JPEG_QUALITY, 82])
            now = time.monotonic()
            if last_fps_time is not None and now > last_fps_time:
                inst = 1.0 / (now - last_fps_time)
                fps_ema = inst if not fps_ema else fps_ema + 0.2 * (inst - fps_ema)
            last_fps_time = now
            with Shared.lock:
                if ok1:
                    Shared.frame_jpeg = frame_buf.tobytes()
                if ok2:
                    Shared.mask_jpeg = mask_buf.tobytes()
                Shared.fps = fps_ema
    except KeyboardInterrupt:
        pass
    finally:
        with Shared.lock:
            Shared.status = "Stopping"
        Shared.stop.set()
        server.shutdown()
        server.server_close()
        camera.stop()
        web_thread.join(timeout=2)
        print("Tuning server stopped. Use Save profile before stopping to keep changed values.", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Interactive detection-to-spray accuracy and timing test.

The camera keeps running in the background, but a frame is processed and given
new detection IDs only when the operator enters ``next``.  ``spray <id>`` uses
the same coordinate, side-selection, nozzle-offset, and angle logic as the
production controller, aims the appropriate servo, then waits for a separate
Enter press before firing that side for exactly 0.5 seconds.

Use water and a paper target.  Results are appended to spray_accuracy_test.csv
so measured hit offsets can be added after each physical test.
"""

import argparse
import csv
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import detection_worldcoord as detector
from deepstream_pipeline import DEEPSTREAM_AVAILABLE, DEEPSTREAM_UNAVAILABLE_REASON, DeepStreamCamera
from spray_actuation import SERVO_LIMITS_DEG, SERVO_SETTLE_S, SprayActuator
import sprayer_control as sprayer_geometry
from sprayer_control import (camera_local_to_global, frame_half_side, lateral_gap,
                             offset_to_angle, sprayer_global_position)


SPRAY_DURATION_S = 0.5
CSV_FIELDS = (
    "detection_id", "detected_at", "side", "world_x_m", "world_y_m",
    "nozzle_lateral_gap_m", "requested_angle_deg", "commanded_angle_deg",
    "aim_delay_s", "operator_confirm_delay_s", "spray_duration_s",
    "spray_started_at", "spray_ended_at", "measured_hit_offset_m", "notes",
)


class WebCameraPreview:
    """Loopback-only browser GUI for the manual accuracy-test workflow."""

    def __init__(self, camera, actuator, args, port):
        self.camera = camera
        self.actuator = actuator
        self.args = args
        self.port = port
        self._jpeg = None
        self._lock = threading.Lock()
        self._action_lock = threading.Lock()
        self._stop = threading.Event()
        self._server = None
        self._thread = threading.Thread(target=self._encode_frames, daemon=True)
        self.detections = {}
        self.pending = None

    @staticmethod
    def _html():
        return """<!doctype html><html><head><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Spray accuracy test</title><style>
body{margin:0;background:#101512;color:#edf4ec;font:16px system-ui,sans-serif}main{max-width:1280px;margin:auto;padding:16px}
h1{margin:0 0 12px}.layout{display:grid;grid-template-columns:minmax(0,2fr) minmax(300px,1fr);gap:16px}img{width:100%;background:#000;border-radius:8px}.panel{background:#1b241d;border-radius:8px;padding:14px}
button{font:inherit;font-weight:700;padding:11px 15px;margin:4px;border:0;border-radius:6px;background:#82c784;color:#102010;cursor:pointer}button.spray{background:#ffb74d}button.confirm{background:#ef5350;color:#fff}button:disabled{opacity:.45;cursor:not-allowed}
table{width:100%;border-collapse:collapse;margin-top:10px}td,th{padding:7px;text-align:left;border-bottom:1px solid #3c4b3e}tr.selected{background:#354a38}#status{min-height:44px;padding:9px;background:#0a0d0b;border-radius:6px;white-space:pre-wrap}.small{color:#b8c7ba;font-size:.9em}@media(max-width:800px){.layout{grid-template-columns:1fr}}</style></head>
<body><main><h1>Spray accuracy test</h1><div class='layout'><section><img src='/camera.mjpg'><p class='small'>Live camera. Detected targets are marked with red IDs after Next.</p></section><section class='panel'>
<label>Tractor lift (m) <input id='lift' type='number' min='0' max='2' step='0.01' value='0.22'></label>
<label>Weed mode <select id='mode'><option value='yolo'>YOLO</option><option value='gi'>GI (green index)</option></select></label>
<button onclick='saveSettings()'>Apply settings</button><br>
<button onclick='nextFrame()'>Next: detect current frame</button><button class='spray' id='aim' disabled onclick='aim()'>Spray: aim selected ID</button><button class='confirm' id='fire' disabled onclick='fireSpray()'>Confirm: spray 0.5 s</button>
<div id='status'>Waiting for camera…</div><table><thead><tr><th>Select</th><th>ID</th><th>X (m)</th><th>Y (m)</th></tr></thead><tbody id='targets'></tbody></table>
</section></div></main><script>
let selected=null; const status=t=>document.getElementById('status').textContent=t;
async function request(path, body={}){const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const raw=await r.text();let d;try{d=JSON.parse(raw)}catch(_){throw Error(`GUI server returned HTML instead of an API response (HTTP ${r.status}). Stop the old runner, restart this script, then reload this page.`)}if(!r.ok)throw Error(d.error||'Request failed');return d}
function render(ds){const t=document.getElementById('targets');t.innerHTML='';selected=null;document.getElementById('aim').disabled=true;document.getElementById('fire').disabled=true;for(const d of ds){const row=document.createElement('tr');row.innerHTML=`<td><input type='radio' name='id'></td><td>${d.id}</td><td>${d.x.toFixed(3)}</td><td>${d.y.toFixed(3)}</td>`;row.onclick=()=>{selected=d.id;[...t.rows].forEach(x=>x.classList.remove('selected'));row.classList.add('selected');row.querySelector('input').checked=true;document.getElementById('aim').disabled=false};t.appendChild(row)} }
async function nextFrame(){try{status('Detecting current frame…');const d=await request('/api/next');render(d.detections);status(`Detected ${d.detections.length} weed(s). Select an ID, then press Spray.`)}catch(e){status('ERROR: '+e.message)}}
async function saveSettings(){try{const d=await request('/api/settings',{tractor_lift_m:Number(document.getElementById('lift').value),weed_mode:document.getElementById('mode').value});status(`Applied: lift ${d.tractor_lift_m.toFixed(2)} m, ${d.weed_mode.toUpperCase()} mode.`)}catch(e){status('ERROR: '+e.message)}}
async function aim(){try{if(selected===null)return;status('Moving servo to calculated target angle…');const d=await request('/api/aim',{id:selected});document.getElementById('fire').disabled=false;status(`ID ${d.id}: ${d.side} servo aimed at ${d.angle.toFixed(1)}°. Verify target, then press Confirm.`)}catch(e){status('ERROR: '+e.message)}}
async function fireSpray(){try{if(selected===null)return;document.getElementById('fire').disabled=true;status('SPRAYING for 0.5 s…');const d=await request('/api/fire',{id:selected});status(`Complete: ID ${d.id}, ${d.side}, ${d.duration.toFixed(3)} s. Logged to CSV.`)}catch(e){status('ERROR: '+e.message)}}
</script></body></html>"""

    @staticmethod
    def _body(request):
        length = int(request.headers.get("Content-Length", "0"))
        return json.loads(request.rfile.read(length) or b"{}")

    @staticmethod
    def _send_json(response, status, data):
        payload = json.dumps(data).encode("utf-8")
        response.send_response(status)
        response.send_header("Content-Type", "application/json")
        response.send_header("Content-Length", str(len(payload)))
        response.end_headers()
        response.wfile.write(payload)

    def _detect(self):
        detections = detect_once(self.camera, self.args)
        with self._lock:
            self.detections = {int(det["id"]): det for det in detections}
            self.pending = None
        return [{"id": int(det["id"]), "x": float(det["x"]), "y": float(det["y"])} for det in detections]

    def _settings(self, body):
        lift_m = float(body.get("tractor_lift_m", self.args.lift_height_cm / 100.0))
        mode = str(body.get("weed_mode", self.args.weed_mode))
        if not 0.0 <= lift_m <= 2.0:
            raise ValueError("tractor_lift_m must be 0..2 metres")
        if mode not in ("yolo", "gi"):
            raise ValueError("weed_mode must be yolo or gi")
        self.args.lift_height_cm = lift_m * 100.0
        self.args.weed_mode = mode
        sprayer_geometry.set_tractor_lift_height(lift_m)
        return {"tractor_lift_m": lift_m, "weed_mode": mode}

    def _aim(self, target_id):
        with self._lock:
            target = self.detections.get(target_id)
        if target is None:
            raise ValueError("Unknown target ID. Press Next and select a detected weed.")
        side, gap, requested, commanded, aimed_at = prepare_spray(target, self.actuator)
        with self._lock:
            self.pending = {"target": target, "side": side, "gap": gap, "requested": requested,
                            "commanded": commanded, "aimed_at": aimed_at}
        return {"id": target_id, "side": side, "angle": commanded}

    def _fire(self, target_id):
        with self._lock:
            pending = self.pending
        if pending is None or int(pending["target"]["id"]) != target_id:
            raise ValueError("Press Spray to aim this selected ID before confirming the spray.")
        started_at = time.monotonic()
        self.actuator.fire_side(pending["side"], duration_s=SPRAY_DURATION_S)
        ended_at = time.monotonic()
        target = pending["target"]
        append_result(self.args.csv, {
            "detection_id": target_id, "detected_at": target["_detected_at"], "side": pending["side"],
            "world_x_m": target["x"], "world_y_m": target["y"], "nozzle_lateral_gap_m": pending["gap"],
            "requested_angle_deg": pending["requested"], "commanded_angle_deg": pending["commanded"],
            "aim_delay_s": pending["aimed_at"] - target["_detected_at"],
            "operator_confirm_delay_s": started_at - pending["aimed_at"], "spray_duration_s": ended_at - started_at,
            "spray_started_at": started_at, "spray_ended_at": ended_at,
        })
        with self._lock:
            self.pending = None
        return {"id": target_id, "side": pending["side"], "duration": ended_at - started_at}

    def start(self):
        preview = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path in ("/", "/index.html"):
                    body = preview._html().encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if self.path != "/camera.mjpg":
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while not preview._stop.is_set():
                        with preview._lock:
                            jpeg = preview._jpeg
                        if jpeg is not None:
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
                        time.sleep(0.05)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def do_POST(self):
                try:
                    with preview._action_lock:
                        if self.path == "/api/next":
                            result = {"detections": preview._detect()}
                        elif self.path == "/api/settings":
                            result = preview._settings(preview._body(self))
                        elif self.path == "/api/aim":
                            result = preview._aim(int(preview._body(self)["id"]))
                        elif self.path == "/api/fire":
                            result = preview._fire(int(preview._body(self)["id"]))
                        else:
                            preview._send_json(self, 404, {"error": "Unknown GUI API endpoint."})
                            return
                    preview._send_json(self, 200, result)
                except (KeyError, TypeError, ValueError) as exc:
                    preview._send_json(self, 400, {"error": str(exc)})
                except Exception as exc:
                    print(f"[GUI] Action failed: {exc}")
                    preview._send_json(self, 500, {"error": str(exc)})

            def log_message(self, _format, *_args):
                pass

        # Loopback-only: the camera feed is accessible only from this machine.
        self._server = ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        self._thread.start()

    def _encode_frames(self):
        while not self._stop.is_set():
            frame, *_ = self.camera.get_latest(timeout=0.2)
            if frame is None:
                continue
            display = frame.copy()
            with self._lock:
                detections = list(self.detections.values())
            for det in detections:
                u, v = (int(round(value)) for value in det["pixel"])
                cv2.circle(display, (u, v), 16, (0, 0, 255), 2)
                cv2.putText(display, f"ID {det['id']}", (u + 18, max(22, v - 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            ok, encoded = cv2.imencode(".jpg", display)
            if ok:
                with self._lock:
                    self._jpeg = encoded.tobytes()

    def stop(self):
        self._stop.set()
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        self._thread.join(timeout=1.0)


def parse_source(value):
    return int(value) if str(value).isdigit() else value


def clamp_angle(side, angle):
    low, high = SERVO_LIMITS_DEG[side]
    return max(low, min(high, angle))


def append_result(path, result):
    new_file = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow({name: result.get(name, "") for name in CSV_FIELDS})


def detect_once(camera, args):
    # Keep this order synchronized with DeepStreamCamera.get_latest_tracking:
    # frame, plants, plant IDs, weed boxes, weed IDs, mask, FPS, sequence.
    frame, plants, plant_ids, weed_boxes, weed_ids, _weed_mask, _fps, sequence = camera.get_latest_tracking(timeout=2.0)
    if frame is None:
        print("[DETECT] No camera frame is available yet.")
        return []
    annotated, detections, _ = detector.process_frame(
        frame, plants, weed_mask=None, exg_threshold=args.exg,
        min_area=args.min_area, max_box_area=args.max_area,
        cluster_area_threshold=args.cluster_area, separation_px=args.separation,
        plant_track_ids=plant_ids, tracked_weed_boxes=weed_boxes,
        weed_track_ids=weed_ids, pipeline_compatible=True,
        weed_mode=args.weed_mode,
    )
    # Deliberately retain this value for an audit trail; frames are only
    # processed at the operator's command, never continuously.
    del annotated
    captured_at = time.monotonic()
    print(f"[DETECT] frame={sequence}, detections={len(detections)}")
    for det in detections:
        print(f"  id={det['id']:>4}  camera=(x={det['x']:+.3f}, y={det['y']:+.3f}) m  pixel={det['pixel']}")
        det["_detected_at"] = captured_at
    return detections


def prepare_spray(target, actuator):
    """Aim using the stationary (0, 0, 0) test-rig pose used by the pipeline."""
    pose = (0.0, 0.0, 0.0)
    world_target = camera_local_to_global(target, pose)
    side = frame_half_side(target)
    nozzle_position = sprayer_global_position(pose, side)
    gap = lateral_gap(world_target, nozzle_position, pose[2])
    requested_angle = offset_to_angle(gap, side, sprayer_geometry.SPRAYER_HEIGHT_M)
    commanded_angle = clamp_angle(side, requested_angle)

    aimed_at = time.monotonic()
    actuator.aim(side, requested_angle)  # driver independently enforces the same limits
    print(
        f"[AIM] id={target['id']} side={side}; lateral gap={gap:+.3f} m; "
        f"angle={requested_angle:.1f}° -> commanded={commanded_angle:.1f}° "
        f"(limit {SERVO_LIMITS_DEG[side][0]:.0f}..{SERVO_LIMITS_DEG[side][1]:.0f}°)."
    )
    return side, gap, requested_angle, commanded_angle, aimed_at


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=detector.VIDEO_SOURCE, help="CSI camera id or video path (default: 0)")
    parser.add_argument("--config", default="config_infer_primary_yolo.txt")
    parser.add_argument("--weed-config", default="config_infer_weed_yolov8.txt")
    parser.add_argument("--csv", default="spray_accuracy_test.csv", help="CSV log path")
    parser.add_argument("--exg", type=int, default=detector.EXG_THRESHOLD)
    parser.add_argument("--min-area", type=int, default=detector.MIN_CONTOUR_AREA)
    parser.add_argument("--max-area", type=int, default=detector.MAX_WEED_BOX_AREA)
    parser.add_argument("--cluster-area", type=int, default=detector.CLUSTER_AREA_THRESHOLD)
    parser.add_argument("--separation", type=int, default=detector.SEPARATION_KERNEL_PX)
    parser.add_argument("--lift-height-cm", type=float, default=sprayer_geometry.TRACTOR_LIFT_HEIGHT_M * 100.0,
                        help="common tractor lift added to camera and nozzle heights (default: 22 cm)")
    parser.add_argument("--weed-mode", choices=("yolo", "gi"), default="yolo",
                        help="YOLO uses the secondary DeepStream weed model; GI uses the earlier ExG path")
    parser.add_argument("--simulate", action="store_true", help="permit a no-hardware dry run when UART is unavailable")
    parser.add_argument("--preview-port", type=int, default=8081, help="loopback GUI port (default: 8081)")
    args = parser.parse_args()

    if not DEEPSTREAM_AVAILABLE:
        print(f"ERROR: DeepStream is unavailable ({DEEPSTREAM_UNAVAILABLE_REASON}).")
        return 2

    if args.lift_height_cm < 0:
        parser.error("--lift-height-cm cannot be negative")
    sprayer_geometry.set_tractor_lift_height(args.lift_height_cm / 100.0)
    detector.reset_detection_ids()
    actuator = SprayActuator()
    if not actuator._link.available and not args.simulate:
        print("ERROR: ESP32 UART is unavailable. Refusing to run a hardware test.")
        if actuator._link.last_error:
            print(f"UART error: {actuator._link.last_error}")
        print("Fix the connection or use --simulate for a dry run.")
        actuator.shutdown()
        return 2

    camera = None
    preview = None
    try:
        camera = DeepStreamCamera(
            source=parse_source(args.source), config_path=args.config, exg_threshold=args.exg,
            weed_config_path=args.weed_config,
            weed_min_area=args.min_area, weed_max_box_area=args.max_area,
            weed_cluster_area=args.cluster_area, weed_separation_px=args.separation,
            enable_optical_flow=False, enable_weed_mask=False, use_updated=True,
            enable_native_weed_bboxes=False,
            enable_weed_yolo=True,
        )
        camera.start()
        preview = WebCameraPreview(camera, actuator, args, args.preview_port)
        preview.start()
        print(f"GUI ready at http://localhost:{args.preview_port}")
        print("Use Ctrl+C here to stop; the GUI controls Next, target selection, aim, and confirmed 0.5 s spray.")
        while True:
            time.sleep(1.0)
    except (KeyboardInterrupt, EOFError):
        print("\nTest stopped by operator.")
    finally:
        if preview is not None:
            preview.stop()
        if camera is not None:
            camera.stop()
        actuator.shutdown()
        print("Valves and pump OFF; servos parked.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

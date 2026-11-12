"""
web_gui.py -- browser-based version of weed_sprayer_gui.py.

Runs the exact same backend pipeline (DeepStreamCamera -> detector.process_frame
-> SprayerController) as the Tkinter app, but serves it as a web page instead
of a native window, so it can be viewed and controlled from any browser --
including over HTTPS once reverse-proxied (see the bottom of this file for
deployment notes). Does not import or modify weed_sprayer_gui.py; it is a
standalone sibling that talks to the same underlying modules.

Run:
    python3 web_gui.py

Then browse to http://<jetson-ip>:8000 (see bottom of file for HTTPS).

SECURITY: this panel can trigger a physical sprayer.  It intentionally has no
application login for an isolated, firewall-controlled deployment.  Never put
the port directly on the public internet; use a VPN or authenticated reverse
proxy outside this process when remote access is required.
"""
import csv
import os
import threading
import time

import cv2
from flask import Flask, Response, jsonify, request

import detection_worldcoord as detector
import sprayer_control as sprayer_geometry
from sprayer_control import SprayerController

try:
    from deepstream_pipeline import (
        DEEPSTREAM_AVAILABLE,
        DEEPSTREAM_UNAVAILABLE_REASON,
        DeepStreamCamera,
    )
except Exception as exc:  # pragma: no cover - mirrors weed_sprayer_gui.py's own fallback
    DeepStreamCamera = None
    DEEPSTREAM_AVAILABLE = False
    DEEPSTREAM_UNAVAILABLE_REASON = f"{type(exc).__name__}: {exc}"

ANNOTATIONS_FILE = "detection_annotations.csv"
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DEEPSTREAM_CONFIG_PATH = os.path.join(PROJECT_DIR, "config_infer_primary_yolo.txt")
WEED_DEEPSTREAM_CONFIG_PATH = os.path.join(PROJECT_DIR, "config_infer_weed_yolov8.txt")

app = Flask(__name__)


# ---------------------------------------------------------------------------
# Shared state, filled in by the background worker thread and read by routes.
# Mirrors the Tk app's queue.Queue()s / trace_add() pattern, just without Tk.
# ---------------------------------------------------------------------------
class State:
    lock = threading.Lock()
    running = False
    stopping = False
    session_id = 0
    paused = False
    status = "Stopped"
    fps = 0.0
    jpeg = None
    detections = {}   # id -> dict(id, x, y, z, ground_truth, sprayed)
    logs = []          # capped list of recent log lines
    last_detection_signature = ()
    params = {
        "exg": detector.EXG_THRESHOLD,
        "min_area": detector.MIN_CONTOUR_AREA,
        "max_area": detector.MAX_WEED_BOX_AREA,
        "cluster": detector.CLUSTER_AREA_THRESHOLD,
        "separation": detector.SEPARATION_KERNEL_PX,
        "flip": 0,
        "tractor_lift_m": sprayer_geometry.TRACTOR_LIFT_HEIGHT_M,
        "weed_mode": "yolo",
    }
    source = str(detector.VIDEO_SOURCE)
    worker = None
    sprayer = None


def _append_ui_log(message):
    line = f"{time.strftime('%H:%M:%S')} {message}"
    with State.lock:
        State.logs.append(line)
        State.logs[:] = State.logs[-200:]


def log(message):
    """Keep diagnostic output in the terminal; expose actuator events only."""
    print(f"{time.strftime('%H:%M:%S')} {message}", flush=True)
    if message.startswith(("[AIM]", "[ACTUATOR]", "[SPRAY]", "[MISS]")):
        _append_ui_log(message)


# ---------------------------------------------------------------------------
# Worker thread: identical shape to WeedSprayerApp._capture_worker.
# ---------------------------------------------------------------------------
def _capture_worker(source, session_id):
    cam = None
    try:
        with State.lock:
            params = State.params.copy()
        cam = DeepStreamCamera(
            source=source, config_path=DEEPSTREAM_CONFIG_PATH,
            weed_config_path=WEED_DEEPSTREAM_CONFIG_PATH,
            exg_threshold=params["exg"],
            flip_method=params.get("flip", 0),
            enable_optical_flow=True,
            enable_weed_mask=False,  # preview original pixels only
            enable_native_weed_bboxes=False,
            enable_weed_yolo=True,
            use_updated=True,
        )
        with State.lock:
            State.params.update({
                "exg": cam.exg_threshold,
                "min_area": cam.weed_min_area,
                "max_area": cam.weed_max_box_area,
                "cluster": cam.weed_cluster_area,
                "separation": cam.weed_separation_px,
            })
        cam.start()
        kind = "camera" if cam.is_camera else "file/stream"
        log(f"[SYSTEM] DeepStream started ({kind}: {source}); "
            f"encoded={cam.width}x{cam.height}, output={cam.output_width}x{cam.output_height}, "
            f"display-rotation={cam.display_rotation}deg, flip-method={cam.output_flip_method}")
        log("[SYSTEM] Chilli YOLO is green; weed YOLO is red. GI mode uses the earlier ExG path.")

        with State.lock:
            State.status = "Running"

        last_perf_log = 0.0
        last_sequence = -1
        last_fps_time = None
        fps_ema = 0.0

        while True:
            with State.lock:
                # A Stop request invalidates this worker before it has had a
                # chance to leave DeepStream.  Do not let a quick subsequent
                # Start make this old worker resume and contend for the CSI
                # camera with the new one.
                if not State.running or session_id != State.session_id:
                    break
                paused = State.paused
                params = State.params.copy()
            if paused:
                time.sleep(0.05)
                continue
            sprayer_geometry.set_tractor_lift_height(params["tractor_lift_m"])

            (frame, plants, plant_ids, weed_boxes, weed_ids,
             weed_mask, camera_fps, sequence) = cam.get_latest_tracking(timeout=0.2)
            if frame is None or sequence == last_sequence:
                time.sleep(0.002)
                continue
            last_sequence = sequence

            flow_x, flow_y, _ = cam.get_optical_flow()
            if flow_x != 0.0 or flow_y != 0.0:
                State.sprayer.update_optical_flow(flow_x, flow_y)

            started = time.perf_counter()
            annotated, detections, mask = detector.process_frame(
                frame, plants, weed_mask=None,
                exg_threshold=params["exg"], min_area=params["min_area"],
                cluster_area_threshold=params["cluster"],
                separation_px=params["separation"], max_box_area=params["max_area"],
                plant_track_ids=plant_ids, tracked_weed_boxes=weed_boxes,
                weed_track_ids=weed_ids, weed_mode=params["weed_mode"])
            State.sprayer.add_detections(detections)
            signature = (
                tuple(sorted(track_id for track_id in plant_ids if track_id >= 0)),
                tuple(sorted(int(det["id"]) for det in detections)),
            )
            with State.lock:
                changed = signature != State.last_detection_signature
                State.last_detection_signature = signature
            if changed:
                for track_id in signature[0]:
                    _append_ui_log(f"Chilli plant detected: index={track_id}")
                for det in detections:
                    _append_ui_log(
                        f"Weed detected: index={det['id']} side="
                        f"{sprayer_geometry.frame_half_side(det)} sprayed=no"
                    )
            elapsed_ms = (time.perf_counter() - started) * 1000

            now = time.perf_counter()
            if last_fps_time is not None and now - last_fps_time > 0:
                inst_fps = 1.0 / (now - last_fps_time)
                fps_ema = inst_fps if not fps_ema else fps_ema + 0.2 * (inst_fps - fps_ema)
            last_fps_time = now

            # Overlay actuator firing indicator if active
            if State.sprayer and hasattr(State.sprayer, "is_actuator_active") and State.sprayer.is_actuator_active():
                cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 46), (0, 160, 0), -1)
                cv2.putText(annotated, "*** SPRAYING: VALVES & PUMP ACTIVE (100ms) ***", (24, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)

            ok, buf = cv2.imencode(".jpg", annotated)
            with State.lock:
                if ok:
                    State.jpeg = buf.tobytes()
                State.fps = fps_ema
                State.status = f"Running - {len(detections)} detections"
                existing = State.detections
                State.detections = {
                    det["id"]: {
                        **det,
                        "ground_truth": existing.get(det["id"], {}).get("ground_truth", ""),
                        "sprayed": existing.get(det["id"], {}).get("sprayed", ""),
                    }
                    for det in detections
                }

            now_mono = time.monotonic()
            if now_mono - last_perf_log >= 2.0:
                vegetation = 100.0 * cv2.countNonZero(mask) / mask.size if mask is not None else 0.0
                dist_m = State.sprayer.get_distance()
                log(f"[PERF] postprocess={elapsed_ms:.0f} ms, DeepStream={camera_fps:.1f} FPS, "
                    f"dist={dist_m:.2f} m, flow=({flow_x:.1f}, {flow_y:.1f}), "
                    f"weeds={len(detections)}, vegetation={vegetation:.1f}%")
                last_perf_log = now_mono
    except Exception as exc:
        log(f"[ERROR] worker: {exc}")
    finally:
        if cam is not None:
            cam.stop()
        with State.lock:
            # A newer worker owns the state after a later Start request.
            if session_id == State.session_id:
                State.running = False
                State.stopping = False
                State.status = "Stopped"
                State.jpeg = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return INDEX_HTML


@app.route("/stream.mjpg")
def stream():
    def generate():
        while True:
            with State.lock:
                jpg = State.jpeg
            if jpg is not None:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
            time.sleep(0.03)
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/state")
def api_state():
    with State.lock:
        return jsonify({
            "running": State.running,
            "stopping": State.stopping,
            "paused": State.paused,
            "status": State.status,
            "fps": round(State.fps, 1),
            "params": State.params,
            "source": State.source,
            "detections": list(State.detections.values()),
            "logs": State.logs[-100:],
            "actuator_active": State.sprayer.is_actuator_active() if State.sprayer else False,
            "deepstream_available": DEEPSTREAM_AVAILABLE,
            "deepstream_reason": DEEPSTREAM_UNAVAILABLE_REASON,
        })


def _start_pipeline(raw_source):
    if not DEEPSTREAM_AVAILABLE:
        raise RuntimeError(f"DeepStream unavailable: {DEEPSTREAM_UNAVAILABLE_REASON}")
    with State.lock:
        if State.running or State.stopping or (State.worker and State.worker.is_alive()):
            return False
        raw_source = str(raw_source).strip()
        source = int(raw_source) if raw_source.isdigit() else raw_source
        State.source = raw_source
        State.detections = {}
        State.status = "Starting..."
        State.running = True
        State.stopping = False
        State.paused = False
        State.jpeg = None
        State.fps = 0.0
        State.session_id += 1
        session_id = State.session_id
    detector.reset_detection_ids()
    State.sprayer = SprayerController(log_callback=log)
    State.sprayer.reset()
    State.sprayer.start()
    State.worker = threading.Thread(target=_capture_worker, args=(source, session_id), daemon=True)
    State.worker.start()
    return True


@app.route("/api/start", methods=["POST"])
def api_start():
    try:
        started = _start_pipeline((request.json or {}).get("source", State.source))
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 400
    if not started:
        return jsonify({"error": "pipeline is already running or is still stopping"}), 400
    return jsonify({"ok": True})


@app.route("/api/stop", methods=["POST"])
def api_stop():
    with State.lock:
        State.running = False
        State.stopping = True
        State.session_id += 1
        State.paused = False
        State.status = "Stopping..."
    if State.sprayer:
        State.sprayer.stop()
    log("[SYSTEM] Stop requested")
    return jsonify({"ok": True, "status": "Stopping..."})


@app.route("/api/pause", methods=["POST"])
def api_pause():
    with State.lock:
        State.paused = not State.paused
        paused = State.paused
    return jsonify({"paused": paused})


@app.route("/api/settings", methods=["POST"])
def api_settings():
    body = request.json or {}
    with State.lock:
        if "tractor_lift_m" in body:
            lift_m = float(body["tractor_lift_m"])
            if not 0.0 <= lift_m <= 2.0:
                return jsonify({"error": "tractor_lift_m must be 0..2 metres"}), 400
            State.params["tractor_lift_m"] = lift_m
        if "weed_mode" in body:
            mode = str(body["weed_mode"])
            if mode not in ("yolo", "gi"):
                return jsonify({"error": "weed_mode must be yolo or gi"}), 400
            State.params["weed_mode"] = mode
        params = State.params.copy()
    sprayer_geometry.set_tractor_lift_height(params["tractor_lift_m"])
    return jsonify({"ok": True, "params": params})


@app.route("/api/annotate", methods=["POST"])
def api_annotate():
    body = request.json or {}
    weed_id = body.get("id")
    with State.lock:
        item = State.detections.get(weed_id)
        if item is None:
            return jsonify({"error": "unknown detection id"}), 404
        item["ground_truth"] = str(body.get("ground_truth", "")).strip()
        item["sprayed"] = str(body.get("sprayed", "")).strip()
        snapshot = item.copy()
    new_file = not os.path.exists(ANNOTATIONS_FILE)
    with open(ANNOTATIONS_FILE, "a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=(
            "timestamp", "detection_index", "ground_truth", "sprayed",
            "world_x", "world_y", "world_z",
        ))
        if new_file:
            writer.writeheader()
        writer.writerow({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "detection_index": snapshot["id"],
            "ground_truth": snapshot["ground_truth"],
            "sprayed": snapshot["sprayed"],
            "world_x": snapshot["x"],
            "world_y": snapshot["y"],
            "world_z": snapshot.get("z", ""),
        })
    log(f"[ANNOTATION] id={weed_id} ground_truth={snapshot['ground_truth']!r} sprayed={snapshot['sprayed']}")
    return jsonify({"ok": True})


INDEX_HTML = """<!doctype html>
<html><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Weed Detection and Sprayer Control</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 0; background: #f4f4f4; color: #222; }
  header { background: #2a2a2a; color: #fff; padding: 10px 16px; display: flex; gap: 16px; align-items: center; flex-wrap: wrap; }
  header input[type=text] { flex: 1; min-width: 180px; padding: 6px 8px; }
  header button { padding: 6px 14px; cursor: pointer; }
  main { display: flex; gap: 12px; padding: 12px; flex-wrap: wrap; }
  #video-col { flex: 3; min-width: 320px; }
  #video-col img { width: 100%; background: #000; border-radius: 6px; }
  #side-col { flex: 2; min-width: 300px; display: flex; flex-direction: column; gap: 12px; }
  .panel { background: #fff; border-radius: 6px; padding: 10px 12px; box-shadow: 0 1px 3px rgba(0,0,0,.15); }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { border-bottom: 1px solid #eee; padding: 4px 6px; text-align: center; }
  td input { width: 100%; box-sizing: border-box; font-size: 12px; }
  #log { height: 160px; overflow-y: auto; background: #111; color: #7fd; font-family: monospace; font-size: 12px; padding: 6px; border-radius: 4px; }
  .status-ok { color: #2a7; } .status-bad { color: #c33; }
</style>
</head><body>
<header>
  <strong>Weed Sprayer</strong>
  <input id="source" type="text" placeholder="camera / video source">
  <button id="btn-start">Start</button>
  <button id="btn-pause" disabled>Pause</button>
  <button id="btn-stop" disabled>Stop</button>
  <label title="Common vertical lift applied to camera and nozzles">Tractor lift (m)
    <input id="tractor-lift" type="number" min="0" max="2" step="0.01" value="0.22">
  </label>
  <label title="YOLO uses the second DeepStream model; GI uses the earlier green-index/ExG detection path">Weed mode
    <input id="weed-mode" type="range" min="0" max="1" step="1" value="0">
    <span id="weed-mode-value">YOLO</span>
  </label>
  <span id="status">Stopped</span>
  <span id="fps">-- FPS</span>
  <span id="actuator-status" style="background:#444; color:#aaa; padding:4px 10px; border-radius:14px; font-weight:bold; font-size:12px;">Valves: OFF | Pump: OFF</span>
</header>
<main>
  <div id="video-col">
    <div class="panel"><span style="color:#087d35;font-weight:bold;">Green: chilli YOLO + DeepStream track ID</span> &nbsp; <span style="color:#c22;font-weight:bold;">Red: weed + DeepStream NvSORT track ID</span></div>
    <img id="video" src="" alt="stream will appear here once started">
  </div>
  <div id="side-col">
    <div class="panel" style="flex:1; overflow:auto; max-height:260px;">
      <strong>Detections</strong>
      <table>
        <thead><tr><th>ID</th><th>X</th><th>Y</th><th>Ground truth</th><th>Sprayed</th><th></th></tr></thead>
        <tbody id="det-body"></tbody>
      </table>
    </div>
    <div class="panel">
      <strong>Log</strong>
      <div id="log"></div>
    </div>
  </div>
</main>
<script>
const $ = (id) => document.getElementById(id);
async function api(path, opts) {
  const res = await fetch(path, opts);
  return res.json();
}

async function saveSettings() {
  const lift = Number($("tractor-lift").value);
  const weedMode = $("weed-mode").value === "0" ? "yolo" : "gi";
  $("weed-mode-value").textContent = weedMode === "yolo" ? "YOLO" : "GI";
  await api("/api/settings", {method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({tractor_lift_m: lift, weed_mode: weedMode})});
}
$("tractor-lift").onchange = saveSettings;
$("weed-mode").oninput = () => { $("weed-mode-value").textContent = $("weed-mode").value === "0" ? "YOLO" : "GI"; };
$("weed-mode").onchange = saveSettings;


$("btn-start").onclick = async () => {
  await api("/api/start", {method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({source: $("source").value})});
  $("video").src = "/stream.mjpg?" + Date.now();
};
$("btn-stop").onclick = () => api("/api/stop", {method: "POST"});
$("btn-pause").onclick = () => api("/api/pause", {method: "POST"});

async function saveAnnotation(id) {
  const gt = document.getElementById("gt-" + id).value;
  const sp = document.getElementById("sp-" + id).value;
  await api("/api/annotate", {method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({id: Number(id), ground_truth: gt, sprayed: sp})});
}

function renderDetections(rows) {
  const body = $("det-body");
  body.innerHTML = "";
  rows.forEach(d => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${d.id}</td><td>${d.x.toFixed(2)}</td><td>${d.y.toFixed(2)}</td>
      <td><input id="gt-${d.id}" value="${d.ground_truth || ''}"></td>
      <td><input id="sp-${d.id}" value="${d.sprayed || ''}"></td>
      <td><button onclick="saveAnnotation(${d.id})">Save</button></td>`;
    body.appendChild(tr);
  });
}

async function poll() {
  const s = await api("/api/state");
  $("status").textContent = s.status;
  $("status").className = s.running ? "status-ok" : "status-bad";
  $("fps").textContent = s.fps ? `${s.fps.toFixed(1)} FPS` : "-- FPS";
  $("source").value = $("source").value || s.source;
  if (document.activeElement !== $("tractor-lift")) {
    $("tractor-lift").value = s.params.tractor_lift_m;
  }
  if (document.activeElement !== $("weed-mode")) {
    $("weed-mode").value = s.params.weed_mode === "gi" ? "1" : "0";
    $("weed-mode-value").textContent = s.params.weed_mode === "gi" ? "GI" : "YOLO";
  }
  $("btn-start").disabled = s.running || s.stopping;
  $("btn-stop").disabled = !s.running;
  $("btn-pause").disabled = !s.running;
  $("btn-pause").textContent = s.paused ? "Resume" : "Pause";

  const actBadge = $("actuator-status");
  if (s.actuator_active) {
    actBadge.innerText = "🟢 VALVES: ON | PUMP: ON (100ms)";
    actBadge.style.background = "#1b4332";
    actBadge.style.color = "#00e676";
  } else {
    actBadge.innerText = "Valves: OFF | Pump: OFF";
    actBadge.style.background = "#444";
    actBadge.style.color = "#aaa";
  }
  renderDetections(s.detections);
  $("log").innerHTML = s.logs.map(l => l.replace(/</g, "&lt;")).join("<br>");
  $("log").scrollTop = $("log").scrollHeight;
  setTimeout(poll, 800);
}
poll();
</script>
</body></html>
"""


if __name__ == "__main__":
    # Dev server is fine for a handful of viewers on your LAN. For a public
    # deployment, run this behind gunicorn/waitress + nginx (see notes below).
    if os.environ.get("WEBGUI_AUTOSTART", "0").lower() not in ("0", "false", "no"):
        try:
            _start_pipeline(State.source)
        except RuntimeError as exc:
            log(f"[ERROR] automatic startup failed: {exc}")
    app.run(host="0.0.0.0", port=int(os.environ.get("WEBGUI_PORT", "8000")), threaded=True)


# ---------------------------------------------------------------------------
# Getting this onto HTTPS
# ---------------------------------------------------------------------------
# This script only serves plain HTTP on port 8000. Three ways to get HTTPS
# in front of it, roughly fastest-to-set-up first:
#
# 1) Tailscale Funnel (easiest, no domain/cert management, works from
#    anywhere without opening router ports):
#       curl -fsSL https://tailscale.com/install.sh | sh
#       sudo tailscale up
#       sudo tailscale funnel 8000
#    Tailscale gives you a stable https://<device>.<tailnet>.ts.net URL with
#    a real, valid certificate. Good default choice for a small team.
#
# 2) nginx reverse proxy + Let's Encrypt (if you own a domain pointing at
#    this Jetson's public IP):
#       sudo apt install nginx certbot python3-certbot-nginx
#       # /etc/nginx/sites-available/weedsprayer:
#       #   server {
#       #       listen 80;
#       #       server_name weeds.yourdomain.com;
#       #       location / { proxy_pass http://127.0.0.1:8000;
#       #                    proxy_set_header Host $host;
#       #                    proxy_buffering off; }  # needed for the MJPEG stream
#       #   }
#       sudo ln -s /etc/nginx/sites-available/weedsprayer /etc/nginx/sites-enabled/
#       sudo nginx -t && sudo systemctl reload nginx
#       sudo certbot --nginx -d weeds.yourdomain.com
#
# 3) Quick self-signed cert for testing only (browsers will show a warning):
#       openssl req -x509 -newkey rsa:4096 -nodes -keyout key.pem -out cert.pem -days 365
#       app.run(host="0.0.0.0", port=8000, ssl_context=("cert.pem", "key.pem"))
#
# Either way, run this script itself with a real process manager for
# anything beyond quick testing, e.g.:
#       pip install waitress
#       waitress-serve --host=0.0.0.0 --port=8000 web_gui:app

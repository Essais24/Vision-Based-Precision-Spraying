"""Responsive desktop UI for detection_worldcoord.py and sprayer_control.py.

DeepStream (deepstream_pipeline.DeepStreamCamera) is the only inference
backend now -- there's no PyTorch/ultralytics fallback path anymore, so
this app requires DeepStream 9.1 Service Maker to actually start a
stream. It still imports cleanly without it (e.g. for editing on a dev
PC), it just disables the Start button and explains why.
"""

import os
import csv
import queue
import threading
import time
import argparse
import signal
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
from PIL import Image, ImageTk

import detection_worldcoord as detector
from sprayer_control import SprayerController, TRACTOR_LIFT_HEIGHT_M

try:
    from deepstream_pipeline import (
        DEEPSTREAM_AVAILABLE,
        DEEPSTREAM_UNAVAILABLE_REASON,
        DeepStreamCamera,
    )
except Exception as exc:
    DeepStreamCamera = None
    DEEPSTREAM_AVAILABLE = False
    DEEPSTREAM_UNAVAILABLE_REASON = f"{type(exc).__name__}: {exc}"

ANNOTATIONS_FILE = "detection_annotations.csv"
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DEEPSTREAM_CONFIG_PATH = os.path.join(PROJECT_DIR, "config_infer_primary_yolo.txt")
RUNS_DIR = os.path.join(PROJECT_DIR, "runs")
RECORD_FPS = 30.0


class WeedSprayerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Weed Detection and Sprayer Control")
        self.geometry("1280x780")
        self.minsize(1000, 650)

        self.running = threading.Event()
        self.paused = threading.Event()
        self.frames = queue.Queue(maxsize=2)
        self.annotated_frames = queue.Queue(maxsize=1)
        self.logs = queue.Queue()
        self.detections = {}
        self.worker = None
        self.preview_image = None
        self._has_captured_frame = False
        # Recording is deliberately independent of the inference/spray loop.
        # Its small queue drops recording frames under disk pressure rather
        # than ever adding actuator latency.
        self.recording = threading.Event()
        self.record_frames = queue.Queue(maxsize=3)
        self.record_worker = None
        self.record_path = None

        self.source_var = tk.StringVar(value=str(detector.VIDEO_SOURCE))
        self.status_var = tk.StringVar(value="Stopped")
        self.ground_truth_var = tk.StringVar()
        self.sprayed_var = tk.StringVar()
        self.exg_var = tk.IntVar(value=detector.EXG_THRESHOLD)
        # YOLO is the spray-safe default. ExG remains available only as an
        # explicitly selected fallback/testing mode.
        self.use_exg_fallback_var = tk.BooleanVar(value=False)
        self.min_area_var = tk.IntVar(value=detector.MIN_CONTOUR_AREA)
        self.max_area_var = tk.IntVar(value=detector.MAX_WEED_BOX_AREA)
        self.cluster_var = tk.IntVar(value=detector.CLUSTER_AREA_THRESHOLD)
        self.separation_var = tk.IntVar(value=detector.SEPARATION_KERNEL_PX)
        # Zero preserves video display metadata automatically. A non-zero
        # value is only for a physically rotated camera and overrides it.
        self.flip_var = tk.IntVar(value=0)
        self.lift_height_cm_var = tk.IntVar(value=round(TRACTOR_LIFT_HEIGHT_M * 100))

        # -- FPS tracking -------------------------------------------------
        # Updated from the DeepStream worker thread and read back on the
        # main thread inside _drain_queues, the same way self.logs /
        # self.detections already cross threads.
        self.infer_fps_var = tk.StringVar(value="Pipeline: -- FPS")
        self._fps_lock = threading.Lock()
        self._infer_fps = 0.0
        self._last_pipeline_completion = None
        self._fps_alpha = 0.2  # EMA smoothing factor (higher = reacts faster)

        self._params_lock = threading.Lock()
        self._params = {
            "exg": self.exg_var.get(),
            "use_exg_fallback": self.use_exg_fallback_var.get(),
            "min_area": self.min_area_var.get(),
            "max_area": self.max_area_var.get(),
            "cluster": self.cluster_var.get(),
            "separation": self.separation_var.get(),
            "flip": self.flip_var.get(),
            "lift_height_cm": self.lift_height_cm_var.get(),
        }
        for variable in (self.exg_var, self.use_exg_fallback_var, self.min_area_var, self.max_area_var,
                         self.cluster_var, self.separation_var, self.flip_var,
                         self.lift_height_cm_var):
            variable.trace_add("write", self._update_params)
        self.sprayer = SprayerController(log_callback=self._queue_log)

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(30, self._drain_queues)

    def _build_ui(self):
        toolbar = ttk.Frame(self, padding=(10, 10, 10, 6))
        toolbar.grid(row=0, column=0, sticky="ew")
        toolbar.columnconfigure(1, weight=1)
        ttk.Label(toolbar, text="Camera / video source").grid(row=0, column=0, sticky="w")
        ttk.Entry(toolbar, textvariable=self.source_var).grid(row=0, column=1, sticky="ew", padx=8)
        self.start_button = ttk.Button(toolbar, text="Start", command=self.start_stream)
        self.start_button.grid(row=0, column=2, padx=3)
        self.pause_button = ttk.Button(toolbar, text="Pause", command=self.toggle_pause, state="disabled")
        self.pause_button.grid(row=0, column=3, padx=3)
        self.stop_button = ttk.Button(toolbar, text="Stop", command=self.stop_stream, state="disabled")
        self.stop_button.grid(row=0, column=4, padx=3)
        self.record_button = ttk.Button(toolbar, text="Record", command=self.toggle_recording,
                                        state="disabled")
        self.record_button.grid(row=0, column=5, padx=3)
        ttk.Label(toolbar, textvariable=self.status_var).grid(row=0, column=6, padx=(14, 0))
        ttk.Label(toolbar, textvariable=self.infer_fps_var).grid(row=0, column=7, padx=(14, 0))

        self.actuator_status_var = tk.StringVar(value="Valves: OFF | Pump: OFF")
        self.actuator_badge = tk.Label(
            toolbar, textvariable=self.actuator_status_var,
            font=("Helvetica", 9, "bold"), bg="#333333", fg="#aaaaaa",
            padx=8, pady=3, relief="groove"
        )
        self.actuator_badge.grid(row=0, column=8, padx=(14, 0))

        if DEEPSTREAM_AVAILABLE:
            ttk.Label(
                toolbar, text="DeepStream 9.1 (Service Maker) inference backend",
                foreground="#2a7",
            ).grid(row=1, column=0, columnspan=5, sticky="w", pady=(4, 0))
        else:
            self.start_button.configure(state="disabled")
            detail = DEEPSTREAM_UNAVAILABLE_REASON or "pyservicemaker is not installed"
            ttk.Label(
                toolbar,
                text=(f"DeepStream 9.1 Service Maker unavailable ({detail}). "
                      "Install the DeepStream SDK on the Jetson to enable Start."),
                foreground="#c33",
                wraplength=900,
            ).grid(row=1, column=0, columnspan=6, sticky="w", pady=(4, 0))

        body = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        body.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        self.rowconfigure(1, weight=1)
        self.columnconfigure(0, weight=1)

        left = ttk.Frame(body, padding=4)
        right = ttk.Frame(body, padding=4)
        body.add(left, weight=3)
        body.add(right, weight=2)

        self.preview = ttk.Label(left, text="Start a camera or video stream", anchor="center")
        self.preview.pack(fill="both", expand=True)

        controls = ttk.LabelFrame(right, text="Detection Controls", padding=8)
        controls.pack(fill="x")
        ttk.Label(
            controls,
            text="Default: tracked weed-YOLO boxes are red; tracked chilli-YOLO boxes are green.",
            wraplength=300,
        ).pack(anchor="w", pady=(0, 6))
        ttk.Checkbutton(
            controls, text="Use ExG weed fallback (restart stream)",
            variable=self.use_exg_fallback_var,
        ).pack(anchor="w", pady=(0, 6))
        self._slider(controls, "ExG threshold", self.exg_var, 0, 100)
        self._slider(controls, "Minimum area", self.min_area_var, 100, 20000)
        self._slider(controls, "Maximum area", self.max_area_var, 1000, 200000)
        self._slider(controls, "Cluster area", self.cluster_var, 100, 100000)
        self._slider(controls, "Separation pixels", self.separation_var, 1, 41)
        self._slider(controls, "Manual camera flip (restart)", self.flip_var, 0, 4)
        self._slider(controls, "Tractor lift (cm)", self.lift_height_cm_var, 0, 200)

        table_frame = ttk.LabelFrame(right, text="Current Detections", padding=8)
        table_frame.pack(fill="both", expand=True, pady=(8, 0))
        columns = ("id", "world_x", "world_y", "ground_truth", "sprayed")
        self.table = ttk.Treeview(table_frame, columns=columns, show="headings", height=8)
        headings = ("Index", "X", "Y", "Ground Truth", "Sprayed")
        for column, heading in zip(columns, headings):
            self.table.heading(column, text=heading)
            self.table.column(column, width=88, anchor="center")
        self.table.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.table.yview)
        scrollbar.pack(side="right", fill="y")
        self.table.configure(yscrollcommand=scrollbar.set)
        self.table.bind("<<TreeviewSelect>>", self._select_detection)

        annotation = ttk.LabelFrame(right, text="Selected Detection", padding=8)
        annotation.pack(fill="x", pady=(8, 0))
        annotation.columnconfigure(1, weight=1)
        ttk.Label(annotation, text="Ground truth").grid(row=0, column=0, sticky="w")
        ttk.Entry(annotation, textvariable=self.ground_truth_var).grid(row=0, column=1, sticky="ew", padx=(8, 0))
        ttk.Label(annotation, text="Sprayed result").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(annotation, textvariable=self.sprayed_var).grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=(6, 0))
        actions = ttk.Frame(annotation)
        actions.grid(row=2, column=0, columnspan=2, sticky="e", pady=(8, 0))
        ttk.Button(actions, text="Save", command=self.save_annotation).pack(side="left", padx=3)

        log_frame = ttk.LabelFrame(self, text="Log Output", padding=8)
        log_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=(0, 10))
        self.rowconfigure(2, weight=0)
        self.log = tk.Text(log_frame, height=8, state="disabled", wrap="word")
        self.log.pack(fill="both", expand=True)

    @staticmethod
    def _slider(parent, text, variable, minimum, maximum):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text=text, width=16).pack(side="left")
        ttk.Scale(row, from_=minimum, to=maximum, variable=variable).pack(side="left", fill="x", expand=True)
        ttk.Label(row, textvariable=variable, width=6).pack(side="left", padx=(6, 0))

    def _queue_log(self, message):
        self.logs.put(f"{time.strftime('%H:%M:%S')} {message}")

    def _update_params(self, *_args):
        """Copy Tk-owned slider values into worker-safe plain Python values."""
        with self._params_lock:
            self._params = {
                "exg": self.exg_var.get(),
                "use_exg_fallback": self.use_exg_fallback_var.get(),
                "min_area": self.min_area_var.get(),
                "max_area": self.max_area_var.get(),
                "cluster": self.cluster_var.get(),
                "separation": self.separation_var.get(),
                "flip": self.flip_var.get(),
                "lift_height_cm": self.lift_height_cm_var.get(),
            }

    def _detection_params(self):
        with self._params_lock:
            return self._params.copy()

    @staticmethod
    def _apply_lift_height(lift_height_cm):
        from sprayer_control import set_tractor_lift_height
        set_tractor_lift_height(float(lift_height_cm) / 100.0)

    def _parse_source(self):
        source = self.source_var.get().strip()
        return int(source) if source.isdigit() else source

    def start_stream(self):
        if self.running.is_set():
            return
        if not DEEPSTREAM_AVAILABLE:
            messagebox.showerror(
                "DeepStream unavailable",
                "DeepStream 9.1 Service Maker isn't available in this Python "
                f"environment ({DEEPSTREAM_UNAVAILABLE_REASON or 'pyservicemaker missing'}).\n\n"
                "Install the matching DeepStream SDK on the Jetson and verify "
                "with: python3 -c \"import pyservicemaker\"",
            )
            return
        self.detections.clear()
        self._apply_lift_height(self._detection_params()["lift_height_cm"])
        for item in self.table.get_children():
            self.table.delete(item)
        for pending_queue in (self.frames, self.annotated_frames):
            while True:
                try:
                    pending_queue.get_nowait()
                except queue.Empty:
                    break
        detector.reset_detection_ids()
        self.sprayer.reset()
        self.sprayer.start()
        self.running.set()
        self.paused.clear()
        self._has_captured_frame = False
        with self._fps_lock:
            self._infer_fps = 0.0
            self._last_pipeline_completion = None
        self.infer_fps_var.set("Pipeline: -- FPS")

        self.worker = threading.Thread(target=self._capture_worker, daemon=True)
        self.worker.start()
        self.status_var.set("Starting...")
        self.start_button.configure(state="disabled")
        self.pause_button.configure(state="normal", text="Pause")
        self.stop_button.configure(state="normal")
        self.record_button.configure(state="normal", text="Record")

    def _record_pipeline_completion(self, completed_at):
        """EMA throughput of complete, unique frames from DeepStream through
        weed boxes, world coordinates, and the GUI hand-off."""
        with self._fps_lock:
            if self._last_pipeline_completion is not None:
                elapsed = completed_at - self._last_pipeline_completion
                if elapsed > 0:
                    inst_fps = 1.0 / elapsed
                    self._infer_fps = (
                        inst_fps if not self._infer_fps
                        else self._infer_fps + self._fps_alpha * (inst_fps - self._infer_fps)
                    )
            self._last_pipeline_completion = completed_at

    def _capture_worker(self):
        """Single-thread capture+inference path: the DeepStream pipeline
        (Service Maker -> nvstreammux -> nvinfer) runs YOLO internally via
        TensorRT, so this just pulls the latest (frame, plant_boxes) pair
        and overlays red ExG weed boxes plus green YOLO plant boxes on the
        original pixels. The displayed FPS is full completed-frame throughput."""
        cam = None
        try:
            source = self._parse_source()
            params = self._detection_params()
            cam = DeepStreamCamera(source=source, config_path=DEEPSTREAM_CONFIG_PATH,
                                   exg_threshold=params["exg"],
                                   flip_method=params.get("flip", 0),
                                   enable_optical_flow=True,
                                   # Enable the second full-frame weed model by
                                   # default. ExG is an explicit fallback mode.
                                   enable_weed_yolo=not params["use_exg_fallback"],
                                   enable_native_weed_bboxes=False,
                                   enable_weed_mask=False,  # preview original pixels only
                                   use_updated=True)
            cam.start()
            kind = "camera" if cam.is_camera else "file/stream"
            self._queue_log(
                f"[SYSTEM] DeepStream started ({kind}: {source}); "
                f"encoded={cam.width}x{cam.height}, output={cam.output_width}x{cam.output_height}, "
                f"display-rotation={cam.display_rotation}deg, flip-method={cam.output_flip_method}"
            )
            weed_mode = "gi" if params["use_exg_fallback"] else "yolo"
            self._queue_log(
                "[SYSTEM] Weed source: "
                + ("ExG fallback (YOLO weeds disabled)." if weed_mode == "gi"
                   else "YOLOv8 + NvSORT tracker IDs.")
            )
            last_perf_log = 0.0
            last_sequence = -1
            while self.running.is_set():
                if self.paused.is_set() and self._has_captured_frame:
                    time.sleep(0.05)
                    continue
                (frame, plants, plant_ids, weed_boxes, weed_ids, weed_mask,
                 camera_fps, sequence) = cam.get_latest_tracking(timeout=0.2)
                if frame is None or sequence == last_sequence:
                    time.sleep(0.002)
                    continue
                last_sequence = sequence
                frame_height, frame_width = frame.shape[:2]
                if (frame_width, frame_height) != (cam.output_width, cam.output_height):
                    raise RuntimeError(
                        "GUI received a frame with dimensions different from DeepStream output caps: "
                        f"received {frame_width}x{frame_height}, expected "
                        f"{cam.output_width}x{cam.output_height}"
                    )
                self._has_captured_frame = True
                # Raw pixels only: enqueue before GUI overlays and ExG
                # post-processing. The writer owns its copy because the
                # capture loop immediately reuses this frame.
                self._enqueue_recording_frame(frame)
                self._put_latest((frame, None))

                # Track camera motion vectors from DeepStream hardware Optical Flow
                flow_x, flow_y, _ = cam.get_optical_flow()
                if flow_x != 0.0 or flow_y != 0.0:
                    self.sprayer.update_optical_flow(flow_x, flow_y)

                params = self._detection_params()
                self._apply_lift_height(params["lift_height_cm"])
                started = time.perf_counter()
                # Both detector classes arrive atomically with their NvSORT
                # IDs. Weed YOLO is the default; ExG is used only when the
                # operator selected the fallback before starting this stream.
                annotated, detections, mask = detector.process_frame(
                    frame, plants, weed_mask=None,
                    exg_threshold=params["exg"], min_area=params["min_area"],
                    cluster_area_threshold=params["cluster"],
                    separation_px=params["separation"], max_box_area=params["max_area"],
                    plant_track_ids=plant_ids,
                    tracked_weed_boxes=weed_boxes if weed_mode == "yolo" else None,
                    weed_track_ids=weed_ids if weed_mode == "yolo" else None,
                    weed_mode=weed_mode)
                self.sprayer.add_detections(detections)
                self._put_latest((annotated, detections), target=self.annotated_frames)
                elapsed_ms = (time.perf_counter() - started) * 1000
                self._record_pipeline_completion(time.perf_counter())
                now = time.monotonic()
                if now - last_perf_log >= 2.0:
                    vegetation = 100.0 * cv2.countNonZero(mask) / mask.size
                    dist_m = self.sprayer.get_distance()
                    self._queue_log(
                        f"[PERF] postprocess={elapsed_ms:.0f} ms, DeepStream={camera_fps:.1f} FPS, "
                        f"dist={dist_m:.2f} m, flow=({flow_x:.1f}, {flow_y:.1f}), "
                        f"weeds={len(detections)}, vegetation={vegetation:.1f}%")
                    last_perf_log = now
        except Exception as exc:
            self._queue_log(f"[ERROR] DeepStream worker: {exc}")
        finally:
            self._stop_recording()
            if cam is not None:
                cam.stop()
            self.running.clear()

    def _put_latest(self, item, target=None):
        target = target or self.frames
        try:
            target.put_nowait(item)
        except queue.Full:
            try:
                target.get_nowait()
            except queue.Empty:
                pass
            target.put_nowait(item)

    def toggle_pause(self):
        if self.paused.is_set():
            self.paused.clear()
            self.pause_button.configure(text="Pause")
            self.status_var.set("Running")
        else:
            self.paused.set()
            self.pause_button.configure(text="Resume")
            self.status_var.set("Paused")

    def toggle_recording(self):
        if self.recording.is_set():
            self._stop_recording()
            return
        if not self.running.is_set():
            return
        os.makedirs(RUNS_DIR, exist_ok=True)
        self.record_path = os.path.join(
            RUNS_DIR, f"camera_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        )
        self.recording.set()
        self.record_worker = threading.Thread(target=self._record_worker, daemon=True)
        self.record_worker.start()
        self.record_button.configure(text="Stop recording")
        self._queue_log(f"[RECORD] Started raw camera recording: {self.record_path}")

    def _enqueue_recording_frame(self, frame):
        if not self.recording.is_set():
            return
        try:
            self.record_frames.put_nowait(frame.copy())
        except queue.Full:
            # Recording is best-effort by design; never block inference or
            # spraying for a slow filesystem/encoder.
            try:
                self.record_frames.get_nowait()
            except queue.Empty:
                pass
            try:
                self.record_frames.put_nowait(frame.copy())
            except queue.Full:
                pass

    def _record_worker(self):
        writer = None
        path = self.record_path
        try:
            while self.recording.is_set() or not self.record_frames.empty():
                try:
                    frame = self.record_frames.get(timeout=0.2)
                except queue.Empty:
                    continue
                if writer is None:
                    height, width = frame.shape[:2]
                    # Prefer H.264 if the local OpenCV/GStreamer build exposes
                    # it; MPEG-4 is a portable fallback on development hosts.
                    for codec in ("avc1", "mp4v"):
                        candidate = cv2.VideoWriter(
                            path, cv2.VideoWriter_fourcc(*codec), RECORD_FPS, (width, height)
                        )
                        if candidate.isOpened():
                            writer = candidate
                            self._queue_log(f"[RECORD] Encoder: {codec} ({width}x{height} @ {RECORD_FPS:g} FPS)")
                            break
                        candidate.release()
                    if writer is None:
                        raise RuntimeError("could not open an MP4 video encoder")
                writer.write(frame)
        except Exception as exc:
            self._queue_log(f"[RECORD][ERROR] {exc}")
        finally:
            if writer is not None:
                writer.release()

    def _stop_recording(self):
        was_recording = self.recording.is_set()
        self.recording.clear()
        worker = self.record_worker
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=3.0)
        self.record_worker = None
        if hasattr(self, "record_button"):
            self.record_button.configure(text="Record",
                                         state="normal" if self.running.is_set() else "disabled")
        if was_recording:
            self._queue_log(f"[RECORD] Saved raw camera recording: {self.record_path}")

    def stop_stream(self):
        self.running.clear()
        self.paused.clear()
        self.sprayer.stop()
        self.status_var.set("Stopped")
        self.start_button.configure(state="normal" if DEEPSTREAM_AVAILABLE else "disabled")
        self.pause_button.configure(state="disabled", text="Pause")
        self.stop_button.configure(state="disabled")
        self._stop_recording()
        self.infer_fps_var.set("Pipeline: -- FPS")
        self._queue_log("[SYSTEM] Stop requested")

    def _drain_queues(self):
        latest_raw = None
        latest_annotated = None
        while True:
            try:
                latest_raw = self.frames.get_nowait()
            except queue.Empty:
                break
        while True:
            try:
                latest_annotated = self.annotated_frames.get_nowait()
            except queue.Empty:
                break
        if latest_annotated is not None:
            self._show_frame(*latest_annotated)
        elif latest_raw is not None:
            self._show_frame(*latest_raw)
        while True:
            try:
                line = self.logs.get_nowait()
            except queue.Empty:
                break
            self.log.configure(state="normal")
            self.log.insert("end", line + "\n")
            self.log.see("end")
            self.log.configure(state="disabled")
        with self._fps_lock:
            infer_fps = self._infer_fps
        if infer_fps:
            self.infer_fps_var.set(f"Pipeline: {infer_fps:.1f} FPS")

        # Real-time hardware actuator indicator
        if hasattr(self, "sprayer") and hasattr(self.sprayer, "is_actuator_active"):
            if self.sprayer.is_actuator_active():
                self.actuator_status_var.set("🟢 VALVES: ON | PUMP: ON (100ms)")
                self.actuator_badge.configure(bg="#1b4332", fg="#00e676")
            else:
                self.actuator_status_var.set("Valves: OFF | Pump: OFF")
                self.actuator_badge.configure(bg="#333333", fg="#aaaaaa")

        if not self.running.is_set() and self.start_button["state"] == "disabled":
            self.stop_stream()
        self.after(30, self._drain_queues)

    def _show_frame(self, frame, detections):
        # Draw on-screen notification if actuators are actively firing
        if hasattr(self, "sprayer") and hasattr(self.sprayer, "is_actuator_active") and self.sprayer.is_actuator_active():
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 46), (0, 160, 0), -1)
            cv2.putText(frame, "*** SPRAYING: VALVES & PUMP ACTIVE (100ms) ***", (24, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)

        height, width = frame.shape[:2]
        max_width, max_height = 760, 600
        scale = min(max_width / width, max_height / height, 1.0)
        if scale != 1.0:
            frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.preview_image = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.preview.configure(image=self.preview_image, text="")
        if detections is None:
            self.status_var.set("Running - processing frame...")
            return
        self.status_var.set(f"Running - {len(detections)} detections")
        self.detections = {det["id"]: {**det, "ground_truth": "", "sprayed": ""} for det in detections}
        for item in self.table.get_children():
            self.table.delete(item)
        for det in detections:
            self.table.insert("", "end", iid=str(det["id"]), values=(det["id"], f"{det['x']:.2f}", f"{det['y']:.2f}", "", ""))

    def _selected_id(self):
        selected = self.table.selection()
        return int(selected[0]) if selected else None

    def _select_detection(self, _event=None):
        weed_id = self._selected_id()
        if weed_id is None or weed_id not in self.detections:
            return
        # Keep this frame visible while the operator records its result.
        if self.running.is_set() and not self.paused.is_set():
            self.paused.set()
            self.pause_button.configure(text="Resume")
            self.status_var.set("Paused for annotation")
            while True:
                try:
                    self.frames.get_nowait()
                except queue.Empty:
                    break
        item = self.detections[weed_id]
        self.ground_truth_var.set(item["ground_truth"])
        self.sprayed_var.set(item["sprayed"])

    def save_annotation(self):
        weed_id = self._selected_id()
        if weed_id is None:
            messagebox.showwarning("No detection", "Select a detection first.")
            return
        item = self.detections.get(weed_id)
        if item is None:
            return
        item["ground_truth"] = self.ground_truth_var.get().strip()
        item["sprayed"] = self.sprayed_var.get().strip()
        self.table.item(str(weed_id), values=(weed_id, f"{item['x']:.2f}", f"{item['y']:.2f}", item["ground_truth"], item["sprayed"]))
        self._write_annotation(item)
        self._queue_log(f"[ANNOTATION] id={weed_id} ground_truth={item['ground_truth']!r} sprayed={item['sprayed']}")

    @staticmethod
    def _write_annotation(item):
        """Append the operator inputs with the detection's displayed index."""
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
                "detection_index": item["id"],
                "ground_truth": item["ground_truth"],
                "sprayed": item["sprayed"],
                "world_x": item["x"],
                "world_y": item["y"],
                "world_z": item["z"],
            })

    def _on_close(self):
        self.running.clear()
        self._stop_recording()
        self.sprayer.stop()
        self.destroy()


def run_headless(source=detector.VIDEO_SOURCE,
                 exg=detector.EXG_THRESHOLD,
                 min_area=detector.MIN_CONTOUR_AREA,
                 max_area=detector.MAX_WEED_BOX_AREA,
                 cluster=detector.CLUSTER_AREA_THRESHOLD,
                 separation=detector.SEPARATION_KERNEL_PX,
                 flip=0):
    """Headless CLI runner for weed detection and sprayer control over SSH.

    Runs inference, coordinate transformation, and autonomous spraying without
    needing an X11 / GUI display.
    """
    print("=" * 68)
    print("🚀 Weed Detection & Sprayer Controller (Headless / SSH Mode)")
    print(f"   Camera / Source : {source} (Default CSI camera)")
    print(f"   Min Weed Area   : {min_area} px²")
    print(f"   Max Weed Area   : {max_area} px²")
    print(f"   Cluster Area    : {cluster} px²")
    print(f"   ExG Threshold   : {exg}")
    print("=" * 68)

    if not DEEPSTREAM_AVAILABLE:
        print(f"[ERROR] DeepStream 9.1 Service Maker unavailable ({DEEPSTREAM_UNAVAILABLE_REASON}).")
        print("Install the DeepStream SDK on the Jetson to run.")
        sys.exit(1)

    sprayer = SprayerController()
    detector.reset_detection_ids()
    sprayer.reset()
    sprayer.start()

    cam = None
    stop_event = threading.Event()

    def signal_handler(sig, frame):
        print("\n[INFO] Stopping camera and parking actuators...")
        stop_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        src = int(source) if str(source).isdigit() else source
        cam = DeepStreamCamera(source=src, config_path=DEEPSTREAM_CONFIG_PATH,
                               exg_threshold=exg, flip_method=flip,
                               enable_optical_flow=True, enable_weed_mask=False,
                               use_updated=True)
        cam.start()
        print("[SYSTEM] DeepStream pipeline active. Autonomous tracking running.")
        print(">>> Press Ctrl+C in this terminal to stop <<<\n")

        last_perf_log = 0.0
        last_sequence = -1

        while not stop_event.is_set():
            frame, plants, weed_mask, camera_fps, sequence = cam.get_latest(timeout=0.2)
            if frame is None or sequence == last_sequence:
                time.sleep(0.002)
                continue
            last_sequence = sequence

            flow_x, flow_y, _ = cam.get_optical_flow()
            if flow_x != 0.0 or flow_y != 0.0:
                sprayer.update_optical_flow(flow_x, flow_y)

            started = time.perf_counter()
            annotated, detections, mask = detector.process_frame(
                frame, plants, weed_mask=None,
                exg_threshold=exg, min_area=min_area,
                cluster_area_threshold=cluster,
                separation_px=separation, max_box_area=max_area)
            sprayer.add_detections(detections)
            elapsed_ms = (time.perf_counter() - started) * 1000

            now = time.monotonic()
            if now - last_perf_log >= 2.0:
                dist_m = sprayer.get_distance()
                act_state = "🟢 FIRING (100ms)" if sprayer.is_actuator_active() else "IDLE (Valves OFF, Pump OFF)"
                print(f"[STATUS] Pipeline: {camera_fps:.1f} FPS | Postprocess: {elapsed_ms:.0f}ms | "
                      f"Weeds: {len(detections)} | Dist: {dist_m:.2f}m | Actuator: {act_state}")
                last_perf_log = now

    except Exception as exc:
        print(f"[ERROR] Headless loop: {exc}")
    finally:
        if cam is not None:
            cam.stop()
        sprayer.stop()
        print("[SYSTEM] Sprayer controller stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Weed Detection and Autonomous Sprayer Control")
    parser.add_argument(
        "--gui", "-g", "--display", dest="gui", action="store_true", default=False,
        help="Display the desktop GUI window (requires active desktop / X11 session). Defaults to headless for SSH."
    )
    parser.add_argument(
        "--source", "-s", default=detector.VIDEO_SOURCE,
        help="Camera or video source (default: 0 for live CSI camera)."
    )
    parser.add_argument(
        "--min-area", type=int, default=detector.MIN_CONTOUR_AREA,
        help=f"Minimum weed area in pixels (default: {detector.MIN_CONTOUR_AREA})."
    )
    parser.add_argument(
        "--max-area", type=int, default=detector.MAX_WEED_BOX_AREA,
        help=f"Maximum weed box area in pixels (default: {detector.MAX_WEED_BOX_AREA})."
    )
    parser.add_argument(
        "--cluster-area", type=int, default=detector.CLUSTER_AREA_THRESHOLD,
        help=f"Cluster split area threshold in pixels (default: {detector.CLUSTER_AREA_THRESHOLD})."
    )
    parser.add_argument(
        "--exg", type=int, default=detector.EXG_THRESHOLD,
        help=f"ExG vegetation threshold (default: {detector.EXG_THRESHOLD})."
    )
    args = parser.parse_args()

    show_gui = args.gui or os.environ.get("DISPLAY_GUI", "0").lower() in ("1", "true", "yes")

    if show_gui:
        # Check if DISPLAY is available
        if not os.environ.get("DISPLAY"):
            for disp in [":1", ":0"]:
                if os.path.exists(f"/tmp/.X11-unix/X{disp[1:]}"):
                    os.environ["DISPLAY"] = disp
                    break

        if not os.environ.get("DISPLAY"):
            print("[WARN] --gui requested but no X11 DISPLAY is available on this session.")
            print("[WARN] Running in headless mode for SSH...")
            run_headless(source=args.source, exg=args.exg, min_area=args.min_area,
                         max_area=args.max_area, cluster=args.cluster_area)
        else:
            app = None
            try:
                app = WeedSprayerApp()
                app.mainloop()
            except KeyboardInterrupt:
                pass
            finally:
                if app is not None:
                    try:
                        app._on_close()
                    except Exception:
                        pass
    else:
        # Default: Headless CLI runner for SSH
        run_headless(source=args.source, exg=args.exg, min_area=args.min_area,
                     max_area=args.max_area, cluster=args.cluster_area)
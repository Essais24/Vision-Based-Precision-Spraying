"""
sprayer_control.py
========================================================================
ODOMETRY  ->  ROVER POSITION  ->  COORDINATE TRANSFORM  ->  SPRAY COMMAND

This is the second half of the pipeline. It is meant to run as its own
process (so it can be started/stopped independently from a GUI later).
It:

    1. Reads odometry (forward speed + yaw rate) and integrates it into
       a rover pose (x, y, heading) over time -- dead-reckoning.
    2. Tails the shared detections file written by
       `detection_worldcoord.py`, which contains weed positions in the
       CAMERA's local world frame (X, Y, Z on the ground plane, camera
       at origin).
    3. Converts each weed's camera-local coordinates into GLOBAL
       coordinates using the rover pose *at the time the detection was
       made* (i.e. "where was the rover, and which way was it facing,
       when the camera saw this weed").
    4. Continuously compares the rover's current global position to each
       known weed's global position. When the sprayer nozzle (which is
       offset from the camera/rover origin) is lined up with the weed
       within a tolerance, it fires the spray command.
    5. The actual spray "command" is just a print() for now -- swap in
       your GPIO / relay / solenoid trigger code when you wire up the
       Jetson Nano.

--------------------------------------------------------------------------
ASSUMPTIONS / PLACEHOLDERS  (all clearly marked, edit to match your setup)
--------------------------------------------------------------------------
  * `read_odometry()` is a MOCK. Replace it with your real encoder /
    IMU / GPS reading code. It must return (forward_speed_m_s, yaw_rate_deg_s).
  * The rover is assumed to move on a flat 2D plane; heading is a single
    yaw angle. Extend to full 3D/SLAM if you need it later.
  * SPRAYER_OFFSETS are the sprayer nozzles' positions relative to the
    camera. Their vertical offset is measured; their lateral and
    forward components must match the physical rig.
  * Communication with the detection script is via the same JSON-lines
    file (DETECTIONS_FILE) -- replace with a socket/queue/ROS topic for
    a production system.
========================================================================
"""

import json
import math
import time
import threading

from spray_actuation import SprayActuator, offset_to_angle, NOZZLE_LATERAL_OFFSETS_M, SERVO_SETTLE_S
import detection_worldcoord as detector_geometry

# ------------------------------------------------------------------
# 1. CONFIG / PLACEHOLDERS -- edit these for your setup
# ------------------------------------------------------------------
DETECTIONS_FILE = "weed_detections.jsonl"   # shared file written by detection_worldcoord.py

# Measured base mounting heights above ground, metres. A common lift is added
# to both, preserving their physical spacing.
BASE_CAMERA_HEIGHT_M = detector_geometry.BASE_CAMERA_HEIGHT_M
BASE_SPRAYER_HEIGHT_M = 0.28
TRACTOR_LIFT_HEIGHT_M = 0.22
CAMERA_HEIGHT_M = BASE_CAMERA_HEIGHT_M + TRACTOR_LIFT_HEIGHT_M
SPRAYER_HEIGHT_M = BASE_SPRAYER_HEIGHT_M + TRACTOR_LIFT_HEIGHT_M
CAMERA_TO_SPRAYER_Z_M = SPRAYER_HEIGHT_M - CAMERA_HEIGHT_M  # -0.14 m: nozzle below camera


def set_tractor_lift_height(height_m):
    """Apply one common lift value (metres) to camera and both nozzles."""
    global TRACTOR_LIFT_HEIGHT_M, CAMERA_HEIGHT_M, SPRAYER_HEIGHT_M, CAMERA_TO_SPRAYER_Z_M
    height_m = float(height_m)
    if height_m < 0:
        raise ValueError("Tractor lift height cannot be negative")
    TRACTOR_LIFT_HEIGHT_M = height_m
    detector_geometry.set_tractor_lift_height(height_m)
    CAMERA_HEIGHT_M = BASE_CAMERA_HEIGHT_M + height_m
    SPRAYER_HEIGHT_M = BASE_SPRAYER_HEIGHT_M + height_m
    CAMERA_TO_SPRAYER_Z_M = SPRAYER_HEIGHT_M - CAMERA_HEIGHT_M
    for offset in SPRAYER_OFFSETS.values():
        offset["z"] = CAMERA_TO_SPRAYER_Z_M


def frame_half_side(detection):
    """Assign exactly one nozzle from the original camera-frame half.

    The mounted camera is horizontally flipped: u > cx is physically left and
    u < cx is physically right. The exact centre pixel is assigned right so a
    detection always has one deterministic owner.
    """
    pixel = detection.get("pixel")
    if pixel is not None:
        centre_x = float(detection.get("frame_center_x", detector_geometry.K[0, 2]))
        return "left" if float(pixel[0]) > centre_x else "right"
    # Compatibility for old JSONL records lacking pixel coordinates.
    x = float(detection.get("_cam_x", detection["x"]))
    return "left" if (x >= 0 if detector_geometry.CAMERA_X_FLIP else x < 0) else "right"

# Sprayer nozzle positions relative to the camera (camera-local frame),
# metres. X is lateral (right-positive), Y is forward, Z is up.
SPRAYER_OFFSETS = {
    "right": {"x": NOZZLE_LATERAL_OFFSETS_M["right"], "y": 0.0, "z": CAMERA_TO_SPRAYER_Z_M},
    "left": {"x": NOZZLE_LATERAL_OFFSETS_M["left"], "y": 0.0, "z": CAMERA_TO_SPRAYER_Z_M},
}

SPRAY_TOLERANCE_M = 0.05     # how close (metres) the sprayer must be to fire
LOOP_HZ = 20                 # main control-loop rate
ODOMETRY_HZ = 20             # how often we poll/integrate odometry
TARGET_MAX_AGE_S = 15.0      # discard detections that are no longer actionable
MAX_PENDING_WEEDS = 500      # prevent unbounded growth when every frame has new IDs

# STATIC_CAMERA_MODE: set True when the camera/rover is NOT moving (e.g. bench
# testing or a fixed mount). Effects:
#   • read_odometry() returns v=0 (no forward motion)
#   • add_detections() re-uses existing weed IDs when the pixel-centre shifts
#     less than 50% of the weed's reference distance (avoids duplicate accumulation)
#   • _side_spray_loop skips the dist<=SPRAY_TOLERANCE_M check (weed is always
#     "in range" since the nozzle never moves toward it)
# Set False when mounted on a moving tractor.
STATIC_CAMERA_MODE = True



# ------------------------------------------------------------------
# 2. ODOMETRY  (MOCK -- replace with real sensor reading)
# ------------------------------------------------------------------
def read_odometry():
    """
    ASSUMPTION / PLACEHOLDER: this stands in for real wheel-encoder /
    IMU / GPS data acquisition on the Jetson Nano.

    Must return:
        forward_speed_m_s : float  -- rover's forward speed, m/s
        yaw_rate_deg_s     : float  -- rate of heading change, deg/s

    Replace the body of this function with, e.g., reading a serial port
    from an Arduino/encoder board, or a ROS /odom subscription.
    """
    # STATIC MODE: forward speed = 0 (camera not moving; rover is parked).
    # Re-enable the line below and remove the next line when mounting on a tractor.
    # forward_speed_m_s = 2.0 * 1000 / 3600.0  # 2 km/h tractor speed (v_f from notebook)
    forward_speed_m_s = 0.0
    yaw_rate_deg_s = 0.0                      # assume straight-line travel for now
    return forward_speed_m_s, yaw_rate_deg_s


# Optical flow scale factor: converts pixels/frame to meters traveled.
# Calibration approximation: camera height / calibrated focal length.
# 0.42 m / mean(2318.75030, 2318.66170) px = 0.00018114 m/px.
OPTICAL_FLOW_SCALE_M_PER_PX = 0.00018114


# ------------------------------------------------------------------
# 3. ROVER STATE -- dead-reckoning position/heading tracker
# ------------------------------------------------------------------
class RoverState:
    """Tracks the rover's global pose (x, y, heading) via odometry integration."""

    def __init__(self, x0=0.0, y0=0.0, heading0_deg=0.0):
        self.x = x0
        self.y = y0
        self.heading_deg = heading0_deg
        self.total_distance = 0.0
        self._lock = threading.Lock()
        self._last_t = time.time()

    def update(self, forward_speed_m_s, yaw_rate_deg_s):
        """Integrate one odometry sample into the pose. Call at ODOMETRY_HZ."""
        now = time.time()
        dt = now - self._last_t
        self._last_t = now

        with self._lock:
            heading_rad = math.radians(self.heading_deg)
            ds = forward_speed_m_s * dt
            self.x += ds * math.cos(heading_rad)
            self.y += ds * math.sin(heading_rad)
            self.total_distance += abs(ds)
            self.heading_deg += yaw_rate_deg_s * dt

    def update_displacement(self, dx_m, dy_m):
        """Update rover state directly from camera optical flow displacement (dx_m, dy_m).

        Integrates forward/lateral displacement into rover pose and cumulative distance traveled.
        """
        now = time.time()
        self._last_t = now
        with self._lock:
            heading_rad = math.radians(self.heading_deg)
            ds = math.hypot(dx_m, dy_m)
            self.total_distance += ds
            self.x += dx_m * math.cos(heading_rad) - dy_m * math.sin(heading_rad)
            self.y += dx_m * math.sin(heading_rad) + dy_m * math.cos(heading_rad)

    def get_pose(self):
        with self._lock:
            return self.x, self.y, self.heading_deg

    def get_distance(self):
        with self._lock:
            return self.total_distance


def odometry_loop(rover_state, stop_event):
    """Background thread: continuously read odometry and update rover pose."""
    period = 1.0 / ODOMETRY_HZ
    while not stop_event.is_set():
        v, yaw_rate = read_odometry()
        rover_state.update(v, yaw_rate)
        time.sleep(period)


# ------------------------------------------------------------------
# 4. COORDINATE TRANSFORM: camera-local -> global
# ------------------------------------------------------------------
def camera_local_to_global(det, rover_pose_at_detection):
    """
    Convert a weed detection's camera-local (x, y) [ground-plane, camera
    at origin, from detection_worldcoord.py] into a GLOBAL (map) frame
    coordinate, given the rover's pose when the frame was captured.

    det: {"id", "x", "y", "z"}  (camera-local, metres)
    rover_pose_at_detection: (rx, ry, heading_deg)

    IMPORTANT axis convention: detection_worldcoord.py's local frame is
    X = lateral (right-positive), Y = forward -- NOT the (forward, left)
    body-frame convention a plain rotation matrix would assume. Composing
    body-frame forward = local Y and body-frame left = -local X before
    rotating by heading keeps this consistent with that file's output.
    (A previous version of this function assumed local_x = forward,
    local_y = left, which silently misplaced every detection by 90
    degrees -- see sprayer_global_position() below for the matching fix.)
    """
    rx, ry, heading_deg = rover_pose_at_detection
    heading_rad = math.radians(heading_deg)

    local_x, local_y = det["x"], det["y"]  # local_x = lateral (right+), local_y = forward
    global_x = rx + (local_y * math.cos(heading_rad) + local_x * math.sin(heading_rad))
    global_y = ry + (local_y * math.sin(heading_rad) - local_x * math.cos(heading_rad))

    return {"id": det["id"], "x": global_x, "y": global_y, "z": det.get("z", 0.0)}


def sprayer_global_position(rover_pose, side):
    """Global (x, y) of the named side's sprayer nozzle, given the rover
    pose and that side's fixed mount offset. Uses the same (lateral,
    forward) -> global rotation as camera_local_to_global(); see its
    docstring for why a plain rotation matrix isn't correct here."""
    rx, ry, heading_deg = rover_pose
    heading_rad = math.radians(heading_deg)
    offset = SPRAYER_OFFSETS[side]
    ox, oy = offset["x"], offset["y"]  # ox = lateral (right+), oy = forward
    sx = rx + (oy * math.cos(heading_rad) + ox * math.sin(heading_rad))
    sy = ry + (oy * math.sin(heading_rad) - ox * math.cos(heading_rad))
    return sx, sy


def calculate_theta(Zs, Yw0, Yr):
    """
    Angle (degrees) between the sprayer's forward offset (Zs) and the
    lateral gap between the target's initial lateral position (Yw0) and
    the rover's current lateral position (Yr). Kept from the original
    notebook logic; useful if you drive a steerable/aimable nozzle.
    """
    if Yw0 == Yr:
        return 0.0
    return math.degrees(math.atan(Zs / (Yw0 - Yr)))


def lateral_gap(target_global, sprayer_global_pos, heading_deg):
    """Signed lateral distance (metres, right-positive) from the sprayer
    to a global-frame target, along the rover's CURRENT left/right axis.

    NOT simply target_x - sprayer_x -- that's only correct at heading=0.
    In general it's the (target - sprayer) vector projected onto the
    rover's right-facing body axis, which rotates with heading. This is
    the same rotation-frame issue camera_local_to_global() had; see its
    docstring.
    """
    heading_rad = math.radians(heading_deg)
    dx = target_global["x"] - sprayer_global_pos[0]
    dy = target_global["y"] - sprayer_global_pos[1]
    return math.sin(heading_rad) * dx - math.cos(heading_rad) * dy


def spray_direction(det_global, rover_pose):
    """'left' or 'right' of the rover's heading, used to pick which nozzle to fire."""
    rx, ry, heading_deg = rover_pose
    heading_rad = math.radians(heading_deg)
    dx, dy = det_global["x"] - rx, det_global["y"] - ry
    # cross product of heading vector and vector-to-target tells left/right
    cross = math.cos(heading_rad) * dy - math.sin(heading_rad) * dx
    return "left" if cross > 0 else "right"


# ------------------------------------------------------------------
# 5. SPRAY COMMAND -- fires the real solenoid via spray_actuation.py.
# Aiming (servo rotation) happens separately and earlier, in
# SprayerController's per-side loop below -- by the time this runs, the
# corresponding servo should already have had SERVO_SETTLE_S to finish
# rotating. See spray_actuation.SprayActuator.
# ------------------------------------------------------------------
_default_actuator = None


def _get_default_actuator():
    global _default_actuator
    if _default_actuator is None:
        _default_actuator = SprayActuator()
    return _default_actuator


def spray_command(weed_id, direction, actuator=None):
    """Fire the sprayer solenoid on `direction` ("left"/"right")."""
    # One weed is owned by exactly one frame half/nozzle.  Do not use
    # SprayActuator.fire() here: that legacy helper intentionally opens both
    # valves and would spray the same weed from both sides.
    (actuator or _get_default_actuator()).fire_side(direction)
    print(f"[SPRAY] weed_id={weed_id} direction={direction}  "
          f"(t={time.strftime('%H:%M:%S')})")


class SprayerController:
    """Thread-safe sprayer state used by the GUI or a hardware runner.

    The controller intentionally has no UI or file polling.  A caller supplies
    detections and rover poses, so a GUI can keep all controls responsive while
    the actuator logic remains in this module.
    """

    def __init__(self, log_callback=None, command_callback=None, actuator=None):
        self.rover_state = RoverState()
        self.known_weeds = {}
        self.already_sprayed = set()
        self.missed = set()  # weed ids discarded un-sprayed (age-out or overtaken -- see _discard_stale_targets)
        self._lock = threading.Lock()
        self._log_callback = log_callback
        self.actuator = actuator or _get_default_actuator()
        self._command_callback = command_callback or (
            lambda weed_id, direction: spray_command(weed_id, direction, actuator=self.actuator)
        )
        self._stop_event = threading.Event()
        self._odometry_thread = None
        self._control_threads = []
        self._last_aim_angle = {"left": None, "right": None}

    def _log(self, message):
        if self._log_callback:
            self._log_callback(message)
        else:
            print(message)

    def is_actuator_active(self):
        """Returns True if the valves and pump are actively firing."""
        return getattr(self.actuator, "is_active", lambda: False)()

    def set_rover_pose(self, x, y, heading_deg):
        with self.rover_state._lock:
            self.rover_state.x = float(x)
            self.rover_state.y = float(y)
            self.rover_state.heading_deg = float(heading_deg)
            self.rover_state._last_t = time.time()

    def update_optical_flow(self, flow_x_px, flow_y_px, scale_factor=None):
        """Update rover odometry using DeepStream hardware optical flow.

        scale_factor converts flow in pixels/frame to meters traveled.
        Default scale is derived from focal length and height above ground:
            scale ~ altitude_m / focal_length_px
        """
        if scale_factor is None:
            scale_factor = OPTICAL_FLOW_SCALE_M_PER_PX
        # Motion vector inversion: scene moving backward (-Y) means rover moving forward (+X/Z)
        dx_m = -float(flow_x_px) * scale_factor
        dy_m = -float(flow_y_px) * scale_factor
        self.rover_state.update_displacement(dx_m, dy_m)

    def get_distance(self):
        """Return cumulative distance traveled in meters."""
        return self.rover_state.get_distance()

    def add_detections(self, detections):
        """Transform the camera-local detections using the current rover pose.

        A live DeepStream weed-YOLO detection carries its NvSORT ID in both
        ``id`` and ``tracker_id``. That ID is authoritative: it is the key
        used to refresh this exact target's world location, nozzle side,
        aiming angle, and already-sprayed state. Position-only matching below
        is retained exclusively for the ExG/offline fallback, where no native
        tracker ID exists.

        In static-camera mode (v=0) the same physical weed appears every frame
        with a freshly-assigned detection ID. Before inserting, we compare
        each new detection's CAMERA-LOCAL (x, y) against every existing
        known_weed's stored camera-local position. If the Euclidean distance
        between them is less than 50 % of the reference distance of the
        existing weed (i.e. small relative shift in the camera view) we treat
        them as the same physical weed, refresh its timestamp, and skip
        creating a duplicate entry.  The 50 % threshold keeps genuinely
        different weeds separate while absorbing per-frame jitter.
        """
        pose = self.rover_state.get_pose()
        now = time.monotonic()
        with self._lock:
            self._discard_stale_targets(now)
            for detection in detections:
                cam_x = detection["x"]
                cam_y = detection["y"]

                tracker_id = detection.get("tracker_id")
                if tracker_id is not None and int(tracker_id) >= 0:
                    tracker_id = int(tracker_id)
                    # A persistent NvSORT ID identifies the target; do not
                    # merge it with a nearby but distinct weed using the old
                    # geometric fallback. Refresh its target geometry from
                    # the newest tracked box on every frame.
                    if tracker_id in self.already_sprayed:
                        self.known_weeds.pop(tracker_id, None)
                        continue
                    global_detection = camera_local_to_global(detection, pose)
                    global_detection["id"] = tracker_id
                    global_detection["_queued_at"] = now
                    global_detection["_cam_x"] = cam_x
                    global_detection["_cam_y"] = cam_y
                    global_detection["sprayer_side"] = frame_half_side(detection)
                    self.known_weeds[tracker_id] = global_detection
                    continue

                # ---- camera-frame position matching (static-mode re-ID) ----
                matched_id = None
                best_dist = float("inf")
                for existing_id, existing_weed in self.known_weeds.items():
                    ex = existing_weed.get("_cam_x", existing_weed["x"])
                    ey = existing_weed.get("_cam_y", existing_weed["y"])
                    ref_dist = math.hypot(ex, ey) or 1e-6  # distance of weed from camera
                    shift = math.hypot(cam_x - ex, cam_y - ey)
                    if shift < ref_dist * 0.50 and shift < best_dist:
                        best_dist = shift
                        matched_id = existing_id

                if matched_id is not None:
                    # Same physical weed -- just refresh its timestamp so it
                    # doesn't age out while it's still in view.
                    self.known_weeds[matched_id]["_queued_at"] = now
                    self.known_weeds[matched_id]["_cam_x"] = cam_x
                    self.known_weeds[matched_id]["_cam_y"] = cam_y
                    self.known_weeds[matched_id]["sprayer_side"] = frame_half_side(detection)
                    continue

                # New weed not seen before -- store with global coords.
                global_detection = camera_local_to_global(detection, pose)
                global_detection["_queued_at"] = now
                global_detection["_cam_x"] = cam_x   # keep camera-local for matching
                global_detection["_cam_y"] = cam_y
                global_detection["sprayer_side"] = frame_half_side(detection)
                self.known_weeds[global_detection["id"]] = global_detection

            while len(self.known_weeds) > MAX_PENDING_WEEDS:
                oldest_id = min(self.known_weeds, key=lambda weed_id: self.known_weeds[weed_id]["_queued_at"])
                self.known_weeds.pop(oldest_id, None)

    def _discard_stale_targets(self, now):
        expired = [
            weed_id for weed_id, weed in self.known_weeds.items()
            if now - weed.get("_queued_at", now) > TARGET_MAX_AGE_S
        ]
        for weed_id in expired:
            self.known_weeds.pop(weed_id, None)
            self.already_sprayed.discard(weed_id)

    def reset(self):
        """Clear detections and spray history before starting a new stream."""
        with self._lock:
            self.known_weeds.clear()
            self.already_sprayed.clear()
            self.missed.clear()
            with self.rover_state._lock:
                self.rover_state.total_distance = 0.0
        self._last_aim_angle = {"left": None, "right": None}
        self.actuator.park_all()

    def start(self):
        """Start odometry plus one automatic spray loop per side (left/right)."""
        if self._control_threads and any(t.is_alive() for t in self._control_threads):
            return
        self._stop_event.clear()
        self._odometry_thread = threading.Thread(
            target=odometry_loop, args=(self.rover_state, self._stop_event), daemon=True)
        self._control_threads = [
            threading.Thread(target=self._side_spray_loop, args=(side,), daemon=True)
            for side in SPRAYER_OFFSETS
        ]
        self._odometry_thread.start()
        for thread in self._control_threads:
            thread.start()
        self._log("[SPRAYER] Automatic controller started (per-side aim-then-spray)")

    def stop(self):
        self._stop_event.set()
        for thread in [self._odometry_thread, *self._control_threads]:
            if thread is not None:
                thread.join(timeout=1.0)
        self.actuator.park_all()
        self._log("[SPRAYER] Automatic controller stopped")

    def _pending_for_side(self, side, pose):
        """Weeds assigned to `side` by their original frame half, not yet sprayed
        and not yet passed by the rover, nearest-ahead first. Weeds the
        rover has now driven past while still unsprayed -- almost always
        because another same-side weed was being serviced when this one's
        narrow alignment window went by, since one servo can only aim at
        one angle at a time -- are logged as missed here immediately,
        rather than silently disappearing 15s later via TARGET_MAX_AGE_S.
        """
        rx, ry, heading_deg = pose
        heading_rad = math.radians(heading_deg)
        with self._lock:
            self._discard_stale_targets(time.monotonic())
            candidates = [
                weed for weed in self.known_weeds.values()
                if weed["id"] not in self.already_sprayed
                and weed.get("sprayer_side", frame_half_side(weed)) == side
            ]

        def forward_gap(weed):
            dx, dy = weed["x"] - rx, weed["y"] - ry
            return dx * math.cos(heading_rad) + dy * math.sin(heading_rad)

        still_ahead, just_passed = [], []
        for weed in candidates:
            (still_ahead if forward_gap(weed) >= -SPRAY_TOLERANCE_M else just_passed).append(weed)

        for weed in just_passed:
            if weed["id"] not in self.missed:
                self.missed.add(weed["id"])
                self._log(f"[MISS] id={weed['id']} side={side} -- rover passed it unsprayed "
                          f"(servo likely busy with another {side}-side target)")

        still_ahead.sort(key=forward_gap)
        return still_ahead

    ANGLE_MERGE_TOLERANCE_DEG = 5.0  # next target within this many degrees of the
    # servo's current commanded angle is treated as "already aimed" -- skips a
    # redundant aim() + settle wait, since the motor doesn't actually need to move.
    # This is what lets tightly-clustered same-side weeds (same/near-identical
    # lateral offset) be sprayed back-to-back instead of each paying the full
    # SERVO_SETTLE_S penalty. It does NOT help two same-side weeds that need
    # genuinely different angles -- see the class docstring note on that limit.

    def _side_spray_loop(self, side):
        """Algorithm for handling multiple weed detections, per side:

        Track only the single nearest-ahead, not-yet-sprayed weed assigned
        to this side (spray_direction() decides left vs right). As soon as
        a weed becomes that side's nearest target, pre-aim that side's
        servo at it immediately -- well before the rover physically
        reaches it -- UNLESS the servo is already sitting within
        ANGLE_MERGE_TOLERANCE_DEG of the angle this target needs, in which
        case no new aim/settle wait is issued at all. Fire the solenoid
        only once BOTH conditions hold:
        (a) the servo's settle time has elapsed since aim() was called
            (spray_actuation.SprayActuator.ready_to_fire), i.e. the motor
            should have finished rotating, and
        (b) the sprayer's global position is within SPRAY_TOLERANCE_M of
            the weed's global position.
        Then immediately re-aim (or skip re-aiming, per above) at
        whichever weed is now nearest on this side.

        LIMITATION: this is a strict one-target-at-a-time-per-side queue.
        Two same-side weeds needing genuinely different angles, close
        enough together in the rover's direction of travel that the
        second one's ~(SPRAY_TOLERANCE_M / rover_speed) alignment window
        passes before the first one's SERVO_SETTLE_S aim-and-fire cycle
        finishes, WILL be missed -- logged via [MISS] in _pending_for_side
        above, not silently dropped, but not sprayed either. Fixing that
        for real needs either a second nozzle per side, a wider blanket
        spray pattern per shot, or slowing the rover while working through
        a same-side backlog -- none of which this module controls today.
        """
        period = 1.0 / LOOP_HZ
        current_id = None
        while not self._stop_event.is_set():
            pose = self.rover_state.get_pose()
            pending = self._pending_for_side(side, pose)
            target = pending[0] if pending else None

            if target is None:
                current_id = None
            else:
                sx, sy = sprayer_global_position(pose, side)
                angle = offset_to_angle(lateral_gap(target, (sx, sy), pose[2]), side, SPRAYER_HEIGHT_M)

                if target["id"] != current_id:
                    last_angle = self._last_aim_angle[side]
                    already_aimed_there = (
                        last_angle is not None
                        and abs(angle - last_angle) <= self.ANGLE_MERGE_TOLERANCE_DEG
                    )
                    if not already_aimed_there:
                        self.actuator.aim(side, angle)
                        self._last_aim_angle[side] = angle
                        self._log(f"[AIM] side={side} id={target['id']} angle={angle:.1f}deg")
                    else:
                        self._log(f"[AIM] side={side} id={target['id']} angle={angle:.1f}deg "
                                  f"(servo already there, no re-aim needed)")
                    current_id = target["id"]

                dist = math.hypot(target["x"] - sx, target["y"] - sy)
                # In static-camera mode the rover doesn't move, so dist stays
                # constant and would never satisfy SPRAY_TOLERANCE_M.
                # We treat every detected weed as "in range" when static.
                in_range = STATIC_CAMERA_MODE or (dist <= SPRAY_TOLERANCE_M)
                if in_range and self.actuator.ready_to_fire(side):
                    self._spray_now(target["id"], side)
                    current_id = None

            self._stop_event.wait(period)

    def _spray_now(self, weed_id, direction):
        """Fire immediately. Caller must already have aimed that side's
        servo and confirmed it's had time to settle -- shared by the
        automatic per-side loop and the manual spray() path below."""
        with self._lock:
            weed = self.known_weeds.get(weed_id)
            if weed is None or weed_id in self.already_sprayed:
                return False
            self.already_sprayed.add(weed_id)
            self.known_weeds.pop(weed_id, None)
            if len(self.already_sprayed) > MAX_PENDING_WEEDS:
                self.already_sprayed.clear()

        self._log(f"[ACTUATOR][ON] Weed #{weed_id} reached ({direction}) -> VALVES + PUMP ON (400ms)")
        self._command_callback(weed_id, direction)
        self._log(f"[ACTUATOR][OFF] Weed #{weed_id} shot complete -> VALVES + PUMP OFF")
        return True

    def spray(self, weed_id):
        """Manually spray one known detection (e.g. a GUI button). Unlike
        the automatic per-side loop, there's no advance pre-aim here, so
        this blocks for the servo settle time before firing. Returns
        False for invalid/already-sprayed ids."""
        with self._lock:
            weed = self.known_weeds.get(weed_id)
            if weed is None:
                self._log(f"[SPRAY] Unknown detection id={weed_id}")
                return False
            if weed_id in self.already_sprayed:
                self._log(f"[SPRAY] Detection id={weed_id} was already sprayed")
                return False
            pose = self.rover_state.get_pose()
            direction = weed.get("sprayer_side", frame_half_side(weed))
            sx, sy = sprayer_global_position(pose, direction)
            angle = offset_to_angle(lateral_gap(weed, (sx, sy), pose[2]), direction, SPRAYER_HEIGHT_M)
        self.actuator.aim(direction, angle)
        time.sleep(SERVO_SETTLE_S)
        return self._spray_now(weed_id, direction)


# ------------------------------------------------------------------
# 6. DETECTIONS FILE READER (tails the file written by the other script)
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# 6. MAIN CONTROL LOOP
# ------------------------------------------------------------------
def main():
    """Standalone process entry point: tails DETECTIONS_FILE and feeds
    each batch into a SprayerController, which owns odometry + the
    per-side aim-then-spray loops (see SprayerController._side_spray_loop
    for the actual multi-detection algorithm -- this function no longer
    duplicates it).

    NOTE (gap): this process has no access to the DeepStream camera's
    hardware optical flow, since that lives in whatever process owns the
    DeepStreamCamera object (currently only weed_sprayer_gui.py / a
    similar GUI process wires update_optical_flow() in). Run this
    standalone only if you're OK with the mocked constant-speed
    read_odometry() for position tracking; for real optical-flow-based
    tracking, run detection + spraying together in one process instead
    (see the README's "known gaps" section) or publish flow data through
    DETECTIONS_FILE alongside the detections.
    """
    controller = SprayerController()
    controller.reset()
    controller.start()

    print("Sprayer control loop started. Waiting for detections & tracking position...")
    offset = 0
    loop_period = 1.0 / LOOP_HZ
    try:
        while True:
            # --- Non-blocking poll of any newly published detections ---
            with open(DETECTIONS_FILE, "a"):
                pass  # ensure file exists
            with open(DETECTIONS_FILE, "r") as f:
                f.seek(offset)
                new_lines = f.readlines()
                offset = f.tell()

            for line in new_lines:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                controller.add_detections(record["detections"])
                for det in record["detections"]:
                    print(f"[TRACK] weed id={det['id']} camera-local=({det['x']:.3f}, {det['y']:.3f})")

            time.sleep(loop_period)

    except KeyboardInterrupt:
        print("Stopping sprayer control loop...")
    finally:
        controller.stop()


if __name__ == "__main__":
    main()

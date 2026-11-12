"""
spray_actuation.py
========================================================================
Hardware driver for the weed sprayer rig: communicates with ESP32 via UART
(running esp32_sprayer_uart.ino).

All actuation (servos, solenoid valves, and main pump) is managed by the
ESP32 via structured 4-byte UART frames @ 115200 baud.

--------------------------------------------------------------------------
UART PROTOCOL SPECIFICATION (Active-LOW Relay Logic):
--------------------------------------------------------------------------
Frame Format: [0xAA] [CMD_TYPE] [DATA_BYTE_1] [DATA_BYTE_2]

  1. Motor Control:
     CMD_TYPE 0x01: [0x00=Left / 0x01=Right] [Angle 0-180]
     -> Pre-aimed in advance when a weed is detected ahead.

  2. Solenoid Valve Control (0x00 = ON, 0x01 = OFF):
     CMD_TYPE 0x02: [0x00=Left / 0x01=Right] [0x00=On / 0x01=Off]
     - Left Valve ON:   b'\xAA\x02\x00\x00'
     - Left Valve OFF:  b'\xAA\x02\x00\x01'
     - Right Valve ON:  b'\xAA\x02\x01\x00'
     - Right Valve OFF: b'\xAA\x02\x01\x01'

  3. Main Pump Control (0x00 = ON, 0x01 = OFF):
     CMD_TYPE 0x03: [0x00=On / 0x01=Off]     [0x00 (Dummy)]
     - Pump ON:         b'\xAA\x03\x00\x00'
     - Pump OFF:        b'\xAA\x03\x01\x00'

  4. Batch TURN BOTH ON Command (Left Valve ON + Right Valve ON + Pump ON):
     b'\xAA\x02\x00\x00\xAA\x02\x01\x00\xAA\x03\x00\x00'

  5. Batch TURN ALL OFF Command (Left Valve OFF + Right Valve OFF + Pump OFF):
     b'\xAA\x02\x00\x01\xAA\x02\x01\x01\xAA\x03\x01\x00'

--------------------------------------------------------------------------
TIMING:
  - Servo aim is sent in advance as soon as a weed is tracked ahead.
  - When the sprayer reaches the weed, BOTH valve and pump are turned ON:
      b'\xAA\x02\x00\x00\xAA\x02\x01\x00\xAA\x03\x00\x00'
  - Kept ON for 100ms (SPRAY_DURATION_S = 0.20).
  - Sent signal for OFF:
      b'\xAA\x02\x00\x01\xAA\x02\x01\x01\xAA\x03\x01\x00'
========================================================================
"""

import os
import math
import time
import threading
import subprocess

try:
    import serial
    _SERIAL_AVAILABLE = True
except ImportError:
    serial = None
    _SERIAL_AVAILABLE = False
    print("[ACTUATOR][WARN] pyserial not found. Running in simulation mode.")

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
UART_PORT = "/dev/ttyTHS1"       # Jetson 40-pin header UART1 (Pins 8 & 10)
UART_BAUDRATE = 115200          # Matches BAUD_RATE in esp32_sprayer_uart.ino

PACKET_START_BYTE = 0xAA
CMD_MOTOR = 0x01
CMD_VALVE = 0x02
CMD_PUMP  = 0x03

SIDE_CODE = {"left": 0x00, "right": 0x01}

# Active-LOW UART relay states, as specified by the protocol above.
VALVE_STATE_ON  = 0x00
VALVE_STATE_OFF = 0x01
PUMP_STATE_ON   = 0x00
PUMP_STATE_OFF  = 0x01

# Batch frames for explicitly turning both valves on and off.
FRAME_BOTH_ON  = b'\xAA\x02\x00\x00\xAA\x02\x01\x00\xAA\x03\x00\x00'
FRAME_BOTH_OFF = b'\xAA\x02\x00\x01\xAA\x02\x01\x01\xAA\x03\x01\x00'

SERVO_CENTER_ANGLE = 90.0        # vertical: each nozzle hits below its own mount
# Calibrated usable sweep limits.  At the base nozzle height (0.28 m):
#   left  90° -> below left nozzle; 135° -> below camera centre (55.1 cm)
#   right 90° -> below right nozzle;  30° -> below camera centre (50.0 cm)
# The lift-aware aiming calculation stays within these physical endpoint
# commands; it must not command the left linkage beyond its camera-centre stop.
LEFT_MIN_ANGLE_DEG = 90.0
LEFT_MAX_ANGLE_DEG = 135.0
RIGHT_MIN_ANGLE_DEG = 30.0
RIGHT_MAX_ANGLE_DEG = 90.0

# Keep limits in this hardware-driver module so every caller (automatic
# control, calibration, and manual test tools) is protected.
SERVO_LIMITS_DEG = {
    "left": (LEFT_MIN_ANGLE_DEG, LEFT_MAX_ANGLE_DEG),
    "right": (RIGHT_MIN_ANGLE_DEG, RIGHT_MAX_ANGLE_DEG),
}
SERVO_SETTLE_S = 0.3             # Settle time allowed for servo sweep before spraying
SPRAY_DURATION_S = 0.20          # Keep valve and pump ON for 100ms (0.1s)

# Measured lateral position of each nozzle's straight-down spray point relative
# to the camera centreline (right-positive). These are rig measurements, not
# a symmetric nominal value: left is 55.1 cm and right is 50.0 cm away.
NOZZLE_LATERAL_OFFSETS_M = {
    "left": -0.551,
    "right": +0.500,
}

# Physical aiming calibration, measured on the rig:
#   right: 30° hits camera centre, 90° hits directly below the right nozzle
#   left : 135° hits camera centre, 90° hits directly below the left nozzle
# ``lateral_gap_m`` is target lateral position relative to that nozzle
# (right-positive).  The mapping is deliberately per-side because the two
# linkages have different angle spans.
NOZZLE_ANGLE_CALIBRATION = {
    "right": {
        "nozzle_lateral_m": NOZZLE_LATERAL_OFFSETS_M["right"],
        "angle_at_nozzle_deg": RIGHT_MAX_ANGLE_DEG,
        "angle_at_camera_center_deg": RIGHT_MIN_ANGLE_DEG,
    },
    "left": {
        "nozzle_lateral_m": NOZZLE_LATERAL_OFFSETS_M["left"],
        "angle_at_nozzle_deg": LEFT_MIN_ANGLE_DEG,
        "angle_at_camera_center_deg": LEFT_MAX_ANGLE_DEG,
    },
}


# ------------------------------------------------------------------
# ESP32 UART LINK
# ------------------------------------------------------------------
class ESP32Link:
    """Thread-safe UART connection to ESP32 slave."""

    def __init__(self, port=UART_PORT, baudrate=UART_BAUDRATE):
        self.port = port
        self.baudrate = baudrate
        self.available = False
        self.ser = None
        self.last_error = None
        self._lock = threading.Lock()

        if not _SERIAL_AVAILABLE:
            return

        candidates = [port]
        for fallback in ["/dev/ttyUSB0", "/dev/ttyACM0", "/dev/ttyTHS1"]:
            if fallback not in candidates and os.path.exists(fallback):
                candidates.append(fallback)

        for p in candidates:
            try:
                # Set raw mode on Linux tty to disable echo and terminal translations
                subprocess.run(["stty", "-F", p, str(baudrate), "raw", "-echo"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                self.ser = serial.Serial(p, baudrate, timeout=0.1)
                self.port = p
                self.available = True
                print(f"[ACTUATOR] Connected to ESP32 on {p} @ {baudrate} baud (raw mode).")
                # Initial safety clear: ensure all actuators start OFF
                self.both_off()
                break
            except (serial.SerialException, PermissionError, FileNotFoundError) as exc:
                # Retain the OS error so callers can distinguish a missing
                # controller from a permissions problem.  Previously this was
                # swallowed and every failure looked like a harmless
                # simulation fallback.
                self.last_error = f"{p}: {exc}"

        if not self.available:
            print(f"[ACTUATOR][WARN] Could not connect to UART port {port} ({candidates}).")
            if self.last_error:
                print(f"[ACTUATOR][WARN] Last UART error: {self.last_error}")
            print("[ACTUATOR][HINT] If permission denied: sudo usermod -aG dialout $USER")
            print("[ACTUATOR][SIM] Actuators will run in simulation/logging mode.")

    def _send_packet(self, cmd_type, data1, data2):
        packet = bytes([PACKET_START_BYTE, cmd_type, data1, data2])
        if not self.available or self.ser is None:
            return False
        try:
            with self._lock:
                self.ser.write(packet)
                self.ser.flush()
            return True
        except Exception as exc:
            print(f"[ACTUATOR][WARN] UART transmission failed: {exc}")
            return False

    def set_motor(self, side, angle_deg):
        """CMD_TYPE 0x01 (Motor): [0x00=Left / 0x01=Right] [Angle 0-180]"""
        if side not in SIDE_CODE:
            raise ValueError(f"Unknown motor side: {side!r}")
        low, high = SERVO_LIMITS_DEG[side]
        angle_byte = int(round(max(low, min(high, float(angle_deg)))))
        side_byte = SIDE_CODE[side]
        sent = self._send_packet(CMD_MOTOR, side_byte, angle_byte)
        if not sent:
            print(f"[ACTUATOR][SIM] UART -> MOTOR side={side} ({side_byte}) angle={angle_byte} deg")

    def set_valve(self, side, state_on):
        """Set the physical valve, accounting for the confirmed left/pump wiring swap."""
        if side not in SIDE_CODE:
            raise ValueError(f"Unknown valve side: {side!r}")
        side_byte = SIDE_CODE[side]
        state_byte = VALVE_STATE_ON if state_on else VALVE_STATE_OFF
        # The relay inputs are wired swapped: CMD_PUMP controls the physical
        # left valve; CMD_VALVE/LEFT controls the physical pump.
        if side == "left":
            sent = self._send_packet(CMD_PUMP, state_byte, 0x00)
        else:
            sent = self._send_packet(CMD_VALVE, side_byte, state_byte)
        if not sent:
            state_str = "ON" if state_on else "OFF"
            print(f"[ACTUATOR][SIM] UART -> VALVE side={side} ({side_byte}) state={state_str} (0x{state_byte:02X})")

    def set_pump(self, state_on):
        """Set the physical pump (wired to the firmware's LEFT-valve output)."""
        state_byte = PUMP_STATE_ON if state_on else PUMP_STATE_OFF
        sent = self._send_packet(CMD_VALVE, SIDE_CODE["left"], state_byte)
        if not sent:
            state_str = "ON" if state_on else "OFF"
            print(f"[ACTUATOR][SIM] UART -> PUMP state={state_str} (0x{state_byte:02X})")

    def both_on(self):
        """Sends confirmed batch frame to turn BOTH valves and pump ON:
        b'\\xAA\\x02\\x00\\x00\\xAA\\x02\\x01\\x00\\xAA\\x03\\x00\\x00'
        """
        if not self.available or self.ser is None:
            print("[ACTUATOR][SIM] UART -> BOTH ON (Left Valve ON + Right Valve ON + Pump ON)")
            return False
        try:
            with self._lock:
                self.ser.write(FRAME_BOTH_ON)
                self.ser.flush()
            return True
        except Exception as exc:
            print(f"[ACTUATOR][WARN] UART both_on transmission failed: {exc}")
            return False

    def both_off(self):
        """Sends confirmed batch frame to turn both valves and pump OFF:
        b'\\xAA\\x02\\x00\\x01\\xAA\\x02\\x01\\x01\\xAA\\x03\\x01\\x00'
        """
        if not self.available or self.ser is None:
            print("[ACTUATOR][SIM] UART -> BOTH OFF (Left Valve OFF + Right Valve OFF + Pump OFF)")
            return False
        try:
            with self._lock:
                self.ser.write(FRAME_BOTH_OFF)
                self.ser.flush()
            return True
        except Exception as exc:
            print(f"[ACTUATOR][WARN] UART both_off transmission failed: {exc}")
            return False

    def close(self):
        if self.ser is not None and self.ser.is_open:
            try:
                self.both_off()
                self.ser.close()
            except Exception:
                pass


# ------------------------------------------------------------------
# HARDWARE COMPONENT ABSTRACTIONS
# ------------------------------------------------------------------
class Servo:
    """One side's servo motor driven over UART."""

    def __init__(self, name, link):
        self.name = name
        self.link = link
        self.angle = None

    def set_angle(self, angle_deg):
        low, high = SERVO_LIMITS_DEG[self.name]
        angle_deg = max(low, min(high, float(angle_deg)))
        self.angle = angle_deg
        self.link.set_motor(self.name, angle_deg)

    def park(self):
        self.set_angle(SERVO_CENTER_ANGLE)

    def shutdown(self):
        self.park()


class Valve:
    """One side's solenoid valve driven over UART."""

    def __init__(self, name, link):
        self.name = name
        self.link = link
        self.is_on = False

    def on(self):
        self.is_on = True
        self.link.set_valve(self.name, True)

    def off(self):
        self.is_on = False
        self.link.set_valve(self.name, False)

    def pulse(self, duration_s=SPRAY_DURATION_S):
        self.on()
        time.sleep(duration_s)
        self.off()


class Pump:
    """Shared pump with reference counting."""

    def __init__(self, link):
        self.link = link
        self._lock = threading.Lock()
        self._active_count = 0

    def acquire(self):
        with self._lock:
            self._active_count += 1
            if self._active_count == 1:
                self.link.set_pump(True)

    def release(self):
        with self._lock:
            self._active_count = max(0, self._active_count - 1)
            if self._active_count == 0:
                self.link.set_pump(False)

    def on(self):
        self.link.set_pump(True)

    def off(self):
        with self._lock:
            self._active_count = 0
            self.link.set_pump(False)


def offset_to_angle(lateral_gap_m, side, nozzle_height_m=0.28):
    """Map a target's lateral gap from one nozzle to its calibrated angle.

    ``side`` is required: left and right have independently measured endpoint
    angles and must never share a generic 0–180 degree conversion. The
    endpoint calibration was measured at 0.28 m. For a raised rig, the target
    spray-ray deflection uses atan(lateral_gap / nozzle_height), so changing
    the height changes the required servo angle.
    """
    if side not in NOZZLE_ANGLE_CALIBRATION:
        raise ValueError(f"Unknown nozzle side: {side!r}")
    calibration = NOZZLE_ANGLE_CALIBRATION[side]
    nozzle_height_m = float(nozzle_height_m)
    if nozzle_height_m <= 0:
        raise ValueError("Nozzle height must be positive")
    gap_at_camera_center_m = -calibration["nozzle_lateral_m"]
    # Centre-hit reference ray is based on this side's actual nozzle spacing
    # (55.1 cm left, 50.0 cm right), not a generic 15 cm offset.
    reference_ray_angle = math.atan2(gap_at_camera_center_m, 0.28)
    target_ray_angle = math.atan2(float(lateral_gap_m), nozzle_height_m)
    angle = calibration["angle_at_nozzle_deg"] + (
        (target_ray_angle / reference_ray_angle)
        * (calibration["angle_at_camera_center_deg"] - calibration["angle_at_nozzle_deg"])
    )
    low, high = SERVO_LIMITS_DEG[side]
    return max(low, min(high, angle))


# ------------------------------------------------------------------
# SPRAY ACTUATOR SYSTEM
# ------------------------------------------------------------------
class SprayActuator:
    """Coordinates aiming (servos), spraying (valves), and pump over UART.

    Timing sequence:
      1. Ahead of time: aim(side, angle) pre-rotates the servo.
      2. When reached: fire(side, duration_s=0.10) sends the batch ON command
         b'\\xAA\\x02\\x00\\x00\\xAA\\x02\\x01\\x00\\xAA\\x03\\x00\\x00',
         holds for 100ms, and sends the signal for OFF
         b'\\xAA\\x02\\x00\\x01\\xAA\\x02\\x01\\x01\\xAA\\x03\\x01\\x00'.
    """

    def __init__(self, uart_port=UART_PORT):
        self._link = ESP32Link(port=uart_port)
        self.servos = {side: Servo(side, self._link) for side in ("left", "right")}
        self.valves = {side: Valve(side, self._link) for side in ("left", "right")}
        self.solenoids = self.valves  # Alias for backward compatibility
        self.pump = Pump(self._link)
        self._aim_started_at = {"left": None, "right": None}
        self._is_firing = False
        self._state_lock = threading.Lock()

    def is_active(self):
        """True while valves and pump are actively firing (e.g. during 100ms pulse)."""
        with self._state_lock:
            return self._is_firing

    def aim(self, side, angle_deg):
        """Pre-aims the motor for `side` to angle_deg before the rover reaches the weed."""
        self.servos[side].set_angle(angle_deg)
        self._aim_started_at[side] = time.monotonic()

    def ready_to_fire(self, side):
        """True once SERVO_SETTLE_S has elapsed since the last aim() call."""
        started = self._aim_started_at[side]
        return started is not None and (time.monotonic() - started) >= SERVO_SETTLE_S

    def fire(self, side=None, duration_s=SPRAY_DURATION_S):
        """Fires when the sprayer has reached the weed.

        Sequence:
          1. Sets is_active() -> True for GUI and logs
          2. Sends ON command for both valve and pump:
             b'\\xAA\\x02\\x00\\x00\\xAA\\x02\\x01\\x00\\xAA\\x03\\x00\\x00'
          3. Keeps both ON for duration_s (100ms = 0.10s by default)
          4. Sends signal for OFF:
             b'\\xAA\\x02\\x00\\x01\\xAA\\x02\\x01\\x01\\xAA\\x03\\x01\\x00'
          5. Sets is_active() -> False
        """
        # Older callers pass a side to this method.  Honour it: previously
        # ``fire("left")`` and ``fire("right")`` both opened both valves.
        if side is not None:
            return self.fire_side(side, duration_s=duration_s)

        with self._state_lock:
            self._is_firing = True
        try:
            self._link.both_on()
            time.sleep(duration_s)
            self._link.both_off()
        finally:
            with self._state_lock:
                self._is_firing = False

    def fire_side(self, side, duration_s=SPRAY_DURATION_S):
        """Fires only a single side's valve + pump for duration_s (100ms by default)."""
        if side not in self.valves:
            raise ValueError(f"Unknown spray side: {side!r}")
        with self._state_lock:
            self._is_firing = True
        try:
            self.pump.acquire()
            try:
                self.valves[side].on()
                time.sleep(duration_s)
            finally:
                try:
                    self.valves[side].off()
                finally:
                    self.pump.release()
        finally:
            with self._state_lock:
                self._is_firing = False

    def all_on(self):
        """Directly sends the confirmed batch ON command."""
        self._link.both_on()

    def all_off(self):
        """Directly sends the batch OFF command."""
        self._link.both_off()

    def park_all(self):
        for servo in self.servos.values():
            servo.park()

    def shutdown(self):
        self.park_all()
        self.all_off()
        self._link.close()

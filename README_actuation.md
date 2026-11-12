# Two-motor / two-solenoid sprayer actuation — what changed and what to confirm

## Files touched
- **`detection_worldcoord.py`** — camera-centric calibration fix + principal-point auto-centering.
- **`sprayer_control.py`** — two-sprayer geometry, per-side aim-then-fire control loop, fixed a real local→global coordinate bug.
- **`spray_actuation.py`** *(new)* — hardware driver: two servos (PWM) + two solenoids (GPIO), generalized from your `servo_control.py`.
- `servo_control.py` and `weed_sprayer_gui.py` were **not modified**.

---

## 1. Hardware pins — CONFIRM before powering anything

| Role | Interface | Value | Status |
|---|---|---|---|
| Right servo | sysfs PWM | `/sys/class/pwm/pwmchip2/pwm0/` | ✅ your validated path from `servo_control.py` |
| Left servo | sysfs PWM | `/sys/class/pwm/pwmchip0/pwm0/` | ⚠️ **guess** — common second PWM export on Orin devkits, unverified on your board |
| Right solenoid | GPIO (BOARD numbering) | pin 15 | ⚠️ **guess** — pick any free header pin |
| Left solenoid | GPIO (BOARD numbering) | pin 16 | ⚠️ **guess** — pick any free header pin |

All four are set in `spray_actuation.py`'s `PWM_PATHS` / `SOLENOID_PINS` dicts at the top of the file. Before wiring:
```bash
ls /sys/class/pwm/          # confirm pwmchip0 exists and is exported; if not, enable via jetson-io
```
If `pwmchip0` isn't available on your carrier board, tell me the real second PWM channel and I'll swap it in.

## 2. Other placeholder constants (all in `spray_actuation.py` / `sprayer_control.py`)

| Constant | Current value | What it means |
|---|---|---|
| `SPRAYER_OFFSETS` | X/Y remain `{"right": {x:+0.15, y:+0.05}, "left": {x:-0.15, y:+0.05}}` (m); both Z values are now −0.14 m | The camera is 0.42 m and both nozzles are 0.28 m above ground. Measure each nozzle's signed lateral X and forward Y coordinates before replacing their placeholders. |
| `MOUNT_REACH_M` | 0.15 m | Measured camera-centre to each nozzle's straight-down ground point. Angle mapping is side-specific: right 30°=centre/90°=right nozzle; left 135°=centre/90°=left nozzle. |
| `SERVO_SETTLE_S` | 0.6 s | Worst-case 0→180° travel time (open loop, no encoder) — **time your actual servo** and set with margin |
| `SPRAY_DURATION_S` | 0.10 s | How long the solenoid stays open per shot |
| `CAMERA_HEIGHT_M` | 0.42 m | Measured camera mounting height above ground |

## 3. Calibration fix (`detection_worldcoord.py`)

Two real bugs, not just missing values:

- **Principal point (`cx`,`cy`)** was `320, 320`, almost certainly from a smaller calibration resolution. Now auto-corrected to the real frame's center the first time a frame is seen (`align_principal_point()`, called from `main()`). Set `AUTO_CENTER_PRINCIPAL_POINT = False` once you have a real calibrated principal point you don't want overridden.
- **Extrinsics translation `t` had nonzero X/Y** (`-9.93, -4.90`), meaning your old calibration's world origin was an arbitrary chessboard point, not the camera — contradicting the file's own "camera-centric" assumption. `SIMPLE_CALIBRATION = True` now forces `R = identity`, `t = [0, 0, CAMERA_HEIGHT_M]`, so the camera is genuinely at world (0,0). **When you redo real calibration, keep `t_x = t_y = 0` by construction** — only fit `R` for actual tilt/pitch and `t_z` for height, rather than fitting `t` freely against a freestanding chessboard.

## 4. A real bug found while testing: local→global rotation

`camera_local_to_global()` and `sprayer_global_position()` were rotating detections as if the local frame were (forward, left), but `detection_worldcoord.py` actually outputs (X = lateral-right, Y = forward) — swapped axes. Concretely: with the rover facing 90°, a weed detected 1m to the camera's right was landing **directly ahead** of the rover in the map frame instead of to its side. Both functions are fixed now (verified with a unit test — see conversation). This wasn't something I introduced; it predates this change and would have silently broken position tracking and motor routing as soon as the rover made any turns.

## 5. Algorithm for multiple weed detections (your question)

**Before:** a single shared tolerance check fired every weed within `SPRAY_TOLERANCE_M` of one shared nozzle position, all in the same loop tick — fine for a fixed nozzle, but physically impossible for an aimed servo (can't hold two angles at once).

**Now**, per side, independently:
1. Among not-yet-sprayed weeds routed to that side (`spray_direction()`), take the **nearest-ahead** one (smallest forward gap from the rover, ignoring ones already passed).
2. As soon as it becomes that side's target, **pre-aim** the servo at it immediately — well before the rover physically reaches it. If the next target needs an angle within `ANGLE_MERGE_TOLERANCE_DEG` (5°) of where the servo already is, the aim/settle step is skipped entirely, since the motor doesn't need to move — this is what lets tightly-clustered same-side weeds get sprayed back-to-back instead of each paying the full settle penalty.
3. Fire the solenoid only once **both** hold: the servo's settle time has elapsed since aiming (`ready_to_fire()`), and the sprayer's position is within tolerance of the weed.
4. Immediately re-aim at whichever weed is now nearest on that side.

### Multiple weeds at the same horizontal position, specifically
- **Opposite sides, same forward position** — handled fully in parallel; left/right are two independent threads with their own servo+solenoid, no conflict.
- **Same side, same/near-identical lateral offset** — handled efficiently via the angle-merge skip above (#2).
- **Same side, genuinely different lateral offsets, same forward position** — this is a real hardware limit, not fixable in software alone: one servo can only hold one angle. At the mocked rover speed (2 km/h) and current constants, the sprayer is only "within tolerance" of a given weed for **~90ms**, but a fresh servo aim+settle takes **600ms** — so while the first weed on that side is being serviced, the rover travels ~0.33m, almost certainly carrying it past the second weed's window. This now gets **logged explicitly** (`[MISS] id=... side=... -- rover passed it unsprayed`) instead of silently aging out after 15s, so you'll actually see when it happens. Fixing it for real needs a second nozzle per side, a wider blanket spray pattern per shot, or slowing the rover through a same-side backlog — none of which this module controls today.

This is a strict one-target-at-a-time-per-side queue ordered by distance — within a side, weeds are handled strictly in the order the rover will reach them, and the motor is always given its full settle time before spraying (unless already aimed there), per your instruction.

## 5b. A second bug found while testing the "same horizontal" case

The servo-angle math (`offset_to_angle(target["x"] - sx)`) used the raw global-X difference as "lateral offset" — correct only when the rover's heading is exactly 0°. At any other heading this silently reported the wrong lateral gap (verified: a weed truly 0.3m to the right reported as 0.0m offset at heading 0 after a sign/axis mixup, and more generally wrong at any nonzero heading). Replaced with a proper `lateral_gap()` helper that projects the target-minus-sprayer vector onto the rover's current left/right axis — the same class of rotation-frame bug as the `camera_local_to_global` fix, caught here in new code via the same kind of unit test. Verified against a heading=90° case (see conversation).


## 6. What's still missing / needs your decision

1. **Optical-flow rover tracking exists but isn't wired everywhere.** `DeepStreamCamera.get_optical_flow()` → `SprayerController.update_optical_flow()` is real, working, hardware-based position tracking — but today it's only called from `weed_sprayer_gui.py`'s worker thread. If you run `detection_worldcoord.py` and `sprayer_control.py` as the two separate processes their docstrings describe, the sprayer process falls back to the mocked constant-speed `read_odometry()` and never sees real optical flow. **Recommend:** run detection + spraying in one process (like the GUI does) so timing-critical aiming has live position data. If you need them split across processes, I can publish flow data through the shared JSONL file instead — let me know which you want.
2. **`SIMPLE_CALIBRATION` is a stand-in, not real calibration.** It assumes zero camera tilt. Real-world accuracy needs your actual pitch angle in `R` once you calibrate.
3. **No hardware-in-the-loop testing was possible** — I don't have your Jetson. Both new/changed files degrade to console-logging no-ops when PWM/GPIO/DeepStream aren't available (same pattern the rest of this codebase already uses), so they're safe to `python3 -m py_compile` and dry-run on any machine, but the actual aiming/firing behavior, `SERVO_SETTLE_S`, and `MOUNT_REACH_M` all need verification on the real rig.
4. **No encoder/limit-switch feedback on the servos.** Aiming is fully open-loop (timed settle only). If a servo stalls or the arm is obstructed, the code has no way to know and will still fire the solenoid after `SERVO_SETTLE_S`. Worth adding a limit switch or position feedback if false-positive spraying (wrong spot) is a concern.

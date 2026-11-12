#!/usr/bin/env python3
"""Manual left/right sprayer angle calibration utility.

This tool deliberately does not infer calibration values: the current rig has
no servo-position feedback or spray-spot camera, so only a physical test can
establish each nozzle's safe minimum, maximum, and centre angles.  It aims one
selected servo and fires only that side's valve when you explicitly press
Enter.

Use water and paper/card targets while calibrating.  Keep hands clear of the
linkage and spray direction.
"""

import argparse
import sys
import time

from spray_actuation import SPRAY_DURATION_S, SERVO_LIMITS_DEG, SERVO_SETTLE_S, SprayActuator


SIDE_NAMES = {"l": "left", "r": "right"}


def read_side():
    """Read a side selection, or return None when the operator exits."""
    while True:
        value = input("Side [l=left, r=right, q=quit]: ").strip().lower()
        if value in ("q", "quit", "exit"):
            return None
        if value in SIDE_NAMES:
            return SIDE_NAMES[value]
        print("Enter l, r, or q.")


def read_angle(side):
    """Read a valid ESP32 servo angle."""
    while True:
        low, high = SERVO_LIMITS_DEG[side]
        value = input(f"Servo angle [{low:.0f}-{high:.0f}]: ").strip()
        try:
            angle = float(value)
        except ValueError:
            print(f"Enter a number from {low:.0f} to {high:.0f}.")
            continue
        if low <= angle <= high:
            return angle
        print(f"Angle must be between {low:.0f} and {high:.0f}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--duration", type=float, default=SPRAY_DURATION_S,
        help=f"single-side spray duration in seconds (default: {SPRAY_DURATION_S})",
    )
    parser.add_argument(
        "--settle", type=float, default=SERVO_SETTLE_S,
        help=f"wait after aiming before enabling the spray prompt (default: {SERVO_SETTLE_S})",
    )
    parser.add_argument(
        "--simulate", action="store_true",
        help="allow a dry-run when the ESP32 UART controller is unavailable",
    )
    args = parser.parse_args()
    if args.duration <= 0 or args.settle < 0:
        parser.error("--duration must be positive and --settle cannot be negative")

    print("Manual sprayer-angle calibration")
    print("Use WATER only. Each test: select side, enter angle, then press Enter to spray that side.")
    print("Find and write down, for EACH side: safe min angle, centre-hit angle, safe max angle.")

    actuator = SprayActuator()
    if not actuator._link.available:
        if not args.simulate:
            print("ERROR: ESP32 UART controller is unavailable; calibration will not move motors.")
            if actuator._link.last_error:
                print(f"UART error: {actuator._link.last_error}")
            print("Fix the UART connection/permissions, then log out and back in before retrying.")
            print("Use --simulate only for a dry-run with no hardware motion.")
            actuator.shutdown()
            return 2
        print("WARNING: simulation mode; this session will not move or spray hardware.")

    try:
        while True:
            side = read_side()
            if side is None:
                break
            angle = read_angle(side)

            actuator.aim(side, angle)
            print(f"Aiming {side} servo at {angle:.1f} degrees; waiting {args.settle:.2f}s.")
            time.sleep(args.settle)
            confirm = input(
                f"Press Enter to spray {side} for {args.duration:.2f}s at {angle:.1f}deg; "
                "type s to skip: "
            ).strip().lower()
            if confirm:
                print("Spray skipped. The servo remains aimed for visual inspection.")
                continue

            print(f"SPRAY: {side}, angle={angle:.1f}deg, duration={args.duration:.2f}s")
            actuator.fire_side(side, duration_s=args.duration)
            print("Spray complete. Measure the wet-spot centre and record the angle.")
    except (KeyboardInterrupt, EOFError):
        print("\nCalibration stopped by operator.")
    finally:
        print("Turning valves/pump off and parking both servos.")
        actuator.shutdown()


if __name__ == "__main__":
    sys.exit(main() or 0)

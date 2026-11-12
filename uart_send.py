#!/usr/bin/env python3
"""
uart_send.py - type hex bytes, they go straight to the ESP32 UART.

  python3 uart_send.py                      # /dev/ttyTHS1 @ 115200
  python3 uart_send.py --port /dev/ttyUSB0

At the prompt:
  AA 03 00 00 AA 02 00 00     send these bytes (spaces optional: AA030000AA020000)
  off   send documented ALL OFF   AA 02 00 01 AA 02 01 01 AA 03 01 00
  on    send documented BOTH ON   AA 02 00 00 AA 02 01 00 AA 03 00 00
  lp    pump ON then left valve ON    AA 02 00 00 AA 03 00 00
  rp    pump ON then right valve ON   AA 02 00 00 AA 02 01 00
  q     quit (sends ALL OFF first)

Nothing is switched off automatically after a send. You decide when to send off.
Any bytes the ESP32 sends back are printed after each send.
"""
import argparse
import os
import subprocess
import sys
import time

try:
    import serial
except ImportError:
    sys.exit("pyserial is not installed. Run: pip install pyserial")

ALL_OFF = bytes.fromhex("AA020001 AA020101 AA030100")
SHORTCUTS = {
    "off": ALL_OFF,
    "on": bytes.fromhex("AA020000 AA020100 AA030000"),
    "lp": bytes.fromhex("AA020000 AA030000"),
    "rp": bytes.fromhex("AA020000 AA020100"),
}
FALLBACKS = ["/dev/ttyUSB0", "/dev/ttyACM0", "/dev/ttyTHS1"]


def hexs(b):
    return " ".join(f"{x:02X}" for x in b)


def open_port(port, baud):
    last = None
    for p in [port] + [f for f in FALLBACKS if f != port and os.path.exists(f)]:
        try:
            subprocess.run(["stty", "-F", p, str(baud), "raw", "-echo"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            ser = serial.Serial(p, baud, timeout=0.2)
            print(f"Connected to {p} @ {baud} baud.")
            if p.startswith(("/dev/ttyUSB", "/dev/ttyACM")):
                time.sleep(2.0)  # opening a USB port usually resets the ESP32
            return ser
        except Exception as exc:  # noqa: BLE001
            last = f"{p}: {exc}"
    sys.exit(f"Could not open a serial port. Last error: {last}\n"
             "If permission denied: sudo usermod -aG dialout $USER")


def send(ser, data):
    ser.reset_input_buffer()
    ser.write(data)
    ser.flush()
    print(f"  sent: {hexs(data)}")
    time.sleep(0.15)
    reply = ser.read(256)
    if reply:
        print(f"  reply: {hexs(reply)}   {reply!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyTHS1")
    ap.add_argument("--baud", type=int, default=115200)
    args = ap.parse_args()

    ser = open_port(args.port, args.baud)
    print(__doc__)
    try:
        while True:
            try:
                text = input("uart> ").strip()
            except EOFError:
                break
            low = text.lower()
            if not text:
                continue
            if low == "q":
                break
            if low in SHORTCUTS:
                send(ser, SHORTCUTS[low])
                continue
            try:
                data = bytes.fromhex(text.replace("0x", " "))
            except ValueError:
                print("  Not valid hex. Example: AA 03 00 00 AA 02 00 00")
                continue
            if data:
                send(ser, data)
    except KeyboardInterrupt:
        print()
    finally:
        try:
            ser.write(ALL_OFF)
            ser.flush()
            ser.close()
            print("ALL OFF sent, port closed.")
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    main()

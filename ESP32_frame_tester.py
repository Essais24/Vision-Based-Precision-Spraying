#!/usr/bin/env python3
#!/usr/bin/env python3
"""
esp32_frame_tester.py
=====================
Standalone interactive tester for the ESP32 sprayer UART protocol.
Nothing here imports spray_actuation.py / servo_control.py / sprayer_control.py,
and none of those files are touched.

GOAL
  Find out which byte sequences make
    - the LEFT  solenoid + the pump actuate together
    - the RIGHT solenoid + the pump actuate together

FRAME FORMAT (4 bytes):   [0xAA] [CMD] [DATA1] [DATA2]
  CMD 0x02 = valve : DATA1 = side (0=Left, 1=Right), DATA2 = state
  CMD 0x03 = pump  : DATA1 = state, DATA2 = dummy
  (documented: state 0x00 = ON, 0x01 = OFF -- active-LOW relays)
  CMD 0x01 (servo motor) is NOT tested here, so the servos will not move.

WHAT IT TESTS (all enumerated from the frame format above)
  A. The documented sequences, first.
  B. Every single valve frame and every single pump frame (8 frames).
  C. Every valve frame (4) combined with every pump frame (4), sent together
     as one two-frame message. Order can be pump-first, valve-first or both.
  D. Both valves + each pump frame (3-frame messages).

HOW IT WORKS
  1. Shows the next test and its hex bytes.
  2. You press Enter -> the bytes are sent, held for --hold seconds, then the
     documented ALL-OFF message is sent automatically.
  3. You type what you saw/heard:
        l = left valve        r = right valve        p = pump
        v = a valve, but you can't tell which one
        all = everything (left valve + right valve + pump)
        n = nothing
        rep = repeat the test        Enter = don't record
     Combine letters for mixed cases, e.g. lp, rp, or vp (a valve + pump).
     v can't be combined with l or r.
  4. At the end (or on q) it prints which sequences gave exactly Left+Pump,
     exactly Right+Pump, ALL, a valve (+ pump) with the side unknown, and
     everything else, and saves every answer to a CSV file.

SAFETY
  - Run the pump with water in the tank. Short holds (default 1.0 s) help.
  - The ALL-OFF message uses the documented OFF values. If a test leaves the pump
    running (because its state encoding is not what the docs say), press x to
    resend ALL OFF, or cut power / reset the ESP32.
  - Ctrl+C always sends ALL OFF before exiting.

USAGE
  python3 esp32_frame_tester.py                      # /dev/ttyTHS1 @ 115200
  python3 esp32_frame_tester.py --port /dev/ttyUSB0
  python3 esp32_frame_tester.py --dry-run            # print only, no serial
  python3 esp32_frame_tester.py --order pump-first    # fewer tests
  python3 esp32_frame_tester.py --gap-ms 20           # send frames separately
                                                      # 20 ms apart, not in one write
  python3 esp32_frame_tester.py --byte-gap-ms 5       # diagnostic: one byte at a time
  python3 esp32_frame_tester.py --rx-window-ms 250    # retain ESP32 replies in the CSV

COMMANDS AT THE PROMPT
  Enter  send this test          s  skip          b  go back one
  j N    jump to test N          l  list all      x  send ALL OFF now
  h HEX  send raw bytes, e.g.  h AA 02 00 00 AA 03 00 00
  q      quit and show summary
"""

import argparse
import csv
import os
import subprocess
import sys
import time
from datetime import datetime

try:
    import serial
except ImportError:
    serial = None

START = 0xAA
CMD_VALVE = 0x02
CMD_PUMP = 0x03
LEFT, RIGHT = 0x00, 0x01
SIDE_NAME = {LEFT: "LEFT", RIGHT: "RIGHT"}

DEFAULT_PORT = "/dev/ttyTHS1"
FALLBACK_PORTS = ["/dev/ttyUSB0", "/dev/ttyACM0", "/dev/ttyTHS1"]
DEFAULT_BAUD = 115200

# Documented batch frames (from spray_actuation.py).
BOTH_ON = bytes.fromhex("AA 02 00 00  AA 02 01 00  AA 03 00 00")
ALL_OFF = bytes.fromhex("AA 02 00 01  AA 02 01 01  AA 03 01 00")


def frame(cmd, d1, d2):
    return bytes([START, cmd, d1, d2])


def hexs(data):
    return " ".join(f"{b:02X}" for b in data)


class Test:
    def __init__(self, group, label, data):
        self.group = group
        self.label = label
        self.data = data


def valve_frames():
    out = []
    for side in (LEFT, RIGHT):
        for state in (0x00, 0x01):
            out.append((frame(CMD_VALVE, side, state),
                        f"{SIDE_NAME[side]} valve state=0x{state:02X}"))
    return out


def pump_frames():
    out = []
    for d1 in (0x00, 0x01):
        for d2 in (0x00, 0x01):
            out.append((frame(CMD_PUMP, d1, d2), f"pump d1=0x{d1:02X} d2=0x{d2:02X}"))
    return out


def build_tests(order):
    tests, seen = [], set()

    def add(group, label, data):
        if data in seen:
            return
        seen.add(data)
        tests.append(Test(group, label, data))

    # A. documented sequences
    add("A documented", "BOTH valves + pump (documented batch ON)", BOTH_ON)
    add("A documented", "LEFT valve + pump (pump ON, then left valve ON, as fire_side does)",
        frame(CMD_PUMP, 0x00, 0x00) + frame(CMD_VALVE, LEFT, 0x00))
    add("A documented", "RIGHT valve + pump (pump ON, then right valve ON, as fire_side does)",
        frame(CMD_PUMP, 0x00, 0x00) + frame(CMD_VALVE, RIGHT, 0x00))

    # B. single frames
    for data, label in valve_frames():
        add("B single frame", label, data)
    for data, label in pump_frames():
        add("B single frame", label, data)

    # C. every valve frame with every pump frame
    for vdata, vlabel in valve_frames():
        for pdata, plabel in pump_frames():
            if order in ("pump-first", "both"):
                add("C valve + pump", f"{plabel}  THEN  {vlabel}", pdata + vdata)
            if order in ("valve-first", "both"):
                add("C valve + pump", f"{vlabel}  THEN  {plabel}", vdata + pdata)

    # D. both valves ON + each pump frame
    for pdata, plabel in pump_frames():
        add("D both valves + pump",
            f"LEFT valve 0x00 + RIGHT valve 0x00  THEN  {plabel}",
            frame(CMD_VALVE, LEFT, 0x00) + frame(CMD_VALVE, RIGHT, 0x00) + pdata)
    return tests


class Link:
    """Serial link, or a print-only stand-in when no hardware is available."""

    def __init__(self, port, baud, dry_run):
        self.ser = None
        self.dry = True
        if dry_run:
            print("[DRY RUN] No serial port will be opened.")
            return
        if serial is None:
            print("[WARN] pyserial is not installed, so this is a dry run. "
                  "Install with: pip install pyserial")
            return
        candidates = [port] + [p for p in FALLBACK_PORTS if p != port and os.path.exists(p)]
        last_err = None
        for p in candidates:
            try:
                subprocess.run(["stty", "-F", p, str(baud), "raw", "-echo"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                self.ser = serial.Serial(p, baud, timeout=0.1)
                self.dry = False
                print(f"[OK] Connected to {p} @ {baud} baud.")
                if p.startswith(("/dev/ttyUSB", "/dev/ttyACM")):
                    time.sleep(2.0)  # opening a USB port usually resets the ESP32
                return
            except Exception as exc:  # noqa: BLE001
                last_err = f"{p}: {exc}"
        print(f"[WARN] Could not open any port ({candidates}). Last error: {last_err}")
        print("[WARN] Running as a dry run. If permission denied: sudo usermod -aG dialout $USER")

    def read_for(self, window_ms):
        """Collect every byte the controller emits during a short time window."""
        if self.dry or self.ser is None or window_ms <= 0:
            return b""
        deadline = time.monotonic() + window_ms / 1000.0
        reply = bytearray()
        while time.monotonic() < deadline:
            waiting = self.ser.in_waiting
            if waiting:
                reply.extend(self.ser.read(waiting))
                # Give a line-oriented firmware a moment to emit its last bytes.
                deadline = max(deadline, time.monotonic() + 0.015)
            else:
                time.sleep(0.005)
        return bytes(reply)

    def clear_input(self):
        if not self.dry and self.ser is not None:
            self.ser.reset_input_buffer()

    def send(self, data, gap_ms=0, byte_gap_ms=0):
        if self.dry:
            return
        if byte_gap_ms:
            for value in data:
                self.ser.write(bytes([value]))
                self.ser.flush()
                time.sleep(byte_gap_ms / 1000.0)
        elif gap_ms and len(data) > 4:
            for i in range(0, len(data), 4):
                self.ser.write(data[i:i + 4])
                self.ser.flush()
                time.sleep(gap_ms / 1000.0)
        else:
            self.ser.write(data)
            self.ser.flush()

    def close(self):
        if self.ser is not None:
            try:
                self.send(ALL_OFF)
                self.ser.close()
            except Exception:  # noqa: BLE001
                pass


def parse_observation(text):
    """Turn what you typed into a stored code.

      n        -> 'none'   (nothing happened)
      all      -> 'LPR'    (left valve + right valve + pump)
      v        -> 'V'      (a valve, but which one is unknown)
      l r p    -> combine freely: lp -> 'LP', rp -> 'PR', lrp -> 'LPR'
      vp       -> 'PV'     (a valve + pump, side unknown)
    Returns None if the text can't be understood (including v mixed with l/r).
    """
    text = text.strip().lower()
    if text == "n":
        return "none"
    if text == "all":
        return "LPR"
    cleaned = text.replace(" ", "")
    if not cleaned or any(c not in "lrpv" for c in cleaned):
        return None
    letters = set(cleaned)
    if "v" in letters and ("l" in letters or "r" in letters):
        return None
    return "".join(sorted(c.upper() for c in letters))


def describe(obs):
    names = {"L": "left valve", "R": "right valve", "P": "pump",
             "V": "a valve (side not identified)"}
    if obs == "none":
        return "nothing"
    if obs == "LPR":
        return "ALL (left valve + right valve + pump)"
    return " + ".join(names[c] for c in obs)


def save_row(path, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["timestamp", "test_no", "group", "label", "hex_bytes", "observed"])
        w.writerow(row)


def save_trace_row(path, row):
    """Keep transport diagnostics separate so older results CSVs remain readable."""
    new = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["timestamp", "test_no", "hex_bytes", "gap_ms", "byte_gap_ms",
                        "pre_off_reply", "test_reply", "off_reply"])
        w.writerow(row)


def run_test(link, test, number, args):
    """Send, hold, switch off, then ask what was seen. Returns observation or None."""
    while True:
        print(f"   TX  {hexs(test.data)}")
        # A separate OFF/reset before every trial is important: otherwise a
        # latched relay from the preceding trial can be mistaken for this one.
        link.clear_input()
        link.send(ALL_OFF, args.gap_ms, args.byte_gap_ms)
        pre_reply = link.read_for(args.rx_window_ms)
        if pre_reply:
            print(f"   RX after pre-OFF: {hexs(pre_reply)}")
        if args.pre_off_settle:
            print(f"   Settling OFF state for {args.pre_off_settle:.2f} s ...")
            time.sleep(args.pre_off_settle)
        try:
            link.clear_input()
            link.send(test.data, args.gap_ms, args.byte_gap_ms)
            test_reply = link.read_for(args.rx_window_ms)
            if test_reply:
                print(f"   RX after test: {hexs(test_reply)}")
            print(f"   ON for {args.hold:.1f} s ... watch the relays and pump")
            time.sleep(args.hold)
        finally:
            link.clear_input()
            link.send(ALL_OFF, args.gap_ms, args.byte_gap_ms)
            off_reply = link.read_for(args.rx_window_ms)
            print("   ALL OFF sent")
            if off_reply:
                print(f"   RX after OFF: {hexs(off_reply)}")
        print("   What actuated?  l=left valve  r=right valve  p=pump  "
              "v=a valve (not sure which)")
        print("                   all=everything  n=nothing  rep=repeat  Enter=skip recording")
        while True:
            ans = input("   > ").strip()
            if ans == "" or ans.lower() == "rep":
                break
            obs = parse_observation(ans)
            if obs is not None:
                break
            print("   Not understood. Use l, r, p, v, all, n or rep (v can't be mixed with l or r).")
        if ans.lower() == "rep":
            continue
        if ans == "":
            return None
        timestamp = datetime.now().isoformat(timespec="seconds")
        save_row(args.results, [timestamp, number, test.group, test.label, hexs(test.data), obs])
        save_trace_row(args.trace_results, [timestamp, number, hexs(test.data), args.gap_ms,
                                             args.byte_gap_ms, hexs(pre_reply),
                                             hexs(test_reply), hexs(off_reply)])
        print(f"   Recorded: {describe(obs)}")
        return obs


def print_summary(records):
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    if not records:
        print("No observations were recorded.")
        return

    def show(title, wanted):
        hits = [r for r in records if r["obs"] == wanted]
        print(f"\n{title}")
        if not hits:
            print("   none found yet")
        for r in hits:
            print(f"   #{r['no']:>2}  {hexs(r['data'])}")
            print(f"        {r['label']}")

    show("Exactly LEFT valve + PUMP:", "LP")
    show("Exactly RIGHT valve + PUMP:", "PR")
    show("ALL (left valve + right valve + pump):", "LPR")
    show("A valve + PUMP, side not identified (vp):", "PV")
    show("A valve only, side not identified (v):", "V")
    others = [r for r in records if r["obs"] not in ("LP", "PR", "LPR", "PV", "V")]
    print("\nEverything else recorded:")
    if not others:
        print("   (nothing)")
    for r in others:
        print(f"   #{r['no']:>2}  {describe(r['obs']):<24} {hexs(r['data'])}")
    print("\nAll answers are also saved in the CSV results file.")


def main():
    ap = argparse.ArgumentParser(description="Interactive ESP32 sprayer frame tester")
    ap.add_argument("--port", default=DEFAULT_PORT)
    ap.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    ap.add_argument("--hold", type=float, default=1.0, help="seconds to keep each test ON")
    ap.add_argument("--gap-ms", type=int, default=0,
                    help="if >0, send each 4-byte frame in a separate write with this gap")
    ap.add_argument("--byte-gap-ms", type=int, default=0,
                    help="diagnostic: send every byte in its own write with this gap")
    ap.add_argument("--pre-off-settle", type=float, default=0.25,
                    help="seconds to wait after the reset/OFF before each test")
    ap.add_argument("--rx-window-ms", type=int, default=100,
                    help="milliseconds to collect ESP32 serial replies after each transmission")
    ap.add_argument("--order", choices=["both", "pump-first", "valve-first"], default="both",
                    help="order of frames in valve+pump tests")
    ap.add_argument("--results", default="esp32_frame_results.csv")
    ap.add_argument("--trace-results", default=None,
                    help="CSV for UART replies/timing (default: <results>.trace.csv)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.trace_results is None:
        root, ext = os.path.splitext(args.results)
        args.trace_results = f"{root}.trace{ext or '.csv'}"

    tests = build_tests(args.order)
    link = Link(args.port, args.baud, args.dry_run)
    link.send(ALL_OFF, args.gap_ms, args.byte_gap_ms)
    records = []
    print(f"\n{len(tests)} tests ready. Results file: {args.results}")
    print(f"UART diagnostic trace: {args.trace_results}")
    print("Type l to list them all, q to quit.\n")

    i = 0
    try:
        while i < len(tests):
            t = tests[i]
            print(f"[{i + 1}/{len(tests)}] {t.group}")
            print(f"   {t.label}")
            print(f"   bytes: {hexs(t.data)}")
            cmd = input("   Enter=send  s=skip  b=back  j N=jump  l=list  h HEX=raw  x=all off  q=quit > ").strip()
            low = cmd.lower()
            if low == "":
                obs = run_test(link, t, i + 1, args)
                if obs is not None:
                    records.append({"no": i + 1, "label": t.label, "data": t.data, "obs": obs})
                i += 1
            elif low == "s":
                i += 1
            elif low == "b":
                i = max(0, i - 1)
            elif low == "l":
                for n, x in enumerate(tests, 1):
                    print(f"   {n:>2}. [{x.group}] {hexs(x.data)}   {x.label}")
            elif low == "x":
                link.send(ALL_OFF)
                print("   ALL OFF sent")
            elif low.startswith("j"):
                try:
                    n = int(low[1:].strip())
                    if 1 <= n <= len(tests):
                        i = n - 1
                    else:
                        print(f"   Pick a number from 1 to {len(tests)}.")
                except ValueError:
                    print("   Usage: j 12")
            elif low.startswith("h"):
                try:
                    raw = bytes.fromhex(cmd[1:].strip())
                except ValueError:
                    print("   Could not read that as hex. Example: h AA 02 00 00 AA 03 00 00")
                    continue
                if not raw:
                    continue
                custom = Test("raw", "raw bytes typed by you", raw)
                obs = run_test(link, custom, 0, args)
                if obs is not None:
                    records.append({"no": 0, "label": custom.label, "data": raw, "obs": obs})
            elif low == "q":
                break
            else:
                print("   Unknown command.")
        else:
            print("\nAll tests done.")
    except (KeyboardInterrupt, EOFError):
        print("\nInterrupted.")
    finally:
        link.close()
    print_summary(records)


if __name__ == "__main__":
    main()
